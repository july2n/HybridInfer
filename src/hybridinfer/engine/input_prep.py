"""GPU input preparation shared by prefill, decode and mixed batches."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(tokens, computed, tables, slots, offsets, counts,
             input_ids, positions, mapping, lengths, cu_k,
             TOKEN_WIDTH: tl.constexpr, TABLE_WIDTH: tl.constexpr,
             BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    start = tl.load(computed + slot)
    count = tl.load(counts + row)
    offset = tl.load(offsets + row)
    local = tl.program_id(1) * TILE + tl.arange(0, TILE)
    pos = start + local
    mask = local < count
    token = tl.load(tokens + slot * TOKEN_WIDTH + pos, mask, other=0)
    block = tl.load(tables + slot * TABLE_WIDTH + pos // BLOCK_SIZE, mask, other=-1)
    tl.store(input_ids + offset + local, token, mask)
    tl.store(positions + offset + local, pos, mask)
    tl.store(mapping + offset + local, tl.where(block < 0, -1, block * BLOCK_SIZE + pos % BLOCK_SIZE), mask)
    if tl.program_id(1) == 0:
        tl.store(lengths + row, start + count)
        # The cumulative K lengths are derived on GPU by a subsequent cumsum.
        tl.store(cu_k + row, start + count)


@triton.jit
def _advance(computed, slots, counts, N: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = i < N
    slot = tl.load(slots + i, mask, other=0)
    count = tl.load(counts + i, mask, other=0)
    value = tl.load(computed + slot, mask, other=0)
    tl.store(computed + slot, value + count, mask)


def prepare_inputs(state, slots, offsets, counts, num_tokens, max_query_len, block_size):
    device = slots.device
    ids = torch.empty(num_tokens, dtype=torch.int64, device=device)
    positions = torch.empty_like(ids)
    mapping = torch.empty(num_tokens, dtype=torch.int32, device=device)
    lengths = torch.empty(slots.numel(), dtype=torch.int32, device=device)
    k_lengths = torch.empty_like(lengths)
    _prepare[(slots.numel(), triton.cdiv(max_query_len, 256))](
        state.tokens.tensor, state.computed.tensor, state.block_tables.tensor,
        slots, offsets, counts, ids, positions, mapping, lengths, k_lengths,
        state.capacity, state.block_tables.tensor.shape[1], block_size, 256,
    )
    cu_k = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), k_lengths.cumsum(0, dtype=torch.int32)))
    tables = state.block_tables.tensor.index_select(0, slots)
    return ids, positions, mapping, lengths, cu_k, tables


def advance(state, slots, counts):
    _advance[(triton.cdiv(slots.numel(), 128),)](state.computed.tensor, slots, counts, slots.numel(), 128)


@triton.jit
def _commit(tokens, computed, slots, sampled, emit, last_tokens,
            WIDTH: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = row < N
    slot = tl.load(slots + row, mask, other=0)
    position = tl.load(computed + slot, mask, other=0)
    token = tl.load(sampled + row, mask, other=0)
    valid = tl.load(emit + row, mask, other=0)
    tl.store(tokens + slot * WIDTH + position, token, mask & (valid != 0))
    tl.store(last_tokens + slot, token, mask & (valid != 0))


def commit_sampled(state, slots, sampled, emit, last_tokens):
    _commit[(triton.cdiv(slots.numel(), 128),)](
        state.tokens.tensor, state.computed.tensor, slots, sampled, emit,
        last_tokens, state.capacity, slots.numel(), 128,
    )
