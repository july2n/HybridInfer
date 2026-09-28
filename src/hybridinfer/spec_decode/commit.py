"""Commit only valid output tokens to resident GPU history."""
import torch
import triton
import triton.language as tl


@triton.jit
def _commit(tokens, computed, last, slots, output, lengths, endpoints,
            CAPACITY: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots+row)
    length = tl.load(lengths+row)
    endpoint = tl.load(endpoints+row)
    positions = tl.arange(0, BLOCK)
    values = tl.load(output+row*WIDTH+positions, positions < length, other=0)
    # The old anchor at C is already resident. Outputs occupy C+1..C+m.
    tl.store(tokens+slot*CAPACITY+endpoint-length+1+positions,
             values, positions < length)
    tl.store(computed+slot, endpoint)
    tl.store(last+slot, tl.load(output+row*WIDTH+length-1))


@torch.inference_mode()
def commit_batch(state, last_tokens, slots, result):
    if result.token_ids.is_cuda:
        _commit[(slots.numel(),)](
            state.tokens.tensor, state.computed.tensor, last_tokens, slots,
            result.token_ids, result.lengths, result.computed,
            state.tokens.tensor.shape[1], result.token_ids.shape[1],
            triton.next_power_of_2(result.token_ids.shape[1]),
        )
    else:
        positions = torch.arange(result.token_ids.shape[1])[None, :]
        mask = positions < result.lengths[:, None]
        rows = slots[:, None].expand_as(mask)
        columns = result.computed[:, None]-result.lengths[:, None]+1+positions
        state.tokens.tensor[rows[mask], columns[mask]] = result.token_ids[mask]
        state.computed.tensor[slots] = result.computed.to(state.computed.tensor.dtype)
        last_tokens[slots] = result.token_ids.gather(1, result.lengths[:, None]-1)[:, 0]
