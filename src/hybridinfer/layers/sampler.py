"""Tiled Gumbel-max sampling without vocabulary-sized probabilities."""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _sample_tiles(logits, stride: tl.constexpr, slots, temperatures, seeds,
                  positions, maxima, winners, VOCAB: tl.constexpr,
                  PARTS: tl.constexpr, TILE: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + row)
    temperature = tl.load(temperatures + slot)
    seed = tl.load(seeds + slot)
    position = tl.load(positions + slot)
    token = part * TILE + tl.arange(0, TILE)
    value = tl.load(logits + row * stride + token, token < VOCAB, other=-float("inf")).to(tl.float32)
    if temperature > 0:
        uniform = tl.rand(seed, (position.to(tl.uint32) * VOCAB + token).to(tl.uint32))
        uniform = tl.minimum(tl.maximum(uniform, 4.656612873077393e-10), 0.9999999403953552)
        value = value / temperature - tl.log(-tl.log(uniform))
    maximum = tl.max(value, 0)
    winner = tl.min(tl.where((value == maximum) & (token < VOCAB), token, 2147483647), 0)
    tl.store(maxima + row * PARTS + part, maximum)
    tl.store(winners + row * PARTS + part, winner)


@triton.jit
def _reduce_tiles(maxima, winners, output, PARTS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, TILE)
    value = tl.load(maxima + row * PARTS + part, part < PARTS, other=-float("inf"))
    token = tl.load(winners + row * PARTS + part, part < PARTS, other=2147483647)
    maximum = tl.max(value, 0)
    tl.store(output + row, tl.min(tl.where(value == maximum, token, 2147483647), 0))


class Sampler(nn.Module):
    def sample(self, logits, slots, temperatures, seeds, positions):
        rows, vocab = logits.shape
        parts = triton.cdiv(vocab, 1024)
        maxima = torch.empty((rows, parts), dtype=torch.float32, device=logits.device)
        winners = torch.empty((rows, parts), dtype=torch.int32, device=logits.device)
        output = torch.empty(rows, dtype=torch.int64, device=logits.device)
        _sample_tiles[(rows, parts)](
            logits, logits.stride(0), slots, temperatures, seeds, positions,
            maxima, winners, vocab, parts, 1024,
        )
        _reduce_tiles[(rows,)](maxima, winners, output, parts, triton.next_power_of_2(parts))
        return output

    def forward(self, logits, temperatures):
        rows = logits.shape[0]
        slots = torch.arange(rows, dtype=torch.int64, device=logits.device)
        seeds = torch.randint(0, 2**31, (rows,), dtype=torch.int64, device=logits.device)
        positions = torch.zeros(rows, dtype=torch.int32, device=logits.device)
        return self.sample(logits, slots, temperatures, seeds, positions)
