"""GPU state updates with immutable, per-submission pinned staging storage."""
import torch
import triton
import triton.language as tl


def to_device(values, dtype, device):
    # Never mutate/reuse this host allocation. PyTorch's pinned allocator
    # tracks the asynchronous copy before making its storage reusable.
    return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True).to(
        device=device, non_blocking=True,
    )


@triton.jit
def _apply_writes(dst, indices, values, count: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < count
    index = tl.load(indices + i, mask, other=0)
    value = tl.load(values + i, mask, other=0)
    tl.store(dst + index, value, mask)


class StagedWriteTensor:
    """Store the base tensor on GPU; upload only changed elements.

    Writes to the same address within a submission are coalesced on CPU.
    apply_write queues copies and one update kernel on the compute stream.
    """

    def __init__(self, shape, dtype, device="cuda", fill=0):
        self.tensor = torch.full(shape, fill, dtype=dtype, device=device)
        self._writes = {}

    def stage_write(self, row, start, values):
        width = self.tensor.shape[1] if self.tensor.ndim == 2 else 1
        if row < 0 or row >= self.tensor.shape[0] or start < 0:
            raise IndexError("state write outside allocated row")
        if start + len(values) > width:
            raise IndexError("state write exceeds row capacity")
        for i, value in enumerate(values):
            self._writes[row * width + start + i] = value

    def apply_write(self):
        if not self._writes:
            return
        indices = to_device(list(self._writes), torch.int64, self.tensor.device)
        values = to_device(list(self._writes.values()), self.tensor.dtype, self.tensor.device)
        _apply_writes[(triton.cdiv(len(self._writes), 256),)](
            self.tensor, indices, values, len(self._writes), 256,
        )
        self._writes.clear()
