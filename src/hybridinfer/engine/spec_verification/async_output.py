"""Independent variable-length output storage and completion ownership."""
import torch

from hybridinfer.sampling.batch_verifier import results_from_payload


class AsyncVerificationOutput:
    def __init__(self, acceptance, plans, runner, on_ready=None):
        self._gpu = acceptance.payload()
        self._plans = tuple(plans)
        self._on_ready = on_ready
        self._result = None
        self._event = None
        if self._gpu.is_cuda:
            self._cpu = torch.empty_like(self._gpu, device='cpu', pin_memory=True)
            self._event = torch.cuda.Event(blocking=True)
            if runner.async_output:
                stream = runner.output_copy_stream
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self._cpu.copy_(self._gpu, non_blocking=True)
                    self._gpu.record_stream(stream)
                    self._event.record(stream)
            else:
                self._cpu.copy_(self._gpu)
                self._event.record()
        else:
            self._cpu = self._gpu.clone()

    def get_output(self):
        if self._result is None:
            if self._event is not None:
                self._event.synchronize()
            self._result = results_from_payload(self._cpu, self._plans)
            if self._on_ready is not None:
                self._on_ready(self._result)
            self._on_ready = None
            self._gpu = self._cpu = None
        return self._result
