"""Target forward hooks used by feature-backed draft proposers."""
from hybridinfer.utils.context import get_context


def forward_with_draft(runner, input_ids, positions, is_prefill):
    proposer = runner.draft_proposer
    hidden = proposer.forward_target(input_ids, positions)
    context = get_context()
    proposer.record(runner.batch_slots_gpu[:context.batch_descriptor.num_reqs],
                    positions, hidden, context.prefill_slices if is_prefill else None)
    return runner.compute_logits(hidden, is_prefill)


def forward_verification_target(runner, input_ids, positions):
    return runner.model.compute_logits(runner.model(input_ids, positions))
