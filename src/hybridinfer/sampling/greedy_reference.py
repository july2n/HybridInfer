"""CPU reference for greedy prediction-row alignment and endpoint selection."""
from hybridinfer.spec_decode.interfaces import VerificationResult


def accept_greedy(plan, predictions, *, remaining_output_tokens,
                  max_model_len, eos=-1, ignore_eos=False):
    predictions = tuple(predictions)
    if len(predictions) != len(plan.candidates) + 1:
        raise ValueError("verification requires K+1 prediction rows")
    if remaining_output_tokens < 1 or plan.computed_length >= max_model_len:
        raise ValueError("no output or context budget")
    if plan.trial_end > max_model_len:
        raise ValueError("trial exceeds model context")
    output = []
    accepted = 0
    reason = None
    for i, prediction in enumerate(predictions):
        matches = i < len(plan.candidates) and plan.candidates[i] == prediction
        output.append(prediction)
        accepted += int(matches)
        if not ignore_eos and prediction == eos:
            reason = "eos"
        elif len(output) >= remaining_output_tokens:
            reason = "max_tokens"
        elif plan.computed_length + len(output) >= max_model_len:
            reason = "context"
        if reason or not matches:
            break
    return VerificationResult(tuple(output), accepted, len(predictions),
                              plan.computed_length + len(output), bool(reason), reason)


def verify_logits(plan, logits, **kwargs):
    if logits.ndim != 2 or logits.shape[0] != len(plan.input_tokens):
        raise ValueError("logits must contain one vocabulary row per trial input")
    return accept_greedy(plan, logits.argmax(-1).tolist(), **kwargs)
