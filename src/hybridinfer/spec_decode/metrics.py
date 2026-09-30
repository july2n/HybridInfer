"""Counters shared by speculative verification backends."""


def new_spec_metrics():
    return dict(rounds=0, draft_tokens=0, accepted_tokens=0,
                output_tokens=0, trial_tokens=0, replay_tokens=0,
                copy_seconds=0., verify_seconds=0.,
                restore_seconds=0., commit_seconds=0.)
