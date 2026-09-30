"""One dispatch point for checkpoint-backed draft proposers."""


def create_draft_proposer(runner):
    spec = runner.config.speculative
    if spec is None or not spec.enabled or spec.method == 'ngram':
        return None
    if spec.method == 'mtp':
        from .mtp import MTPProposer
        return MTPProposer(runner)
    if spec.method == 'eagle3':
        from .eagle3 import Eagle3Proposer
        return Eagle3Proposer(runner)
    if spec.method in ('dflash', 'dspark'):
        from .block import BlockProposer
        return BlockProposer(runner)
    raise ValueError(f'Unknown draft method: {spec.method}')
