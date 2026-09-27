"""Longest suffix match, then nearest historical occurrence with a continuation."""
from .interfaces import BackendCapabilities, DraftProposal


class NgramProposer:
    capabilities = BackendCapabilities()

    def __init__(self, config):
        self.config = config

    def propose(self, contexts):
        tokens, offsets, ids = [], [0], []
        for context in contexts:
            ids.append(context.request_id)
            tokens.extend(self._propose_one(context))
            offsets.append(len(tokens))
        return DraftProposal(tuple(ids), tuple(tokens), tuple(offsets))

    def _propose_one(self, context):
        # Reserve one prediction for the correction/bonus and one input for
        # the uncomputed anchor. Never extend the search with draft tokens.
        limit = min(self.config.max_draft_tokens,
                    context.remaining_output_tokens - 1,
                    context.max_model_len - context.computed_length - 1,
                    context.verification_budget - 1)
        if limit <= 0:
            return ()
        history = context.token_ids
        for n in range(min(self.config.ngram_max, len(history)-1),
                       self.config.ngram_min-1, -1):
            suffix = history[-n:]
            for start in range(len(history)-n-1, -1, -1):
                if history[start:start+n] == suffix:
                    return history[start+n:start+n+limit]
        return ()

    def on_commit(self, result):
        # Stateless: the next proposal reads only the committed history.
        pass
