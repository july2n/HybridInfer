"""Private GDN trial state, independent of the public prefix snapshot pool."""


class GDNTransaction:
    def __init__(self, layers, source_slot, trial_slot):
        self.layers = layers
        self.source_slot = source_slot
        self.trial_slot = trial_slot
        self.original = []
        self.committed = False

    def __enter__(self):
        for layer in self.layers:
            for pool in (layer.conv_states, layer.recurrent_states):
                original = pool[self.source_slot].clone()
                self.original.append((pool, original))
                pool[self.trial_slot].copy_(original)
        return self

    def commit_trial(self):
        """Commit the private state after an ordinary anchor fallback."""
        for pool, _ in self.original:
            pool[self.source_slot].copy_(pool[self.trial_slot])
        self.committed = True

    def finish_endpoint_commit(self):
        """Mark original-trial endpoint selection complete; retain rollback on error."""
        self.committed = True

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self.committed:
            for pool, original in self.original:
                pool[self.source_slot].copy_(original)
        # Compute-stream ordering protects reuse of private trial slots.
        return False
