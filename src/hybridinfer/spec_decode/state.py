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

    def commit(self, *, all_inputs_committed, replay):
        if all_inputs_committed:
            for pool, _ in self.original:
                pool[self.source_slot].copy_(pool[self.trial_slot])
        else:
            replay()
        self.committed = True

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self.committed:
            for pool, original in self.original:
                pool[self.source_slot].copy_(original)
        # The single private slot is reused by the next synchronous transaction.
        return False
