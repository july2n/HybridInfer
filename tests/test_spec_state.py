from types import SimpleNamespace
import unittest
import torch

from hybridinfer.spec_decode.state import GDNTransaction
from hybridinfer.engine.block_manager import BlockManager
from hybridinfer.engine.sequence import Sequence


class TransactionTests(unittest.TestCase):
    def pools(self):
        return [SimpleNamespace(conv_states=torch.arange(12.).reshape(3, 4),
                                recurrent_states=torch.arange(24.).reshape(3, 2, 4))]

    def test_full_accept_only_changes_source(self):
        layers = self.pools()
        inactive = layers[0].recurrent_states[1].clone()
        with GDNTransaction(layers, 0, 2) as txn:
            for pool in (layers[0].conv_states, layers[0].recurrent_states):
                self.assertTrue(torch.equal(pool[0], pool[2]))
                pool[2].add_(10)
            txn.commit_trial()
        self.assertTrue(torch.equal(layers[0].recurrent_states[1], inactive))
        self.assertTrue(torch.equal(layers[0].recurrent_states[0], layers[0].recurrent_states[2]))

    def test_selected_endpoint_is_preserved_on_commit(self):
        layers = self.pools()
        original = layers[0].recurrent_states[0].clone()
        with GDNTransaction(layers, 0, 2) as txn:
            layers[0].recurrent_states[2].add_(100)
            layers[0].recurrent_states[0].copy_(original+3)
            txn.finish_endpoint_commit()
        self.assertTrue(torch.equal(layers[0].recurrent_states[0], original+3))

    def test_trial_and_commit_exceptions_restore_source(self):
        for after_commit in (False, True):
            layers = self.pools()
            original = [x.clone() for x in (layers[0].conv_states[0], layers[0].recurrent_states[0])]
            with self.assertRaises(RuntimeError):
                with GDNTransaction(layers, 0, 2) as txn:
                    if after_commit:
                        layers[0].conv_states[0].add_(99)
                        layers[0].recurrent_states[0].add_(99)
                        txn.finish_endpoint_commit()
                    raise RuntimeError('trial/commit')
            for pool, saved in zip((layers[0].conv_states, layers[0].recurrent_states), original):
                self.assertTrue(torch.equal(pool[0], saved))


class PageTests(unittest.TestCase):
    def setUp(self):
        self.old = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old

    def test_cross_page_reserve_trim_and_no_publication(self):
        manager = BlockManager(8, 4)
        seq = Sequence([1, 2, 3, 4])
        manager.allocate(seq, 0)
        seq.num_cached_tokens = 3
        history = seq.token_ids.copy()
        self.assertTrue(manager.reserve_trial(seq, 9))
        self.assertEqual(len(seq.block_table), 3)
        self.assertEqual(seq.token_ids, history)
        self.assertFalse(manager.hash_to_block_id)
        manager.trim_trial(seq, 4)
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(len(manager.free_block_ids), 7)
        self.assertTrue(all(manager.blocks[i].ref_count == 0 for i in manager.free_block_ids))

    def test_resource_failure_is_atomic(self):
        manager = BlockManager(1, 4)
        seq = Sequence([1, 2, 3, 4])
        manager.allocate(seq, 0)
        seq.num_cached_tokens = 3
        self.assertFalse(manager.reserve_trial(seq, 7))
        self.assertEqual(seq.block_table, [0])
        manager.blocks[0].ref_count = 2
        self.assertFalse(manager.reserve_trial(seq, 4))
        self.assertEqual(seq.block_table, [0])

    def test_shared_complete_prefix_is_not_writable(self):
        manager = BlockManager(4, 4)
        seq = Sequence([1, 2, 3, 4, 5])
        manager.allocate(seq, 0)
        seq.num_cached_tokens = 4
        manager.blocks[seq.block_table[0]].ref_count = 2
        self.assertTrue(manager.reserve_trial(seq, 7))
        self.assertEqual(manager.blocks[seq.block_table[0]].ref_count, 2)

    def test_empty_and_batch_append(self):
        seq = Sequence([1])
        seq.append_tokens([])
        self.assertEqual(seq.last_token, 1)
        seq.append_tokens([2, 3])
        self.assertEqual((seq.token_ids, seq.last_token, seq.num_tokens), ([1, 2, 3], 3, 3))
