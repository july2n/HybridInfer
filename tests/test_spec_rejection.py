import unittest
import torch
from hybridinfer.spec_decode.interfaces import VerificationPlan
from hybridinfer.spec_decode.metadata import VerificationBatch
from hybridinfer.sampling.batch_verifier import accept_greedy_batch
from hybridinfer.sampling.rejection_sampler import accept_random_batch, residual_distribution, stream_seed


class RejectionTests(unittest.TestCase):
    def test_mixed_greedy_random_and_realized_q(self):
        batch = VerificationBatch.from_plans([
            VerificationPlan(7, 5, 0, (1, 2)), VerificationPlan(3, 9, 0, (2,))])
        logits = torch.tensor([[0., 20., 0.], [0., 0., 20.], [20., 0., 0.],
                               [0., 0., 20.], [20., 0., 0.]])
        q = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
        limits = dict(remaining_output_tokens=[3, 2], max_model_len=20, ignore_eos=[True, True])
        result = accept_random_batch(batch, batch.tensors('cpu'), logits,
            temperatures=[0, 1], seeds=[42, 17], draft_probs=q, **limits)
        self.assertEqual(result.token_ids.tolist(), [[1, 2, 0], [2, 0, -1]])
        self.assertEqual(result.computed.tolist(), [8, 11])
        greedy = accept_greedy_batch(batch, batch.tensors('cpu'), logits.argmax(-1), **limits)
        self.assertTrue(torch.equal(greedy.payload(), result.payload()))

    def test_certain_rejection_residual_eos_and_truncation(self):
        batch = VerificationBatch.from_plans([VerificationPlan(1, 10, 0, (1, 2))])
        logits = torch.tensor([[100., -100., -100.], [0., 0., 100.], [0., 0., 100.]])
        result = accept_random_batch(batch, batch.tensors('cpu'), logits,
            temperatures=[1], seeds=[42], remaining_output_tokens=[1], max_model_len=20, eos=0)
        self.assertEqual(result.lengths.tolist(), [1])
        self.assertEqual(result.accepted.tolist(), [0])
        self.assertEqual(result.token_ids.tolist(), [[0, -1, -1]])
        self.assertEqual(result.reasons.tolist(), [1])

    def test_random_sampler_rejects_exhausted_or_misaligned_budgets(self):
        batch = VerificationBatch.from_plans([VerificationPlan(1, 10, 0, (1,))])
        for remaining in [[0], [], [1, 2]]:
            with self.assertRaises(ValueError):
                accept_random_batch(batch, batch.tensors('cpu'), torch.zeros(2, 3),
                    temperatures=[1], seeds=[42], remaining_output_tokens=remaining, max_model_len=20)

    def test_distribution_preservation_and_independent_rng_domains(self):
        # Exhaustive discrete marginal: q*min(1,p/q) plus residual mass is p.
        p = torch.tensor([[.2, .5, .3], [.1, .2, .7], [.5, .5, 0.]])
        q = torch.tensor([[.7, .2, .1], [0., 1., 0.], [.5, .5, 0.]])
        accepted_mass = torch.minimum(p, q)
        reject_mass = 1-accepted_mass.sum(-1, keepdim=True)
        actual = accepted_mass+reject_mass*residual_distribution(p, q)
        torch.testing.assert_close(actual, p)
        seeds = {stream_seed(42, position, domain) for position in range(100)
                 for domain in ['draft', 'accept', 'correction', 'bonus']}
        self.assertEqual(len(seeds), 400)

    def test_actual_sampler_marginal_for_point_mass_and_probabilistic_drafts(self):
        from hybridinfer.sampling.rejection_sampler import categorical, random_uniform, rejection_decisions
        p = torch.tensor([.2, .5, .3])
        for q in [torch.tensor([.7, .2, .1]), torch.tensor([0., 1., 0.])]:
            candidate = categorical(q, random_uniform(42, 10, 'draft', 'cpu', (20000,)))
            accepted = rejection_decisions(p[candidate], q[candidate],
                random_uniform(42, 10, 'accept', 'cpu', (20000,)))
            correction = categorical(residual_distribution(p, q),
                random_uniform(42, 10, 'correction', 'cpu', (20000,)))
            output = torch.where(accepted, candidate, correction)
            frequencies = torch.bincount(output, minlength=3).float()/output.numel()
            torch.testing.assert_close(frequencies, p, rtol=0, atol=.015)

    def test_request_reordering_does_not_change_random_results(self):
        plans = [VerificationPlan(7, 5, 0, (1, 2)), VerificationPlan(3, 9, 0, (2,))]
        logits = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.],
                               [0., 0., 1.], [1., 0., 0.]])
        def run(order):
            b = VerificationBatch.from_plans([plans[i] for i in order])
            pieces = [logits[:3], logits[3:]]
            return accept_random_batch(b, b.tensors('cpu'), torch.cat([pieces[i] for i in order]),
                temperatures=[1, 1], seeds=[[42, 17][i] for i in order],
                remaining_output_tokens=[3, 3], max_model_len=20, ignore_eos=[True, True])
        self.assertTrue(torch.equal(run([0, 1]).payload(), run([1, 0]).payload().flip(0)))


class DeviceContractTests(unittest.TestCase):
    def test_device_offsets_q_and_host_adapter(self):
        from hybridinfer.spec_decode.interfaces import DeviceDraftProposal
        p = DeviceDraftProposal((7, 3), torch.tensor([1, 2]), (0, 2, 2))
        self.assertEqual(p.to_host().tokens_for(0), (1, 2))
        self.assertIsNone(p.probabilities)
        with self.assertRaises(ValueError):
            DeviceDraftProposal((7,), torch.tensor([1]), (0, 2))
        with self.assertRaises(ValueError):
            DeviceDraftProposal((7,), torch.tensor([1]), (0, 1), torch.ones(2, 3))
