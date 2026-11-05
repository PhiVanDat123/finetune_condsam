import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from optimizer.condsam import CondSAM
from optimizer.condsam_training import train_step, evaluate, run
from optimizer.gram import gram_accumulate


class CondSAMTests(unittest.TestCase):
    def test_gram_and_direction_formula(self):
        rows = torch.randn(11, 4)
        out = torch.zeros(4, 4)
        gram_accumulate(rows, out, alpha=2)
        torch.testing.assert_close(out, 2 * rows.T @ rows)
        g = torch.randn(3, 4)
        x, y = torch.randn(4), torch.randn(3)
        state = dict(A=torch.eye(4), x=x, y=y, spectral=torch.tensor(2.), ab=torch.tensor(12.))
        s = y @ g @ x
        expected = 4 / (2 * (s + (1e-8 if s >= 0 else -1e-8))) * torch.outer(y, x)
        expected -= 12 / (2 * g.numel() * (g.square().sum() + 1e-8)) * g
        torch.testing.assert_close(CondSAM._direction(g, state, 1e-8), expected)

    def test_step_norm_checkpoint_and_bn(self):
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4),
                                    torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(16, 6))
        policy = CondSAM(model.parameters(), rho=0.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        x, y = torch.randn(4, 3, 2, 2), torch.tensor([0, 1, 2, 3])
        before = model[0].weight.detach().clone()
        for _ in range(2):
            model.train()
            train_step(model, x, y, optimizer, policy)
        self.assertFalse(torch.equal(before, model[0].weight))
        self.assertEqual(model[1].num_batches_tracked.item(), 2)
        self.assertTrue(model[1].track_running_stats)
        self.assertIn('A', policy.state[model[0].weight])
        restored = CondSAM(model.parameters())
        restored.load_state_dict(policy.state_dict())
        torch.testing.assert_close(restored.state[model[0].weight]['A'], policy.state[model[0].weight]['A'])
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        deltas = policy.perturbations()
        norm = sum(d.double().square().sum() for d in deltas.values()).sqrt()
        self.assertAlmostEqual(norm.item(), 0.1, places=6)
        for p in model.parameters():
            p.grad = None
        self.assertEqual(policy.perturbations(), {})
        metrics = evaluate(model, [(x, y)], torch.device('cpu'))
        self.assertLessEqual(metrics['top1'], metrics['top5'])
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(log_dir=tmp, eval_only=True, model_name='tiny', dataset='synthetic')
            torch.save(dict(state_dict=model.state_dict(), args=vars(args)), Path(tmp) / 'last.pt')
            run(args, model, SimpleNamespace(test=[(x, y)]))
            self.assertTrue((Path(tmp) / 'eval.json').exists())

    def test_failed_second_pass_restores_weights_and_bn(self):
        class Failing(torch.nn.Sequential):
            calls = 0
            def forward(self, x):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('second pass failed')
                return super().forward(x)
        model = Failing(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 6))
        policy = CondSAM(model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        before = [p.detach().clone() for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError, 'second pass'):
            train_step(model, torch.randn(4, 3), torch.tensor([0, 1, 2, 3]), optimizer, policy)
        for p, original in zip(model.parameters(), before):
            torch.testing.assert_close(p, original, rtol=0, atol=0)
        self.assertTrue(model[0].track_running_stats)
        self.assertFalse(model[1]._forward_hooks)


if __name__ == '__main__':
    unittest.main()
