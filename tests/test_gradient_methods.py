import pytest

torch = pytest.importorskip('torch')

from runnetworks import RunNetworks


class TinyMultitaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.rebuild_head = torch.nn.Parameter(torch.tensor([1.0]))
        self.mask_head = torch.nn.Parameter(torch.tensor([1.0]))


def _runner():
    runner = RunNetworks.__new__(RunNetworks)
    runner.model = TinyMultitaskModel()
    runner._gradient_balance_ema = None
    return runner


def _losses(runner):
    reconstruction = (runner.model.inc * torch.tensor([1.0, 0.0])).sum()
    reconstruction = reconstruction + runner.model.rebuild_head.sum()
    segmentation = (runner.model.inc * torch.tensor([-1.0, 1.0])).sum()
    segmentation = segmentation + runner.model.mask_head.sum()
    return {
        'total': reconstruction + segmentation,
        'multiscale': reconstruction,
        'refinement': reconstruction * 0.0,
        'mask': segmentation,
    }


def test_pcgrad_projects_conflicting_shared_gradients():
    runner = _runner()
    losses = _losses(runner)
    result = runner._apply_multitask_gradients(
        loss_dict=losses,
        total_loss=losses['total'],
        method='pcgrad',
        balance_cfg={},
    )
    assert result['is_conflict'] is True
    assert runner.model.inc.grad is not None
    assert torch.isfinite(runner.model.inc.grad).all()
    assert runner.model.rebuild_head.grad.item() == pytest.approx(1.0)
    assert runner.model.mask_head.grad.item() == pytest.approx(1.0)


def test_balance_matches_shared_gradient_norms_before_clamping():
    runner = _runner()
    losses = _losses(runner)
    result = runner._apply_multitask_gradients(
        loss_dict=losses,
        total_loss=losses['total'],
        method='balance',
        balance_cfg={'min_weight': 0.1, 'max_weight': 10.0, 'ema_beta': 0.0},
    )
    assert result['segmentation_weight'] == pytest.approx(2.0 ** -0.5)
    assert result['is_conflict'] is True
