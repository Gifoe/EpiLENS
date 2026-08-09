import torch

from epilens.models import BCRNet, PRQNet, cdel_probability
from epilens.objectives import bcr_loss


def test_primary_models_return_one_score_per_valid_channel():
    torch.manual_seed(1)
    features = torch.randn(2, 3, 5, 36)
    valid = torch.ones(2, 3, 5, dtype=torch.bool)
    valid[:, :, 4] = False
    prq = PRQNet()(features, valid)
    bcr = BCRNet()(features, valid)
    assert prq["probability_nez"].shape == (5,)
    assert bcr["logit_ez"].shape == (5,)
    assert prq["channel_valid"].tolist() == [True, True, True, True, False]
    fused = cdel_probability(prq["probability_nez"], bcr["logit_ez"])
    torch.testing.assert_close(
        fused, 0.8 * prq["probability_nez"] + 0.2 * (1 - torch.sigmoid(bcr["logit_ez"]))
    )


def test_bcr_objective_is_finite_and_differentiable():
    logits = torch.tensor([1.2, 0.7, -0.2, -0.8], requires_grad=True)
    labels_nez = torch.tensor([0, 0, 1, 1])
    total, parts = bcr_loss(logits, labels_nez, torch.ones(4, dtype=torch.bool))
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in parts.values())
    assert logits.grad is not None
