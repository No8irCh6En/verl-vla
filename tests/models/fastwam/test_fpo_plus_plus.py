from __future__ import annotations

import pytest
import torch

from verl_vla.models.fastwam.trainable_model import robust_cfm_element_loss


def test_robust_cfm_kernel_matches_official_mse_scaled_huber_definition():
    prediction = torch.tensor([0.0, 0.5, 2.0, -3.0])
    target = torch.zeros_like(prediction)

    loss = robust_cfm_element_loss(prediction, target, delta=1.0)

    torch.testing.assert_close(loss, torch.tensor([0.0, 0.25, 3.0, 5.0]))


def test_robust_cfm_kernel_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="identical shapes"):
        robust_cfm_element_loss(torch.zeros(2), torch.zeros(3), delta=1.0)
    with pytest.raises(ValueError, match="positive"):
        robust_cfm_element_loss(torch.zeros(2), torch.zeros(2), delta=0.0)
