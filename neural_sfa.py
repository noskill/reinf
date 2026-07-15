"""Neural Slow Feature Analysis (SFA) implementation.

This module defines the *NeuralSFA* model – a small convolutional encoder
followed by global average pooling and a recurrent MLP head.  The head receives
the pooled CNN features from the *current* frame **concatenated** with the SFA
embedding of the *previous* frame (``y_{t-1}``) and predicts the embedding for
the current timestep (``y_t``).

The accompanying helper ``sfa_loss`` implements the composite SFA objective

    ``L_sfa = L_slowness + L_variance + L_correlation``

as described in the user prompt:

```
L_slowness   = mean(||y_t - y_{t-1}||_2^2)
L_variance   = mean(diag(C) - 1)
L_correlation= mean(||C ⊙ (1 - I)||_F^2)
```

where ``C`` is the covariance matrix of zero-mean embeddings in the **batch**
and ``⊙`` denotes element-wise multiplication.

The implementation purposefully keeps external dependencies minimal – only
``torch`` and ``torchvision`` are required.
"""

# adopted from https://github.com/noskill/sfa-gen

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralSFA(nn.Module):
    """CNN + MLP network that outputs SFA embeddings.

    Parameters
    ----------
    emb_dim: int, default 32
        Dimensionality of the resulting SFA embedding ``y``.
    cnn_channels: Tuple[int, ...], default (32, 64, 128)
        Number of channels for each convolutional block.  The length of the
        tuple determines the number of conv blocks.
    img_channels: int, default 3
        Number of channels in the input images (3 for RGB).
    """

    def __init__(
        self,
        emb_dim: int = 32,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        img_channels: int = 3,
    ) -> None:  # noqa: D401, N802
        super().__init__()

        # --------------------
        # Convolutional encoder
        # --------------------
        layers = []
        in_ch = img_channels
        for out_ch in cnn_channels:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(True))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # -------------
        # Recurrent MLP
        # -------------
        mlp_in_dim = in_ch + emb_dim  # pooled CNN features + y_{t-1}
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, emb_dim),
        )

        self.emb_dim = emb_dim

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor, y_prev: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D401, N802
        """Compute embedding for *x* using *y_prev* as additional context.

        If ``y_prev`` is ``None`` a tensor of zeros is used.
        """

        # CNN feature extraction.
        h = self.encoder(x)  # (B, C, H', W')
        h = self.pool(h).flatten(1)  # (B, C)

        # Previous embedding (initialise with zeros if absent).
        if y_prev is None:
            y_prev = torch.zeros(x.size(0), self.emb_dim, device=x.device, dtype=x.dtype)

        z = torch.cat([h, y_prev], dim=1)
        y = self.mlp(z)
        return y


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _covariance(y: torch.Tensor) -> torch.Tensor:
    """Compute *batch* covariance matrix of zero-mean vectors ``y``.

    Input shape is ``(B, D)``.  The returned tensor has shape ``(D, D)``.
    """

    y_centered = y - y.mean(dim=0, keepdim=True)
    cov = (y_centered.T @ y_centered) / y_centered.size(0)
    return cov


def sfa_loss(
    y: torch.Tensor,
    y_prev: torch.Tensor,
    *,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: D401, N802
    """Composite SFA loss *and* optional individual terms.

    ``y`` and ``y_prev`` must have shape ``(B, D)`` and correspond to the
    *current* and *previous* embeddings for **matching** samples.

    Parameters
    ----------
    return_components: bool, default ``False``
        If *True*, returns a 4-tuple ``(total, L_slowness, L_var, L_corr)``.
    """

    if y.shape != y_prev.shape:
        raise ValueError("y and y_prev must have the same shape")

    # ----------
    # Slowness
    # ----------
    l_slowness = F.mse_loss(y, y_prev, reduction="mean")

    # ----------
    # Covariance
    # ----------
    cov = _covariance(y)

    # Diagonal / off-diagonal masks
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)

    # Variance term: push diagonals towards 1.
    l_var = (cov.diag() - 1).pow(2).mean()

    # Correlation term: minimise squared off-diagonal elements.
    off_diag = cov * (1 - eye)
    l_corr = off_diag.pow(2).mean()

    total = l_slowness + l_var + l_corr
    assert torch.isfinite(total).all()
    assert y.abs().max() < 100


    if return_components:
        return total, l_slowness, l_var, l_corr
    return total


# ---------------------------------------------------------------------------
# Minimal smoke-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    model = NeuralSFA()
    x0 = torch.randn(8, 3, 64, 64)
    x1 = torch.randn(8, 3, 64, 64)

    y0 = model(x0)
    y1 = model(x1, y0)

    loss = sfa_loss(y1, y0)
    print("loss:", loss.item())
