from typing import List, Tuple

import lightning.pytorch as L
import torch


class LastObservationForecaster(L.LightningModule):
    """Spits out the forecasts using the last observation.

    :param horizon: horizon of the forecast.
    """

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon

    def _last_observation(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., -1:, :]

    def predict_step(
        self, batch: List, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch

        y_hat = self._last_observation(x)

        y_hat = y_hat.repeat(1, self.horizon, 1)

        return x.squeeze(-1), y_hat.squeeze(-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.type(self.dtype)
        return (
            x.squeeze(-1),
            self._last_observation(x).repeat(1, self.horizon, 1).squeeze(-1),
        )
