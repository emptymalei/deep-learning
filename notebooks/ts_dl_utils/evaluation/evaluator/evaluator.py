from typing import Dict, List, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
)


class Evaluator:
    """Evaluate the predictions

    :param step: which prediction step to be evaluated.
    :param gap: gap between input history and target/prediction.
    """

    def __init__(self, step: int = 0, gap: int = 0):
        self.step = step
        self.gap = gap

    @staticmethod
    def get_one_history(
        predictions: List, idx: int, batch_idx: int = 0
    ) -> torch.Tensor:
        return predictions[batch_idx][0][idx, ...]

    @staticmethod
    def get_one_pred(predictions: List, idx: int, batch_idx: int = 0) -> torch.Tensor:
        return predictions[batch_idx][1][idx, ...]

    @staticmethod
    def get_y(predictions: List, step: int) -> List[torch.Tensor]:
        return [i[1][..., step] for i in predictions]

    def y(self, predictions: List, batch_idx: int = 0) -> torch.Tensor:
        return self.get_y(predictions, self.step)[batch_idx].detach()

    @staticmethod
    def get_y_true(dataloader: DataLoader, step: int) -> torch.Tensor:
        return [i[1].squeeze(-1)[..., step] for i in dataloader]

    def y_true(self, dataloader: DataLoader, batch_idx: int = 0) -> torch.Tensor:
        return self.get_y_true(dataloader, step=self.step)[batch_idx].detach()

    def get_one_sample(
        self, predictions: List, idx: int, batch_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.get_one_history(predictions, idx, batch_idx),
            self.get_one_pred(predictions, idx, batch_idx),
        )

    def plot_one_sample(
        self, ax: mpl.axes.Axes, predictions: List, idx: int, batch_idx: int = 0
    ):
        history, pred = self.get_one_sample(predictions, idx, batch_idx)

        x_raw = np.arange(len(history) + len(pred) + self.gap)
        x_history = x_raw[: len(history)]
        x_pred = x_raw[len(history) + self.gap :]
        x = np.concatenate([x_history, x_pred])

        y = np.concatenate([history, pred])

        ax.plot(x, y, marker=".", label=f"input ({idx})")

        ax.axvspan(x_pred[0], x_pred[-1], color="orange", alpha=0.1)

    @property
    def metric_collection(self) -> MetricCollection:
        return MetricCollection(
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            MeanSquaredError(),
            SymmetricMeanAbsolutePercentageError(),
        )

    @staticmethod
    def metric_dataframe(metrics: Dict) -> pd.DataFrame:
        return pd.DataFrame(
            [{k: float(v) for k, v in metrics.items()}], index=["values"]
        ).T

    def metrics(
        self, predictions: List, dataloader: DataLoader, batch_idx: int = 0
    ) -> pd.DataFrame:
        truths = self.y_true(dataloader)
        preds = self.y(predictions, batch_idx=batch_idx)

        return self.metric_dataframe(self.metric_collection(preds, truths))
