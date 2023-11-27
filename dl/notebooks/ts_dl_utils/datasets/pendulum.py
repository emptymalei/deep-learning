import math
from functools import cached_property
from typing import Dict, List

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Pendulum:
    r"""Class for generating time series data for a pendulum.

    The pendulum is modelled as a damped harmonic oscillator, i.e.,

    $$
    \theta(t) = \theta(0) \cos(2 \pi t / p)\exp(-\beta t),
    $$

    where $\theta(t)$ is the angle of the pendulum at time $t$.
    The period $p$ is calculated using

    $$
    p = 2 \pi \sqrt(L / g),
    $$

    with $L$ being the length of the pendulum
    and $g$ being the surface gravity.

    :param length: Length of the pendulum.
    :param gravity: Acceleration due to gravity.
    """

    def __init__(self, length: float, gravity: float = 9.81) -> None:
        self.length = length
        self.gravity = gravity

    @cached_property
    def period(self) -> float:
        """Calculate the period of the pendulum."""
        return 2 * math.pi * math.sqrt(self.length / self.gravity)

    def __call__(
        self,
        num_periods: int,
        num_samples_per_period: int,
        initial_angle: float = 0.1,
        beta: float = 0,
    ) -> Dict[str, List[float]]:
        """Generate time series data for the pendulum.

        Returns a list of floats representing the angle
        of the pendulum at each time step.

        :param num_periods: Number of periods to generate.
        :param num_samples_per_period: Number of samples per period.
        :param initial_angle: Initial angle of the pendulum.
        """
        time_step = self.period / num_samples_per_period
        steps = []
        time_series = []
        for i in range(num_periods * num_samples_per_period):
            t = i * time_step
            angle = (
                initial_angle
                * math.cos(2 * math.pi * t / self.period)
                * math.exp(-beta * t)
            )
            steps.append(t)
            time_series.append(angle)

        return {"t": steps, "theta": time_series}


class PendulumDataModule(L.LightningDataModule):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        dataframe: pd.DataFrame,
        test_fraction: float = 0.3,
        val_fraction: float = 0.1,
        batch_size: int = 32,
    ):
        super().__init__()
        self.history_length = history_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction

        self.train_dataset, self.val_dataset = self.split_train_val(
            self.train_val_dataset
        )

    @cached_property
    def df_length(self):
        return len(self.dataframe)

    @cached_property
    def df_test_length(self):
        return int(self.df_length * self.test_fraction)

    @cached_property
    def df_train_val_length(self):
        return self.df_length - self.df_test_length

    @cached_property
    def train_val_dataframe(self):
        return self.dataframe.iloc[: self.df_train_val_length]

    @cached_property
    def test_dataframe(self):
        return self.dataframe.iloc[self.df_train_val_length :]

    @cached_property
    def train_val_dataset(self):
        return DataFrameDataset(
            dataframe=self.train_val_dataframe,
            history_length=self.history_length,
            horizon=self.horizon,
        )

    @cached_property
    def test_dataset(self):
        return DataFrameDataset(
            dataframe=self.test_dataframe,
            history_length=self.history_length,
            horizon=self.horizon,
        )

    def split_train_val(self, dataset: Dataset):
        return torch.utils.data.random_split(
            dataset, [1 - self.val_fraction, self.val_fraction]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=len(self.test_dataset), shuffle=False
        )
