from functools import cached_property
from typing import Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class DataFrameDataset(Dataset):
    """A dataset from a pandas dataframe.

    For a given pandas dataframe, this generates a pytorch
    compatible dataset by sliding in time dimension.

    ```python
    ds = DataFrameDataset(
        dataframe=df, history_length=10, horizon=2
    )
    ```

    :param dataframe: input dataframe with a DatetimeIndex.
    :param history_length: length of input X in time dimension
        in the final Dataset class.
    :param horizon: number of steps to be forecasted.
    :param gap: gap between input history and prediction
    """

    def __init__(
        self, dataframe: pd.DataFrame, history_length: int, horizon: int, gap: int = 0
    ):
        super().__init__()
        self.dataframe = dataframe
        self.history_length = history_length
        self.horzion = horizon
        self.gap = gap
        self.dataframe_rows = len(self.dataframe)
        self.length = (
            self.dataframe_rows - self.history_length - self.horzion - self.gap + 1
        )

    def moving_slicing(self, idx: int, gap: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        x, y = (
            self.dataframe[idx : self.history_length + idx].values,
            self.dataframe[
                self.history_length
                + idx
                + gap : self.history_length
                + self.horzion
                + idx
                + gap
            ].values,
        )
        return x, y

    def _validate_dataframe(self) -> None:
        """Validate the input dataframe.

        - We require the dataframe index to be DatetimeIndex.
        - This dataset is null aversion.
        - Dataframe index should be sorted.
        """

        if not isinstance(
            self.dataframe.index, pd.core.indexes.datetimes.DatetimeIndex
        ):
            raise TypeError(
                "Type of the dataframe index is not DatetimeIndex"
                f": {type(self.dataframe.index)}"
            )

        has_na = self.dataframe.isnull().values.any()

        if has_na:
            logger.warning("Dataframe has null")

        has_index_sorted = self.dataframe.index.equals(
            self.dataframe.index.sort_values()
        )

        if not has_index_sorted:
            logger.warning("Dataframe index is not sorted")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(idx, slice):
            if (idx.start < 0) or (idx.stop >= self.length):
                raise IndexError(f"Slice out of range: {idx}")
            step = idx.step if idx.step is not None else 1
            return [
                self.moving_slicing(i, self.gap)
                for i in range(idx.start, idx.stop, step)
            ]
        else:
            if idx >= self.length:
                raise IndexError("End of dataset")
            return self.moving_slicing(idx, self.gap)

    def __len__(self) -> int:
        return self.length


class DataFrameDataModule(L.LightningDataModule):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        dataframe: pd.DataFrame,
        gap: int = 0,
        test_fraction: float = 0.3,
        val_fraction: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.history_length = history_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.gap = gap
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.num_workers = num_workers

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
            gap=self.gap,
        )

    @cached_property
    def test_dataset(self):
        return DataFrameDataset(
            dataframe=self.test_dataframe,
            history_length=self.history_length,
            horizon=self.horizon,
            gap=self.gap,
        )

    def split_train_val(self, dataset: Dataset):
        return torch.utils.data.random_split(
            dataset, [1 - self.val_fraction, self.val_fraction]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=len(self.test_dataset), shuffle=False
        )
