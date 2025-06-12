# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: deep-learning
#     language: python
#     name: deep-learning
# ---

# # Creating Time Series Datasets
#
# In this notebook, we explain how to create a time series dataset for PyTorch using the moving slicing technique.
#
# The class `DataFrameDataset` is also included in our `ts_dl_utils` package.

# +
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

# -


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


# ## Examples

# We create a sample dataframe with one single variable "y"

df = pd.DataFrame(np.arange(15), columns=["y"])
df

# ### history_length=10, horizon=1

ds_1 = DataFrameDataset(dataframe=df, history_length=10, horizon=1)

list(ds_1)

# ### history_length=10, horizon=2

ds_2 = DataFrameDataset(dataframe=df, history_length=10, horizon=2)

list(ds_2)

# ### history_length=10, horizon=1, gap=1

ds_1_gap_1 = DataFrameDataset(dataframe=df, history_length=10, horizon=1, gap=1)

list(ds_1_gap_1)

# ### history_length=10, horizon=1, gap=2

ds_1_gap_2 = DataFrameDataset(dataframe=df, history_length=10, horizon=1, gap=2)

list(ds_1_gap_2)

# ### history_length=10, horizon=2, gap=1

ds_2_gap_1 = DataFrameDataset(dataframe=df, history_length=10, horizon=2, gap=1)

list(ds_2_gap_1)

# ### history_length=10, horizon=2, gap=2

ds_2_gap_2 = DataFrameDataset(dataframe=df, history_length=10, horizon=2, gap=2)

list(ds_2_gap_2)
