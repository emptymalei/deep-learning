# Time Series Forecasting with Deep Learning

> [!NOTE]
>
> **In one minute**
>
> Deep learning models train on batches, so a time series first has to be cut into fixed-length inputs and targets. The page shows the DataFrameDataset class of the ts\_dl\_utils package, which slides along the time axis with a history length, a horizon and an optional gap, and what its first data point looks like.
>
> **Read first:** [The Time Delay Embedding Representation](<https://dl.leima.is/time-series/timeseries-data.time-delayed-embedding/index.md>)

In the chapter [Deep Learning Fundamentals](<https://dl.leima.is/deep-learning-fundamentals>), we discussed some deep learning models. In this chapter, we will discuss how to apply deep learning models to time series forecasting problems.

## Creating Dataset for Deep Learning Models

Deep learning models usually require batches of data to train. For time series data, we need to slice along the time axis to create batches. In section [The Time Delay Embedding Representation](<https://dl.leima.is/time-series/timeseries-data.time-delayed-embedding/index.md>), we discussed methods to represent time series data. In this section, we provide an example.

In our [`ts_dl_utils` package](<https://dl.leima.is/utilities/notebooks-and-utilities/index.md>), we provide a class called `DataFrameDataset`. This class moves along the time axis and cuts the time series into multiple data points.

````
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset


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
````

For example, give a time series dataset,

| index | y |
| ---: | ---: |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |
| 10 | 10 |
| 11 | 11 |
| 12 | 12 |
| 13 | 13 |
| 14 | 14 |

The first data point of `DataFrameDataset(dataframe=df, history_length=10, horizon=1)` will be

```
(array([[0],
         [1],
         [2],
         [3],
         [4],
         [5],
         [6],
         [7],
         [8],
         [9]]),
  array([[10]]))
```

## Try It: Slicing a Series into Data Points

The widget uses the same 15-row dataframe as the example above. Change `history_length`, `horizon` and `gap` to see which rows become the input `x` and the target `y` of each data point, and how many data points the dataset has.

*Interactive demo: enable JavaScript to slice the example dataframe into data points.*

---

**Related:** [Pendulum Dataset](<https://dl.leima.is/time-series-deep-learning/timeseries.dataset.pendulum/index.md>) · [Notebooks and Utilities for Tutorials](<https://dl.leima.is/utilities/notebooks-and-utilities/index.md>)
