# The Time Delay Embedding Representation

> [!NOTE]
>
> **In one minute**
>
> Time delay embedding, also called rolling, rewrites a series as a table of fixed-length windows: each row holds p consecutive past values and the value to predict next. Neural networks need such fixed-length inputs, and Takens' theorem explains why the representation preserves the dynamics of the original series; the pendulum data illustrates it.

The time delay embedding representation of time series data is widely used in deep learning forecasting models[^/time-series/timeseries-data.time-delayed-embedding/#bib-Hewamalage2022-sc]. This is also called rolling in many time series analyses [^/time-series/timeseries-data.time-delayed-embedding/#bib-Zivot2006-es].

For simplicity, we only write down the representation for a problem with time series \\(y\_{1}, \\cdots, y\_{t}\\), and forecasting \\(y\_{t+1}\\). We rewrite the series into a matrix, in an autoregressive way,

\\\[ \\begin{align} \\mathbf Y = \\begin{bmatrix} y\_1 &amp; y\_2 &amp; \\cdots &amp; y\_p &amp;\\Big| &amp; {\\color{red}y\_{p+1}} \\\\ y\_{1+1} &amp; y\_{1+2} &amp; \\cdots &amp; y\_{1+p} &amp;\\Big| &amp; {\\color{red}y\_{1+p+1}} \\\\ \\vdots &amp; \\vdots &amp; \\ddots &amp; \\vdots &amp;\\Big| &amp; {\\color{red}\\vdots} \\\\ y\_{i-p+1} &amp; y\_{i-p+2} &amp; \\cdots &amp; y\_{i} &amp;\\Big| &amp; {\\color{red}y\_{i+1}} \\\\ \\vdots &amp; \\vdots &amp; \\ddots &amp; \\vdots &amp;\\Big| &amp; {\\color{red}\\vdots} \\\\ y\_{t-p+1} &amp; y\_{t-p+2} &amp; \\cdots &amp; y\_{t} &amp;\\Big| &amp; {\\color{red}y\_{t+1}} \\\\ \\end{bmatrix} \\end{align} \\\]

which indicates that we will use everything on the left, a matrix of shape \\((t-p+1,p)\\), to predict the vector on the right (in red). This is a useful representation when building deep learning models as many of the neural networks require fixed-length inputs.

## Try It: Rolling a Window over a Series

Choose the history length \\(p\\) and the forecast horizon \\(H\\), then step through the rows of the table. Each row is one training example: the blue values are the inputs and the orange values are the targets.

An interactive chart that slides an input window and a target window along a series and shows the resulting rows of the table. It needs JavaScript; the matrix above shows the same construction.

## Taken's Theorem

The reason that time delayed embedding representation is useful is that it is a representation of the original time series that preserves the dynamics of the original time series, if any. The math behind it is the Taken's theorem [^/time-series/timeseries-data.time-delayed-embedding/#bib-Takens1981-kh].

To illustrate the idea, we take our [pendulum dataset](<https://dl.leima.is/time-series-deep-learning/timeseries.dataset.pendulum/index.md>) as an example. The pendulum dataset describes a damped pendulum, for the math and visualizations please refer to [the corresponding page](<https://dl.leima.is/time-series-deep-learning/timeseries.dataset.pendulum/index.md>). Here we apply the time delay embedding representation to the pendulum dataset by setting both the history length and the target length to 1, so that we can better visualize it.

We plot out the delayed embedding representation of the pendulum dataset. The x-axis is the value of the pendulum angle at time \\(t\\), and the y-axis is the value of the pendulum angle at time \\(t+1\\). The animation shows how the delayed embedding representation evolves over time and shows attractor behavior. If a model can capture this dynamics, it can make good predictions.

![Delayed Embedding](<https://dl.leima.is/notebooks/results/pendulum_dataset/delayed_embedding_animation.gif>)

The notebook for more about the dataset itself is [here](<https://dl.leima.is/notebooks/pendulum_dataset/index.md>).

````
from functools import cached_property
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from ts_dl_utils.datasets.dataset import DataFrameDataset
from ts_dl_utils.datasets.pendulum import Pendulum


ds_de = DataFrameDataset(dataframe=df["theta"][:200], history_length=1, horizon=1)

class DelayedEmbeddingAnimation:
    """Builds an animation for univariate time series
    using delayed embedding.

    ```python
    fig, ax = plt.subplots(figsize=(10, 10))

    dea = DelayedEmbeddingAnimation(dataset=ds_de, fig=fig, ax=ax)
    ani = dea.build(interval=10, save_count=dea.time_steps)
    ani.save("results/pendulum_dataset/delayed_embedding_animation.mp4")
    ```

    :param dataset: a PyTorch dataset, input and target should have only length 1
    :param fig: figure object from matplotlib
    :param ax: axis object from matplotlib
    """
    def __init__(
        self, dataset: DataFrameDataset, fig: mpl.figure.Figure, ax: mpl.axes.Axes
    ):
        self.dataset = dataset
        self.ax = ax
        self.fig = fig

    @cached_property
    def data(self) -> List[Tuple[float, float]]:
        return [(i[0][0], i[1][0]) for i in self.dataset]

    @cached_property
    def x(self):
        return [i[0] for i in self.data]

    @cached_property
    def y(self):
        return [i[1] for i in self.data]

    def data_gen(self):
        for i in self.data:
            yield i

    def animation_init(self) -> mpl.axes.Axes:
        ax.plot(
            self.x,
            self.y,
        )
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlabel("t")
        ax.set_ylabel("t+1")

        return self.ax

    def animation_run(self, data: Tuple[float, float]) -> mpl.axes.Axes:
        x, y = data
        self.ax.scatter(x, y)
        return self.ax

    @cached_property
    def time_steps(self):
        return len(self.data)

    def build(self, interval: int = 10, save_count: int = 10):
        return animation.FuncAnimation(
            self.fig,
            self.animation_run,
            self.data_gen,
            interval=interval,
            init_func=self.animation_init,
            save_count=save_count,
        )


fig, ax = plt.subplots(figsize=(10, 10))

dea = DelayedEmbeddingAnimation(dataset=ds_de, fig=fig, ax=ax)

ani = dea.build(interval=10, save_count=dea.time_steps)

gif_writer = animation.PillowWriter(fps=5, metadata=dict(artist="Lei Ma"), bitrate=100)

ani.save("results/pendulum_dataset/delayed_embedding_animation.gif", writer=gif_writer)
# ani.save("results/pendulum_dataset/delayed_embedding_animation.mp4")
````

In some advanced deep learning models, delayed embedding plays a crucial role. For example, Large Language Models (LLM) can perform good forecasts by taking in delayed embedding of time series[^/time-series/timeseries-data.time-delayed-embedding/#bib-Rasul2023-ep].

---

**Related:** [Time Series Forecasting Tasks](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>) · [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>)

[^/time-series/timeseries-data.time-delayed-embedding/#bib-Hewamalage2022-sc]: Hewamalage H., Ackermann K., Bergmeir C.. Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices. 2022. [http://arxiv.org/abs/2203.10716](<http://arxiv.org/abs/2203.10716>)

[^/time-series/timeseries-data.time-delayed-embedding/#bib-Zivot2006-es]: Zivot E., Wang J.. Modeling Financial Time Series with S-PLUS. *Springer New York*. 2006. [https://link.springer.com/book/10.1007/978-0-387-32348-0](<https://link.springer.com/book/10.1007/978-0-387-32348-0>)

[^/time-series/timeseries-data.time-delayed-embedding/#bib-Takens1981-kh]: Takens F.. Detecting strange attractors in turbulence. *Lecture Notes in Mathematics*. 1981. [http://dx.doi.org/10.1007/bfb0091924](<http://dx.doi.org/10.1007/bfb0091924>)

[^/time-series/timeseries-data.time-delayed-embedding/#bib-Rasul2023-ep]: Rasul K., Ashok A., Williams A.R. et al.. Lag-Llama: Towards foundation models for time series forecasting. *arXiv \[cs.LG\]*. 2023. [http://arxiv.org/abs/2310.08278](<http://arxiv.org/abs/2310.08278>)
