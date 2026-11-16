# Notebooks and Utilities for Tutorials

> [!NOTE]
>
> **In one minute**
>
> How to run the notebooks of this book: install the Python environment with poetry, create a Jupyter kernel named deep-learning, and use the ts\_dl\_utils package, which holds the helpers shared by the tutorials.

All the notebooks are located in the folder `dl/notebooks`. To run these notebooks, we need to set up our Python environment first. We use [poetry](<https://python-poetry.org/>) to manage our Python environment. For the argument of this choice, please refer to [Engineering Tips](<https://dl.leima.is/engineering/python/index.md#python-environment-management>).

## Install Requirements and Create Jupyter Kernel

First of all, we need to install all the requirements,

```
poetry install
```

or install certain groups using

```
poetry install --with notebook,visualization,torch,darts
```

To create a Jupyter kernel for the notebooks, run

```
poetry run ipython kernel install --user --name=deep-learning
```

and a Jupyter kernel named `deep-learning` will be created.

## Utilities

We have a few utilities that we use in our tutorials. Most of them are located in the package `ts_dl_utils` located in the folder `dl/notebooks/ts_dl_utils`.

In principle, the notebooks we provided should work without installing this package. The package is also installed in the environment if you run `poetry install` just in case one uses the kernel `deep-learning` created above to run some personal notebooks located in other folders.

---

**Related:** [Python](<https://dl.leima.is/engineering/python/index.md>) · [Time Series Forecasting with Deep Learning](<https://dl.leima.is/time-series-deep-learning/index.md>)
