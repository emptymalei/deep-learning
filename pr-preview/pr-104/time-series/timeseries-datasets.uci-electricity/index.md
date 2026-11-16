# Time Series Dataset: Electricity

> [!NOTE]
>
> **In one minute**
>
> Electricity consumption of 370 clients from the UCI repository, in 15-minute steps from 2011 to the start of 2015 with no missing values, plus code to load and clean it.

This dataset is provided [as the "ElectricityLoadDiagrams20112014 Data Set" on the UCI website](<https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014>). It is the time series of electricity consumption of 370 points/clients.

We download the time series data in zip format using [this link](<https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip>).

We [find](<https://deepnote.com/workspace/lm-3917ee58-3e0d-43ba-a6c8-13241298300c/project/time-series-notebooks-deae214e-e319-4268-ac24-de1038ff0b94>) that

- in total 140256 rows and 370 series,
- the earliest time is 2011-01-01 00:15:00,
- the latest time is 2015-01-01 00:00:00,
- a fixed time interval of 15 minutes.

We only plot out three series. We only plot every 100 time steps.

![](<https://dl.leima.is/time-series/assets/timeseries-datasets.uci-electricity/uci_electricity_examples.png>)

We fine no missing values.

![](<https://dl.leima.is/time-series/assets/timeseries-datasets.uci-electricity/uci_electricity_missingno.png>)

## Loading and Basic Cleaning

We provide some code to load the data from the UCI website.

```
import requests, zipfile, io
import pandas as pd

# Download from remote URL
data_uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"

r = requests.get(data_uri)

z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("data/uci_electricity/")

# Load as pandas dataframe
df = pd.read_csv(
    "data/uci_electricity/LD2011_2014.txt", delimiter=";", decimal=','
).rename(columns={"Unnamed: 0": "date"}).set_index("date")
df.index = pd.to_datetime(df.index)
```

---

**Related:** [Time Series Datasets](<https://dl.leima.is/time-series/timeseries-datasets/index.md>)
