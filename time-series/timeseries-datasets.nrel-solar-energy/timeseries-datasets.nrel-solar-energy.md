# Time Series Dataset: Solar Energy


We download the time series data from [this link](https://www.nrel.gov/grid/solar-power-data.html).

> NREL's Solar Power Data for Integration Studies are synthetic solar photovoltaic (PV) power plant data points for the United States representing the year 2006.

When we downloaded Alabama on 2022-11-05, and loaded `Actual_30.45_-88.25_2006_UPV_70MW_5_Min.csv` as an example. We [found](https://deepnote.com/workspace/lm-3917ee58-3e0d-43ba-a6c8-13241298300c/project/time-series-notebooks-deae214e-e319-4268-ac24-de1038ff0b94) a lot of `0` entries (which is expected as the will be no power during dark nights).

|  Power is Zero  | Number of Records |
|:----:|:----:|
| False |   57868 |
| True  |   47252 |

The dataset contains multiple files with each file containing a time series with a time step of 5 minutes ([naming convention explained here](https://www.nrel.gov/grid/solar-power-data.html)).


=== "Example Plots"

    ![](assets/timeseries-datasets.nrel-solar-energy/nrel_solar_power_alabama_upv_70mw_example.png)

=== "Power Distribution in this Example"

    ![](assets/timeseries-datasets.nrel-solar-energy/nrel_solar_power_alabama_upv_70mw_example_distribution.png)

=== "Missing Values"

    ![](assets/timeseries-datasets.nrel-solar-energy/nrel_solar_power_alabama_upv_70mw_example_missingno.png)
