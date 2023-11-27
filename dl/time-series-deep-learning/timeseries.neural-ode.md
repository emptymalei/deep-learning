# Time Series Forecasting with Neural ODE

!!! info ":material-code-json: Jupyter Notebook Available"
    We have a [:notebook: notebook](../../notebooks/neuralode_timeseries) for this section which includes all the code used in this section.


!!! info ":simple-abstract: Introduction to Neural Networks"
    We explain the theories of neural networks in [this section](../dynamical-systems/neural-ode.md). Please read it first if you are not familiar with neural ode.


In the section [Neural ODE](../dynamical-systems/neural-ode.md), we have introduced the concept of neural ODE. In this section, we will show how to use neural ODE to do time series forecasting.

## A Nueral ODE Model

We built a single hidden layer neural network as the field,

```mermaid
graph TD
    input["Input (100)"]
    input_layer["Hidden Layer (100)"]
    output_layer["Output Layer (100)"]
    hidden_layer["Hidden Layer (256)"]
    output["Output (1)"]

    input --> input_layer
    input_layer --> hidden_layer
    hidden_layer --> output_layer
    output_layer --> output
```

The model is built using the package called [torchdyn](https://github.com/DiffEqML/torchdyn) [@Poli_TorchDyn_Implicit_Models].

!!! note "Packages"

    Apart from the torchdyn package we used here, there is another package called [torchdiffeq](https://github.com/rtqichen/torchdiffeq) [@Chen_torchdiffeq_2021] which is developed by the authors of neural ode.


## Result

We trained the model using history length of 100 and only forecast one step ahead. The result is shown below.

![Results](../assets/timeseries.neural-ode/neuralode_univariate_results.png)

The metrics are also computed and listed below.

| Metric | Value |
| --- | --- |
| Mean Absolute Error | 0.0049 |
| Mean Squared Error | 2.4341e-05 |
| Symmetric Mean Absolute Percentage Error | 0.0315 |
