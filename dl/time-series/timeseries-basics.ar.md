# AR

Autoregressive (AR) models are simple model to model time series. A general AR(p) model is described by the following process:

$$
s(t) = \phi_0 + \sum_{i=1}^p \phi_i s(t-i) + \epsilon.
$$


## AR(1)

A first order AR model, aka AR(1), is as simple as

$$
s(t) = \phi_0 + \phi_1 s(t-1) + \epsilon.
$$



By staring at this equation, we can build up our intuitions.

| $\phi_0$ | $\phi_1$ | $\epsilon$ | Behavior |
|:----:|:----:|:----:|:----:|
| - | $0$ | - | constant + noise |
| $0$  |  $1$  | -  |  constant + noise  |
| $0$ |  $\phi_1>1$ or $0\le\phi_1 \lt 1$  |  - |  exponential + noise |


!!! note "Exponential Behavior doesn't Always Approach Positive Inifinity"

    For example, the combination $\phi_0=0$ and $\phi_1>1$ without noise leads to exponential growth if the initial series value is positive. However, it approaches negative infinity if the initial series is negative.



=== "Constant"

    ![](assets/timeseries-basics.ar/ar1-phi0-0-phi1-1-std-0.1.png)

=== "Decay"

    ![Decay](assets/timeseries-basics.ar/ar1-phi0-0-phi1-0.9-std-0.1.png)

=== "Exponential"

    ![](assets/timeseries-basics.ar/ar1-phi0-0-phi1-1.1-std-0.1.png)

=== "Linear"

    ![](assets/timeseries-basics.ar/ar1-phi0-0.1-phi1-1-std-0.1.png)


[^Kumar2022]: Kumar A. Autoregressive (AR) models with Python examples. In: Data Analytics [Internet]. 25 Apr 2022 [cited 11 Aug 2022]. Available: https://vitalflux.com/autoregressive-ar-models-with-python-examples/
