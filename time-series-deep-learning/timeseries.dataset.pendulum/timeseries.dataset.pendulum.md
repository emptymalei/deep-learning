---
tags:
  - WIP
---
# Pendulum Dataset

We create a synthetic dataset based on the physical model called pendulum. The pendulum is modeled as a damped harmonic oscillator, i.e.,

$$
\theta(t) = \theta(0) \cos(2 \pi t / p)\exp(-\beta t),
$$

where $\theta(t)$ is the angle of the pendulum at time $t$.
The period $p$ is calculated using

$$
p = 2 \pi \sqrt(L / g),
$$

with $L$ being the length of the pendulum and $g$ being the surface gravity.

=== "Pendulum Angle"

    ![Pendulum data](assets/timeseries.transformer/transformer_series_data_pendulum.png)

=== ":material-code-json: Physics and Pytorch Dataset"

    ```python
    import math
    from functools import cached_property
    from typing import Dict, List

    import pandas as pd

    class Pendulum:
        """Class for generating time series data for a pendulum.

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

    pen = Pendulum(length=100)
    df = pd.DataFrame(pen(10, 400, initial_angle=1, beta=0.001))

    _, ax = plt.subplots(figsize=(10, 6.18))
    df.plot(x="t", y="theta", ax=ax)
    ```

We take this time series and ask our model to forecast the next step (**forecast horizon is 1**).


!!! info "PyTorch Dataset and Lightning DataModule"

    In our tutorials, we will use Pytorch lightning excessively.
    We defined some useful modules in our [:material-language-python: `ts_dl_utils` package](../../utilities/notebooks-and-utilities)
    and [:notebook: this notebook](../../notebooks/pendulum_dataset).
