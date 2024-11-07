# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Time Series Data Generation

# +
import numpy as np
import pandas as pd
import plotly.express as px


# +
def profile_sin(t: np.ndarray, lambda_min: float, lambda_max: float) -> np.ndarray:
    """generate a sin wave profile for
    the expected number of visitors
    in every 10min for each hour during a day

    :param t: time in minutes
    :param lambda_min: minimum number of visitors
    :param lambda_max: maximum number of visitors
    """
    amplitude = lambda_max - lambda_min
    t_rescaled = (t - t.min()) / t.max() * np.pi

    return amplitude * np.sin(t_rescaled) + lambda_min


class KioskVisitors:
    """generate number of visitors for a kiosk store

    :param daily_profile: expectations of visitors
        in every 10min for each hour during a day
    """

    def __init__(self, daily_profile: np.ndarray):
        self.daily_profile = daily_profile
        self.daily_segments = len(daily_profile)

    def __call__(self, n_days: int) -> pd.DataFrame:
        """generate number of visitors for n_days

        :param n_days: number of days to generate visitors
        """
        visitors = np.concatenate(
            [np.random.poisson(self.daily_profile) for _ in range(n_days)]
        )

        df = pd.DataFrame(
            {
                "visitors": visitors,
                "time": np.arange(len(visitors)),
                "expectation": np.tile(self.daily_profile, n_days),
            }
        )

        return df


# -

# Create a sin profile

# +
t = np.arange(0, 12 * 60 / 5, 1)

daily_profile = profile_sin(t, lambda_min=0.5, lambda_max=10)
# -

# Generate a time series data representing the number of visitors to a Kiosk.

kiosk_visitors = KioskVisitors(daily_profile=daily_profile)

df_visitors = kiosk_visitors(n_days=10)

px.line(
    df_visitors,
    x="time",
    y=["visitors", "expectation"],
)
