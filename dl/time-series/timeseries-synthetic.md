# Synthetic Time Series

Synthetic time series data is useful in time series modeling, such as forecasting.

Real world time series data often comes with complex dynamics in the [data generating process](timeseries-generating-process.md). Benchmarking models using real world data often doesn't reflect the special designs in forecasting models. Synthetic time series data provides a good playground for benchmarking models as it can provide useful insights.

Another application of synthetic data is to improve model performance. Synthetic data can be used to [augment the training data](timeseries-data.data-augmentation.md)[^Bandara2020] as well as in transfer learning[^Rotem2022].

A third application of synthetic data is data sharing without compromising  privacy and business secret[^Lin2019].

Though being useful, synthesizing proper artificial time series data can be very complicated as there are enormous amount of diverse theories associated with time series data. In this chapter, we will explain the basic ideas and demonstrate our generic programming framework for synthetic time series. With the basics explored, we will focus on a special cases of synthetic time series: time series with interactions.


[^Rotem2022]: Rotem Y, Shimoni N, Rokach L, Shapira B. Transfer learning for time series classification using synthetic data generation. arXiv [cs.LG]. 2022. Available: http://arxiv.org/abs/2207.07897
[^Bandara2020]: Bandara K, Hewamalage H, Liu Y-H, Kang Y, Bergmeir C. Improving the Accuracy of Global Forecasting Models using Time Series Data Augmentation. arXiv [cs.LG]. 2020. Available: http://arxiv.org/abs/2008.02663
[^Lin2019]: Lin Z, Jain A, Wang C, Fanti G, Sekar V. Using GANs for Sharing Networked Time Series Data: Challenges, Initial Promise, and Open Questions. arXiv [cs.LG]. 2019. Available: http://arxiv.org/abs/1909.13403
