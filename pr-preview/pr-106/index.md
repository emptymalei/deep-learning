# Time Series Forecasting using Deep Learning

> [!NOTE]
>
> Order my new book: [Time Series with PyTorch: Modern Deep Learning Toolkit for Real-World Forecasting Challenges](<https://www.amazon.com/Time-PyTorch-Real-World-Forecasting-Challenges-ebook/dp/B0DK5LR6XL/>). [More about the book](<https://time-series.xyz>).

Forecasting the future is an extremely valuable superpower. The forecasting game has been dominated by statisticians who are real experts in time series problems. As the amount of data increases, many of the statistical methods are not squeezing enough out of the massive datasets. Consequently, time series forecasting using deep learning emerges and became a fast-growing field. It is trendy, not only in LinkedIn debates but also in academic papers. We plotted the number of related publications per year using [the keyword "deep learning forecasting" on dimensions.ai](<https://app.dimensions.ai/analytics/publication/overview/timeline?search_mode=content&amp;search_text=deep learning forecasting&amp;search_type=kws&amp;search_field=full_search&amp;year_from=2015&amp;year_to=2021>)) [^/#bib-dimensionsai].

![deep-learning-forecasting-dimension-ai](<https://dl.leima.is/assets/images/deep-learning-forecasting-dimension-ai.png>)

This chart is obtained on 2022-08-06, from Digital Science’s Dimensions platform, available at https://app.dimensions.ai

On the other hand, deep learning methods are not yet winning all the games of forecasting. Time series forecasting is a complicated problem with a great variety of data generating processes (DGP). Some argue that we don't need deep learning to forecast since well tuned statistical models and trees are already performing well and are faster and more interpretable than deep neural networks[^/#bib-Elsayed2021-ug] [^/#bib-Grinsztajn2022-mu]. Ensembles of statistical models performing great, even outperforming many deep learning models on the [M3 data](<https://forecasters.org/resources/time-series-data/m3-competition/>)[^/#nixtla-m3-ensemble].

However, deep learning models are picking up speed. In the [M5 competition](<https://mofc.unic.ac.cy/m5-competition/>), deep learning "have shown forecasting potential, motivating further research in this direction"[^/#bib-Makridakis2022-hb]. As the complexity and size of time series data are growing and more and more deep learning forecasting models are being developed, forecasting with deep learning is on the path to be an important alternative to statistical forecasting methods.

In [Coding Tips](<https://dl.leima.is/engineering/index.md>), we provide coding tips to help some readers set up the development environment. In [Deep Learning Fundamentals](<https://dl.leima.is/deep-learning-fundamentals/index.md>), we introduce the fundamentals of deep neural networks and their practices. For completeness, we also provide code and derivations for the models. With these two parts, we introduce time series data and statistical forecasting models in [Time Series Forecasting Fundamentals](<https://dl.leima.is/time-series/index.md>), where we discuss methods to analyze time series data, several universal data generating processes of time series data, and some statistical forecasting methods. Finally, we fulfill our promise in the title in [Time Series Forecasting with Deep Learning](<https://dl.leima.is/time-series-deep-learning/index.md>).

## Where to Start

Pick the path that fits what you came for.

- **I want to forecast with deep learning.** [Time Series Forecasting with Deep Learning](<https://dl.leima.is/time-series-deep-learning/index.md>) → [Forecasting with MLP](<https://dl.leima.is/time-series-deep-learning/timeseries.feedforward/index.md>) → [Forecasting with RNN](<https://dl.leima.is/time-series-deep-learning/timeseries.rnn/index.md>) → [Forecasting with Transformers](<https://dl.leima.is/time-series-deep-learning/timeseries.transformer/index.md>)
- **I am new to time series.** [Terminologies of Time Series Data](<https://dl.leima.is/time-series/timeseries-data.analysis/index.md>) → [Time Series Forecasting Tasks](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>) → [Naive Forecasts](<https://dl.leima.is/time-series/timeseries-forecast.naive/index.md>) → [Statistical Models](<https://dl.leima.is/time-series/timeseries-basics.statistical-models/index.md>)
- **I need to judge a forecast.** [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>) → [Time Series Forecasting Metrics](<https://dl.leima.is/time-series/timeseries-metrics.forecasting/index.md>) → [CRPS](<https://dl.leima.is/time-series/timeseries-metrics.forecasting.crps/index.md>)
- **I want the theory behind the models.** [Neural Networks](<https://dl.leima.is/deep-learning-fundamentals/neural-net/index.md>) → [Recurrent Neural Networks](<https://dl.leima.is/deep-learning-fundamentals/recurrent-neural-networks/index.md>) → [Vanilla Transformers](<https://dl.leima.is/transformers/transformers.vanilla/index.md>) → [Diffusion Models](<https://dl.leima.is/energy-based-models/ebm.diffusion/index.md>)

Or see the whole book at once: the **[Map of the Book](<https://dl.leima.is/map/index.md>)** shows every page as a dot and every link between two pages as a line.

## Blueprint

The following is my first version of the blueprint.

- [x] Engineering Tips

  - [x] Environment, VSCode, Git, ...
  - [x] Python Project Tips
- [x] Fundamentals of Time Series Forecasting

  - [x] Time Series Data and Terminologies
  - [x] Transformation of Time Series
  - [x] Two-way Fixed Effects
  - [x] Time Delayed Embedding
  - [x] Data Generating Process (DGP)
  - [x] DGP: Langevin Equation
  - [x] Kindergarten Models for Time Series Forecasting

    - [x] Statistical Models
    - [x] Statistical Model: AR
    - [x] Statistical Model: VAR
  - [x] Synthetic Datasets

    - [x] Synthetic Time Series
    - [x] Creating Synthetic Dataset
  - [x] Data Augmentation
  - [x] Forecasting

    - [x] Time Series Forecasting Tasks
    - [x] Naive Forecasts
  - [x] Evaluation and Metrics

    - [x] Time Series Forecasting Evaluation
    - [x] Time Series Forecasting Metrics

      - [x] CRPS
  - [x] Hierarchical Time Series

    - [x] Hierarchical Time Series Data
    - [x] Hierarchical Time Series Reconciliation
  - [x] Some Useful Datasets
- [x] Trees

  - [x] Tree-based Models
  - [x] Random Forest
  - [x] Gradient Boosted Trees
  - [x] Forecasting with Trees
- [ ] Fundamentals of Deep Learning

  - [x] Deep Learning Introduction
  - [x] Learning from Data
  - [x] Neural Networks
  - [x] Recurrent Neural Networks
  - [x] Convolutional Neural Networks
  - [x] Transformers
  - [x] Dynamical Systems

    - [x] Why Dynamical Systems
    - [x] Neural ODE
  - [x] Energy-based Models

    - [x] Diffusion Models
  - [x] Generative Models

    - [x] Autoregressive Model
    - [x] Auto-Encoder
    - [x] Variational Auto-Encoder
    - [x] Flow
    - [x] Generative Adversarial Network (GAN)
- [ ] Time Series Forecasting with Deep Learning

  - [x] A Few Datasets
  - [x] Forecasting with MLP
  - [x] Forecasting with RNN
  - [x] Forecasting with Transformers

    - [ ] TFT
    - [ ] DLinear
    - [ ] NLinear
  - [ ] Forecasting with CNN
  - [ ] Forecasting with VAE
  - [ ] Forecasting with Flow
  - [ ] Forecasting with GAN
  - [x] Forecasting with Neural ODE
  - [x] Forecasting with Diffusion Models
- [ ] Extras Topics, Supplementary Concepts, and Code

  - [x] DTW and DBA
  - [x] f-GAN
  - [x] Info-GAN
  - [ ] Spatial-temporal Models, e.g., GNN
  - [ ] Conformal Prediction
  - [ ] Graph Neural Networks
  - [ ] Spiking Neural Networks
  - [x] Deep Infomax
  - [x] Contrastive Predictive Coding
  - [ ] MADE
  - [ ] MAF
  - [ ] ...

[^/#nixtla-m3-ensemble]: Nixtla. statsforecast/experiments/m3 at main · Nixtla/statsforecast. In: GitHub \[Internet\]. \[cited 12 Dec 2022\]. Available: [https://github.com/Nixtla/statsforecast/tree/main/experiments/m3](<https://github.com/Nixtla/statsforecast/tree/main/experiments/m3 >)

[^/#bib-dimensionsai]: Hook D.W., Porter S.J., Herzog C.. Dimensions: Building Context for Search and Evaluation. *Frontiers in Research Metrics and Analytics*. 2018. [https://app.dimensions.ai/details/publication/pub.1106289502](<https://app.dimensions.ai/details/publication/pub.1106289502>)

[^/#bib-Elsayed2021-ug]: Elsayed S., Thyssens D., Rashed A. et al.. Do We Really Need Deep Learning Models for Time Series Forecasting?. 2021. [http://arxiv.org/abs/2101.02118](<http://arxiv.org/abs/2101.02118>)

[^/#bib-Grinsztajn2022-mu]: Grinsztajn L., Oyallon E., Varoquaux G.. Why do tree-based models still outperform deep learning on tabular data?. 2022. [http://arxiv.org/abs/2207.08815](<http://arxiv.org/abs/2207.08815>)

[^/#bib-Makridakis2022-hb]: Makridakis S., Spiliotis E., Assimakopoulos V.. M5 accuracy competition: Results, findings, and conclusions. *International journal of forecasting*. 2022. [https://www.sciencedirect.com/science/article/pii/S0169207021001874](<https://www.sciencedirect.com/science/article/pii/S0169207021001874>)
