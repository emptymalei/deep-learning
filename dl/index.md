---
hide:
  - navigation
---

!!! info ""

    Pre-order my new book: [Time Series with PyTorch: Modern Deep Learning Toolkit for Real-World Forecasting Challenges](https://www.amazon.com/Time-PyTorch-Real-World-Forecasting-Challenges-ebook/dp/B0DK5LR6XL/).
    

# Time Series Forecasting using Deep Learning

Forecasting the future is an extremely valuable superpower. The forecasting game has been dominated by statisticians who are real experts in time series problems. As the amount of data increases, many of the statistical methods are not squeezing enough out of the massive datasets. Consequently, time series forecasting using deep learning emerges and became a fast-growing field. It is trendy, not only in LinkedIn debates but also in academic papers. We plotted the number of related publications per year using [the keyword "deep learning forecasting" on dimensions.ai](https://app.dimensions.ai/analytics/publication/overview/timeline?search_mode=content&search_text=deep%20learning%20forecasting&search_type=kws&search_field=full_search&year_from=2015&year_to=2021)) [@dimensionsai].

<figure markdown>
  ![deep-learning-forecasting-dimension-ai](assets/images/deep-learning-forecasting-dimension-ai.png)
  <figcaption>This chart is obtained on 2022-08-06, from Digital Science’s Dimensions platform, available at https://app.dimensions.ai</figcaption>
</figure>

On the other hand, deep learning methods are not yet winning all the games of forecasting. Time series forecasting is a complicated problem with a great variety of data generating processes (DGP). Some argue that we don't need deep learning to forecast since well tuned statistical models and trees are already performing well and are faster and more interpretable than deep neural networks[@Elsayed2021-ug][@Grinsztajn2022-mu]. Ensembles of statistical models performing great, even outperforming many deep learning models on the [M3 data](https://forecasters.org/resources/time-series-data/m3-competition/)[^nixtla-m3-ensemble].

However, deep learning models are picking up speed. In the [M5 competition](https://mofc.unic.ac.cy/m5-competition/), deep learning "have shown forecasting potential, motivating further research in this direction"[@Makridakis2022-hb]. As the complexity and size of time series data are growing and more and more deep learning forecasting models are being developed, forecasting with deep learning is on the path to be an important alternative to statistical forecasting methods.

In [Coding Tips](engineering/index.md), we provide coding tips to help some readers set up the development environment. In [Deep Learning Fundamentals](deep-learning-fundamentals/index.md), we introduce the fundamentals of deep neural networks and their practices. For completeness, we also provide code and derivations for the models. With these two parts, we introduce time series data and statistical forecasting models in [Time Series Forecasting Fundamentals](time-series/index.md), where we discuss methods to analyze time series data, several universal data generating processes of time series data, and some statistical forecasting methods. Finally, we fulfill our promise in the title in [Time Series Forecasting with Deep Learning](time-series-deep-learning/index.md).


## :material-floor-plan: Blueprint

The following is my first version of the blueprint.

- [x] Engineering Tips
    - [x] Environment, VSCode, Git, ...
    - [x] Python Project Tips
- [x] Fundamentals of Time Series Forecasting
    - [x]  Time Series Data and Terminologies
    - [x]  Transformation of Time Series
    - [x]  Two-way Fixed Effects
    - [x]  Time Delayed Embedding
    - [x]  Data Generating Process (DGP)
    - [x]  DGP: Langevin Equation
    - [x]  Kindergarten Models for Time Series Forecasting
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

[^nixtla-m3-ensemble]: Nixtla. statsforecast/experiments/m3 at main · Nixtla/statsforecast. In: GitHub [Internet]. [cited 12 Dec 2022]. Available: https://github.com/Nixtla/statsforecast/tree/main/experiments/m3
