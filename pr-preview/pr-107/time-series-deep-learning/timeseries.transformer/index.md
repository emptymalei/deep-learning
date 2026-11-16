# Transformers for Time Series Forecasting

> [!NOTE]
>
> **In one minute**
>
> An encoder-only transformer forecasts the univariate pendulum series: a linear layer shapes the input, the encoder processes it and a second linear layer produces the forecast. It beats the naive forecast one step and three steps ahead, and the page closes with the transformer variants designed for forecasting and the packages that implement them.
>
> **Read first:** [Vanilla Transformers](<https://dl.leima.is/transformers/transformers.vanilla/index.md>) · [Pendulum Dataset](<https://dl.leima.is/time-series-deep-learning/timeseries.dataset.pendulum/index.md>)

> [!NOTE]
>
> **Jupyter Notebook Available**
>
> We have a [📓 notebook](<https://dl.leima.is/notebooks/transformer_timeseries_univariate/index.md>) for this section which includes all the code used in this section.

> [!NOTE]
>
> **Introduction to Transformers**
>
> We explain the theories of transformers in [this section](<https://dl.leima.is/transformers/transformers.vanilla/index.md>). Please read it first if you are not familiar with transformers.

[Transformer](<https://dl.leima.is/transformers/transformers.vanilla/index.md>) is a good candidate for time series forecasting due to its sequence modeling capability[^/time-series-deep-learning/timeseries.transformer/#bib-Ahmed2022-wl][^/time-series-deep-learning/timeseries.transformer/#bib-Wen2022-fc]. In this section, we will introduce some basic ideas of transformer-based models for time series forecasting.

## Transformer for Univariate Time Series Forecasting

We take a simple [Univariate time series forecasting](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>) task as an example. There are implementations of transformers for multivariate time series forecasting with all sorts of covariates, but we focus on univariate forecasting problem for simplicity.

### Dataset

In this example, we use the [pendulum physics dataset](<https://dl.leima.is/time-series-deep-learning/timeseries.dataset.pendulum/index.md>).

### Model

We built a naive transformer that only has an encoder. The input is passed to a linear layer to convert the tensor to the shape accepted by the encoder. The tensor is then passed to the encoder. The output of the encoder is passed to another linear layer to convert the tensor to the shape of the output.

```
flowchart TD

input_linear_layer[Linear Layer for Input]
positional_encoder[Positional Encoder]
encoder[Encoder]
output_linear_layer[Linear Layer for Output]

input_linear_layer --> positional_encoder
positional_encoder --> encoder
encoder --> output_linear_layer
```

> [!NOTE]
>
> **Decoder is Good for Covariates**
>
> A decoder in a transformer model is good for capturing [future covariates](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>). In our problem, we do not have any covariates at all.

> [!NOTE]
>
> **Positional Encoder**
>
> In this experiment, we do not include positional encoder as it introduces more complexities but it doesn't help that much in our case[^/time-series-deep-learning/timeseries.transformer/#bib-Zhang2019-pq].

### Evaluations

> [!NOTE]
>
> **Training**
>
> The details for model training can be found in this [📓 notebook](<https://dl.leima.is/notebooks/transformer_timeseries_univariate/index.md>). We will skip the details but show the loss curve here.
>
> ![Training](<https://dl.leima.is/time-series-deep-learning/assets/timeseries.transformer/vanilla_transformer_univariate_pendulum_training.png>)

We trained the model using a history length of 50 and plotted the forecasts for a test dataset that was held out from training. The forecasts are plotted in red and the ground truth is plotted in blue.

![Vanilla Transformer Result (Univariate, Horizon 1)](<https://dl.leima.is/time-series-deep-learning/assets/timeseries.transformer/transformer_univariate_forecasting_result.png>)

The forecasts roughly captured the patterns of the pendulum. To quantify the results, we compute a few metrics.

| Metric | Vanilla Transformer | Naive |
| --- | --- | --- |
| Mean Absolute Error | **0.050232** | 0.092666 |
| Mean Squared Error | **0.003625** | 0.010553 |
| Symmetric Mean Absolute Percentage Error | **0.108245** | 0.376550 |

## Multi-horizon Forecasting

We perform a similar experiment for multi-horizon forecasting (horizon=3). We plot out some samples. In the plot, the orange-shaded regions are the predictions.

![Transformer m step result](<https://dl.leima.is/time-series-deep-learning/assets/timeseries.transformer/transformer_univariate_m_step_forecasting_result.png>)

To verify that the forecasts make sense, we also plot out a few samples.

![Transformer m step samples](<https://dl.leima.is/time-series-deep-learning/assets/timeseries.transformer/transformer_univariate_m_step_forecasting_samples.png>)

The following is a table of the metrics.

| Metric | Vanilla Transformer | Naive |
| --- | --- | --- |
| Mean Absolute Error | **0.057219** | 0.109485 |
| Mean Squared Error | **0.004241** | 0.014723 |
| Symmetric Mean Absolute Percentage Error | **0.112247** | 0.423563 |

## Generalization

The vanilla transformer has its limitations. For example, it doesn't capture the correlations between the series that well. There are many variants of transformers that are designed just for time series forecasting[^/time-series-deep-learning/timeseries.transformer/#bib-Lim2019-hd][^/time-series-deep-learning/timeseries.transformer/#bib-Wu2021-zp][^/time-series-deep-learning/timeseries.transformer/#bib-Zhou2020-ov][^/time-series-deep-learning/timeseries.transformer/#bib-Nie2022-ww][^/time-series-deep-learning/timeseries.transformer/#bib-Zhou2022-ry][^/time-series-deep-learning/timeseries.transformer/#bib-Liu2023-wf].

A few forecasting packages implemented transformers for time series forecasting. For example, the [neuralforecast](<https://github.com/Nixtla/neuralforecast>) package by Nixtla has [implemented](<https://nixtla.mintlify.app/neuralforecast/models.html#c-transformer-based>) TFT, Informer, AutoFormer, FEDFormer, and PatchTST, as of November 2023. An alternative is [darts](<https://github.com/unit8co/darts>). These packages provide documentation and we encourage the reader to check them out for more complicated use cases of transformer-based models.

---

**Related:** [Time Series Forecasting Tasks](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>) · [Forecasting with RNN](<https://dl.leima.is/time-series-deep-learning/timeseries.rnn/index.md>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Ahmed2022-wl]: Ahmed S., Nielsen I.E., Tripathi A. et al.. Transformers in time-series analysis: A tutorial. 2022. [http://dx.doi.org/10.1007/s00034-023-02454-8](<http://dx.doi.org/10.1007/s00034-023-02454-8>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Wen2022-fc]: Wen Q., Zhou T., Zhang C. et al.. Transformers in time series: A survey. 2022. [http://arxiv.org/abs/2202.07125](<http://arxiv.org/abs/2202.07125>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Zhang2019-pq]: Zhang Y., Jiang Q., Li S. et al.. You may not need order in time series forecasting. *arXiv \[cs.LG\]*. 2019. [http://arxiv.org/abs/1910.09620](<http://arxiv.org/abs/1910.09620>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Lim2019-hd]: Lim B., Arik S.O., Loeff N. et al.. Temporal fusion transformers for interpretable multi-horizon time series forecasting. 2019. [http://arxiv.org/abs/1912.09363](<http://arxiv.org/abs/1912.09363>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Wu2021-zp]: Wu H., Xu J., Wang J. et al.. Autoformer: Decomposition Transformers with Auto-Correlation for long-term series forecasting. 2021. [https://github.com/thuml/Autoformer](<https://github.com/thuml/Autoformer>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Zhou2020-ov]: Zhou H., Zhang S., Peng J. et al.. Informer: Beyond efficient transformer for long sequence time-series forecasting. 2020. [http://arxiv.org/abs/2012.07436](<http://arxiv.org/abs/2012.07436>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Nie2022-ww]: Nie Y., Nguyen N.H., Sinthong P. et al.. A time series is worth 64 words: Long-term forecasting with transformers. 2022. [http://arxiv.org/abs/2211.14730](<http://arxiv.org/abs/2211.14730>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Zhou2022-ry]: Zhou T., Ma Z., Wen Q. et al.. FEDformer: Frequency Enhanced Decomposed Transformer for long-term series forecasting. 2022. [http://arxiv.org/abs/2201.12740](<http://arxiv.org/abs/2201.12740>)

[^/time-series-deep-learning/timeseries.transformer/#bib-Liu2023-wf]: Liu Y., Hu T., Zhang H. et al.. ITransformer: Inverted Transformers are effective for time series forecasting. 2023. [http://arxiv.org/abs/2310.06625](<http://arxiv.org/abs/2310.06625>)
