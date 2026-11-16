# TimeGrad Using Diffusion Model

> [!NOTE]
>
> **In one minute**
>
> TimeGrad (Rasul et al., 2021) turns the denoising diffusion model into a probabilistic forecaster for multivariate series. The forecast is factorized autoregressively over time, an RNN state carries the time dynamics into the denoising network, and repeating the reverse process many times gives the quantiles.
>
> **Read first:** [Denoising Diffusion Probabilistic Models](<https://dl.leima.is/energy-based-models/ebm.diffusion/index.md>) · [Recurrent Neural Networks](<https://dl.leima.is/deep-learning-fundamentals/recurrent-neural-networks/index.md>)

Rasul et al., (2021) proposed a probabilistic forecasting model using denoising diffusion models.

## Autoregressive

> [!NOTE]
>
> **Multivariate Forecasting Problem**
>
> Given an input sequence \\(\\mathbf x\_{t-K: t}\\), we forecast \\(\\mathbf x\_{t+1:t+H}\\).
>
> See this section for more [Time Series Forecasting Tasks](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md#forecasting-multivariate-time-series>).

> [!NOTE]
>
> **Notation**
>
> We use \\(x^0\\) to denote the actual time series. The super script \\({}^{0}\\) will be used to represent the non-diffused values.

To apply the [denoising diffusion model](<https://dl.leima.is/energy-based-models/ebm.diffusion/index.md>) in a multivariate forecasting problem, we define our forecasting task as the following autoregressive problem,

\\\[ q(\\mathbf x^0\_{t\_0:T} \\vert \\mathbf x^0\_{1:t\_0 - 1}) = \\Pi\_{t=t\_0}^T q(\\mathbf x^0\_t \\vert \\mathbf x^0\_{1:t-1}). \\\]

![AR](<https://dl.leima.is/time-series-deep-learning/assets/timeseries.deep-learning.timegrad/ar-denoising-diffusion-model-problem.jpg>)

At each time step \\(t\\), we build a denoising diffusion model.

## Time Dynamics

Note that in the denoising diffusion model, we minimize

\\\[ \\operatorname{min}\_\\theta \\mathbb E\_{q(\\mathbf x^0)} \\left\[ -\\log p\_\\theta (\\mathbf x^0) \\right\] \\\]

The above loss becomes that of the denoising model for a single time step. Explicitly,

\\\[ \\operatorname{min}\_\\theta \\mathbb E\_{q(\\mathbf x^0\_t )} \\left\[ -\\log p\_\\theta (\\mathbf x^0\_t) \\right\]. \\\]

Time dynamics can be easily captured by some RNN. To include the time dynamics, we use the RNN state built using the time series data of the previous time step \\(\\mathbf h\_{t-1}\\)[^/time-series-deep-learning/timeseries.deep-learning.timegrad/#Rasul2021]

\\\[ \\operatorname{min}\_\\theta \\mathbb E\_{q(\\mathbf x^0\_t )} \\left\[ -\\log p\_\\theta (\\mathbf x^0\_t \\vert \\mathbf h\_{t-1}) \\right\]. \\\]

Apart from the usual time dimension \\(t\\), the autoregressive denoising diffusion model has another dimension to optimize: the diffusion step \\(n\\) for each time \\(t\\).

The loss for each time step \\(t\\) is[^/time-series-deep-learning/timeseries.deep-learning.timegrad/#Rasul2021]

\\\[ \\begin{equation} \\mathcal L\_t = \\mathbb E\_{\\mathbf x^0\_t, \\epsilon, n} \\left\[ \\lVert \\epsilon - \\epsilon\_\\theta ( \\sqrt{\\bar \\alpha\_n} \\mathbf x^0\_t + \\sqrt{1-\\bar \\alpha\_n}\\epsilon, \\mathbf h\_{t-1}, n ) \\rVert^2 \\right\]. \\label{eq:ddpm-loss} \\end{equation} \\\]

That being said, we just need to minimize \\(\\mathcal L\_t\\) for each time step \\(t\\).

## Training Algorithm

The input data is sliced into fixed length time series \\(\\mathbf x\_t^0\\). Since Eq \\eqref{eq:ddpm-loss} shows that a loss can be calculated for arbitrary \\(n\\) without depending on any previous diffusion steps \\(n-1\\), the training can be done by both random sampling in \\(\\mathbf x\_t^0\\) and \\(n\\). See Rasul et al. (2021)[^/time-series-deep-learning/timeseries.deep-learning.timegrad/#Rasul2021].

![Training](<https://dl.leima.is/time-series-deep-learning/assets/timeseries.deep-learning.timegrad/ar-ddpm-train.png>)

## How to Forecast

After training, we obtain the time dynamics encoding \\(\\mathbf h\_T\\), with which the denoising steps can be calculated using the reverse process

\\\[ \\mathbf x^{n-1}\_{T+1} = \\frac{1}{\\sqrt{\\alpha\_n}} \\left( \\mathbf x^n\_{T+1} - \\frac{\\beta\_n}{\\sqrt{1 - \\bar\\alpha\_n}} \\epsilon\_\\theta( \\mathbf x^n\_{T+1}, \\mathbf h\_{T}, n ) \\right) + \\sqrt{\\Sigma\_\\theta} \\mathbf z, \\\]

where \\(\\mathbf z \\sim \\mathcal N(\\mathbf 0, \\mathbf I)\\).

For example,

\\\[ \\mathbf x^{0}\_{T+1} = \\frac{1}{\\sqrt{\\alpha\_1}} \\left( \\mathbf x^1\_{T+1} - \\frac{\\beta\_1}{\\sqrt{1 - \\bar\\alpha\_1}} \\epsilon\_\\theta( \\mathbf x^1\_{T+1}, \\mathbf h\_{T}, 1 ) \\right) + \\sqrt{\\Sigma\_\\theta} \\mathbf z. \\\]

## It is Probabilistic

The quantiles is calculated by repeating many times for each forecasted time step[^/time-series-deep-learning/timeseries.deep-learning.timegrad/#Rasul2021].

## Code

An implementation of the model can be found in the package [pytorch-ts](<https://github.com/zalandoresearch/pytorch-ts>) [^/time-series-deep-learning/timeseries.deep-learning.timegrad/#bib-Rasul_PyTorchTS].

---

**Related:** [Time Series Forecasting Tasks](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>) · [Autoregressive Model](<https://dl.leima.is/self-supervised/generative/autoregressive/index.md>)

[^/time-series-deep-learning/timeseries.deep-learning.timegrad/#Rasul2021]: Rasul K, Seward C, Schuster I, Vollgraf R. Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting. arXiv \[cs.LG\]. 2021. Available: [http://arxiv.org/abs/2101.12072](<http://arxiv.org/abs/2101.12072 >)

[^/time-series-deep-learning/timeseries.deep-learning.timegrad/#bib-Rasul_PyTorchTS]: Rasul K.. PyTorchTS. [https://github.com/zalandoresearch/pytorch-ts](<https://github.com/zalandoresearch/pytorch-ts>)
