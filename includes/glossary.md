<!-- markdownlint-disable-file MD041 -->
<!-- Glossary: appended to every page by pymdownx.snippets (auto_append in
mkdocs.yml). Each term becomes a tooltip wherever it appears as a word. -->

*[MAE]: Mean absolute error: the average size of the forecast errors, in the data's units.
*[MSE]: Mean squared error: the average squared forecast error, so large errors count much more.
*[RMSE]: Root mean squared error: the square root of the MSE, back in the data's units.
*[MAPE]: Mean absolute percentage error: the average error relative to the true value; undefined when the truth is 0.
*[sMAPE]: Symmetric MAPE: the error relative to the average size of truth and forecast; between 0% and 200%.
*[wMAPE]: Weighted MAPE: percentage errors weighted, for example by volume.
*[MASE]: Mean absolute scaled error: the MAE divided by the in-sample MAE of a naive forecast.
*[RMSLE]: Root mean squared logarithmic error: the RMSE of log(1 + value), for relative errors on positive data.
*[CRPS]: Continuous ranked probability score: how far a forecast distribution is from the observed value; lower is better.
*[CDF]: Cumulative distribution function: the probability that the variable is at most x.
*[RNN]: Recurrent neural network: a network that reads a sequence step by step, carrying a hidden state.
*[LSTM]: Long short-term memory: a recurrent network with gates that control what it remembers.
*[GRU]: Gated recurrent unit: a simpler gated recurrent network than the LSTM.
*[VAE]: Variational autoencoder: a generative model that encodes data into a distribution over a latent space.
*[GAN]: Generative adversarial network: a generator and a discriminator trained against each other.
*[ELBO]: Evidence lower bound: the training objective of a VAE, a lower bound on the data log-likelihood.
*[DTW]: Dynamic time warping: aligns two series by locally stretching or compressing time.
*[MLP]: Multilayer perceptron: a feedforward neural network of fully connected layers.
*[CNN]: Convolutional neural network: a network built from shared, sliding filters.
*[ODE]: Ordinary differential equation.
*[ARIMA]: Autoregressive integrated moving average: a classical statistical forecasting model.
