# Data Augmentation for Time Series

> [!NOTE]
>
> **In one minute**
>
> Real datasets are often too small or too imbalanced to train deep models well. Augmentation creates extra training series by random transformations such as jittering, by mixing patterns from several series (for example DBA), or with generative models from AR to GANs. The page then covers how to use the synthetic data in training.

In deep learning, our dataset should help the optimization mechanism locate a good spot in the parameter space. However, real-world data is not necessarily diverse enough that covers the required situations with enough records. For example, some datasets may be extremely imbalanced class labels which leads to poor performance in classification tasks [^/time-series/timeseries-data.data-augmentation/#bib-Hasibi2019-in]. Another problem with a limited dataset is that the trained model may not generalize well [^/time-series/timeseries-data.data-augmentation/#bib-Iwana2020-oc][^/time-series/timeseries-data.data-augmentation/#bib-Shorten2019-ty].

We will cover two topics in this section: Augmenting the dataset and application of the augmented data to model training.

## Augmenting the Dataset

There are many different ways of augmenting time series data [^/time-series/timeseries-data.data-augmentation/#bib-Iwana2020-oc][^/time-series/timeseries-data.data-augmentation/#bib-Wen2020-ez]. We categorize the methods into the following groups:

- Random transformations, e.g., jittering;
- Pattern mixing, e.g., DBA;[^/time-series/timeseries-data.data-augmentation/#bib-Petitjean2011-sj]
- Generative models, e.g.,

  - phenomenological generative models such as AR [^/time-series/timeseries-data.data-augmentation/#bib-Kang2019-cl],
  - first principle models such as economical models [^/time-series/timeseries-data.data-augmentation/#bib-Stock2016-mh],
  - deep generative models such as TimeGAN or TS GAN [^/time-series/timeseries-data.data-augmentation/#bib-Yoon_undated-gs][^/time-series/timeseries-data.data-augmentation/#bib-Brophy2021-vn].

We also treat the first two methods, random transformations and pattern mixing as basic methods.

### Basic Methods

In the following table, we group some of the data augmentation methods by two dimensions, the category of the method, and the domain of where the method is applied.

|  | Projected Domain | Time Scale | Magnitude |
| --- | --- | --- | --- |
| Random Transformation | Frequency Masking, Frequency Warping, Fourier Transform, STFT | Permutation, Slicing, Time Warping, Time Masking, Cropping | Jittering, Flipping, Scaling, Magnitude Warping |
| Pattern Mixing | EMDA[^/time-series/timeseries-data.data-augmentation/#bib-Takahashi2017-yz], SFM[^/time-series/timeseries-data.data-augmentation/#bib-Cui2014-de] | Guided Warping[^/time-series/timeseries-data.data-augmentation/#bib-Iwana2020-fe] | DFM[^/time-series/timeseries-data.data-augmentation/#bib-Stock2016-mh], Interpolation, DBA[^/time-series/timeseries-data.data-augmentation/#bib-Petitjean2011-sj] |

For completeness, we will explain some of the methods in more detail in the following.

#### Perturbation in Fourier Domain

In the Fourier domain, for each the amplitude \\(A\_f\\) and phase \\(\\phi\_f\\) at a specific frequency, we can perform[^/time-series/timeseries-data.data-augmentation/#bib-Gao2020-qr]

- magnitude replacement using a Gaussian distribution, and
- phase shift by adding Gaussian noise.

We perform such perturbations at some chosen frequency.

#### Slicing, Permutation, and Bootstrapping

We can slice a series into small segments. With the slices, we can perform different operations to create new series.

- Window Slicing (**WS**): In a classification task, we can take the slices from the original series and assign the same class label to the slice [^/time-series/timeseries-data.data-augmentation/#bib-Le_Guennec2016-zi]. The slices can also be interpolated to match the length of the original series [^/time-series/timeseries-data.data-augmentation/#bib-Iwana2020-oc].
- Permutation: We take the slices and permute them to form a new series [^/time-series/timeseries-data.data-augmentation/#bib-Um2017-oq].
- Moving Block Bootstrapping (**MBB**): First, we remove the trend and seasonability. Then we draw blocks of fixed length from the residual of the series until the desired length of the series is met. Finally, we combine the newly formed residual with trend and seasonality to form a new series [^/time-series/timeseries-data.data-augmentation/#bib-Bergmeir2016-eh].

#### Warping

Both the time scale and magnitude can be warped. For example,

- Time Warping: We distort time intervals by taking a range of data points and upsample or downsample it [^/time-series/timeseries-data.data-augmentation/#bib-Wen2020-ez].
- Magnitude Warping: the magnitude of the time series is rescaled.

> [!NOTE]
>
> **Dynamic Time Warping (**DTW**)**
>
> Given two sequences, \\(S^{(1)}\\) and \\(S^{(2)}\\), the Dynamic Time Warping (**DTW**) algorithm finds the best way to align two sequences. During this alignment process, we quantify the misalignment using a distance similar to the Levenshtein distance, where the distance between two series \\(S^{(1)}\_{1:i}\\) (with \\(i\\) elements) and \\(S^{(2)}\_{1:j}\\) (with \\(j\\) elements) is[^/time-series/timeseries-data.data-augmentation/#bib-Petitjean2011-sj]
>
> \\\[ \\begin{align} D(S^{(1)}\_{1:i}, S^{(2)}\_{1:j}) =&amp; d(S^{(1)}\_i, S^{(2)}\_j)\\\\ &amp; + \\operatorname{min}\\left\[ D(S^{(1)}\_{1:i-1}, S^{(2)}\_{1:j-1}), D(S^{(1)}\_{1:i}, S^{(2)}\_{1:j-1}), D(S^{(1)}\_{1:i-1}, S^{(2)}\_{1:j}) \\right\], \\end{align} \\\]
>
> where \\(S^{(1)}\_i\\) is the \\(i\\)the element of the series \\(S^{(1)}\\), \\(d(x,y)\\) is a predetermined distance, e.g., Euclidean distance. This definition reveals the recursive nature of the DTW distance.
>
> > **Notations in the Definition: \\(S\_{1:i}\\) and \\(S\_{i}\\)**
> >
> > The notation \\(S\_{1:i}\\) stands for a series that contains the elements starting from the first to the \\(i\\)th in series \\(S\\). For example, we have a series
> >
> > \\\[ S^1 = \[s^1\_1, s^1\_2, s^1\_3, s^1\_4, s^1\_5, s^1\_6\]. \\\]
> >
> > The notation \\(S^1\_{1:4}\\) represents
> >
> > \\\[ S^1\_{1:4} = \[s^1\_1, s^1\_2, s^1\_3, s^1\_4\]. \\\]
> >
> > The notation \\(S\_i\\) indicates the \\(i\\)th element in \\(S\\). For example,
> >
> > \\\[ S^1\_4 = s^1\_4. \\\]
> >
> > If we map these two notations to Python,
> >
> > - \\(S\_{1:i}\\) is equivalent to `S[0:i]`, and
> > - \\(S\_i\\) is equivalent to `S[i-1]`.
> >
> > Note that the indices in Python look strange. This is also the reason we choose to use subscripts not square brackets in our definition.
>
> > **Levenshtein Distance**
> >
> > Given two words, e.g., \\(w^{a} = \\mathrm{cats}\\) and \\(w^{b} = \\mathrm{katz}\\). Suppose we can only use three operations: insertions, deletions and substitutions. The Levenshtein distance calculates the number of such operations needed to change from the first word \\(w^a\\) to the second one \\(w^b\\) by applying single-character edits. In this example, we need two replacements, i.e., `"c" -> "k"` and `"s" -> "z"`.
> >
> > The Levenshtein distance can be solved using recursive algorithms [^/time-series/timeseries-data.data-augmentation/#trekhleb].
>
> DTW is very useful when comparing series with different lengths. For example, most error metrics require the actual time series and predicted series to have the same length. In the case of different lengths, we can perform DTW when calculating these metrics[^/time-series/timeseries-data.data-augmentation/#darts-dtw-metric].
>
> The forecasting package darts provides a [demo of DTW](<https://unit8co.github.io/darts/examples/12-Dynamic-Time-Warping-example.html>).

> [!NOTE]
>
> **DTW Barycenter Averaging**
>
> [DTW](<https://dl.leima.is/concepts/timeseries-data.dtw/index.md>) Barycenter Averaging (**DBA**) constructs a series \\(\\bar{\\mathcal S}\\) out of a set of series \\(\\{\\mathcal S^{(\\alpha)}\\}\\) so that \\(\\bar{\\mathcal S}\\) is the barycenter of \\(\\{\\mathcal S^{(\\alpha)}\\}\\) measured by Dynamic Time Warping (**DTW**) distance [^/time-series/timeseries-data.data-augmentation/#bib-Petitjean2011-sj].
>
> ## Barycenter Averaging Based on DTW Distance
>
> Petitjean et al proposed a time series averaging algorithm based on DTW distance which is dubbed DTW Barycenter Averaging (**DBA**).
>
> > **DBA Implementation**
> >
> > [https://github.com/fpetitjean/DBA](<https://github.com/fpetitjean/DBA>)

### Series Mixing

Another class of data augmentation methods is mixing the series. For example, we take two randomly drawn series and average them using DTW Barycenter Averaging (**DBA**) [^/time-series/timeseries-data.data-augmentation/#bib-Petitjean2011-sj]. (DTW, dynamic time warping, is an algorithm to calculate the distance between sequential datasets by matching the data points on each of the series [^/time-series/timeseries-data.data-augmentation/#bib-Petitjean2011-sj][^/time-series/timeseries-data.data-augmentation/#bib-Hewamalage2019-tv].) To augment a dataset, we can choose from a list of strategies [^/time-series/timeseries-data.data-augmentation/#bib-Bandara2020-yp][^/time-series/timeseries-data.data-augmentation/#bib-Forestier2017-uk]:

- Average All series using different sets of weights to create new synthetic series.
- Average Selected series based on some strategies. For example, Forestier et al proposed choosing an initial series and combining it with its nearest neighbors [^/time-series/timeseries-data.data-augmentation/#bib-Forestier2017-uk].
- Average Selected with Distance is Average Selected but neighbors that are far from the initial series are down-weighted [^/time-series/timeseries-data.data-augmentation/#bib-Forestier2017-uk].

Some other similar methods are

- Equalized Mixture Data Augmentation (**EMDA**) calculates the weighted average of spectrograms of the same class label[^/time-series/timeseries-data.data-augmentation/#bib-Takahashi2017-yz].
- Stochastic Feature Mapping (**SFM**) is a data augmentation method in audio data[^/time-series/timeseries-data.data-augmentation/#bib-Cui2014-de].

### Data Generating Process

Time series data can also be augmented using some assumed data generating process (**DGP**). Some methods, such as GRATIS [^/time-series/timeseries-data.data-augmentation/#bib-Kang2019-cl], utilize simple generic methods such as AR/MAR. Some other methods, such as Gaussian Trees [^/time-series/timeseries-data.data-augmentation/#bib-Cao2014-mt], utilize more complicated hidden structures using graphs, which can approximate more complicated data generating processes. These methods do not necessarily reflect the actual data generating process but the data is generated using some parsimonious phenomenological models. Some other methods are more tuned toward detailed mechanisms. There are also methods using generative deep neural networks such as [GAN](<https://dl.leima.is/self-supervised/adversarial/gan/index.md>).

#### Dynamic Factor Model (**DFM**)

For example, we have a series \\(X(t)\\) which depends on a latent variable \\(f(t)\\)[^/time-series/timeseries-data.data-augmentation/#bib-Stock2016-mh],

\\\[ X(t) = \\mathbf A f(t) + \\eta(t), \\\]

where \\(f(t)\\) is determined by a differential equation

\\\[ \\frac{f(t)}{dt} = \\mathbf B f(t) + \\xi(t). \\\]

In the above equations, \\(\\eta(t)\\) and \\(\\xi(t)\\) are the irreducible noise.

The above two equations can be combined into one first-order differential equation.

Once the model is fit, it can be used to generate new data points. However, we will have to understand whether the data is generated in such processes.

## Applying the Synthetic Data to Model Training

Once we prepared the synthetic dataset, there are two strategies to include them in our model training [^/time-series/timeseries-data.data-augmentation/#bib-Bandara2020-yp].

| Strategy | Description |
| --- | --- |
| Pooled Strategy | Synthetic data + original data -\> model |
| Transfer Strategy | Synthetic data -\> pre-trained model; pre-trained model + original data -\> model |

The pooled strategy takes the synthetic data and original data then feeds them together into the training pipeline. The transfer strategy uses the synthetic data to pre-train the model, then uses transfer learning methods (e.g., freeze weights of some layers) to train the model on the original data.

---

**Related:** [Synthetic Time Series](<https://dl.leima.is/time-series/timeseries-synthetic/index.md>) · [Generating Processes for Time Series](<https://dl.leima.is/time-series/timeseries-datasets.dgp/index.md>)

[^/time-series/timeseries-data.data-augmentation/#trekhleb]: trekhleb. javascript-algorithms/src/algorithms/string/levenshtein-distance at master · trekhleb/javascript-algorithms. In: GitHub \[Internet\]. \[cited 27 Jul 2022\]. Available: [https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance](<https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance >)

[^/time-series/timeseries-data.data-augmentation/#darts-dtw-metric]: Unit8. Metrics — darts documentation. In: Darts \[Internet\]. \[cited 7 Mar 2023\]. Available: [https://unit8co.github.io/darts/generated\_api/darts.metrics.metrics.html?highlight=dtw\#darts.metrics.metrics.dtw\_metric](<https://unit8co.github.io/darts/generated_api/darts.metrics.metrics.html?highlight=dtw#darts.metrics.metrics.dtw_metric >)

[^/time-series/timeseries-data.data-augmentation/#bib-Hasibi2019-in]: Hasibi R., Shokri M., Dehghan M.. Augmentation Scheme for Dealing with Imbalanced Network Traffic Classification Using Deep Learning. 2019. [http://arxiv.org/abs/1901.00204](<http://arxiv.org/abs/1901.00204>)

[^/time-series/timeseries-data.data-augmentation/#bib-Iwana2020-oc]: Iwana B.K., Uchida S.. An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks. 2020. [http://arxiv.org/abs/2007.15951](<http://arxiv.org/abs/2007.15951>)

[^/time-series/timeseries-data.data-augmentation/#bib-Shorten2019-ty]: Shorten C., Khoshgoftaar T.M.. A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*. 2019. [https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0](<https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0>)

[^/time-series/timeseries-data.data-augmentation/#bib-Wen2020-ez]: Wen Q., Sun L., Yang F. et al.. Time Series Data Augmentation for Deep Learning: A Survey. 2020. [http://arxiv.org/abs/2002.12478](<http://arxiv.org/abs/2002.12478>)

[^/time-series/timeseries-data.data-augmentation/#bib-Petitjean2011-sj]: Petitjean F., Ketterlin A., Gan\\c carski P.. A global averaging method for dynamic time warping, with applications to clustering. *Pattern recognition*. 2011. [https://www.sciencedirect.com/science/article/pii/S003132031000453X](<https://www.sciencedirect.com/science/article/pii/S003132031000453X>)

[^/time-series/timeseries-data.data-augmentation/#bib-Kang2019-cl]: Kang Y., Hyndman R.J., Li F.. GRATIS: GeneRAting TIme Series with diverse and controllable characteristics. 2019. [http://arxiv.org/abs/1903.02787](<http://arxiv.org/abs/1903.02787>)

[^/time-series/timeseries-data.data-augmentation/#bib-Stock2016-mh]: Stock J.H., Watson M.W.. Chapter 8 - Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics. *Handbook of Macroeconomics*. 2016. [https://www.sciencedirect.com/science/article/pii/S1574004816300027](<https://www.sciencedirect.com/science/article/pii/S1574004816300027>)

[^/time-series/timeseries-data.data-augmentation/#bib-Yoon_undated-gs]: Yoon J., Jarrett D., Schaar M.. Time-series generative adversarial networks. *Advances in Neural Information Processing Systems*. 2019. [https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html](<https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html>)

[^/time-series/timeseries-data.data-augmentation/#bib-Brophy2021-vn]: Brophy E., Wang Z., She Q. et al.. Generative adversarial networks in time series: A survey and taxonomy. 2021. [http://arxiv.org/abs/2107.11098](<http://arxiv.org/abs/2107.11098>)

[^/time-series/timeseries-data.data-augmentation/#bib-Takahashi2017-yz]: Takahashi N., Gygli M., Van Gool L.. AENet: Learning Deep Audio Features for Video Analysis. 2017. [http://arxiv.org/abs/1701.00599](<http://arxiv.org/abs/1701.00599>)

[^/time-series/timeseries-data.data-augmentation/#bib-Cui2014-de]: Cui X., Goel V., Kingsbury B.. Data Augmentation for deep neural network acoustic modeling. *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. 2014. [http://dx.doi.org/10.1109/ICASSP.2014.6854671](<http://dx.doi.org/10.1109/ICASSP.2014.6854671>)

[^/time-series/timeseries-data.data-augmentation/#bib-Iwana2020-fe]: Iwana B.K., Uchida S.. Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher. 2020. [http://arxiv.org/abs/2004.08780](<http://arxiv.org/abs/2004.08780>)

[^/time-series/timeseries-data.data-augmentation/#bib-Gao2020-qr]: Gao J., Song X., Wen Q. et al.. RobustTAD: Robust Time Series Anomaly Detection via Decomposition and Convolutional Neural Networks. 2020. [http://arxiv.org/abs/2002.09545](<http://arxiv.org/abs/2002.09545>)

[^/time-series/timeseries-data.data-augmentation/#bib-Le_Guennec2016-zi]: Le Guennec A., Malinowski S., Tavenard R.. Data augmentation for time series classification using convolutional neural networks. *ECML/PKDD workshop on advanced analytics and learning on temporal data*. 2016. [https://halshs.archives-ouvertes.fr/halshs-01357973/document](<https://halshs.archives-ouvertes.fr/halshs-01357973/document>)

[^/time-series/timeseries-data.data-augmentation/#bib-Um2017-oq]: Um T.T., Pfister F.M.J., Pichler D. et al.. Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks. 2017. [http://arxiv.org/abs/1706.00527](<http://arxiv.org/abs/1706.00527>)

[^/time-series/timeseries-data.data-augmentation/#bib-Bergmeir2016-eh]: Bergmeir C., Hyndman R.J., Ben\\'\\itez J.M.. Bagging exponential smoothing methods using STL decomposition and Box–Cox transformation. *International journal of forecasting*. 2016. [https://www.sciencedirect.com/science/article/pii/S0169207015001120](<https://www.sciencedirect.com/science/article/pii/S0169207015001120>)

[^/time-series/timeseries-data.data-augmentation/#bib-Hewamalage2019-tv]: Hewamalage H., Bergmeir C., Bandara K.. Recurrent Neural Networks for Time Series Forecasting: Current Status and Future Directions. 2019. [http://arxiv.org/abs/1909.00590](<http://arxiv.org/abs/1909.00590>)

[^/time-series/timeseries-data.data-augmentation/#bib-Bandara2020-yp]: Bandara K., Hewamalage H., Liu Y. et al.. Improving the Accuracy of Global Forecasting Models using Time Series Data Augmentation. 2020. [http://arxiv.org/abs/2008.02663](<http://arxiv.org/abs/2008.02663>)

[^/time-series/timeseries-data.data-augmentation/#bib-Forestier2017-uk]: Forestier G., Petitjean F., Dau H.A. et al.. Generating Synthetic Time Series to Augment Sparse Datasets. *2017 IEEE International Conference on Data Mining (ICDM)*. 2017. [http://dx.doi.org/10.1109/ICDM.2017.106](<http://dx.doi.org/10.1109/ICDM.2017.106>)

[^/time-series/timeseries-data.data-augmentation/#bib-Cao2014-mt]: Cao H., Tan V.Y.F., Pang J.Z.F.. A parsimonious mixture of Gaussian trees model for oversampling in imbalanced and multimodal time-series classification. *IEEE transactions on neural networks and learning systems*. 2014. [http://dx.doi.org/10.1109/TNNLS.2014.2308321](<http://dx.doi.org/10.1109/TNNLS.2014.2308321>)
