# Data Augmentation for Time Series


In deep learning, our dataset should help the optimization mechanism locate a good spot in the parameter space. However, real-world data is not necessarily diverse enough that covers the required situations with enough records. For example, some datasets may be extremely imbalanced class labels which leads to poor performance in classification tasks [@Hasibi2019-in]. Another problem with a limited dataset is that the trained model may not generalize well [@Iwana2020-oc; @Shorten2019-ty].

We will cover two topics in this section: Augmenting the dataset and application of the augmented data to model training.

## Augmenting the Dataset

There are many different ways of augmenting time series data [@Iwana2020-oc; @Wen2020-ez]. We categorize the methods into the following groups:

- Random transformations, e.g., jittering;
- Pattern mixing, e.g., DBA;[@Petitjean2011-sj]
- Generative models, e.g.,
    - phenomenological generative models such as AR [@Kang2019-cl],
    - first principle models such as economical models [@Stock2016-mh],
    - deep generative models such as TimeGAN or TS GAN [@Yoon_undated-gs; @Brophy2021-vn].

We also treat the first two methods, random transformations and pattern mixing as basic methods.

### Basic Methods

In the following table, we group some of the data augmentation methods by two dimensions, the category of the method, and the domain of where the method is applied.

|   |  Projected Domain  | Time Scale | Magnitude |
|---|---|---|---|
| Random Transformation | Frequency Masking, Frequency Warping, Fourier Transform, STFT  | Permutation, Slicing, Time Warping, Time Masking, Cropping   | Jittering, Flipping, Scaling, Magnitude Warping  |
| Pattern Mixing  | EMDA[@Takahashi2017-yz], SFM[@Cui2014-de]  | Guided Warping[@Iwana2020-fe]  | DFM[@Stock2016-mh], Interpolation, DBA[@Petitjean2011-sj]  |

For completeness, we will explain some of the methods in more detail in the following.

#### Perturbation in Fourier Domain

In the Fourier domain, for each the amplitude $A_f$ and phase $\phi_f$ at a specific frequency, we can perform[@Gao2020-qr]

- magnitude replacement using a Gaussian distribution, and
- phase shift by adding Gaussian noise.

We perform such perturbations at some chosen frequency.

#### Slicing, Permutation, and Bootstrapping

We can slice a series into small segments. With the slices, we can perform different operations to create new series.

- Window Slicing (**WS**): In a classification task, we can take the slices from the original series and assign the same class label to the slice [@Le_Guennec2016-zi]. The slices can also be interpolated to match the length of the original series [@Iwana2020-oc].
- Permutation: We take the slices and permute them to form a new series [@Um2017-oq].
- Moving Block Bootstrapping (**MBB**): First, we remove the trend and seasonability. Then we draw blocks of fixed length from the residual of the series until the desired length of the series is met. Finally, we combine the newly formed residual with trend and seasonality to form a new series [@Bergmeir2016-eh].

#### Warping

Both the time scale and magnitude can be warped. For example,

- Time Warping: We distort time intervals by taking a range of data points and upsample or downsample it [@Wen2020-ez].
- Magnitude Warping: the magnitude of the time series is rescaled.

### Series Mixing

Another class of data augmentation methods is mixing the series. For example, we take two randomly drawn series and average them using DTW Barycenter Averaging (**DBA**) [@Petitjean2011-sj]. (DTW, dynamic time warping, is an algorithm to calculate the distance between sequential datasets by matching the data points on each of the series [@Petitjean2011-sj; @Hewamalage2019-tv].) To augment a dataset, we can choose from a list of strategies [@Bandara2020-yp; @Forestier2017-uk]:

- Average All series using different sets of weights to create new synthetic series.
- Average Selected series based on some strategies. For example, Forestier et al proposed choosing an initial series and combining it with its nearest neighbors [@Forestier2017-uk].
- Average Selected with Distance is Average Selected but neighbors that are far from the initial series are down-weighted [@Forestier2017-uk].


Some other similar methods are

- Equalized Mixture Data Augmentation (**EMDA**) calculates the weighted average of spectrograms of the same class label[@Takahashi2017-yz].
- Stochastic Feature Mapping (**SFM**) is a data augmentation method in audio data[@Cui2014-de].



### Data Generating Process

Time series data can also be augmented using some assumed data generating process (**DGP**). Some methods, such as GRATIS [@Kang2019-cl], utilize simple generic methods such as AR/MAR. Some other methods, such as Gaussian Trees [@Cao2014-mt], utilize more complicated hidden structures using graphs, which can approximate more complicated data generating processes. These methods do not necessarily reflect the actual data generating process but the data is generated using some parsimonious phenomenological models. Some other methods are more tuned toward detailed mechanisms. There are also methods using generative deep neural networks such as [GAN](../self-supervised/adversarial/gan.md).

#### Dynamic Factor Model (**DFM**)

For example, we have a series $X(t)$ which depends on a latent variable $f(t)$[@Stock2016-mh],

$$
X(t) = \mathbf A f(t) + \eta(t),
$$

where $f(t)$ is determined by a differential equation

$$
\frac{f(t)}{dt} = \mathbf B f(t) + \xi(t).
$$

In the above equations, $\eta(t)$ and $\xi(t)$ are the irreducible noise.

The above two equations can be combined into one first-order differential equation.

Once the model is fit, it can be used to generate new data points. However, we will have to understand whether the data is generated in such processes.

## Applying the Synthetic Data to Model Training

Once we prepared the synthetic dataset, there are two strategies to include them in our model training [@Bandara2020-yp].

| Strategy  |  Description |
|---|---|
| Pooled Strategy  | Synthetic data + original data -> model  |
| Transfer Strategy | Synthetic data -> pre-trained model; pre-trained model + original data -> model  |

The pooled strategy takes the synthetic data and original data then feeds them together into the training pipeline. The transfer strategy uses the synthetic data to pre-train the model, then uses transfer learning methods (e.g., freeze weights of some layers) to train the model on the original data.
