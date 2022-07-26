# Data Augmentation for Time Series


In deep learning, our dataset should help the optimization mechanism locate a good spot in the parameter space. However, real-world data is not necessarily diverse enough that covers the required situations with enough records. For example, some datasets maybe extremely imbalanced class labels which leads to poor performance in classification tasks.[@Hasibi2019-in] Another problem with a limited dataset is that the trained model may not generalize well.[@Iwana2020-oc; @Shorten2019-ty]

We will cover two topics in this section: Augmenting the dataset and application of the augmented data to model training.

## Augmenting the Dataset

There are many different ways of augmenting time series data. We list some of the methods based on the reviews by Iwana2020 and Wen2020.[@Iwana2020-oc; @Wen2020-ez]

### Basic Methods

|   |  Projected Domain  | Time Domain | Magnitude |
|---|---|---|---|
| Random Transformation | Frequency Masking, Frequency Warping, Fourier Transform, STFT  | Permutation, Slicing, Time Warping, Time Masking, Cropping   | Jittering, Flipping, Scaling, Magnitude Warping  |
| Pattern Mixing  | EMDA, SFM  | Guided Warping  | DFM, Interpolation, DTW  |


#### Perturbation in Fourier Domain

In Fourier domain, for each the amplitude $A_f$ and phase $\phi_f$ at a specific frequency, we can perform[@Gao2020-qr]
    - magnitude replacement using a Gaussian distribution, and
    - phase shift by adding a Gaussian noise.

We perform such perturbations at some chosen frequency.

#### Slicing, Permutation, and Bootstrapping

We can slice a series into small segments. With the slices, we can perform different operations to create new series.

- Window Slicing (**WS**): In a classification tasks, we can take the slices from the original series and assign the same class label to the slice.[@Le_Guennec2016-zi] The slices can also be interpolated to match the length of the original series.[@Iwana2020-oc]
- Permutation: We take the slices and permute them to form a new series.[@Um2017-oq]
- Moving Block Bootstrapping (**MBB**): First, we remove the trend and seasonability. Then we draw blocks of fixed length from the residual of the series, until the desired length of series is met. Finally, we combine the newly formed residual with trend and seasonality to form a new series.[@Bergmeir2016-eh]

#### Warping

Both the time scale and magnitude can be warped. For example,

- Time Warping: We distort time intervals by taking a range of data points and up sample or down sample it.[@Wen2020-ez]
- Magnitude Warping: the magnitude of the time series is rescaled.

### Series Mixing

Another class of data augmentation methods is mixing the series. For example, we take two random drawn series and average them using DTW Barycenter Averaging (**DBA**).[@Petitjean2011-sj] (DTW, dynamic time warping, is an algorithm to calculate the distance between sequential datasets by matching the data points on each of the series.[@Petitjean2011-sj; @Hewamalage2019-tv]) To augment a dataset, we can choose from a list of strategies:[@Bandara2020-yp; @Forestier2017-uk]

- Average All series using different sets of weights to create new synthetic series.
- Average Selected series based on some strategies. For example, Forestier et al proposed to choose an initial series and combine it with its nearest neighbors.[@Forestier2017-uk]
- Average Selected with Distance is Average Selected but neighbors that are far from the initial series is down weighted.[@Forestier2017-uk]


Some other similar methods are

- Equalized Mixture Data Augmentation (**EMDA**) calculates the weighted average of spectrograms of the same class label.[@Takahashi2017-yz]
- Stochastic Feature Mapping (**SFM**) is a data augmentation method in audio data.[@Cui2014-de]



### Data Generating Process

Time series data can also be augmented using some assumed data generating process (**DGP**). Some methods, such as GRATIS[@Kang2019-cl], utilizes simple generic methods such as AR/MAR. Some other methods, such as Gaussian Trees[@Cao2014-mt], utilizes more complicated hidden structure using graphs, which can approximate more complicated data generating process. These methods do not necessarily reflect the actual data generating process but the data is generated using some parsimonious phenomenological models. Some other methods are more tuned towards the detailed mechanisms. There are also methods using generative deep neural networks such as [GAN](../self-supervised/adversarial/gan.md).

#### Dynamic Factor Model (**DFM**)

For example, we have a series $X(t)$ which depends on a latent variable $f(t)$,[@Stock2016-mh]

$$
X(t) = \mathbf A f(t) + \eta(t),
$$

where $f(t)$ is determined by a differential equation

$$
\frac{f(t)}{dt} = \mathbf B f(t) + \xi(t).
$$

In the above equations, $\eta(t)$ and $\xi(t)$ are the irreducible noise.

The above two equatioins can be combined into one first-order differential equation.

Once the model is fit, it can be used to generate new data points. However, we will have to understand whether the data is generated in such processes.

## Applying the Augmented Data to Model Training
