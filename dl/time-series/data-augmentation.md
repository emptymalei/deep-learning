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


- In Fourier domain, for each the amplitude and phase at a specific frequency, we can perform[@Gao2020-qr]
    - magnitude replacement using a Gaussian distribution, and
    - phase shift by adding a Gaussian noise.

    We perform such perturbations at some chosen frequency.

- Permutation: Slice the series into segments then permute the segments.
- Time Warping: Distort time intervals.

- Equalized Mixture Data Augmentation (EMDA): weighted average of spectrograms of the same class label.[@Takahashi2017-yz]

- Stochastic Feature Mapping (SFM) is a data augmentation method in audio data.[@Cui2014-de]


??? note "DFM: dynamic factor model"

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


### Generative Models


- Gaussian Trees [@Cao2014-mt]

## Applying the Augmented Data to Model Training
