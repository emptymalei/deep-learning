# Data Augmentation for Time Series


In deep learning, our dataset should help the optimization mechanism locate a good spot in the parameter space. However, real-world data is not necessarily diverse enough that covers the required situations with enough records. For example, some datasets maybe extremely imbalanced class labels which leads to poor performance in classification tasks.[@Hasibi2019-in] Another problem with a limited dataset is that the trained model may not generalize well.[@Iwana2020-oc; @Shorten2019-ty]

We will cover two topics in this section: Augmenting the dataset and application of the augmented data to model training.

## Augmenting the Dataset

There are many different ways of augmenting time series data. We list some of the methods based on the reviews by Iwana2020 and Wen2020.[@Iwana2020-oc; @Wen2020-ez]


|   |  Frequency Domain  | Time Domain | Magnitude | Time-Frequency Domain |
|---|---|---|---|---|
| Random Transformation | Frequency Masking, Frequency Warping, Fourier Transform  | Permutation, Slicing, Time Warping, Time Masking   | Jittering, Rotation, Scaling, Magnitude Warping  |  |
| Pattern Mixing  | EMDA, SFM  | Guided Warping  | DFM, Interpolation, Time Aligned Averaging  |  |


## Applying the Augmented Data to Model Training
