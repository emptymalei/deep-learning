# Energy-based Models

Energy-based models (EBM) establish relations between different possible values of variables using "energy functions"[@Le_Cun2006-ta].
In an EBM, any input data point can be assigned a probability density[^Lippe].
Similar to statistical physics, we can create such probability densities using a partition function.
As easy as it sounds, such probability densities usually require a scalar function similar to the energy function in statistical physics.
When building the objective functions, we require the configurations that should have the same target label to have low energy, or higher probability density, i.e., to be compatible.


[^Lippe]: Lippe P. Tutorial 9: Deep Autoencoders — UvA DL Notebooks v1.1 documentation. In: UvA Deep Learning Tutorials [Internet]. [cited 20 Sep 2021]. Available: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
