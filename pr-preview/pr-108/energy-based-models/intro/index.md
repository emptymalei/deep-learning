# Energy-based Models

> [!NOTE]
>
> **In one minute**
>
> Energy-based models give every configuration of the variables an energy, a scalar from which a probability density follows through a partition function, as in statistical physics. Compatible configurations get low energy, that is, high probability density.

Energy-based models (EBM) establish relations between different possible values of variables using "energy functions"[^/energy-based-models/intro/#bib-Le_Cun2006-ta]. In an EBM, any input data point can be assigned a probability density[^/energy-based-models/intro/#Lippe]. Similar to statistical physics, we can create such probability densities using a partition function. As easy as it sounds, such probability densities usually require a scalar function similar to the energy function in statistical physics. When building the objective functions, we require the configurations that should have the same target label to have low energy, or higher probability density, i.e., to be compatible.

---

**Related:** [Denoising Diffusion Probabilistic Models](<https://dl.leima.is/energy-based-models/ebm.diffusion/index.md>)

[^/energy-based-models/intro/#Lippe]: Lippe P. Tutorial 9: Deep Autoencoders — UvA DL Notebooks v1.1 documentation. In: UvA Deep Learning Tutorials \[Internet\]. \[cited 20 Sep 2021\]. Available: [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial\_notebooks/tutorial9/AE\_CIFAR10.html](<https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html >)

[^/energy-based-models/intro/#bib-Le_Cun2006-ta]: Le Cun Y., Chopra S., Hadsell R. et al.. A Tutorial on Energy-Based Learning. 2006.
