# f-GAN

The essence of [GAN](gan.md#divergence) is comparing the generated distribution $p_G$ and the data distribution $p_\text{data}$. The vanilla GAN considers the Jensen-Shannon divergence $\operatorname{D}_\text{JS}(p_\text{data}\Vert p_{G})$.

Meanwhile, there exist more generic forms of such divergences, which are called [f-divergence](../appendices/f-divergence.md)[^f-divergence_wiki]. f-GAN [^Nowozin2016].


## Variational Divergence Minimization


The Variational Divergence Minimization (VDM) extends the variational estimation of f-divergence[^Nowozin2016]. VDM searches for the saddle point of an objective $F(\theta, \omega)$, i.e., min w.r.t. $\theta$ and max w.r.t $\omega$, where $\theta$ is the parameter set of the generator, and $\omega$ is the parameter set of the variational approximation to estimate f-divergence.




!!! note "$T$"
    The function $T$ is used to estimate the lower bound of f-divergence[^Nowozin2016].



[^f-divergence_wiki]: Contributors to Wikimedia projects. F-divergence. In: Wikipedia [Internet]. 17 Jul 2021 [cited 6 Sep 2021]. Available: https://en.wikipedia.org/wiki/F-divergence#Instances_of_f-divergences


[^Nowozin2016]: Nowozin S, Cseke B, Tomioka R. f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization. arXiv [stat.ML]. 2016. Available: http://arxiv.org/abs/1606.00709
