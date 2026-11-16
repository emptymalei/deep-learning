# Autoregressive Model

> [!NOTE]
>
> **In one minute**
>
> An autoregressive model describes each variable conditioned on the ones before it. The page also sorts out the different notations the literature uses for those preceding variables.

![Generative AR basics](<https://dl.leima.is/self-supervised/generative/assets/autoregressive/generative-ar-basics-1.jpg>)

An autoregressive (AR) model is autoregressive,

\\\[ \\begin{equation} \\log p\_\\theta (x) = \\sum\_{t=1}^T \\log p\_\\theta ( x\_{t} \\mid \\{x\_{\<t}\\} ). \\end{equation} \\\]

> [!NOTE]
>
> **Notations and Conventions**
>
> In AR models, we have to mention the preceding nodes (\\(\\{x\_{\<t}\\}\\)) of a specific node (\\(x\_{t}\\)). For \\(t=5\\), the relations between \\(\\{x\_{\<5}\\}\\) and \\(x\_5\\) are shown in the following illustration.
>
> ![](<https://dl.leima.is/self-supervised/generative/assets/autoregressive/ar-notations.jpg>)
>
> There are different notations for such relations.
>
> - In Uria et al., the authors use \\(p(x\_{o\_d}\\mid \\mathbf x\_{o\_{\<d}})\\) [^/self-supervised/generative/autoregressive/#Uria2016].
> - In Liu et al. and Papamakarios et al., the authors use \\(p(x\_{t}\\mid \\mathbf x\_{1:t-1})\\) [^/self-supervised/generative/autoregressive/#Liu2020][^/self-supervised/generative/autoregressive/#Papamakarios2017].
> - In Germain et al., the authors use \\(p(x\_t\\mid \\mathbf x\_{\<t})\\) [^/self-supervised/generative/autoregressive/#Germain2015].
>
> In the current review, we expanded the vector notation \\(\\mathbf x\_{\<t}\\) into a set notation as it is not necessarily a vector.

[^/self-supervised/generative/autoregressive/#Uria2016]: Uria B, Côté M-A, Gregor K, Murray I, Larochelle H. Neural Autoregressive Distribution Estimation. arXiv \[cs.LG\]. 2016. Available: [http://arxiv.org/abs/1605.02226](<http://arxiv.org/abs/1605.02226 >)

[^/self-supervised/generative/autoregressive/#Triebe2019]: Triebe O, Laptev N, Rajagopal R. AR-Net: A simple Auto-Regressive Neural Network for time-series. arXiv \[cs.LG\]. 2019. Available: [http://arxiv.org/abs/1911.12436](<http://arxiv.org/abs/1911.12436 >)

[^/self-supervised/generative/autoregressive/#Ho2019]: Ho G. George Ho. In: Eigenfoo \[Internet\]. 9 Mar 2019 \[cited 19 Sep 2021\]. Available: [https://www.eigenfoo.xyz/deep-autoregressive-models/](<https://www.eigenfoo.xyz/deep-autoregressive-models/ >)

[^/self-supervised/generative/autoregressive/#Papamakarios2017]: Papamakarios G, Pavlakou T, Murray I. Masked Autoregressive Flow for Density Estimation. arXiv \[stat.ML\]. 2017. Available: [http://arxiv.org/abs/1705.07057](<http://arxiv.org/abs/1705.07057 >)

[^/self-supervised/generative/autoregressive/#Germain2015]: Germain M, Gregor K, Murray I, Larochelle H. MADE: Masked autoencoder for distribution estimation. 32nd International Conference on Machine Learning, ICML 2015. 2015;2: 881–889. Available: [http://arxiv.org/abs/1502.03509](<http://arxiv.org/abs/1502.03509 >)

[^/self-supervised/generative/autoregressive/#Liu2020]: Liu X, Zhang F, Hou Z, Wang Z, Mian L, Zhang J, et al. Self-supervised Learning: Generative or Contrastive. arXiv \[cs.LG\]. 2020. Available: [http://arxiv.org/abs/2006.08218](<http://arxiv.org/abs/2006.08218 >)

[^/self-supervised/generative/autoregressive/#Lippe]: Lippe P. Tutorial 12: Autoregressive Image Modeling — UvA DL Notebooks v1.1 documentation. In: UvA Deep Learning Tutorials \[Internet\]. \[cited 20 Sep 2021\]. Available: [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial\_notebooks/tutorial12/Autoregressive\_Image\_Modeling.html](<https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html >)

[^/self-supervised/generative/autoregressive/#rogen]: rogen-george. rogen-george/Deep-Autoregressive-Model. In: GitHub \[Internet\]. \[cited 20 Sep 2021\]. Available: [https://github.com/rogen-george/Deep-Autoregressive-Model](<https://github.com/rogen-george/Deep-Autoregressive-Model >)
