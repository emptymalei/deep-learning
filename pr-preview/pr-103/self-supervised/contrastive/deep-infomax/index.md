# Deep Infomax

> [!NOTE]
>
> **In one minute**
>
> Deep InfoMax trains an encoder by maximizing an estimate of the mutual information between the input and its encoding. Doing this for the whole input is often not enough, so it is also done between local patches of the feature maps and the encoding, which pushes the encoding to use information that is shared across the input.
>
> **Read first:** [Mutual Information](<https://dl.leima.is/concepts/mutual-information/index.md>)

> [!TIP]
>
> **Max Global Mutual Information**
>
> Why not just use the global mutual information of the input and encoder output as the objective?
>
> > ... maximizing MI between the complete input and the encoder output (i.e., global MI) is often insufficient for learning useful representations.
> >
> > \-- Hjelm et al[^/self-supervised/contrastive/deep-infomax/#Devon2018]

Mutual information maximization is performed on the input of the encoder \\(X\\) and the encoded feature \\(\\hat X=E\_\\theta (X)\\),

\\\[ \\operatorname{arg\~max}\_\\theta I(X;E\_\\theta (X)). \\\]

Being a quantity that is notoriously hard to compute, mutual information \\(I(X;E\_\\theta (X))\\) is usually estimated using its lower bound, which depends on a choice of a functional \\(T\_\\omega\\). Thus the objective will be maximizing a parametrized mutual information estimation,

\\\[ \\operatorname{arg\~max}\_{\\theta, \\omega} \\hat I\_\\omega(X;E\_\\theta (X)) \\\]

> [!NOTE]
>
> **Local or Global**
>
> Two approaches to apply mutual information on encoders:
>
> - Global mutual information of full input and full encoding. This is useful for reconstruction of the input.
> - Local mutual information of local patches of input full encoding. This is useful for classification.

## Local Mutual Information

To compare local features to the encoder output, we need to extract values from inside the encoder, i.e.,

\\\[ E\_{\\theta\_f, \\theta\_C} = f\_{\\theta\_f} \\circ C\_{\\theta\_C}. \\\]

The first step, \\(C\_{\\theta\_C}\\) is to map the input into feature maps, the second step, \\(f\_{\\theta\_f}\\) maps the feature maps into the encoding. The feature map \\(C\_{\\theta\_C}\\) is splitted into patches, \\(C\_{\\theta\_C}=\\left\\{ C\_\\theta^{(i)} \\right\\}\\). The objective is

\\\[ \\operatorname{arg\~max}\_{\\theta\_f, \\theta\_C, \\omega}\\mathbb E\_{i} \\left\[ \\hat I\_\\omega( C\_{\\theta\_C}^{(i)} ;E\_\\theta (X)) \\right\]. \\\]

![Deep InfoMAX Local](<https://dl.leima.is/self-supervised/contrastive/assets/deep-infomax/deep-infomax-local-mi-architecture.jpg>)

> [!TIP]
>
> **Why does local mutual information help**
>
> Hjelm et al explained the idea behind choosing local mutual information[^/self-supervised/contrastive/deep-infomax/#Devon2018].
>
> Global mutual information doesn't specify what is the meaningful information. Some very local noise can also be treated as meaningful information too.
>
> Local mutual information splits the input into patches, and calculate the mutual information between each patch and the encoding. If the model only uses some information from a few local patches, the mutual information objective will be small after averaging all the patches. Thus local mutual information forces the model to use information that is global in the input.

## Code

- [rdevon/DIM](<https://github.com/rdevon/DIM>): by the authors
- [DuaneNielsen/DeepInfomaxPytorch](<https://github.com/DuaneNielsen/DeepInfomaxPytorch>): a clean implementation

---

**Related:** [Contrastive Predictive Coding](<https://dl.leima.is/self-supervised/contrastive/contrastive-predictive-coding/index.md>) · [Alignment and Uniformity](<https://dl.leima.is/concepts/alignment-and-uniformity/index.md>)

[^/self-supervised/contrastive/deep-infomax/#Devon2018]: Devon Hjelm R, Fedorov A, Lavoie-Marchildon S, Grewal K, Bachman P, Trischler A, et al. Learning deep representations by mutual information estimation and maximization. arXiv \[stat.ML\]. 2018. Available: [http://arxiv.org/abs/1808.06670](<http://arxiv.org/abs/1808.06670 >)

[^/self-supervised/contrastive/deep-infomax/#Newell2020]: Newell A, Deng J. How Useful is Self-Supervised Pretraining for Visual Tasks? arXiv \[cs.CV\]. 2020. Available: [http://arxiv.org/abs/2003.14323](<http://arxiv.org/abs/2003.14323 >)
