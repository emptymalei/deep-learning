---
summary: >-
  A good representation keeps similar instances close together and separates
  different ones. Wang and Isola (2020) call these two properties alignment
  and uniformity and describe both on a hypersphere.
related:
  - self-supervised/contrastive/contrastive-predictive-coding.md
  - self-supervised/contrastive/deep-infomax.md
---

# Alignment and Uniformity


A good representation should be able to

- separate different instances, and
- cluster similar instances.

Wang et al proposed two concepts that matches the above two ideas, alignment and uniformity, on a hypersphere[^Wang2005].

![From Wang et al](assets/alignment-and-uniformity/wang2005-alignment-uniformity.png)
> From Wang et al[^Wang2005].



[^Wang2005]: Wang T, Isola P. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. arXiv [cs.LG]. 2020. Available: http://arxiv.org/abs/2005.10242
