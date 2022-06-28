# Mutual Information

Mutual information is

$$
I(X;Y) = \mathbb E_{p_{XY}} \ln \frac{P_{XY}}{P_X P_Y}.
$$

Mutual information is closed related to [KL divergence](kl-divergence.md),

$$
I(X;Y) = D_{\mathrm{KL}} \left(  P_{XY}(x,y) \parallel  P_X(x) P_{Y}(y) \right).
$$