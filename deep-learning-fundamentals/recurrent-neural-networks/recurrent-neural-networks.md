# Recurrent Neural Networks

In the section [Neural Networks](neural-net.md), we discussed the feedforward neural network.

??? info "Biological Neural Networks"
    Biological neural networks contain recurrent units. There are theories that employ recurrent networks to explain our memory[@Grossberg:2013].

## Recurrent Neural Network Architecture

A recurrent neural network (RNN) can be achieved by including loops in the network, i.e., the output of a unit is fed back to itself. As an example, we show a single unit in the following figure.

![Basic RNN](../assets/recurrent-neural-networks/rnn-simple.jpg)

On the left, we have the unfolded (unrolled) RNN, while the representation on the right is the compressed form. A simplified mathematical representation of the RNN is as follows:

$$
h(t) = f( W_h h(t-1) + W_x x(t) )
$$

where $h(t)$ represents the state of the unit at time $t$, $x(t)$ is the input at time $t$.

Based on our intuition of differential equations, such a dynamical system usually just blows up or diminishes for a large number of iterations. This is the famous vanishing gradient problem in RNN[@Pascanu2012-qv]. One solution to this is to introduce memory in the iterations, e.g., long short-term memory (LSTM) [@Hochreiter1997-gm].

In the basic example of RNN shown above, the output of the hidden state is fed to itself in the next iteration. In theory, the value to feed back to the unit and how the input and output are calculated can be quite different in different setups [^Amidi&Amidi][^Karpathy2015]. In the section [Forecasting with RNN](../time-series-deep-learning/timeseries.rnn.md), we will show some examples of the different setups.



[^Amidi&Amidi]: Amidi A, Amidi S. CS 230. In: Recurrent Neural Networks Cheatsheet [Internet]. [cited 22 Nov 2023]. Available: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks

[^Karpathy2015]: Karpathy A. The Unreasonable Effectiveness of Recurrent Neural Networks. In: Andrej Karpathy blog [Internet]. 2015 [cited 22 Nov 2023]. Available: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
