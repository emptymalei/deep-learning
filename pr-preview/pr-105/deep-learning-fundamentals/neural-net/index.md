# Neural Networks

> [!NOTE]
>
> **In one minute**
>
> An artificial neuron responds to its inputs more or less strongly, and a network composes many neurons into complicated functions. The page builds this intuition first, then explains why neural networks are universal approximators.
>
> **Read first:** [Learning from Data](<https://dl.leima.is/deep-learning-fundamentals/learning/index.md>)

Neural networks have been a buzzword for machine learning in recent years. As indicated in the name, artificial neural networks are artificial neurons connected in a network. In this section, we discuss some intuitions and theories of neural networks.

> [!NOTE]
>
> **Artificial vs Biological**
>
> Neuroscientists also discuss neural networks, or neuronal networks in their research. Those are different concepts from the artificial neural networks we are going to discuss here. In this book, we use the term neural networks to refer to artificial neural networks, unless otherwise specified.

## Intuitions

We start with some intuitions of neural networks before discussing the theoretical implications.

### What is an Artificial Neuron

What an artificial neuron does is respond to stimulations. This response could be strong or weak depending on the strength of the stimulations. Here is an example.

![Neuron response](<https://dl.leima.is/deep-learning-fundamentals/assets/neural-net/neuron-response.png>)

Using one simple single neuron, we do not have much to build. It is just the function we observe above. However, by connecting multiple neurons in a network, we could compose complicated functions and generalize the scaler function to multi-dimensional functions.

Before we connect this neuron to a network, we study a few transformations first. The response function can be shifted, scaled, or inverted. The following figure shows the effect of these transformations.

![transformations](<https://dl.leima.is/deep-learning-fundamentals/assets/neural-net/transformation-activation.jpg>)

### Artificial Neural Network

A simple network is a collection of neurons that respond to stimulations, which could come from the responses of other neurons.

![simple neural network](<https://dl.leima.is/deep-learning-fundamentals/assets/neural-net/neural-network-simple.png>)

A given input signal is spread onto three different neurons. The neurons respond to this signal separately before being combined with different weights. In the language of math, given input \\(x\\), output \\(y(x)\\) is

\\\[y(x) = \\sum\_{k=1}^{3} x v\_k \* \\mathrm{activation}( w\_k \* x + u\_k )\\\]

where \\(\\mathrm{activation}\\) is the activation function, i.e., the response behavior of the neuron. This is a single-layer structure.

> [!NOTE]
>
> **\\(\\mathrm{activation} \\to \\sigma\\)**
>
> In the following discussions, we will use \\(\\sigma\\) as a drop in replacement for \\(\\mathrm{activation}\\).

To extend this naive and shallow network, we could

- increase the number of neurons on one layer, i.e., go wider, or
- extend the number of layers, i.e., go deeper, or
- add interactions between neurons, or
- include recurrent connections in the network.

![multilayer](<https://dl.leima.is/deep-learning-fundamentals/assets/neural-net/multilayer.jpg>)

### Composition Effects

To build up intuitions of how multiple neurons work together, we take an example of a network with two neurons. We will solve two problems:

1. Find out if a hotel room is hot or cold.
2. Find out if the hotel room is comfortable to stay.

The first task can be solved using o single neuron. Suppose our input to the neuron is the temperature of the room. The output of the neuron is a binary value, 1 for hot and 0 for cold. The following figure shows the response of the neuron.

![single neuron classification](<https://dl.leima.is/deep-learning-fundamentals/assets/neural-net/one-neuron-classification.png>)

In the figure above, we use red for "hot" and blue for "cold". In this example, the temperature being \\(T\_1\\) means the room is cold, while that being \\(T\_2\\) and \\(T\_3\\) indicate hot rooms.

However, moving on the the second problem, such monotonic functions won't work. It is only comfortable to stay in the hotel room if the temperature is neither too high nor too low. Now consider two neurons in a network. One neuron has a monotonic increasing response to the temperature, while the other has a monotonic decreasing response. The following figure shows the combined response of the two neurons. We observe that the combined response is high only when the temperature is in a certain range.

![two neuron classification](<https://dl.leima.is/deep-learning-fundamentals/assets/neural-net/two-neuron-network.png>)

Suppose we have three rooms with temperatures \\(T\_1\\), \\(T\_2\\), \\(T\_3\\) respectively. Only \\(T\_2\\) falls into the region of large output value which corresponds to the habitable temperature.

![two neuron classification](<https://dl.leima.is/deep-learning-fundamentals/assets/neural-net/two-neuron-classification-result.png>)

> [!NOTE]
>
> **Mathematical Formulation**
>
> The above example can be formulated as $$ f(x) = \\sum\_k v\_k \\sigma(w\_k x + u\_k) $$ where \\(\\sigma\\) is some sort of monotonic activation function.
>
> It is a form of single hidden layer feedforward network. We will discuss this in more detail in the following sections.

These two examples hint that neural networks are good at classification tasks. Neural networks excel at a variety of tasks. Since this book is about time series, we will demonstrate the power of neural networks in time series analysis.

## Universal Approximators

Even a single hidden layer feedforward network can approximate any measurable function, regardless of the activation function[^/deep-learning-fundamentals/neural-net/#bib-Hornik1989-nc]. In the case of the commonly discussed sigmoid function as an activation function, a neural network for real numbers becomes

\\\[ \\sum\_k v\_k \\sigma(w\_k x + u\_k) \\\]

It is a good approximation of continuous functions[^/deep-learning-fundamentals/neural-net/#bib-Cybenko1989-ro].

> [!NOTE]
>
> **Kolmogorov's Theorem**
>
> Kolmogorov's theorem shows that one can use a finite number of carefully chosen continuous functions to mix up by sums and multiplication with weights to a continuous multivariable function on a compact set[^/deep-learning-fundamentals/neural-net/#bib-Hassoun2021-sh].

## Neural Networks Can be Complicated

In practice, we observe a lot of problems when the number of neurons grows, e.g., the convergence during the training slows down if we have too many layers in the network (the vanishing gradient problem) [^/deep-learning-fundamentals/neural-net/#bib-Hochreiter2008].

> [!NOTE]
>
> **Training**
>
> We have not yet discussed how to adjust the parameters in a neural network. The process is called training. The most popular method is backpropagation[^/deep-learning-fundamentals/neural-net/#Nielsen].

The reader should understand that a good neural network model is not only about these naive examples but is about many different topics. For example, to solve the vanishing gradient problem, new architectures are proposed, e.g., residual blocks[^/deep-learning-fundamentals/neural-net/#bib-He2015-ie], new optimization techniques were proposed[^/deep-learning-fundamentals/neural-net/#bib-Huang2016-uz], and theories such as information highway also became the key to the success of deep neural networks[^/deep-learning-fundamentals/neural-net/#bib-Srivastava2015-te].

---

**Related:** [Recurrent Neural Networks](<https://dl.leima.is/deep-learning-fundamentals/recurrent-neural-networks/index.md>)

[^/deep-learning-fundamentals/neural-net/#Nielsen]: Nielsen MA. How the backpropagation algorithm works. In: Neural networks and deep learning \[Internet\]. \[cited 22 Nov 2023\]. Available: [http://neuralnetworksanddeeplearning.com/chap2.html](<http://neuralnetworksanddeeplearning.com/chap2.html >)

[^/deep-learning-fundamentals/neural-net/#bib-Hornik1989-nc]: Hornik K., Stinchcombe M., White H.. Multilayer feedforward networks are universal approximators. *Neural networks: the official journal of the International Neural Network Society*. 1989. [https://linkinghub.elsevier.com/retrieve/pii/0893608089900208](<https://linkinghub.elsevier.com/retrieve/pii/0893608089900208>)

[^/deep-learning-fundamentals/neural-net/#bib-Cybenko1989-ro]: Cybenko G.. Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals, and Systems*. 1989. [http://link.springer.com/10.1007/BF02551274](<http://link.springer.com/10.1007/BF02551274>)

[^/deep-learning-fundamentals/neural-net/#bib-Hassoun2021-sh]: Hassoun M.. Fundamentals of Artificial Neural Networks. *The MIT Press, Massachusetts Institute of Technology*. 2021. [https://mitpress.mit.edu/9780262514675/fundamentals-of-artificial-neural-networks/](<https://mitpress.mit.edu/9780262514675/fundamentals-of-artificial-neural-networks/>)

[^/deep-learning-fundamentals/neural-net/#bib-Hochreiter2008]: Hochreiter S., Bengio Y., Frasconi P. et al.. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. *A Field Guide to Dynamical Recurrent Neural Networks*. 2001.

[^/deep-learning-fundamentals/neural-net/#bib-He2015-ie]: He K., Zhang X., Ren S. et al.. Deep Residual Learning for Image Recognition. 2015. [http://arxiv.org/abs/1512.03385](<http://arxiv.org/abs/1512.03385>)

[^/deep-learning-fundamentals/neural-net/#bib-Huang2016-uz]: Huang G., Sun Y., Liu Z. et al.. Deep networks with stochastic depth. 2016. [http://arxiv.org/abs/1603.09382](<http://arxiv.org/abs/1603.09382>)

[^/deep-learning-fundamentals/neural-net/#bib-Srivastava2015-te]: Srivastava R.K., Greff K., Schmidhuber J.. Highway Networks. 2015. [http://arxiv.org/abs/1505.00387](<http://arxiv.org/abs/1505.00387>)
