# Learning from Data

> [!NOTE]
>
> **In one minute**
>
> Learning from data means extracting compressed knowledge about the world from data. Predictions can come from induction followed by deduction (first find a model, then apply it) or from transduction (predict directly from the data).

Learning from data is a practice of extracting compressed knowledge about the world from data. There are many frameworks of learning. For example, the induction, deduction, and transduction schema shows different possible paths to produce predictions.

```
graph LR
    P[Prediction]
    D[Data]
    M[Model]
    D --"Induction"--> M
    M --"Deduction"--> P
    D --"Transduction"--> P
```

There are two different approaches to making predictions based on some given data.

1. Perform induction to find a good model from the data, this is called induction. Once we have a model, we can use it to make predictions, this is called deduction.
2. Directly make predictions from the data, this is called transduction.

> [!NOTE]
>
> ***The Nature of Statistical Learning Theory***
>
> Vapnik's seminal book *The Nature of Statistical Learning Theory* is a very good read for the fundamentals of learning theories[^/deep-learning-fundamentals/learning/#bib-Vapnik2010-xf].
>
> Vapnik also discussed some of the key ideas in a book chapter *Estimation of Dependences Based on Empirical Data*[^/deep-learning-fundamentals/learning/#bib-Vapnik2006-pn].

In the context of machine learning, Abu-Mostafa, Magdon-Ismail, and Lin summarized the machine learning problem using the following chart [^/deep-learning-fundamentals/learning/#bib-Abu-Mostafa2012-rk]. Ultimately, we need to find an approximation \\(g\\) of the true map \\(f\\) from features \\(\\mathcal X\\) to targets \\(\\mathcal Y\\) on a specific probability distribution of features \\(P\\). This process is done by using an algorithm to select a hypothesis that works.

```
flowchart LR
    X[Data Samples]
    A[Algorithm]
    H[Hypotheses Set]
    SH[Selected Hypothesis]

    X --> A
    H --> A
    A --> SH
```

Based on this framework, a machine learning process usually consists of three core components[^/deep-learning-fundamentals/learning/#bib-Domingos2012-wn].

1. Representation: Encoded data and the problem representation.
2. Evaluation: An objective function to be evaluated that guides the model.
3. Optimization: An algorithm to optimize the model so it learns what we want it to do.

We will reuse this framework again and again in the following sections of this chapter.

---

**Related:** [Neural Networks](<https://dl.leima.is/deep-learning-fundamentals/neural-net/index.md>)

[^/deep-learning-fundamentals/learning/#bib-Vapnik2010-xf]: Vapnik V.. The nature of statistical learning theory. *Springer*. 2010. [https://www.springer.com/gp/book/9780387987804](<https://www.springer.com/gp/book/9780387987804>)

[^/deep-learning-fundamentals/learning/#bib-Vapnik2006-pn]: Vapnik V.. Estimation of dependences based on empirical data. *Springer*. 2006. [https://sci-hub.se/https://link.springer.com/book/10.1007/0-387-34239-7](<https://sci-hub.se/https://link.springer.com/book/10.1007/0-387-34239-7>)

[^/deep-learning-fundamentals/learning/#bib-Abu-Mostafa2012-rk]: Abu-Mostafa Y.S., Magdon-Ismail M., Lin H.. Learning from data: A short course. *AMLBook*. 2012. [https://www.semanticscholar.org/paper/Learning-From-Data-Abu-Mostafa-Magdon-Ismail/1c0ed9ed3201ef381cc392fc3ca91cae6ecfc698](<https://www.semanticscholar.org/paper/Learning-From-Data-Abu-Mostafa-Magdon-Ismail/1c0ed9ed3201ef381cc392fc3ca91cae6ecfc698>)

[^/deep-learning-fundamentals/learning/#bib-Domingos2012-wn]: Domingos P.. A few useful things to know about machine learning. *Communications of the ACM*. 2012. [https://dl.acm.org/doi/10.1145/2347736.2347755](<https://dl.acm.org/doi/10.1145/2347736.2347755>)
