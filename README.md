# Deep Learning

When I switched to data science, I built [my digital garden, datumorphism](https://datumorphism.leima.is/). I deliberately designed this digital garden as my second brain. As a result, most of the articles are fragments of knowledge and require context to understand them.

Making bricks is easy but assembling them into a house is not easy. So I have decided to use this repository to practice my house-building techniques.

I do not have a finished blueprint yet. But I have a framework in my mind: I want to consolidate some of my thoughts and learnings in a good way. However, I do not want to compile a reference book, as [datumorphism](https://datumorphism.leima.is/) already serves this purpose. I should create stories.


## How to Contribute

Create python environment (>=3.7):

```python
conda create -n deep-learning python=3.8 pip
```

Activate environment:

```python
conda activate deep-learning
```

Validate the environment:

```python
which python
```


Install requirements:

```python
pip install -r requirements.txt
```



### Optional Requirements

> The pdf generation is done by the [mkdocs-with-pdf](https://github.com/orzih/mkdocs-with-pdf) plugin.

To generate PDF locally, please install [cairo, Pango and GDK-PixBuf ](https://doc.courtbouillon.org/weasyprint/latest/first_steps.html#macos).


#### Install pango on Mac

When installing pango on Mac using homebrew, the path for `DYLD_LIBRARY_PATH` are not automatically updated. So we need to add the correct path for pango, harfbuzz, and fontconfig. For example,

```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/itsme/homebrew/Cellar/pango/1.48.8/lib:/Users/itsme/homebrew/Cellar/harfbuzz/2.8.2/lib:/Users/itsme/homebrew/Cellar/fontconfig/2.13.1/lib
```
