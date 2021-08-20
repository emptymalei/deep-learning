# probability-estimation


## Development

Create python environment (>=3.7):

```python
conda create -n cpe python=3.7 pip
```

Activate environment:

```python
conda activate cpe
```

Validate the environment:

```python
which python
```


Install requirements:

```python
pip install -r requirements.docs.txt
```



### Optional Requirements

> The pdf generation is done by the [mkdocs-with-pdf](https://github.com/orzih/mkdocs-with-pdf) plugin.

To generate PDF locally, please install [cairo, Pango and GDK-PixBuf ](https://doc.courtbouillon.org/weasyprint/latest/first_steps.html#macos).


#### Install pango on Mac

When installing pango on Mac using homebrew, the path for `DYLD_LIBRARY_PATH` are not automatically updated. So we need to add the correct path for pango, harfbuzz, and fontconfig. For example,

```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/Users/itsme/homebrew/Cellar/pango/1.48.8/lib:/Users/itsme/homebrew/Cellar/harfbuzz/2.8.2/lib:/Users/itsme/homebrew/Cellar/fontconfig/2.13.1/lib
```



