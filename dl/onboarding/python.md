# Python

Python will be our primary programming language. Thus we assume the readers have a good understanding of the Python language in this section. We will cover the following topics.

- Environment management;
- Dependency management;
- `pre-commit`.


## Environment Management

Python is notorious in environment management. We recommend using `conda` to manage our environments.

!!! note "`conda` cheatsheet"

    The most useful commands for conda are the following.

    1. Create an environment: `conda create -n my-env-name python=3.9 pip`, where
         1. `my-env-name` is the name of the environment,
         2. `python=3.9` specifies the version of Python,
         3. `pip` is telling `conda` to install `pip` in this new environment.
    2. Activate an environment: `conda activate my-env-name`
    3. List all available environments: `conda env list`


    Anaconda provides [a nice cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf).



??? note "Alternative to `conda`"

    [`pyenv`](https://github.com/pyenv/pyenv) is also a good tool for managing different versions and environments of python.


## Dependency Management

We have a few choices to specify the dependencies. The most used method at the moment is `requirements.txt`.


??? info "`conda`'s `environment.yml`"

    Alternatively, `conda` [provides its own requirement specification using `environment.yaml`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). However, this method doesn't make things better. If we ever need to make it more complicated than `requirements.txt`, `pyproject.toml` is the way to go.


??? info "Modern Python with `pyproject.toml` and `poetry`"

    Python introduced `pyproject.toml` in [PEP518](https://peps.python.org/pep-0518/). `pyproject.toml` can be used together with [`poetry`](https://python-poetry.org) to [sepcify dependencies](https://python-poetry.org/docs/pyproject/#packages).

    Both `conda` and `pyenv` have trouble solving the actual full dependency graphs of all the packages used. This problem is solved easily by `poetry`. However, this also means `poetry` can be very slow as it has to load many different versions of the packages.

    We highly recommend using `poetry` in a formal project. However, tutorials on how to use `poetry` is not within the scope of this book.


## Python Styles and `pre-commit`

In a Python project, it is important to have some certain conventions or styles. To be self-consistent, one could follow some style guides for python. There are official proposals, such as [PEP8](https://peps.python.org/pep-0008/), and "third party" style guides, such as [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).[^pep8][^gpsg] And use [pre-commit](https://pre-commit.com/).


=== "Some pre-commit Configs"

    `pre-commit` [officially provides some hooks](https://github.com/pre-commit/pre-commit-hooks) already, e.g., `trailing-whitespace`.[^pch]

    We also recommend using some other hooks, namely,

    - `black`, which formats the code based on pre-defined styles,
    - `isort`, which orders the Python imports[^isort],
    - `mypy`, which is a linter for Python.

=== "An Example Config"

    The following is an example `.pre-commit-config.yaml` file for a Python project.

    ```yaml
    repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
        rev: v4.2.0
        hooks:
        - id: check-added-large-files
        - id: debug-statements
        - id: detect-private-key
        - id: end-of-file-fixer
        - id: requirements-txt-fixer
        - id: trailing-whitespace
    - repo: https://github.com/pre-commit/mirrors-mypy
        rev: v0.960
        hooks:
        - id: mypy
            args:
            - "--no-strict-optional"
            - "--ignore-missing-imports"
    - repo: https://github.com/ambv/black
        rev: 22.6.0
        hooks:
        - id: black
        language: python
        args:
            - "--line-length=120"
    - repo: https://github.com/pycqa/isort
        rev: 5.10.1
        hooks:
        - id: isort
            name: isort (python)
    ```

## Write docstrings

Writing [docstrings](https://peps.python.org/pep-0257/) for functions and classes can help our future self understand them more easily. There are different styles for docstrings. Two of the popular ones are

- [reStructuredText Docstring Format](https://peps.python.org/pep-0287/), and
- [Google style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).

## Test Saves Time

Adding tests to your code can save you time. We will not list all these benefits of having tests. But tests can help us debug our code and ship results more confidently. For example, suppose we are developing a function and spot a bug. One of the best ways of debugging it is to write a test and put a debugger at the suspicious bug in the code. With the help of IDEs such as Visual Studio Code, this process can save us a lot of time in debugging.

??? info "Use `pytest`"

    Use [pytest](https://pytest.org). RealPython provides a [good short introduction](https://realpython.com/pytest-python-testing/#what-makes-pytest-so-useful).


[^isort]: Pre Commit. In: isort [Internet]. [cited 22 Jul 2022]. Available: https://pycqa.github.io/isort/docs/configuration/pre-commit.html
[^pch]: pre-commit-config-pre-commit-hooks.yaml. In: Gist [Internet]. [cited 22 Jul 2022]. Available: https://gist.github.com/lynnkwong/f7591525cfc903ec592943e0f2a61ed9
[^pep8]: Guido van Rossum, Barry Warsaw, Nick Coghlan. PEP 8 – Style Guide for Python Code. In: peps.python.org [Internet]. 5 Jul 2001 [cited 23 Jul 2022]. Available: https://peps.python.org/pep-0008/
[^gpsg]: Google Python Style Guide. In: Google Python Style Guide [Internet]. [cited 22 Jul 2022]. Available: https://google.github.io/styleguide/pyguide.html
