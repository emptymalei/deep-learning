# Python

We assume the readers have a good understanding of the Python programming language, as Python will be the primary programming language for demos and tutorials in this book. For engineering tips, we will cover a few topics here, including

- Environment management;
- Dependency management;
- [`pre-commit`](https://pre-commit.com/).

!!! info "TL;DR"

    1. Use [pyenv](https://github.com/pyenv/pyenv) to manage Python versions;
    2. Use [poetry](https://github.com/python-poetry/poetry) to manage dependencies;
    3. Always set up [`pre-commit``](https://pre-commit.com/) for your git repository.


## Python Environment Management

Environment management is never easy, and the same is true for the Python ecosystem. People have developed a lot of tools to make environment management easier. As you could imagine, this also means we have a zoo of tools to choose from.

There are three things to manage in a Python project:

1. Python version,
2. Dependencies of the project, and
3. An environment where we install our dependencies.

Some tools can manage all three, and some tools focus on one or two of them. We discuss two popular sets of tools: `conda` and `pyenv` + `poetry`.

### `conda`

Many data scientists started with the simple and out-of-the-box choice called [`conda`](https://conda.io). `conda` is an all-in-one toolkit to manage Python versions, environments, and project dependencies.

!!! note "`conda` cheatsheet"

    The most useful commands for conda are the following.

    1. Create an environment: `conda create -n my-env-name python=3.9 pip`, where
         1. `my-env-name` is the name of the environment,
         2. `python=3.9` specifies the version of Python,
         3. `pip` at the end is telling `conda` to install `pip` in this new environment.
    2. Activate an environment: `conda activate my-env-name`
    3. Install new dependency: `conda install pandas`
    4. List all available environments: `conda env list`

    Anaconda provides [a nice cheatsheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf).


### `pyenv` + `poetry`

`conda` is powerful, but it is too powerful for a simple Python project. As of 2024, if you ask around, many Python developers will recommend [`poetry`](https://github.com/python-poetry/poetry).

`poetry` manages dependencies and environments. We just need a tool like [`pyenv`](https://github.com/pyenv/pyenv) to manage Python versions.

!!! note "The `poetry` workflow"

    To work with poetry in an existing project `my_kuhl_project`

    1. `poetry init` to initialize the project and follow the instructions;
    2. `poetry env use 3.10` to specify the Python version. In this example, we use `3.10`;
    3. `poetry add pandas` to add a package called `pandas`.

    Everything we specified will be written into the `pyproject.toml` file.

    `poetry` provides [a nice tutorial](https://python-poetry.org/docs/basic-usage/) on its website.


## Dependency Specifications

We have a few choices to specify the dependencies. The most used method at the moment is `requirements.txt`. However, specifying dependencies in `pyproject.toml` is a much better choice.

Python introduced `pyproject.toml` in [PEP518](https://peps.python.org/pep-0518/) which can be used together with [`poetry`](https://python-poetry.org) to [manage dependencies](https://python-poetry.org/docs/pyproject/#packages).

While tutorials on how to use `poetry` are beyond the scope of this book, we highly recommend using `poetry` in a formal project.

!!! warning "`poetry` is sometimes slow"

    `poetry` can be very slow as it has to load many different versions of the packages to try out in some cases[^slowpoetry][^poery_faq].


!!! note "`conda` with `pip`"

    If one insists on using `conda`, here we provide a few tips for `conda` users.

    `conda` [provides its own requirement specification using `environment.yaml`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). However, many projects still prefer `requirements.txt` even though `conda`'s `environment.yaml` is quite powerful.

    To use `requirements.txt` and `pip`, we always install `pip` when creating a new environment, e.g., `conda create -n my-env-name python=3.9 pip`.

    Once the environment is activated (`conda activate my-env-name`), we can use `pip` to install dependencies, e.g., `pip install -r requirements.txt`.


## Python Styles and `pre-commit`

In a Python project, it is important to have certain conventions or styles. To be consistent, one could follow some style guides for Python. There are official proposals, such as [PEP8](https://peps.python.org/pep-0008/), and "third party" style guides, such as [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) [^pep8][^gpsg].

We also recommend [`pre-commit`](https://pre-commit.com/). `pre-commit` helps us manage git hooks to be executed before each commit. Once installed, every time we run `git commit -m "my commit message here"`, a series of commands will be executed first based on the configurations.

=== ":simple-abstract: Some pre-commit Configs"

    `pre-commit` [officially provides some hooks](https://github.com/pre-commit/pre-commit-hooks) already, e.g., `trailing-whitespace` [^pch].

    We also recommend the following hooks,

    - `black`, which formats the code based on pre-defined styles,
    - `isort`, which orders the Python imports[^isort],
    - `mypy`, which is a linter for Python.

=== ":material-cursor-default-click: An Example Config"

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
            args: ["--profile", "black"]
    ```

## Write docstrings

Writing [docstrings](https://peps.python.org/pep-0257/) for functions and classes can help our future self understand them more easily. There are different styles for docstrings. Two of the popular ones are

- [reStructuredText Docstring Format](https://peps.python.org/pep-0287/), and
- [Google style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).

## Test Saves Time

Adding tests to our code can save us time. We will not list all these benefits of having tests. But tests can help us debug our code and ship results more confidently. For example, suppose we are developing a function and spot a bug. One of the best ways of debugging it is to write a test and put a debugger breakpoint at the suspicious line of the code. With the help of IDEs such as Visual Studio Code, this process can save us a lot of time in debugging.

??? info "Use `pytest`"

    Use [pytest](https://pytest.org). RealPython provides a [good short introduction](https://realpython.com/pytest-python-testing/#what-makes-pytest-so-useful). The Alan Turing Institue provides [some lectures on testing and pytest](https://alan-turing-institute.github.io/rse-course/html/module05_testing_your_code/index.html).


[^isort]: Pre Commit. In: isort [Internet]. [cited 22 Jul 2022]. Available: https://pycqa.github.io/isort/docs/configuration/pre-commit.html
[^pch]: pre-commit-config-pre-commit-hooks.yaml. In: Gist [Internet]. [cited 22 Jul 2022]. Available: https://gist.github.com/lynnkwong/f7591525cfc903ec592943e0f2a61ed9
[^pep8]: Guido van Rossum, Barry Warsaw, Nick Coghlan. PEP 8 – Style Guide for Python Code. In: peps.python.org [Internet]. 5 Jul 2001 [cited 23 Jul 2022]. Available: https://peps.python.org/pep-0008/
[^gpsg]: Google Python Style Guide. In: Google Python Style Guide [Internet]. [cited 22 Jul 2022]. Available: https://google.github.io/styleguide/pyguide.html
[^slowpoetry]: Poetry is extremely slow when resolving the dependencies · Issue #2094 · python-poetry/poetry. In: GitHub [Internet]. [cited 23 Jul 2022]. Available: https://github.com/python-poetry/poetry/issues/2094
[^poery_faq]: FAQ. In: Poetry - Python dependency management and packaging made easy [Internet]. [cited 29 Jan 2024]. Available: https://python-poetry.org/docs/faq/#why-is-the-dependency-resolution-process-slow
