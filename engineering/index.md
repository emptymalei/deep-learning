# Coding Tips

In this book, we use Python as our programming language. In the main chapters, we will focus on the theories and actual code and skip the basic concepts. To make sure we are on the same page, we shove all the tech stack related topics into this chapter for future reference. It is not necessary to read this chapter before reading the main chapters. However, we recommend the readers go through this chapter at some point to make sure they are not missing some basic engineering concepts.

!!! info

    This chapter is not aiming to be a comprehensive note on these technologies but a few key components that may be missing in many research-oriented tech stacks. We assume the readers have worked with the essential technologies in a Python-based deep learning project.


## Good References for Coding in Research

Some skills only take a while to learn but people benefit from them for their whole life. Managing code falls exactly into this bucket, for programmers.

[The Good Research Code Handbook](https://goodresearch.dev) is a very good and concise guide to building good coding habits. This should be a good first read.

The Alan Turing Institute also has a [Research Software Engineering with Python course](https://alan-turing-institute.github.io/rse-course/html/index.html). This is a comprehensive generic course for boosting Python coding skills in research.


!!! note "A Checklist of Tech Stack"

    We provide a concise list of tools for coding. Most of them are probably already integrated into most people's workflows. Hence we provide no descriptions but only the list itself.

    !!! warning ""
        In the following diagrams, we highlight the recommended tools using orange color. Clicking on them takes us to the corresponding website.

    The first set of checklists is to help us set up a good coding environment.

    ```mermaid
    flowchart TD
    classDef highlight fill:#f96;

    env["Setting up Coding Environment"]
    git["fa:fa-star Git"]:::highlight
    precommit["pre-commit"]:::highlight
    ide["Integrated Development Environment (IDE)"]
    vscode["Visual Studio Code"]:::highlight
    pycharm["PyCharm"]
    jupyter["Jupyter Notebooks"]
    python["Python Environment"]
    py_env["Python Environment Management"]
    conda["Anaconda"]
    pyenv_venv["Pyenv + venv + pip"]
    pyenv_poetry["Pyenv + poetry"]
    poetry["Poetry"]:::highlight
    pyenv["pyenv"]:::highlight
    venv["venv"]

    click git "https://git-scm.com/" "Git"
    click precommit "https://pre-commit.com/" "pre-commit"
    click vscode "https://code.visualstudio.com/" "Visual Studio Code"
    click jupyter "https://jupyter.org/" "Jupyter Lab"
    click pycharm "https://www.jetbrains.com/pycharm/" "PyCharm"
    click conda "https://www.anaconda.com/" "Anaconda"
    click pyenv "https://github.com/pyenv/pyenv" "pyenv"
    click venv "https://docs.python.org/3/library/venv.html" "venv"
    click poetry "https://python-poetry.org/" "poetry"

    env --- git
    git --- precommit

    env --- ide
    ide --- vscode
    ide --- jupyter
    ide --- pycharm

    env --- python
    python --- py_env
    py_env --- conda
    py_env --- pyenv_venv
    py_env --- pyenv_poetry

    pyenv_venv --- pyenv
    pyenv_venv --- venv

    pyenv_poetry --- pyenv
    pyenv_poetry --- poetry
    ```


    The second set of checklists is to boost our code quality.

    ```mermaid
    flowchart TD
    classDef highlight fill:#f96;

    python["Python Code Quality"]
    test["Test Your Code"]
    formatter["Formatter"]
    linter["Linter"]
    pytest["pytest"]:::highlight
    black["black"]:::highlight
    isort["isort"]:::highlight
    pylint["pylint"]
    flake8["flake8"]
    pylama["pylama"]
    mypy["mypy"]:::highlight

    click pytest "https://pytest.org/" "pytest"
    click black "https://github.com/psf/black" "black"
    click isort "https://github.com/pycqa/isort"
    click mypy "http://mypy-lang.org/"
    click pylint "https://pylint.pycqa.org/"
    click flake8 "https://flake8.pycqa.org/en/latest/"
    click pylama "https://github.com/klen/pylama"

    python --- test
    test --- pytest

    python --- formatter
    formatter --- black
    formatter --- isort

    python --- linter
    linter --- mypy
    linter --- pylint
    linter --- flake8
    linter ---pylama
    ```

    Finally, we also mention the primary python packages used here.

    ```mermaid
    flowchart TD
    classDef highlight fill:#f96;

    dataml["Data and Machine Learning"]
    pandas["Pandas"]:::highlight
    pytorch["PyTorch"]:::highlight
    lightning["PyTorch Lightning"]:::highlight
    much_more["and more ..."]

    click pandas "https://pandas.pydata.org/"
    click pytorch "https://pytorch.org/"
    click lightning "https://www.pytorchlightning.ai/"

    dataml --- pandas
    dataml --- pytorch
    dataml --- lightning
    dataml --- much_more
    ```
