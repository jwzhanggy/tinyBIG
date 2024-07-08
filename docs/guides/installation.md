# Installation of the {{ toolkit }} Library

## Prerequisites

### Python

It is recommended that you use Python 3.10 - 3.12. You can download and install the latest Python 
from the [python official website](https://www.python.org/downloads/).

### Package Manager

To install the `tinybig` binaries, you will need to use pip. 

#### pip

If you installed Python via Homebrew or the Python website, pip (or pip3) was installed with it.

To install pip, you can refer to the [pip official website](https://pip.pypa.io/en/stable/installation/).

To upgrade your pip, you can use the following command:
```shell
python -m pip install --upgrade pip
```

<!--
#### Anaconda

To install Anaconda, you can download graphical installer provided at the 
[Anaconda official website](https://www.anaconda.com/download/success). 
-->

--------------------

## Dependency

The `tinybig` library is developed based on several dependency packages. 
The updated dependency [requirement.txt](https://github.com/jwzhanggy/tinyBIG/blob/main/requirements.txt) of `tinybig`
can be downloaded from the [project github repository](https://github.com/jwzhanggy/tinyBIG).

After downloading the requirement.txt, you can install all these dependencies with the pip command:

=== "install command"
    ```shell
    pip install -r requirements.txt
    ```

=== "requirement.txt"
    ``` yaml linenums="1"
    torch==2.2.2
    numpy==1.26.3
    pyyaml==6.0.1
    scipy==1.13.1
    tqdm==4.66.4
    torchvision==0.17.2
    torchtext==0.17.2
    scikit-learn==1.5.1
    matplotlib==3.9.1
    ```

--------------------

## Installation

The `tinybig` library has been published at both PyPI and the project github repository.

### Install from PyPI

To install `tinybig` from PyPI, use the following command:

```shell
pip install tinybig
```
<!--
### Anaconda

To install PyTorch via Anaconda, use the following conda command:

```shell
caonda install tinybig
```
-->
### Install from Source Code

You can also install `tinybig` from the source code, which has been released at the 
[project github repository](https://github.com/jwzhanggy/tinyBIG). 

You can download the public repository either from the project github webpage or via the following command:
```shell
git clone https://github.com/jwzhanggy/tinyBIG.git
```

After entering the downloaded source code directory, `tinybig` can be installed with the following command:

```shell
python setup.py install
```

If you don't have `setuptools` installed locally, please consider to first install `setuptools`:
```shell
pip install setuptools 
```

--------------------

## Verification

If you have successfully installed both `tinybig` and the dependency packages, now you can use `tinybig` in your projects.

To ensure that `tinybig` was installed correctly, we can verify the installation by running the sample python code as follows:

```python
>>> import torch
>>> import tinybig as tb
>>> expansion_func = tb.expansion.taylor_expansion()
>>> expansion_func(torch.Tensor([[1, 2]]))
```
The output should be something like:
```python
tensor([[1., 2., 1., 2., 2., 4.]])
```