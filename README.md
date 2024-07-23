<p align="center">
  <a href="https://www.tinybig.org">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/tinybig.png" alt="function_data" height="200">
  </a>
</p>

--------------------------------------------------------------------------------

### Introduction

`tinybig` is a Python library developed by the IFM Lab for deep function learning model designing and building.

* RPN Paper: https://arxiv.org/abs/2407.04819
* Official Website: https://www.tinybig.org/
* PyPI: https://pypi.org/project/tinybig/
* IFM Lab: https://www.ifmlab.org/index.html
* Project Description in Chinese: [项目中文介绍](docs/中文简介)

### Citation

If you find `tinybig` and RPN useful in your work, please cite the RPN paper as follows:
```
@article{Zhang2024RPN,
    title={RPN: Reconciled Polynomial Network Towards Unifying PGMs, Kernel SVMs, MLP and KAN},
    author={Jiawei Zhang},
    year={2024},
    eprint={2407.04819},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

### Installation

You can install `tinybig` either via `pip` or directly from the github source code.

#### Install via Pip

```shell
pip install tinybig
```

#### Install from Source

```shell
git clone https://github.com/jwzhanggy/tinyBIG.git
```

After entering the downloaded source code directory, tinybig can be installed with the following command:

```shell
python setup.py install
```

If you don't have `setuptools` installed locally, please consider to first install `setuptools`:
```shell
pip install setuptools 
```

### Install Dependency

Please download the [requirements.txt](https://github.com/jwzhanggy/tinyBIG/blob/main/requirements.txt) file, and install all the dependency packages:
```shell
pip install -r requirements.txt
```

### Verification

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

### Tutorials

|                                      Tutorial ID                                      |           Tutorial Title           |      Last Update      |
|:-------------------------------------------------------------------------------------:|:----------------------------------:|:---------------------:|
|               [Tutorial 0](https://www.tinybig.org/guides/quick_start/)               |        Quickstart Tutorial         |     July 6, 2024      |
| [Tutorial 1](https://www.tinybig.org/tutorials/beginner/module/expansion_function/)  |      Data Expansion Functions      |     July 7, 2024      |
|                                      Tutorial 2                                       | Extended and Nested Data Expansion |          TBD          |

### Examples

|                              Example ID                               |                    Example Title                    | Released Date  |
|:---------------------------------------------------------------------:|:---------------------------------------------------:|:--------------:|
|        [Example 0](https://www.tinybig.org/examples/text/kan/)        |            Failure of KAN on Sparse Data            |  July 9, 2024  |
|  [Example 1](https://www.tinybig.org/examples/function/elementary/)   |          Elementary Function Approximation          |  July 7, 2024  |
|   [Example 2](https://www.tinybig.org/examples/function/composite/)   |          Composite Function Approximation           |  July 8, 2024  |
|    [Example 3](https://www.tinybig.org/examples/function/feynman/)    |           Feynman Function Approximation            |  July 8, 2024  |
|      [Example 4](https://www.tinybig.org/examples/image/mnist/)       |  MNIST Classification with Identity Reconciliation  |  July 8, 2024  |
| [Example 5](https://www.tinybig.org/examples/image/mnist_dual_lphm/)  | MNIST Classification with Dual LPHM Reconciliation  |  July 8, 2024  |
|     [Example 6](https://www.tinybig.org/examples/image/cifar10/)      |         CIFAR10 Object Detection in Images          |  July 8, 2024  |
|       [Example 7](https://www.tinybig.org/examples/text/imdb/)        |             IMDB Review Classification              |  July 9, 2024  |
|      [Example 8](https://www.tinybig.org/examples/text/agnews/)       |             AGNews Topic Classification             |  July 9, 2024  |
|       [Example 9](https://www.tinybig.org/examples/text/sst2/)        |           SST-2 Sentiment Classification            |  July 9, 2024  |
|     [Example 10](https://www.tinybig.org/examples/tabular/iris/)      |    Iris Species Inference (Naive Probabilistic)     |  July 9, 2024  |
|   [Example 11](https://www.tinybig.org/examples/tabular/diabetes/)    |      Diabetes Diagnosis (Comb. Probabilistic)       |  July 9, 2024  |
|   [Example 12](https://www.tinybig.org/examples/tabular/banknote/)    |   Banknote Authentication (Comb. Probabilistic)	    |  July 9, 2024  |

### Library Organizations

| Components                                                                              | Descriptions                                                                                   |
|:----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| [`tinybig`](https://www.tinybig.org/documentations/tinybig/)                            | a deep function learning library like torch.nn, deeply integrated with autograd                |
| [`tinybig.expansion`](https://www.tinybig.org/documentations/expansion/)                | a library providing the "data expansion functions" for multi-modal data effective expansions   |
| [`tinybig.reconciliation`](https://www.tinybig.org/documentations/reconciliation/)      | a library providing the "parameter reconciliation functions" for parameter efficient learning  |
| [`tinybig.remainder`](https://www.tinybig.org/documentations/remainder/)                | a library providing the "remainder functions" for complementary information addition           |
| [`tinybig.module`](https://www.tinybig.org/documentations/module/)                      | a library providing the basic building blocks for RPN model designing and implementation       |
| [`tinybig.model`](https://www.tinybig.org/documentations/model/)                        | a library providing the RPN models for addressing various deep function learning tasks         |
| [`tinybig.config`](https://www.tinybig.org/documentations/config/)                      | a library providing model component instantiation from textual configuration descriptions      |
| [`tinybig.learner`](https://www.tinybig.org/documentations/learner/)                    | a library providing the learners that can be used for RPN model training and testing           |
| [`tinybig.data`](https://www.tinybig.org/documentations/data/)                          | a library providing multi-modal datasets for solving various deep function learning tasks      |
| [`tinybig.output`](https://www.tinybig.org/documentations/output/)                      | a library providing the processing method interfaces for output processing, saving and loading |
| [`tinybig.metric`](https://www.tinybig.org/documentations/metric/)                      | a library providing the  metrics that can be used for RPN model performance evaluation         |
| [`tinybig.util`](https://www.tinybig.org/documentations/util/)                          | a library of utility functions for RPN model design, implementation and learning               | 


### License & Copyright

Copyright © 2024 [IFM Lab](https://www.ifmlab.org/). All rights reserved.

* `tinybig` source code is published under the terms of the MIT License. 
* `tinybig`'s documentation and the RPN papers are licensed under a Creative Commons Attribution-Share Alike 4.0 Unported License ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). 

