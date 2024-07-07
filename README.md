![tinybig.png](docs/assets/img/tinybig.png)

--------------------------------------------------------------------------------

### Introduction

`tinybig` is a Python package developed by the IFM Lab for deep function learning model designing and building.

* Official Website: https://www.tinybig.org/
* PyPI: https://pypi.org/project/tinybig/


### Citation

* The RPN Paper: https://github.com/jwzhanggy/tinyBIG/blob/main/docs/assets/files/rpn_paper.pdf

* The RPN Paper at arXiv: TBD

If you find `tinybig` and RPN useful in your work, please cite the RPN paper as follows:
```
@article{Zhang2024RPN,
  title={RPN: Reconciled Polynomial Network Towards Unifying PGMs, Kernel SVMs, MLP and KAN},
  author={Jiawei Zhang},
  journal={ArXiv},
  year={2024},
  volume={},
}
```

### Installation

#### Pip

```shell
pip install tinybig
```

#### Source

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

#### Dependency

Please download the [requirements.txt](https://github.com/jwzhanggy/tinyBIG/blob/main/requirements.txt) file, and install all the dependency packages:
```shell
pip install -r requirements.txt
```


### tinybig Organizations

| Components                                                                             | Descriptions                                                                                  |
|:---------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| [`tinybig`](https://www.tinybig.org/documentations/tinybig/)                           | a deep function learning library like torch.nn, deeply integrated with autograd               |
| [`tinybig.expansion`](https://www.tinybig.org/documentations/expansion/)               | a library providing the "data expansion functions" for multi-modal data effective expansions  |
| [`tinybig.reconciliation`](https://www.tinybig.org/documentations/reconciliation/)     | a library providing the "parameter reconciliation functions" for parameter efficient learning |
| [`tinybig.remainder`](https://www.tinybig.org/documentations/remainder/)               | a library providing the "remainder functions" for complementary information addition          |
| [`tinybig.module`](https://www.tinybig.org/documentations/module/)                     | a library providing the basic building blocks for RPN model designing and implementation      |
| [`tinybig.model`](https://www.tinybig.org/documentations/model/)                       | a library providing the RPN models for addressing various deep function learning tasks        |
| [`tinybig.config`](https://www.tinybig.org/documentations/config/)                     | a library providing model component instantiation from textual configuration descriptions     |
| [`tinybig.learner`](https://www.tinybig.org/documentations/learner/)                   | a library providing the learners that can be used for RPN model training and testing          |
| [`tinybig.data`](https://www.tinybig.org/documentations/data/)                         | a library providing multi-modal datasets for solving various deep function learning tasks     |
| [`tinybig.metric`](https://www.tinybig.org/documentations/metric/)                     | a library providing the  metrics that can be used for RPN model performance evaluation        |
| [`tinybig.util`](https://www.tinybig.org/documentations/util/)                         | a library of utility functions for RPN model design, implementation and learning              | 

### tinybig Tutorials

|                                      Tutorial ID                                      |           Tutorial Title           |      Last Update       |
|:-------------------------------------------------------------------------------------:|:----------------------------------:|:----------------------:|
|               [Tutorial 0](https://www.tinybig.org/guides/quick_start/)               |        Quickstart Tutorial         |     July 6, 20204      |
| [Tutorial 1](https://www.tinybig.org/tutorials/kickstart/module/expansion_function/)  |      Data Expansion Functions      |      July 7, 2024      |
|                                      Tutorial 2                                       | Extended and Nested Data Expansion |          TBD           |

### License & Copyright

Copyright © 2024 [IFM Lab](https://www.ifmlab.org/). All rights reserved.

* tinybig source code is published under the terms of the MIT License. 
* tinybig's documentation is licensed under a Creative Commons Attribution-Share Alike 3.0 Unported License ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)). 