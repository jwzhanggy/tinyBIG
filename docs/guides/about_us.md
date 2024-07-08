# About us

{{toolkit}} is a website hosting the documentations, tutorials, examples and the latest updates about the `tinybig` library.

## What is `tinybig`?

`tinybig` is a Python library developed by the [IFM Lab](https://www.ifmlab.org/) for deep function learning model building.

* Official Website: [https://www.tinybig.org/](https://www.tinybig.org/)
* Github Repository: [https://github.com/jwzhanggy/tinyBIG](https://github.com/jwzhanggy/tinyBIG)
* PyPI Package: [https://pypi.org/project/tinybig/](https://pypi.org/project/tinybig/)
* IFM Lab [https://www.ifmlab.org/](https://www.ifmlab.org/)

## Citing Us

`tinybig` is developed based on the RPN paper from IFM Lab, which can be downloaded via the following links:

* The RPN Paper: [PDF Link](https://github.com/jwzhanggy/tinyBIG/blob/main/docs/assets/files/rpn_paper.pdf)

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

## Library Organization

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


## License & Copyright

Copyright © 2024 [IFM Lab](https://www.ifmlab.org/). All rights reserved.

* `tinybig` source code is published under the terms of the MIT License. 
* `tinybig`'s documentation and the RPN papers are licensed under a Creative Commons Attribution-Share Alike 4.0 Unported License ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). 