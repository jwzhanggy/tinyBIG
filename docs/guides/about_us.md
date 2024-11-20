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

* RPN 1 Paper (2024): [https://arxiv.org/abs/2407.04819](https://arxiv.org/abs/2407.04819)
* RPN 2 Paper (2024): [https://arxiv.org/abs/2411.11162](https://arxiv.org/abs/2411.11162)

If you find `tinybig` library and RPN papers useful in your work, please cite the RPN papers as follows:
```
@article{Zhang2024RPN_version1,
    title={RPN: Reconciled Polynomial Network Towards Unifying PGMs, Kernel SVMs, MLP and KAN},
    author={Jiawei Zhang},
    year={2024},
    eprint={2407.04819},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@article{Zhang2024RPN_version2,
    title={RPN 2: On Interdependence Function Learning Towards Unifying and Advancing CNN, RNN, GNN, and Transformer},
    author={Jiawei Zhang},
    year={2024},
    eprint={2411.11162},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Library Organization

| Components                                                                            | Descriptions                                                                                     |
|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| [`tinybig`](https://www.tinybig.org/documentations/tinybig/)                          | a deep function learning library like torch.nn, deeply integrated with autograd                  |
| [`tinybig.model`](https://www.tinybig.org/documentations/model/)                      | a library providing the RPN models for addressing various deep function learning tasks           |
| [`tinybig.module`](https://www.tinybig.org/documentations/module/)                    | a library providing the basic building modules for RPN model designing and implementation        |
| [`tinybig.layer`](https://www.tinybig.org/documentations/layer/)                      | a library providing the pre-defined rpn layers for rpn model designing                           |
| [`tinybig.head`](https://www.tinybig.org/documentations/head/)                        | a library providing the pre-defined rpn head for the rpn layer designing                         |
| [`tinybig.config`](https://www.tinybig.org/documentations/config/)                    | a library providing the configurations for model setups and instantiation                        |
| [`tinybig.expansion`](https://www.tinybig.org/documentations/expansion/)              | a library providing the "data expansion functions" for effective data expansions                 |
| [`tinybig.compression`](https://www.tinybig.org/documentations/compression/)          | a library providing the "data compression functions" for effective data compressions             |
| [`tinybig.transformation`](https://www.tinybig.org/documentations/transformation/)    | a library providing the "data transformation functions" for effective data transformations       |
| [`tinybig.reconciliation`](https://www.tinybig.org/documentations/reconciliation/)    | a library providing the "parameter reconciliation functions" for parameter efficient learning    |
| [`tinybig.remainder`](https://www.tinybig.org/documentations/remainder/)              | a library providing the "remainder functions" for complementary information addition             |
| [`tinybig.interdependence`](https://www.tinybig.org/documentations/interdependence/)  | a library providing the "interdependence functions" for interdependence relationships modeling   |
| [`tinybig.fusion`](https://www.tinybig.org/documentations/fusion/)                    | a library providing the "fusion functions" for multi-source input fusion                         |
| [`tinybig.koala`](https://www.tinybig.org/documentations/koala/)                      | a library providing the interdiciplinary methods and functions about other subjects and areas    |
| [`tinybig.data`](https://www.tinybig.org/documentations/data/)                        | a library providing multi-modal datasets for solving various deep function learning tasks        |
| [`tinybig.output`](https://www.tinybig.org/documentations/output/)                    | a library providing the processing method interfaces for output processing, saving and loading   |
| [`tinybig.loss`](https://www.tinybig.org/documentations/loss/)                        | a library providing the  loss functions that can be used for RPN model learning                  |
| [`tinybig.metric`](https://www.tinybig.org/documentations/metric/)                    | a library providing the  metrics that can be used for RPN model performance evaluation           |
| [`tinybig.optimizer`](https://www.tinybig.org/documentations/optimizer/)              | a library providing the optimizer that can be used for model optimization and learning           |
| [`tinybig.learner`](https://www.tinybig.org/documentations/learner/)                  | a library providing the learners that can be used for RPN model training and testing             |
| [`tinybig.visual`](https://www.tinybig.org/documentations/visual/)                    | a library of visualization functions for RPN model visualization and rendering                   | 
| [`tinybig.util`](https://www.tinybig.org/documentations/util/)                        | a library of utility functions for RPN model design, implementation and learning                 | 
| [`tinybig.zootopia`](https://www.tinybig.org/documentations/zootopia/)                | a library of the RPN model based diverse AI applications                                         |


## License & Copyright

Copyright © 2024 [IFM Lab](https://www.ifmlab.org/). All rights reserved.

* `tinybig` source code is published under the terms of the MIT License. 
* `tinybig` documentation and the RPN papers are licensed under a Creative Commons Attribution-Share Alike 4.0 Unported License ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). 