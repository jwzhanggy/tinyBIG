# Tutorial on Data Interdependence Functions 

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: November 30, 2024; Latest Revision: December 1, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/data_interdependence_tutorial.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/configs/data_interdependence_function_config.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/data_interdependence_tutorial.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>
----------------------------------------

In this tutorial, you will learn

* what is data interdependence function,
* how to model data interdependence in `tinybig`,
* how to calculate data interdependence matrices,
* and how to create data interdependence function from config.

Many materials used in this tutorial are prepared based on the Section 5.1 of `[2]`, and readers are also recommended to 
refer to that section of the paper for more detailed technical descriptions when you are working on this tutorial.

**References**:

`[2] Jiawei Zhang. RPN 2: On Interdependence Function Learning Towards Unifying and Advancing CNN, RNN, GNN, and Transformer. ArXiv abs/2411.11162 (2024).`

----------------------------------------

## 1. What is Data Interdependence Function?

**Data interdependence functions** denotes a new family of interdependence functions capable of modeling a wide range of 
interdependence relationships among both attributes and instances. 
These functions can be defined using input data batches, underlying geometric and topological structures, 
optional learnable parameters, or a hybrid combination of these elements. 

Formally, based on the (optional) input data batch $\mathbf{X} \in {R}^{b \times m}$ (with $b$ instances and each instance with $m$ attributes), 
we define the interdependence functions modeling the interdependence relationships among instances and attributes in the data batch as follows:

\begin{equation}
\xi_a: {R}^{b \times m} \to {R}^{m \times m'} \text{, and }
\xi_i: {R}^{b \times m} \to {R}^{b \times b'},
\end{equation}

where $m'$ and $b'$ denote the output dimensions of their respective interdependence functions, respectively.

## 2. Examples of Data Interdependence Functions.

In the `tinybig` library, several different families of data interdependence functions have been implemented, whose detailed information
is also available at the reconciliation function [documentation pages](../../../documentations/interdependence/index.md).

In the following figure, we illustrate some example of them, including their names, formulas, and the corresponding 
output space dimensions. In the following parts of this tutorial, we will walk you through some of them to 
help you get familiar with some of these functions implemented in the `tinybig` library.

![data_interdependence_functions.png](img/data_interdependence_functions.png)

## 3. Identity Data Interdependence Functions.

A notable special case of the data interdependence functions is the **identity interdependence function**. 
This function outputs the interdependence matrix as a diagonal constant identity (or eye) matrix, 
formally represented as:

\begin{equation}
\xi_a(\mathbf{X}) = \mathbf{I} \in {R}^{m \times m'} \text{, and }
\xi_i(\mathbf{X}) = \mathbf{I} \in {R}^{b \times b'}
\end{equation}

where the output interdependence matrix is, by default, a square matrix with $m'=m$ and $b'=b$.

=== "Attribute Interdependence Function in Python"
    ```python linenums="1"
    from tinybig.util import set_random_seed
    
    random_seed = 42
    set_random_seed(random_seed=random_seed)
    
    import torch
    from tinybig.interdependence import identity_interdependence
    
    b, m = 2, 4
    X = torch.randint(0, 10, (b, m), dtype=torch.float, device='cpu')
    
    attribute_interdep_func = identity_interdependence(
        name='identity_interdependence', 
        interdependence_type='attribute', 
        b=b, m=m
    )

    m_prime = attribute_interdep_func.calculate_m_prime(m=m)
    attribute_A = attribute_interdep_func.calculate_A(x=X, device='cpu')
    attribute_xi_X = attribute_interdep_func(x=X, device='cpu')
    
    print('m_prime:', m_prime)
    print('X:', X)
    print('attribute_A:', attribute_A)
    print('attribute_xi_X:', attribute_xi_X)
    ```
=== "Instance Interdependence Function in Python"
    ```python linenums="1"
    from tinybig.util import set_random_seed

    random_seed = 42
    set_random_seed(random_seed=random_seed)
    
    import torch
    from tinybig.interdependence import identity_interdependence
    
    b, m = 2, 4
    X = torch.randint(0, 10, (b, m), dtype=torch.float, device='cpu')
    
    instance_interdep_func = identity_interdependence(
        name='identity_interdependence', 
        interdependence_type='instance', 
        b=b, m=m
    )

    b_prime = instance_interdep_func.calculate_b_prime(b=b)
    instance_A = instance_interdep_func.calculate_A(x=X, device='cpu')
    instance_xi_X = instance_interdep_func(x=X, device='cpu')
    
    print('b_prime:', b_prime)
    print('X:', X)
    print('instance_A:', instance_A)
    print('instance_xi_X:', instance_xi_X)
    ```
???+ quote "Identity interdependence function outputs"
    ```
    m_prime: 4
    X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    attribute_A: tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    attribute_xi_X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    ===========================================
    b_prime: 2
    X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    instance_A: tensor([[1., 0.],
            [0., 1.]])
    instance_xi_X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    ```

## 4. Kernel based Data Interdependence Functions.

The kernel-based interdependence functions compute the pairwise numerical scores for row (or column) vectors within 
the input data batch, thereby constructing a comprehensive interdependence matrix.

Formally, given a data batch $\mathbf{X} \in {R}^{b \times m}$, we define the numerical metric-based attribute 
interdependence function as:

\begin{equation}
\xi_a(\mathbf{X}) = \mathbf{A} \in {R}^{m \times m} \text{, where } \mathbf{A}(i, j) = \text{kernel} \left(\mathbf{X}(:, i), \mathbf{X}(:, j)\right).
\end{equation}

The notation "kernel($\cdot$, $\cdot$)" denotes the statistical or numerical kernel function. Based on the kernel function,
we can also define the instance interdependence function with the row-vectors of data batch $\mathbf{X}$.

=== "Instance Interdependence Function in Python"
    ```python linenums="1"
    from tinybig.util import set_random_seed
    
    random_seed = 42
    set_random_seed(random_seed=random_seed)
    
    import torch
    from tinybig.interdependence import numerical_kernel_based_interdependence
    from tinybig.koala.linear_algebra import euclidean_distance_kernel
    
    b, m = 2, 4
    X = torch.randint(0, 10, (b, m), dtype=torch.float, device='cpu')
    
    stat_interdep_func = numerical_kernel_based_interdependence(
        name='statistical_kernel_based_interdependence', 
        interdependence_type='attribute', 
        kernel=euclidean_distance_kernel,
        b=b, m=m
    )
    
    m_prime = stat_interdep_func.calculate_m_prime(m=m)
    attribute_A = stat_interdep_func.calculate_A(x=X, device='cpu')
    attribute_xi_X = stat_interdep_func(x=X, device='cpu')
    
    print('m_prime:', m_prime)
    print('X:', X)
    print('attribute_A:', attribute_A)
    print('attribute_xi_X:', attribute_xi_X)
    ```
???+ quote "Kernel based interdependence function outputs"
    ```
    m_prime: 4
    X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    attribute_A: tensor([[1.0000e+00, 6.1027e-03, 7.3834e-04, 5.9106e-02],
            [6.1027e-03, 1.0000e+00, 6.1027e-03, 4.2329e-02],
            [7.3834e-04, 6.1027e-03, 1.0000e+00, 1.1423e-02],
            [5.9106e-02, 4.2329e-02, 1.1423e-02, 1.0000e+00]])
    attribute_xi_X: tensor([[2.2836, 7.2181, 6.0899, 4.4831],
            [6.2669, 5.2059, 0.0806, 4.5663]])
    ```

## 5. Parameterized Data Interdependence Functions.

In addition to the above interdependence function solely defined based on the input data batch, 
another category of fundamental interdependence functions is the parameterized interdependence function, 
which constructs the interdependence matrix exclusively from learnable parameters.

Formally, given a learnable parameter vector $\mathbf{w} \in {R}^{l_{\xi}}$, 
the parameterized interdependence function transforms it into a matrix of desired dimensions $m \times m'$ as follows:

\begin{equation}
\xi(\mathbf{w}) = \text{reshape}(\mathbf{w}) = \mathbf{W} \in {R}^{m \times m'}.
\end{equation}

This parameterized interdependence function operates independently of any data batch, 
deriving the output interdependence matrix solely from the learnable parameter vector $\mathbf{w}$, 
whose requisite length of vector $\mathbf{w}$ is $l_{\xi} = m \times m'$.

=== "Instance Interdependence Function in Python"
    ```python linenums="1"
    from tinybig.util import set_random_seed
    
    random_seed = 42
    set_random_seed(random_seed=random_seed)
    
    import torch
    from tinybig.interdependence import parameterized_interdependence
    
    b, m = 2, 4
    X = torch.randint(0, 10, (b, m), dtype=torch.float, device='cpu')
    
    para_interdep_func = parameterized_interdependence(
        name='parameterized_interdependence', 
        interdependence_type='attribute', 
        b=b, m=m
    )
    
    l_xi = para_interdep_func.calculate_l()
    
    print('l_xi:', l_xi)
    w = torch.nn.Parameter(torch.randn(1, l_xi), requires_grad=True)
    
    m_prime = para_interdep_func.calculate_m_prime(m=m)
    attribute_A = para_interdep_func.calculate_A(x=X, w=w, device='cpu')
    attribute_xi_X = para_interdep_func(x=X, w=w, device='cpu')
    
    print('m_prime:', m_prime)
    print('X:', X)
    print('attribute_A:', attribute_A)
    print('attribute_xi_X:', attribute_xi_X)
    ```
???+ quote "Parameterized interdependence function outputs"
    ```
    l_xi: 16
    m_prime: 4
    X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    attribute_A: tensor([[ 1.7878, -0.4780, -0.2429, -0.9342],
            [-0.2483, -1.2082, -0.4777,  0.5201],
            [-1.5673, -0.2394,  2.3228, -0.9634],
            [ 2.0024,  0.4664,  1.5730, -0.9228]], grad_fn=<ViewBackward0>)
    attribute_xi_X: tensor([[ 0.4435, -8.9845, 16.3996, -7.6989],
            [17.4951, -7.0436,  2.4466, -6.6956]], grad_fn=<MmBackward0>)
    ```

## 6. Parameterized Bilinear Interdependence Functions.

In addition to the numerical metrics discussed above, we have also introduced another quantitative measure, 
namely the **bilinear form**, in the RPN 2 paper. 
The bilinear form enumerates all potential interactions between vector elements to compute interdependence scores.

Formally, given a data batch $\mathbf{X} \in {R}^{b \times m}$, we can represent the parameterized 
bilinear form-based interdependence function as follows:

\begin{equation}\label{equ:bilinear_interdependence_function}
\xi(\mathbf{X} | \mathbf{w}) = \mathbf{X}^\top \mathbf{W} \mathbf{X} = \mathbf{A} \in {R}^{m \times m},
\end{equation}

where $\mathbf{W} = \text{reshape}(\mathbf{w}) \in {R}^{b \times b}$ denotes the parameter matrix reshaped from the 
learnable parameter vector $\mathbf{w} \in {R}^{l_{\xi}}$ with length $l_{\xi} = b^2$.

As introduced in the RPN 2 paper, the parameter matrix $\mathbf{W}$ can also be fabricated with the reconciliation
functions, which can reduce the number of required parameters in the interdependence function, e.g., low-rank or lphm
reconciliations. An example of the low-rank parameterized bilinear interdependence function is shown below.

=== "Instance Interdependence Function in Python"
    ```python linenums="1"
    from tinybig.util import set_random_seed
    
    random_seed = 42
    set_random_seed(random_seed=random_seed)
    
    import torch
    from tinybig.interdependence import lowrank_parameterized_bilinear_interdependence
    
    b, m = 2, 4
    X = torch.randint(0, 10, (b, m), dtype=torch.float, device='cpu')
    
    bilinear_interdep_func = lowrank_parameterized_bilinear_interdependence(
        name='lowrank_parameterized_bilinear_interdependence', 
        interdependence_type='attribute', 
        r=1, b=b, m=m
    )
    
    l_xi = bilinear_interdep_func.calculate_l()
    
    print('l_xi:', l_xi)
    w = torch.nn.Parameter(torch.randn(1, l_xi), requires_grad=True)
    
    m_prime = bilinear_interdep_func.calculate_m_prime(m=m)
    attribute_A = bilinear_interdep_func.calculate_A(x=X, w=w, device='cpu')
    attribute_xi_X = bilinear_interdep_func(x=X, w=w, device='cpu')
    
    print('m_prime:', m_prime)
    print('X:', X)
    print('attribute_A:', attribute_A)
    print('attribute_xi_X:', attribute_xi_X)
    ```
???+ quote "Parameterized bilinear interdependence function outputs"
    ```
    l_xi: 4
    m_prime: 4
    X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    attribute_A: tensor([[  -1.9793,  -41.2637,  -44.5661,  -21.1267],
            [  -5.1732, -107.8507, -116.4822,  -55.2187],
            [  -3.9643,  -82.6473,  -89.2617,  -42.3147],
            [  -3.0814,  -64.2413,  -69.3826,  -32.8910]], grad_fn=<MmBackward0>)
    attribute_xi_X: tensor([[  -76.2820, -1590.3319, -1717.6089,  -814.2365],
            [  -50.0671, -1043.8013, -1127.3385,  -534.4174]],
           grad_fn=<MmBackward0>)
    ```

## 7. Data Interdependence Function instantiation from Configs.

Besides the above manual function definitions, we will also briefly introduce how to instantiate the 
reconciliation function instances from their configurations.
For some complex function configurations, we can also save the configuration detailed information into a file, 
which can be loaded for the function instantiation.

Please save the following `data_interdependence_function_config.yaml` to the directory `./configs/` that your code can access:

=== "Expansion Function in Python"
    ```python linenums="1"
    from tinybig.util import set_random_seed

    random_seed = 42
    set_random_seed(random_seed=random_seed)
    
    import torch
    
    b, m = 2, 4
    X = torch.randint(0, 10, (b, m), dtype=torch.float, device='cpu')
    
    from tinybig.config.base_config import config
    
    config_obj = config(name='data_interdependence_function_config')
    func_configs = config_obj.load_yaml(cache_dir='./configs', config_file='data_interdependence_function_config.yaml')
    
    bilinear_interdep_func = config.instantiation_from_configs(
        configs=func_configs['data_interdependence_function_configs'], 
        class_name='data_interdependence_function_class', 
        parameter_name='data_interdependence_function_parameters'
    )
    
    l_xi = bilinear_interdep_func.calculate_l()
    
    print('l_xi:', l_xi)
    w = torch.nn.Parameter(torch.randn(1, l_xi), requires_grad=True)
    
    m_prime = bilinear_interdep_func.calculate_m_prime(m=m)
    attribute_A = bilinear_interdep_func.calculate_A(x=X, w=w, device='cpu')
    attribute_xi_X = bilinear_interdep_func(x=X, w=w, device='cpu')
    
    print('m_prime:', m_prime)
    print('X:', X)
    print('attribute_A:', attribute_A)
    print('attribute_xi_X:', attribute_xi_X)
    ```

=== "./configs/data_interdependence_function_config.yaml"
    ```yaml linenums="1"
    data_interdependence_function_configs:
      data_interdependence_function_class: tinybig.interdependence.lowrank_parameterized_bilinear_interdependence
      data_interdependence_function_parameters:
        name: lowrank_parameterized_bilinear_interdependence
        r: 1
        b: 2
        m: 4
        interdependence_type: attribute
        require_parameters: True
        require_data: True
        device: cpu
    ```
???+ quote "Reconciliation function instantiation from Configs"
    ```
    l_xi: 4
    m_prime: 4
    X: tensor([[2., 7., 6., 4.],
            [6., 5., 0., 4.]])
    attribute_A: tensor([[  -1.9793,  -41.2637,  -44.5661,  -21.1267],
            [  -5.1732, -107.8507, -116.4822,  -55.2187],
            [  -3.9643,  -82.6473,  -89.2617,  -42.3147],
            [  -3.0814,  -64.2413,  -69.3826,  -32.8910]], grad_fn=<MmBackward0>)
    attribute_xi_X: tensor([[  -76.2820, -1590.3319, -1717.6089,  -814.2365],
            [  -50.0671, -1043.8013, -1127.3385,  -534.4174]],
           grad_fn=<MmBackward0>)
    ```

## 8. Conclusion.

In this tutorial, we briefly discussed part of the data interdependence functions implemented in the `tinybig` library.
Based on a randomly generated toy data batch, we use several concrete examples to illustrate how to use these data
interdependence functions to calculate the interdependence matrix and compute the transformed data batch.
Furthermore, we have also introduced how to use the configuration files to instantiate function instances.

Besides the data interdependence functions introduced above, there also exist another type of interdependence functions
defined based on the data modality specific underlying structures, e.g., grid, chain and graph, which will be introduced
in the following tutorial articles instead.





