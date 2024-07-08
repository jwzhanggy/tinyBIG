# Tutorial on Data Expansion Functions ([Jupyter Note](../../../notes/expansion_tutorial.ipynb))

Author: Jiawei Zhang <br>
(Released: July 6, 2024; Latest Revision: July 7, 2024.)
----------------------------------------

In this tutorial, you will learn

* what is data expansion function,
* how to do data expansion in `tinybig`,
* the optional data processing functions in the expansion,
* and visualize the expansion outputs.

Many materials used in this tutorial are prepared based on the Section 5.1 of `[1]`, and you are also recommended to 
refer to that section of the paper for more detailed technical descriptions when you are working on this tutorial.

**References**:

`[1] Jiawei Zhang. RPN: Reconciled Polynomial Network. Towards Unifying PGMs, Kernel SVMs, MLP and KAN.`

----------------------------------------

## 1. What is Data Expansion Function?

Formally, **data expansion function** is one component function used in the {{our}} model for expanding the input data 
vectors from the input space to a high-dimensional intermediate space:

\begin{equation}
    \kappa: R^m \to R^D,
\end{equation}

where $m$ and $D$ denote the input space and expansion space dimensions, respectively.

Data expansion provides high-order signals and features about the input that cannot be directly learned from the original 
input space. For the data instances that cannot be separated in the input space, with such high-dimensional features, 
the {{our}} model can also learn better parameters to approximate their underlying distributions for easier separations.

The currently released `tinybig` package only implements data expansion functions. Meanwhile, the upcoming new versions of 
`tinybig` in development will support both "data expansion functions" and "data compression function", which can both be 
referred to as the **data transformation functions**. 

In this tutorial, for many places, we will use the function names "data transformation function" and 
"data expansion function" interchangeably without distinguishing their differences.

## 2. Examples of Data Expansion Functions?

In {{toolkit}}, several different families of data expansion functions have been implemented, whose detailed information
is also available at the expansion function [documentation pages](../../../documentations/expansion/index.md).

In the following figure, we illustrate some example of them, including their names, formulas, and the corresponding 
expansion dimension calculations. In the following parts of this tutorial, we will walk you through some of them to 
help you get familiar with their usages.

![data_expansion_functions.png](../../img/data_expansion_functions.png)

## 3. Taylor's Expansion

Formally, given a vector $\mathbf{x} = [x_1, x_2, \cdots, x_m] \in R^m$ of dimension $m$, its Taylor's expansion 
polynomials with orders no greater than $d$ can be represented as follows:

\begin{equation}
\kappa (\mathbf{x} | d) = [P_1(\mathbf{x}), P_2(\mathbf{x}), \cdots, P_d(\mathbf{x}) ] \in R^D,
\end{equation}

where the output dimension $D = \sum_{i=1}^d m^i$. 

In the above Taylor's expansion, notation $P_d(\mathbf{x})$ represents the list of potential polynomials composed by 
the product of the vector elements $x_1$, $x_2$, $\cdots$, $x_m$ with sum of the degrees equals $d$, i.e.,

\begin{equation}
P_d(\mathbf{x}) = [x_1^{d_1} x_2^{d_2} \cdots x_m^{d_m}]_{d_1, d_2,\cdots, d_m \in \{0, 1, \cdots m\} \land \sum_{i=1}^m d_i = d}.
\end{equation}

Some examples of the multivariate polynomials are provided as follows:

\begin{equation}
\begin{aligned}
P_0(\mathbf{x}) &= [1] \in R^{1},\\
P_1(\mathbf{x}) &= [x_1, x_2, \cdots, x_m] \in R^{m},\\
P_2(\mathbf{x}) &= [x_1^2,  x_1 x_2, x_1 x_3, \cdots, x_1 x_m, x_2 x_1, x_2^2, x_2 x_3, \cdots, x_{m} x_m] \in R^{m^2}.
\end{aligned}
\end{equation}

### 3.1 Taylor's Expansion Function

Taylor's expansion has been implemented in `tinybig`, which can be called and applied to inputs as follows:
```python linenums="1"
from tinybig.expansion import taylor_expansion
import torch

exp_func = taylor_expansion(name='taylor_expansion_for_toy_example', d=2)

x = torch.Tensor([[1, 2, 3]])
print('m:', x.size(1), 'D:', exp_func.calculate_D(m=x.size(1)))

kappa_x = exp_func(x)
print(x, kappa_x)
print(x.shape, kappa_x.shape)
```
???+ quote "Taylor's expansion printing output"
    ```
    m: 3 D: 12
    tensor([[1., 2., 3.]]) tensor([[1., 2., 3., 1., 2., 3., 2., 4., 6., 3., 6., 9.]])
    torch.Size([1, 3]) torch.Size([1, 12])

    ```

As reminded before in the [Quickstart tutorial](../../../guides/quick_start.md), the current expansion functions in `tinybig` will only accept 2D tensors 
with shape $(B, m)$ as inputs, where $B$ denotes the batch size and $m$ denotes the input dimension length.


### 3.2 Taylor's Expansion Function Instantiation from Configs

Besides the manual definition of the expansion functions, `tinybig` also allows the function instantiation from the 
configurations. 

For instance, the data expansion function defined in the previous subsection can also be instantiated from its configs 
`data_transformation_configs` as follows:

```python linenums="1"
data_transformation_configs = {
    'data_transformation_class': 'tinybig.expansion.taylor_expansion',
    'data_transformation_parameters':{
        'name': 'taylor_expansion_from_configs',
        'd': 2
    }
}
```

```python linenums="1"
from tinybig.util import get_obj_from_str
import torch

exp_func = get_obj_from_str(data_transformation_configs['data_transformation_class'])(**data_transformation_configs['data_transformation_parameters'])

x = torch.Tensor([[1, 2, 3]])
kappa_x = exp_func(x)
print(x, kappa_x)
```
???+ quote "Taylor's expansion instantiation from configs printing output"
    ```
    tensor([[1., 2., 3.]]) tensor([[1., 2., 3., 1., 2., 3., 2., 4., 6., 3., 6., 9.]])
    ```
In the following tutorials, to make the code more descent, we will just use the configuration files to instantiate the 
other functions, modules and models implemented in the {{toolkit}} toolkit.


## 4. Optional processing functions for expansions

Besides doing the expansions, `tinybig` also allows the data expansion functions to apply optional pre- and post-processing
functions to the inputs and outputs of the expansion functions, respectively.

These optional pre- and post-processing functions will provide {{our}} and {{toolkit}} with great flexibility in model 
design and implementation. In this part, we will illustrate how to add these processing functions into the data expansion
functions.

### 4.1 Pre-processing functions

The pre-processing function used in data expansion can be very diverse, including different activation functions and 
normalization functions. You can also define your customized pre-processing function and use them for data expansions.

Below, we provide an example to add the `layer-norm` as a pre-processing function to the Taylor's expansion function:

```python linenums="1"
from tinybig.expansion import taylor_expansion
import torch

preprocess_func = torch.nn.LayerNorm(normalized_shape=3)
exp_func = taylor_expansion(
    name='taylor_expansion_with_preprocessing', 
    d=2, 
    preprocess_functions=preprocess_func
)

x = torch.Tensor([[1, 2, 3]])
kappa_x = exp_func(x)
print(x, kappa_x)
```
???+ quote "Taylor's expansion with pre-processing layer-norm"
    ```
    tensor([[1., 2., 3.]]) tensor([[-1.2247,  0.0000,  1.2247,  1.5000, -0.0000, -1.5000, -0.0000,  0.0000,
          0.0000, -1.5000,  0.0000,  1.5000]], grad_fn=<CatBackward0>)
    ```

What's more, `tinybig` allows you to add multiple pre-processing functions into the data expansion function definition.
Below, we show the Taylor's expansion with both sigmoid and layer-norm as the pre-processing functions:
```python linenums="1"
from tinybig.expansion import taylor_expansion
import torch

sigmoid = torch.nn.Sigmoid()
layer_norm = torch.nn.LayerNorm(normalized_shape=3)
exp_func = taylor_expansion(
    name='taylor_expansion_with_sigmoid_layernorm', 
    d=2, 
    preprocess_functions=[sigmoid, layer_norm]
)

x = torch.Tensor([[1, 2, 3]])
kappa_x = exp_func(x)
print(x, kappa_x)
```
???+ quote "Taylor's expansion with pre-processing sigmoid and layer-norm"
    ```
    tensor([[1., 2., 3.]]) tensor([[-1.3402,  0.2814,  1.0588,  1.7962, -0.3772, -1.4190, -0.3772,  0.0792,
          0.2980, -1.4190,  0.2980,  1.1210]], grad_fn=<CatBackward0>)
    ```

### 4.2 Post-processing functions

Below, we will define the Taylor's expansion functions with post-processing functions. 

Meanwhile, slightly different from the above manual function definition, we propose to define the function 
configuration in a separate file, and load it with `tinybig` for the expansion function instantiation.

Please save the following `expansion_function_postprocessing.yml` to the directory `./configs/` that your code can access:
=== "Expansion Function in Python"
    ```python linenums="1"
    from tinybig.config import config
    from tinybig.util import get_obj_from_str
    
    config_obj = config(name='taylor_expansion_config')
    func_configs = config_obj.load_yaml(cache_dir='./configs', config_file='expansion_function_postprocessing.yaml')
    print(func_configs)
    
    data_transformation_configs = func_configs['data_transformation_configs']
    exp_func = get_obj_from_str(data_transformation_configs['data_transformation_class'])(**data_transformation_configs['data_transformation_parameters'])
    
    x = torch.Tensor([[1, 2, 3]])
    kappa_x = exp_func(x)
    print(x, kappa_x)
    ```

=== "./configs/expansion_function_postprocessing.yml"
    ```yaml linenums="1"
    data_transformation_configs:
      data_transformation_class: tinybig.expansion.taylor_expansion
      data_transformation_parameters:
        name: taylor_expansion_with_preprocessing
        d: 2
        postprocess_function_configs:
          - function_class: torch.nn.Sigmoid
          - function_class: torch.nn.LayerNorm
            function_parameters:
              normalized_shape: 12
    ```
???+ quote "Taylor's expansion with post-processing sigmoid and layer-norm"
    ```
    {'data_transformation_configs': {'data_transformation_class': 'tinybig.expansion.taylor_expansion', 'data_transformation_parameters': {'name': 'taylor_expansion_with_preprocessing', 'd': 2, 'postprocess_function_configs': [{'function_class': 'torch.nn.Sigmoid'}, {'function_class': 'torch.nn.LayerNorm', 'function_parameters': {'normalized_shape': 12}}]}}}
    tensor([[1., 2., 3.]]) tensor([[-1.9707, -0.3362,  0.4473, -1.9707, -0.3362,  0.4473, -0.3362,  0.7686,
          0.9380,  0.4473,  0.9380,  0.9636]],
       grad_fn=<NativeLayerNormBackward0>)
    ```

Careful readers may have already noticed that the `normalized_shape` parameters of the `layer-norm` function in the 
pre-processing and post-processing function lists are different, since they are applied to the input and output vectors
of the expansion functions, respectively. Also, as the parameter `d` of Taylor's expansion function changes, the 
`normalized_shape` of `layer-norm` as the post-processing function may also need to be adjusted accordingly as well.

???+ note "`torch.nn.LayerNorm` vs `torch.nn.functional.layer_norm` vs function string name '`layer_norm`'"
    Current `tinybig` allows you to define these function in different forms, like "`torch.nn.LayerNorm`, `torch.nn.functional.layer_norm`,
    and even just as function string name `"layer_norm"` (which is used in the previous [Quickstart tutorial](../../../guides/quick_start.md)).
    
    All the pre- and post-processing functions (as well as the output-processing and activation functions to be introduced later)
    are all handled by `tinybig.util.process_function_list`, `tinybig.util.func_x` and `tinybig.util.str_func_x`.

    We recommend you to define these functions as objects, like `torch.nn.LayerNorm` together with the parameters as shown above, 
    which will be first instantiated by `tinybig.util.process_function_list` into callable objects, and then executed by 
    `tinybig.util.func_x`, without entering `tinybig.util.str_func_x`.

    The current `tinybig.util.str_func_x` can only handle the string names of a few frequently used functions, which may
    fail to work for the functions whose names or classes have not been recorded yet. 
    

## 5. Expansion Visualization

We have defined the Taylor's expansion function above and also introduced how to add different pre- and post-processing
functions to process the input and output of the expansions.

Below, we will illustrate the obtained expansion results on real-world MNIST image data and visualize them. 
So, we can compare the data vectors before and after the expansion.

We first define an image display function with `matplotlib` (please install `matplotlib` before running the following code):
```python linenums="1"
import matplotlib.pyplot as plt
def show_image(X):
    plt.figure(figsize=(8, 8))
    plt.imshow(X.numpy().squeeze(), cmap='gray')
    plt.show()
```

As introduced in the previous [Quickstart tutorial](../../../guides/quick_start.md), `tinybig` has a built-in class to load
the mnist dataset (after flattening and normalization):
```python linenums="1"
from tinybig.data import mnist
mnist_data = mnist(name='mnist', train_batch_size=64, test_batch_size=64)
mnist_loaders = mnist_data.load(cache_dir='./data/')
X_batch, y_batch = next(iter(mnist_loaders['test_loader']))
x = X_batch[0:1,:]
print(x.shape)
```
???+ quote "MNIST image shape printing output"
    ```
    torch.Size([1, 784])
    ```

By feeding the image data `x` to the `taylor_expansion` function, we can obtain the Taylor's expansion results
```python linenums="1"
from tinybig.expansion import taylor_expansion

exp_func = taylor_expansion(name='taylor_expansion_for_mnist', d=2)
kappa_x = exp_func(x)
raw_image, expansion_image = kappa_x[0,:784], kappa_x[0,784:]
print(raw_image.shape, expansion_image.shape)
```
???+ quote "MNIST image shape printing output"
    ```
    torch.Size([784]) torch.Size([614656])
    ```

By feeding the image data `raw_image` and `expansion_image` to the `show_image` function, we can display the image before 
and after the expansion as follows:
```python linenums="1"
show_image(raw_image.view(28, 28))
```
???+ quote "MNIST raw image display"
    ![mnist_raw_data_vector.png](../../img/mnist_raw_data_vector.png)

```python linenums="1"
show_image(expansion_image.view(784, 784))
```
???+ quote "MNIST expansion image display"
    ![mnist_raw_data_vector.png](../../img/mnist_expansion_data.png)

Compared with the raw image, the above expansion image visualization is less readable, which makes it hard to interpret 
the expansion results.

Below, we will use the `reshape_expansion` function to re-organize the expansion image of size $784 \times 784$ into 
$28 \times 28$ small-sized images, where each image has a size of $28 \times 28$.
```python linenums="1"
def reshape_expansions(expansion):
    grid28x28 = expansion.reshape(28, 28, 28, 28)
    reshaped_expansion = grid28x28.permute(0, 2, 1, 3).reshape(784, 784)
    return reshaped_expansion
```

With the above reshape function, we can process and display the expansion image as follows:
```python linenums="1"
reshaped_expansion_image = reshape_expansions(expansion_image)
print(reshaped_expansion_image.shape)
show_image(reshaped_expansion_image)
```
???+ quote "MNIST reshaped expansion image display"
    ![mnist_raw_data_vector.png](../../img/mnist_expansion_data_reshape.png)

The above image visualization illustrates the expansion effects of each pixel in the image to the whole raw image,
and there are $28 \times 28$ sub-images in the expansion results.

These high-order expansions actually provide some important features about the input data. 
We will discuss more about this in the following tutorials on the {{our}} model and the parameter reconciliation.

## 6. Conclusion

In this tutorial, we discussed the data expansion functions in the {{toolkit}} toolkit. We introduced different ways to 
define the expansion functions, including both manual function definition and configuration file based function 
instantiation. What's more, we also introduced the optional pre- and post-processing functions that can be used in 
the data expansion functions for input and output processing. Finally, we visualize the expansion results on the MNIST
image data, which also helps interpret the expansion functions and their performance.