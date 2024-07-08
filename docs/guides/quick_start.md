# Quickstart Tutorial ([Jupyter Note](../notes/quickstart_tutorial.ipynb))

<!--[![Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)-->

Author: Jiawei Zhang <br>
(Released: July 4, 2024; 1st Revision: July 6, 2024.)<br>
-------------------------

In this quickstart tutorial, we will walk you through the MNIST image classification task with {{our}} 
built based on the Taylor's expansion and dual lphm reconciliation functions via the APIs provided by `tinybig`.

We assume you have correctly installed the latest `tinybig` and its dependency packages already.
If you haven't installed them yet, please refer to the [installation](installation.md) page for the guidance.

This quickstart tutorial is prepared based on 
[the RPN paper](https://github.com/jwzhanggy/tinyBIG/blob/main/docs/assets/files/rpn_paper.pdf) `[1]`. 
We also recommend reading that paper first for detailed technical information about the {{our}} model and {{toolkit}} toolkit. 

**Reference**

`[1] Jiawei Zhang. RPN: Reconciled Polynomial Network. Towards Unifying PGMs, Kernel SVMs, MLP and KAN.`

-------------------------

## Environment Setup

This tutorial was written on a mac with apple silicon, and we will use `'mps'` as the device here, 
and you can change it to `'cpu'` or `'cuda'` according to the device you are using now.
```python linenums="1"
from tinybig.util import set_random_seed
set_random_seed(random_seed=1234)
DEVICE = 'mps' # or 'cpu', or 'cuda'
```

## Loading Datasets

### MNIST Dataloader

In this quickstart tutorial, we will take the MNIST dataset as an example to illustrate how `tinybig` loads data:

```python linenums="1"
from tinybig.data import mnist

mnist_data = mnist(name='mnist', train_batch_size=64, test_batch_size=64)
mnist_loaders = mnist_data.load(cache_dir='./data/')
train_loader = mnist_loaders['train_loader']
test_loader = mnist_loaders['test_loader']
```

??? quote "Data downloading outputs"
    ```
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz
    100%|██████████| 9912422/9912422 [00:00<00:00, 12146011.18it/s]
    Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz
    100%|██████████| 28881/28881 [00:00<00:00, 278204.89it/s]
    Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz
    100%|██████████| 1648877/1648877 [00:04<00:00, 390733.03it/s]
    Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    100%|██████████| 4542/4542 [00:00<00:00, 2221117.96it/s]
    Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw
    ```

The `mnist_data.load(cache_dir='./data/')` method will download the MNIST dataset from torchvision to a local directory `'./data/'`.

With the `train_loader` and `test_loader`, we can access the MNIST image and label mini-batches in the training and 
testing sets:

```python linenums="1"
for X, y in train_loader:
    print('X shape:', X.shape, 'y.shape:', y.shape)
    print('X', X)
    print('y', y)
    break
```

???+ quote "Data batch printing outputs"
    ```
    X shape: torch.Size([64, 784]) y.shape: torch.Size([64])

    X tensor([[-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
            [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
            [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
            ...,
            [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
            [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242],
            [-0.4242, -0.4242, -0.4242,  ..., -0.4242, -0.4242, -0.4242]])
    y tensor([3, 7, 8, 5, 6, 1, 0, 3, 1, 7, 4, 1, 3, 4, 4, 8, 4, 8, 2, 4, 3, 5, 5, 7,
            5, 9, 4, 2, 2, 3, 3, 4, 1, 2, 7, 2, 9, 0, 2, 4, 9, 4, 9, 2, 1, 3, 6, 5,
            9, 4, 4, 8, 0, 3, 2, 8, 0, 7, 3, 4, 9, 4, 0, 5])
    ```

???+ note "Built-in image data transformation"
    Note: the `tinybig.data.mnist` has a built-in method to flatten and normalize the MNIST images from tensors of size $28 \times 28$ into vectors of length $784$ via `torchvision.transforms`:
    ```python linenums="1"
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        torch.flatten
    ])
    ```

## Creating the RPN Model

To model the underlying data distribution mapping $f: R^m \to R^n$, the {{our}} model disentangles the input data from 
model parameters into three component functions:

* **Data Expansion Function**: $\kappa: R^m \to R^D$,
* **Parameter Reconciliatoin Function**: $\psi: R^l \to R^{n \times D}$,
* **Remainder Function** $\pi: R^m \to R^n$,

where $m$ and $n$ denote the input and output space dimensions, respectively. Notation $D$ denotes the target expansion 
space dimension (determined by the expansion function and input dimension $m$) and $l$ is the number of learnable parameters 
in the model (determined by the reconciliation function and dimensions $n$ and $D$).

So, the underlying mapping $f$ can be approximated by {{our}} as the inner product of the expansion function with
the reconciliation function, subsequentlly summed with the remainder function:
$$
g(\mathbf{x} | \mathbf{w}) = \left \langle \kappa(\mathbf{x}), \psi(\mathbf{w}) \right \rangle + \pi(\mathbf{x}),
$$
where for any input data instance $\mathbf{x} \in R^m$.

### Data Expansion Function

Various data expansion functions have been implemented in `tinybig` already. In this tutorial, we will use the 
Taylor's expansion function as an example to illustrate how data expansion works.

```python linenums="1"
from tinybig.expansion import taylor_expansion

exp_func = taylor_expansion(name='taylor_expansion', d=2, postprocess_functions='layer_norm', device=DEVICE)
x = X[0:1,:]
x = X[0:1,:]
D = exp_func.calculate_D(m=x.size(1))
print('D:', D)

kappa_x = exp_func(x=x)
print('x.shape', x.shape, 'kappa_x.shape', kappa_x.shape)
```

???+ quote "Data expansion printing outputs"
    ```
    Expansion space dimension: 615440
    x.shape torch.Size([1, 784]) kappa_x.shape torch.Size([1, 615440])
    ```

In the above code, we define a Taylor's expansion function of order `d=2` and `'layer_norm'` as the post-processing function.
By applying the expansion function to a data batch with one single data instance, we print output the expansion dimensions
as $D = 784 + 784 \times 784 = 615440$.

???+ note "Expansion function input shapes"
    Note: the expansion function will accept batch inputs as 2D tensors of shape `(B, m)`, with `B` and `m` denote the batch size and input dimension,
    such as, `X[0:1,:]` or `X`. If we feed list, array or 1D tensor, e.g., `X[0,:]`, to the expansion function, it will report errors).

All the expansion functions in `tinybig` has a method `calculate_D(m)`, which can automatically calculates the
target expansion space dimension $D$ based on the input space dimension, i.e., the parameter $m$. The calculated $D$ will
be used later in the reconciliation functions.

### Parameter Reconciliation Function

In `tinybig`, we have implemented different categories of parameter reconciliation functions. Below, we will use the
dual lphm to illustrate how parameter reconciliation works. Several other reconciliation functions will also
be introduced in the tutorial articles.

Assuming we need to build a {{our}} layer with the output dimension $n=64$ here:

```python linenums="1"
from tinybig.reconciliation import identity_reconciliation

rec_func = dual_lphm_reconciliation(name='dual_lphm_reconciliation', p=8, q=784, r=5, device=DEVICE)
l = rec_func.calculate_l(n=64, D=D)
print('Required learnable parameter number:', l)
```

???+ quote "Lorr parameter reconciliation printing outputs"
    ```
    Required learnable parameter number: 7925
    ```

For the parameters, we need to make sure $p$ divides $n$ and $q$ divides $D$. As to the rank parameter $r$, 
it is defined depending on how many parameters we plan to use for the model. 

We use `r=5` here, but you can also try other rank values, e.g., `r=2`, 
which will further reduce the number of parameters but still achieve decent performance.

???+ note "Automatic parameter creation"
    We will not create parameters here, which can be automatically created in the {{our}} head to be used below.

### Remainder Function

By default, we will use the zero remainder in this tutorial, which will not create any learnable parameters:

```python linenums="1"
from tinybig.remainder import zero_remainder

rem_func = zero_remainder(name='zero_remainder', require_parameters=False, enable_bias=False, device=DEVICE)
```

### RPN Head

Based on the above component functions, we can combine them together to define the {{our}} mode. Below, we will first
define the {{our}} head first, which will be used to compose the layers of {{our}}.

```python linenums="1"
from tinybig.module import rpn_head

head = rpn_head(m=784, n=64, channel_num=1, data_transformation=exp_func, parameter_fabrication=rec_func, remainder=rem_func, device=DEVICE)
```

Here, we build a rpn head with one channel of parameters. The parameter `data_transformation` is a general name of 
`data_expansion`, and `parameter_fabrication` can be viewed as equivalent to `parameter_reconciliation`.

We use these general `data_transformation` and `parameter_fabrication` names here not only for their current functionality 
but also to establish a framework that allows for the future expansion of `tinybig`, enabling the addition of new 
functions and components under these broader categorical names.

### RPN Layer

The above head can be used to build the first {{our}} layer of {{our}}: 

```python linenums="1"
from tinybig.module import rpn_layer

layer_1 = rpn_layer(m=784, n=64, heads=[head], device=DEVICE)
```

### Deep RPN Model with Multi-Layers

Via a similar process, we can also define two more {{our}} layers:

```python linenums="1"
layer_2 = rpn_layer(
    m=64, n=64, heads=[
        rpn_head(
            m=64, n=64, channel_num=1,
            data_transformation=taylor_expansion(d=2, postprocess_functions='layer_norm', device=DEVICE),
            parameter_fabrication=dual_lphm_reconciliation(p=8, q=64, r=5, device=DEVICE),
            remainder=zero_remainder(device=DEVICE),
            device=DEVICE
        )
    ],
    device=DEVICE
)

layer_3 = rpn_layer(
    m=64, n=10, heads=[
        rpn_head(
            m=64, n=10, channel_num=1,
            data_transformation=taylor_expansion(d=2, postprocess_functions='layer_norm', device=DEVICE),
            parameter_fabrication=dual_lphm_reconciliation(p=2, q=64, r=5, device=DEVICE),
            remainder=zero_remainder(device=DEVICE),
            device=DEVICE
        )
    ],
    device=DEVICE
)
```

By staking these three layers on top of each other, we can build a deep {{our}} model:

```python linenums="1"
from tinybig.model import rpn

model = rpn(name='3_layer_rpn_model', layers = [layer_1, layer_2, layer_3], device=DEVICE)
```

Later on, in the tutorial on `rpn_config`, we will introduce an easier way to define the model architecture directly 
with the configuration file instead.

## {{our}} Training on MNIST

Below we will train the {{our}} model with the loaded MNIST `train_loader`.

### Learner Setup

`tinybig` provides a built-in leaner module, which can train the input model with the provided optimizer. Below, we will
set up the learner with `torch.nn.CrossEntropyLoss` as the loss function, `torch.optim.AdamW` as the optimizer, and 
`torch.optim.lr_scheduler.ExponentialLR` as the learning rate scheduler:

```python linenums="1"
import torch
from tinybig.learner import backward_learner

optimizer=torch.optim.AdamW(lr=2.0e-03, weight_decay=2.0e-04, params=model.parameters())
lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(gamma=0.95, optimizer=optimizer)
loss = torch.nn.CrossEntropyLoss()
learner = backward_learner(n_epochs=3, optimizer=optimizer, loss=loss, lr_scheduler=lr_scheduler)
```

Here, we train the model for just 3 epochs to quickly assess its performance. 
You can increase the number of epochs to train the model until convergence.

### Training

With the previously loaded MNIST `mnist_loaders`, we can train the {{our}} model built above with the `learner`. 
To monitor the learning performance, we also pass an evaluation metric to the learner to record the training accuracy scores:

```python linenums="1"
from tinybig.metric import accuracy

print('parameter num: ', sum([parameter.numel() for parameter in model.parameters()]))

metric = accuracy(name='accuracy_metric')
training_records = learner.train(model=model, data_loader=mnist_loaders, metric=metric, device=DEVICE)
```

We count the total number of learnable parameters involved in the {{our}} model built above and 
provide the `tqdm` training records as follows:

???+ quote "Model training records"
    ```
    parameter num:  9330

    100%|██████████| 938/938 [00:42<00:00, 21.86it/s, epoch=0/3, loss=0.0519, lr=0.002, metric_score=0.969, time=43.1]
    
    Epoch: 0, Test Loss: 0.12760563759773874, Test Score: 0.9621, Time Cost: 3.982516050338745
    
    100%|██████████| 938/938 [00:43<00:00, 21.74it/s, epoch=1/3, loss=0.0112, lr=0.0019, metric_score=1, time=90.2]    
    
    Epoch: 1, Test Loss: 0.09334634791371549, Test Score: 0.9717, Time Cost: 4.184643030166626
    
    100%|██████████| 938/938 [00:42<00:00, 21.90it/s, epoch=2/3, loss=0.0212, lr=0.0018, metric_score=1, time=137]     
    
    Epoch: 2, Test Loss: 0.08378902525169431, Test Score: 0.9749, Time Cost: 4.574808120727539

    ```

### Testing

Furthermore, by applying the trained model to the testing set, we can obtain the prediction results obtained by the model
as follows:

```python linenums="1"
test_result = learner.test(model=model, test_loader=mnist_loaders['test_loader'], metric=metric, device=DEVICE)
print(metric.__class__.__name__, metric.evaluate(y_true=test_result['y_true'], y_pred=test_result['y_pred'], y_score=test_result['y_score'], ))
```

???+ quote "Model testing results"
    ```
    accuracy 0.9749
    ```

The above results indicate that {{our}} with a 3-layer architecture will achieve a decent testing accuracy score of `0.9749`, 
also it only uses `9330` learnable parameters, much less than that of MLP and KAN with similar architectures.
