from tinybig.expansion import taylor_expansion
import torch

x = torch.Tensor([[1, 2, 3]])
exp_func = taylor_expansion(name='taylor_expansion_for_toy_example', d=2)
print('m:', x.size(1), 'D:', exp_func.calculate_D(m=x.size(1)))

kappa_x = exp_func(x)
print(x, kappa_x)
print(x.shape, kappa_x.shape)

data_transformation_configs = {
    'function_class': 'tinybig.expansion.taylor_expansion',
    'function_parameters':{
        'name': 'taylor_expansion_from_configs',
        'd': 2
    }
}

from tinybig.config.base_config import config
exp_func = config.get_obj_from_str(data_transformation_configs['function_class'])(**data_transformation_configs['function_parameters'])
kappa_x = exp_func(x)
print(x, kappa_x)

from tinybig.expansion import taylor_expansion
import torch

x = torch.Tensor([[1, 2, 3]])
preprocess_func = torch.nn.LayerNorm(normalized_shape=3)
exp_func = taylor_expansion(
    name='taylor_expansion_with_preprocessing', 
    d=2, 
    preprocess_functions=preprocess_func
)

kappa_x = exp_func(x)
print(x, kappa_x)

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

from tinybig.config.base_config import config

config_obj = config(name='taylor_expansion_config')
func_configs = config_obj.load_yaml(cache_dir='./configs', config_file='expansion_function_postprocessing.yaml')
print(func_configs)

data_transformation_configs = func_configs['data_transformation_configs']
print(data_transformation_configs.keys())
exp_func = config.get_obj_from_str(data_transformation_configs['function_class'])(**data_transformation_configs['function_parameters'])

x = torch.Tensor([[1, 2, 3]])
kappa_x = exp_func(x)
print(x, kappa_x)

import matplotlib.pyplot as plt
def show_image(X):
    plt.figure(figsize=(8, 8))
    plt.imshow(X.numpy().squeeze(), cmap='gray')
    plt.show()

from tinybig.data import mnist
mnist_data = mnist(name='mnist', train_batch_size=64, test_batch_size=64)
mnist_loaders = mnist_data.load(cache_dir='./data/')
X_batch, y_batch = next(iter(mnist_loaders['test_loader']))
x = X_batch[0:1,:]
print(x.shape, x)
show_image(x.view(28, 28))

from tinybig.expansion import taylor_expansion

exp_func = taylor_expansion(name='taylor_expansion_for_mnist', d=2)
kappa_x = exp_func(x)
raw_image, expansion_image = kappa_x[0,:784], kappa_x[0,784:]
print(raw_image.shape, expansion_image.shape)

show_image(raw_image.view(28, 28))

show_image(expansion_image.view(784, 784))

def reshape_expansions(expansion):
    grid28x28 = expansion.reshape(28, 28, 28, 28)
    reshaped_expansion = grid28x28.permute(0, 2, 1, 3).reshape(784, 784)
    return reshaped_expansion

reshaped_expansion_image = reshape_expansions(expansion_image)
print(reshaped_expansion_image.shape)
show_image(reshaped_expansion_image)