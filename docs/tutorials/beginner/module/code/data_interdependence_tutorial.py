#%%
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
#%%
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

#%%
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
#%%
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
#%%
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
#%%
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