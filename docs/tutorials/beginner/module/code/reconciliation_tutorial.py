#%%
import torch
from tinybig.reconciliation import identity_reconciliation

rec_func = identity_reconciliation(name='identity_reconciliation')

n, D = 6, 12
l = rec_func.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)
#%%
import torch
from tinybig.reconciliation import duplicated_padding_reconciliation

p, q = 2, 2
rec_func = duplicated_padding_reconciliation(name='duplicated_padding_reconciliation', p=p, q=q)

n, D = 6, 12
l = rec_func.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)

#%%
import torch
from tinybig.reconciliation import lorr_reconciliation

r = 1
rec_func = lorr_reconciliation(name='lorr_reconciliation', r=r)

n, D = 6, 12
l = rec_func.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)
#%%
import torch
from tinybig.reconciliation import hm_reconciliation

p, q = 2, 2
rec_func_manual = hm_reconciliation(name='hm_reconciliation', p=p, q=q)

n, D = 6, 12
l = rec_func_manual.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func_manual(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)
#%%
import torch
from tinybig.reconciliation import hm_reconciliation

rec_func_auto = hm_reconciliation(name='hm_reconciliation')

n, D = 6, 12
l = rec_func_auto.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func_auto(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)
#%%
import torch
from tinybig.reconciliation import lphm_reconciliation

r = 1
rec_func_auto = lphm_reconciliation(name='lphm_reconciliation', r=r)

n, D = 6, 12
l = rec_func_auto.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func_auto(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)
#%%
import torch
from tinybig.reconciliation import dual_lphm_reconciliation

r = 1
rec_func_auto = dual_lphm_reconciliation(name='dual_lphm_reconciliation', r=r)

n, D = 6, 12
l = rec_func_auto.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func_auto(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)
#%%
import torch
from tinybig.config.base_config import config

parameter_reconciliation_configs = {
    'parameter_reconciliation_class': 'tinybig.reconciliation.dual_lphm_reconciliation',
    'parameter_reconciliation_parameters': {
        'name': 'dual_lphm_reconciliation',
        'p': 2,
        'q': 3,
        'r': 1,
    }
}

rec_func = config.get_obj_from_str(parameter_reconciliation_configs['parameter_reconciliation_class'])(**parameter_reconciliation_configs['parameter_reconciliation_parameters'])

n, D = 6, 12
l = rec_func.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)
#%%
from tinybig.config.base_config import config

parameter_reconciliation_configs = {
    'parameter_reconciliation_class': 'tinybig.reconciliation.dual_lphm_reconciliation',
    'parameter_reconciliation_parameters': {
        'name': 'dual_lphm_reconciliation',
        'p': 2,
        'q': 3,
        'r': 1,
    }
}

rec_func = config.instantiation_from_configs(configs=parameter_reconciliation_configs, class_name='parameter_reconciliation_class', parameter_name='parameter_reconciliation_parameters')
#%% md
# 
#%%
import torch
from tinybig.config.base_config import config

config_obj = config(name='dual_lphm_reconciliation_function_config')
func_configs = config_obj.load_yaml(cache_dir='./configs', config_file='reconciliation_function_config.yaml')

rec_func = config.instantiation_from_configs(configs=func_configs['parameter_reconciliation_configs'], class_name='parameter_reconciliation_class', parameter_name='parameter_reconciliation_parameters')

n, D = 6, 12
l = rec_func.calculate_l(n=n, D=D)
print('n:', n, '; D:', D, '; l:', l)

w = torch.nn.Parameter(torch.randn(1, l), requires_grad=True)
W = rec_func(w=w, n=n, D=D)

print("w vector shape:", w.shape, "; W matrix shape:", W.shape)