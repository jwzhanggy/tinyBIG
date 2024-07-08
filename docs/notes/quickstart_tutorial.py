#%%
from tinybig.util import set_random_seed
set_random_seed(random_seed=1234)
DEVICE = 'mps' # or 'cpu', or 'cuda'
#%%
from tinybig.data import mnist
mnist_data = mnist(name='mnist', train_batch_size=64, test_batch_size=64)
mnist_loaders = mnist_data.load(cache_dir='./data/')
train_loader = mnist_loaders['train_loader']
test_loader = mnist_loaders['test_loader']
#%%
for X, y in train_loader:
    print('X shape:', X.shape, 'y.shape:', y.shape)
    print('X', X)
    print('y', y)
    break
#%% md
# 
#%%
from tinybig.expansion import taylor_expansion

exp_func = taylor_expansion(name='taylor_expansion', d=2, postprocess_functions='layer_norm', device=DEVICE)
x = X[0:1,:]
D = exp_func.calculate_D(m=x.size(1))
print('Expansion space dimension:', D)

kappa_x = exp_func(x=x)
print('x.shape', x.shape, 'kappa_x.shape', kappa_x.shape)
#%%
from tinybig.reconciliation import dual_lphm_reconciliation

rec_func = dual_lphm_reconciliation(name='dual_lphm_reconciliation', p=8, q=784, r=5, device=DEVICE)
l = rec_func.calculate_l(n=64, D=D)
print('Required learnable parameter number:', l)
#%%
from tinybig.remainder import zero_remainder

rem_func = zero_remainder(name='zero_remainder', require_parameters=False, enable_bias=False, device=DEVICE)
#%%
from tinybig.module import rpn_head

head = rpn_head(m=784, n=64, channel_num=1, data_transformation=exp_func, parameter_fabrication=rec_func, remainder=rem_func, device=DEVICE)
#%%
from tinybig.module import rpn_layer

layer_1 = rpn_layer(m=784, n=64, heads=[head], device=DEVICE)
#%%
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
#%%
from tinybig.model import rpn

model = rpn(
    layers = [
        layer_1,
        layer_2,
        layer_3
    ],
    device=DEVICE
)
#%%
import torch
from tinybig.learner import backward_learner

optimizer=torch.optim.AdamW(lr=2.0e-03, weight_decay=2.0e-04, params=model.parameters())
lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(gamma=0.95, optimizer=optimizer)
loss = torch.nn.CrossEntropyLoss()
learner = backward_learner(n_epochs=3, optimizer=optimizer, loss=loss, lr_scheduler=lr_scheduler)

#%%
from tinybig.metric import accuracy

print('parameter num: ', sum([parameter.numel() for parameter in model.parameters()]))

metric = accuracy()
training_records = learner.train(model=model, data_loader=mnist_loaders, metric=metric, device=DEVICE)
#%%
test_result = learner.test(model=model, test_loader=mnist_loaders['test_loader'], metric=metric, device=DEVICE)
print(metric.__class__.__name__, metric.evaluate(y_true=test_result['y_true'], y_pred=test_result['y_pred'], y_score=test_result['y_score'], ))