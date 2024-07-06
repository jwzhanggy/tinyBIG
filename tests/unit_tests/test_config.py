#import pytest
import torch
from tinybig.config import rpn_config
from tinybig.util import set_random_seed
from tinybig.learner import backward_learner

config_obj = rpn_config(name='test_rpn_config.yaml')
config = config_obj.load_yaml(cache_dir='./configs', config_file='test_rpn_config.yaml')

# ---- environment setups ----
configs_to_run = 'rpn_main_taylor_hm_zero'
device = config['configurations'].pop('device')
random_seed = config['configurations'].pop('random_seed')
set_random_seed(random_seed)
print('device: ', device, '; random_seed: ', random_seed)
# ---- environment setups ----

# ---- objection initialization setction -----
print('instantiating objects from config...')
object_dict = config_obj.instantiate_object_from_config(config['configurations'])
print(object_dict)
data_obj, model_obj, learner_obj, metric_obj, result_obj = [object_dict[name] for name in
                                                            ['data', 'model', 'learner', 'metric',
                                                             'result']]

print('parameter num: ', sum([parameter.numel() for parameter in model_obj.parameters()]))
# ---- objection initialization setction -----

# ---- running section ----
print('loading dataset...')
data_loader = data_obj.load()

print('**********************************')
print('model training...')
training_records = learner_obj.train(model=model_obj, data_loader=data_loader, device=device,
                                     metric=metric_obj, disable_tqdm=False, display_step=1)
model_obj.save_ckpt(cache_dir='./ckpt', checkpoint_file=f'{configs_to_run}_checkpoint')
print('**********************************')

print('model testing...')
test_result = learner_obj.test(model=model_obj, test_loader=data_loader['test_loader'], device=device,
                               metric=metric_obj)

if metric_obj is not None:
    print('evaluating result...')
    print(metric_obj.__class__.__name__,
          metric_obj.evaluate(y_true=test_result['y_true'], y_pred=test_result['y_pred'],
                              y_score=test_result['y_score'], ))
if result_obj is not None:
    print('saving result...')
    result_obj.save(test_result,
                    result_file='./RPN_{}_result'.format(configs_to_run))
    result_obj.save(training_records,
                    result_file='./RPN_{}_record'.format(configs_to_run))
    # ---- running section ----

print("************** Finish: {} **************".format(configs_to_run))
# ------------------------------------------------------
