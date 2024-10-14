from tinybig.config import config
from tinybig.util import set_random_seed

print('loading configs...')
config_file_name = 'mnist_dual_lphm_configs'
config_obj = config()
config = config_obj.load_yaml(cache_dir='./configs', config_file=config_file_name + '.yaml')
print(config)

print('setting up environments...')
device = config['configurations'].pop('device')
random_seed = config['configurations'].pop('random_seed')
set_random_seed(random_seed)
print('device: ', device, '; random_seed: ', random_seed)

print('instantiating objects from config...')
data_obj, model_obj, learner_obj, metric_obj, result_obj = [config_obj.instantiation_from_configs(config['configurations'][f'{stem}_configs'], device=device, class_name=f'{stem}_class', parameter_name=f'{stem}_parameters') for stem in ['data', 'model', 'learner', 'metric', 'output']]
print('parameter num: ', sum([parameter.numel() for parameter in model_obj.parameters()]))

print('loading dataset...')
data_loader = data_obj.load()

print('training model...')
training_records = learner_obj.train(model=model_obj, data_loader=data_loader, device=device, metric=metric_obj)
model_obj.save_ckpt(cache_dir='./ckpt', checkpoint_file=f'{config_file_name}_checkpoint')

print('testing model...')
test_result = learner_obj.test(model=model_obj, test_loader=data_loader['test_loader'], device=device,
                               metric=metric_obj)

print('evaluating result...')
print(metric_obj.__class__.__name__, metric_obj.evaluate(y_true=test_result['y_true'], y_pred=test_result['y_pred'], y_score=test_result['y_score'], ))

print('saving result...')
result_obj.save(test_result, cache_dir='./result', output_file='{}_result'.format(config_file_name))
result_obj.save(training_records, cache_dir='./result', output_file='{}_record'.format(config_file_name))