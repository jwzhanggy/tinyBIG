from tinybig.config import rpn_config
from tinybig.util import set_random_seed

print('loading configs...')
config_file_name = 'feynman_configs'
config_obj = rpn_config()
config = config_obj.load_yaml(cache_dir='./configs', config_file=config_file_name + '.yaml')
print(config)

print('setting up environments...')
device = config['configurations'].pop('device')
random_seed = config['configurations'].pop('random_seed')
set_random_seed(random_seed)
print('device: ', device, '; random_seed: ', random_seed)

print('instantiating objects from config...')
object_dict = config_obj.instantiate_object_from_config(config['configurations'])
data_obj, model_obj, learner_obj, metric_obj, result_obj = [object_dict[name] for name in ['data', 'model', 'learner', 'metric', 'result']]
print('parameter num: ', sum([parameter.numel() for parameter in model_obj.parameters()]))

print('loading dataset...')
data_loader = data_obj.load()
print(data_loader['str_equation'])

print('training model...')
training_records = learner_obj.train(model=model_obj, data_loader=data_loader, device=device,
                                     metric=metric_obj, disable_tqdm=True, display_step=100)
model_obj.save_ckpt(cache_dir='./ckpt', checkpoint_file=f'{config_file_name}_checkpoint')

print('testing model...')
test_result = learner_obj.test(model=model_obj, test_loader=data_loader['test_loader'], device=device,
                               metric=metric_obj)

print('evaluating result...')
print(metric_obj.__class__.__name__, metric_obj.evaluate(y_true=test_result['y_true'], y_pred=test_result['y_pred'], y_score=test_result['y_score'], ))

print('saving result...')
result_obj.save(test_result, cache_dir='./result', output_file='{}_result'.format(config_file_name))
result_obj.save(training_records, cache_dir='./result', output_file='{}_record'.format(config_file_name))