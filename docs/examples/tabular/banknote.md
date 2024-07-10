# Combinatorial Probabilistic Banknote Authentication

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 9, 2024; latest Revision: July 9, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/banknote_example.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/banknote_configs.yaml"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/banknote_example.py"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a>

</span>
</div>

-----------------------------

In this example, we will build a 1-layer RPN model with `combinatorial_normal_expansion`, `identity_reconciliation` and `linear_remainder`
for diagnosing the banknote disease based on the Banknote Authentication dataset.

We use `mps` as the device for the model config file provided below.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    from tinybig.metric import accuracy
    
    print('loading configs...')
    config_file_name = 'banknote_configs'
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
    data_loader = data_obj.load(train_percentage=0.9, normalize_X=True)
    
    print('training model...')
    training_records = learner_obj.train(model=model_obj, data_loader=data_loader, device=device, metric=metric_obj, disable_tqdm=True, display_step=100)
    model_obj.save_ckpt(cache_dir='./ckpt', checkpoint_file=f'{config_file_name}_checkpoint')
    
    print('testing model...')
    test_result = learner_obj.test(model=model_obj, test_loader=data_loader['test_loader'], device=device,
                                   metric=metric_obj)
    
    print('evaluating result...')
    print(metric_obj.__class__.__name__, metric_obj.evaluate(y_true=test_result['y_true'], y_pred=test_result['y_pred'], y_score=test_result['y_score'], ))
    
    y_rounded_label = [[round(y[0])] for y in test_result['y_score']]
    test_result['y_pred'] = y_rounded_label
    
    acc_metric = accuracy('accuracy_metric')
    print('evaluating rounded prediction labels...')
    print(acc_metric.__class__.__name__, acc_metric.evaluate(y_true=test_result['y_true'], y_pred=test_result['y_pred'], y_score=test_result['y_score'], ))
    
    print('saving result...')
    result_obj.save(test_result, cache_dir='./result', output_file='{}_result'.format(config_file_name))
    result_obj.save(training_records, cache_dir='./result', output_file='{}_record'.format(config_file_name))
    ```

=== "model configs"
    ```yaml linenums="1"
    configurations:
      device: &device mps
      random_seed: 1234
    
      data_configs:
        data_class: tinybig.data.banknote
        data_parameters:
          name: banknote
          train_batch_size: 2000
          test_batch_size: 1000
    
      learner_configs:
        learner_class: tinybig.learner.backward_learner
        learner_parameters:
          name: error_backward_propagation
          n_epochs: 1500
          optimizer_configs:
            optimizer_class: torch.optim.AdamW
            optimizer_parameters:
              lr: 1.0e-02
              weight_decay: 1.0e-05
          loss_configs:
            loss_class: torch.nn.MSELoss
            loss_parameters:
              reduction: mean
    
      model_configs:
        model_class: tinybig.model.rpn
        device: *device
        model_parameters:
          name: reconciled_polynomial_network
          depth: 1
          depth_alloc: [1]
          layer_configs:
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 4
                n: 1
                width: 1
                fusion_strategy: average
                width_alloc: [1]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l: null
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.combinatorial_normal_expansion
                        data_transformation_parameters:
                          name: combinatorial_normal_expansion
                          d: 2
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                      remainder_configs:
                        remainder_class: tinybig.remainder.linear_remainder
                        remainder_parameters:
                          name: linear_remainder
                          require_parameters: True
      metric_configs:
        metric_class: tinybig.metric.mse
        metric_parameters:
          name: mse
    
      result_configs:
        result_class: tinybig.output.rpn_output
        result_parameters:
          name: prediction_output
    ```

???+ quote "rpn with identity reconciliation for mnist classification output"
    ```shell
    training model...
    Epoch: 0, Test Loss: 1.9411336183547974, Test Score: 1.9411336696155832, Time Cost: 0.007844209671020508
    Epoch: 100, Test Loss: 0.36659175157546997, Test Score: 0.366591744963078, Time Cost: 0.009141921997070312
    Epoch: 200, Test Loss: 0.23675186932086945, Test Score: 0.23675185419647374, Time Cost: 0.008933782577514648
    Epoch: 300, Test Loss: 0.17674371600151062, Test Score: 0.1767437075688446, Time Cost: 0.007529020309448242
    Epoch: 400, Test Loss: 0.13803423941135406, Test Score: 0.1380342434864005, Time Cost: 0.0074062347412109375
    Epoch: 500, Test Loss: 0.11002819240093231, Test Score: 0.11002819068762702, Time Cost: 0.007805824279785156
    Epoch: 600, Test Loss: 0.08940108120441437, Test Score: 0.08940108904258437, Time Cost: 0.0074310302734375
    Epoch: 700, Test Loss: 0.07411758601665497, Test Score: 0.07411758835844745, Time Cost: 0.0074651241302490234
    Epoch: 800, Test Loss: 0.06268753856420517, Test Score: 0.06268753147146049, Time Cost: 0.007480144500732422
    Epoch: 900, Test Loss: 0.054036639630794525, Test Score: 0.054036637264635624, Time Cost: 0.007607936859130859
    Epoch: 1000, Test Loss: 0.04742664471268654, Test Score: 0.04742664752789012, Time Cost: 0.008695125579833984
    Epoch: 1100, Test Loss: 0.04236285015940666, Test Score: 0.042362847759040485, Time Cost: 0.007869958877563477
    Epoch: 1200, Test Loss: 0.03850637003779411, Test Score: 0.03850636577531528, Time Cost: 0.00732111930847168
    Epoch: 1300, Test Loss: 0.03560876473784447, Test Score: 0.03560876589215595, Time Cost: 0.007487058639526367
    Epoch: 1400, Test Loss: 0.03347256779670715, Test Score: 0.03347256692348597, Time Cost: 0.00847315788269043
    model checkpoint saving to ./ckpt/banknote_configs_checkpoint...
    
    evaluating result...
    mse 0.03194525463030612
    
    evaluating rounded prediction labels...
    accuracy 0.9710144927536232
    ```

