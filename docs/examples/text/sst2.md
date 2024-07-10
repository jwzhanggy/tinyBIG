# SST2 Sentiment Prediction

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 9, 2024; latest Revision: July 9, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/sst2_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/sst2_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/sst2_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

In this example, we will build a 3-layer RPN model with `identity_expansion`, `lorr_reconciliation` and `zero_remainder`
for predicting the sentiment of articles in the SST2 dataset.

The script code and model configuration files are provided as follows. 
We use `mps` as the device in the config file for this example.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'sst2_configs'
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
    ```

=== "model configs"
    ```yaml linenums="1"
    configurations:
      device: mps
      random_seed: 1234
    
      data_configs:
        data_class: tinybig.data.sst2
        data_parameters:
          name: sst2
          train_batch_size: 64
          test_batch_size: 64
    
      learner_configs:
        learner_class: tinybig.learner.backward_learner
        learner_parameters:
          name: error_backward_propagation
          n_epochs: 12
          optimizer_configs:
            optimizer_class: torch.optim.AdamW
            optimizer_parameters:
              lr: 1.0e-04
              weight_decay: 1.0e-05
          lr_scheduler_configs:
            lr_scheduler_class: torch.optim.lr_scheduler.ExponentialLR
            lr_scheduler_parameters:
              gamma: 0.35
          loss_configs:
            loss_class: torch.nn.CrossEntropyLoss
            loss_parameters:
              reduction: mean
    
      model_configs:
        model_class: tinybig.model.rpn
        model_parameters:
          name: reconciled_polynomial_network
          depth: 3
          depth_alloc: [1, 1, 1]
          layer_configs:
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 10325
                n: 128
                width: 1
                fusion_strategy: average
                width_alloc: [1]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l: null
                      channel_num: 1
                      output_process_function_configs:
                        function_class: torch.nn.Dropout
                        function_parameters:
                          p: 0.4
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.identity_expansion
                        data_transformation_parameters:
                          name: identity_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.lorr_reconciliation
                        parameter_fabrication_parameters:
                          name: lorr_reconciliation
                          r: 5
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
                          require_parameters: False
                          enable_bias: False
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 128
                n: 32
                width: 1
                fusion_strategy: average
                width_alloc: [ 1 ]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l: null
                      channel_num: 1
                      output_process_function_configs:
                        function_class: torch.nn.Dropout
                        function_parameters:
                          p: 0.4
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.identity_expansion
                        data_transformation_parameters:
                          name: identity_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.lorr_reconciliation
                        parameter_fabrication_parameters:
                          name: lorr_reconciliation
                          r: 5
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
                          require_parameters: False
                          enable_bias: False
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 32
                n: 2
                width: 1
                fusion_strategy: average
                width_alloc: [ 1 ]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l: null
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.identity_expansion
                        data_transformation_parameters:
                          name: identity_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.lorr_reconciliation
                        parameter_fabrication_parameters:
                          name: lorr_reconciliation
                          r: 5
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
                          require_parameters: False
                          enable_bias: False
    
      metric_configs:
        metric_class: tinybig.metric.accuracy
        metric_parameters:
          name: accuracy
    
      result_configs:
        result_class: tinybig.output.rpn_output
        result_parameters:
          name: prediction_output
    ```

???+ quote "rpn with identity reconciliation for mnist classification output"
    ```shell
    training model...
    
    100%|██████████| 1053/1053 [00:10<00:00, 98.75it/s, epoch=0/12, loss=0.277, lr=0.0001, metric_score=0.905, time=10.7] 
    
    Epoch: 0, Test Loss: 0.49072653480938505, Test Score: 0.7947247706422018, Time Cost: 0.1188809871673584
    
    100%|██████████| 1053/1053 [00:10<00:00, 102.24it/s, epoch=1/12, loss=0.243, lr=3.5e-5, metric_score=0.905, time=21.1]
    
    Epoch: 1, Test Loss: 0.5090811933789935, Test Score: 0.7970183486238532, Time Cost: 0.07034993171691895
    
    100%|██████████| 1053/1053 [00:10<00:00, 99.44it/s, epoch=2/12, loss=0.161, lr=1.22e-5, metric_score=0.952, time=31.8] 
    
    Epoch: 2, Test Loss: 0.5310905448028019, Test Score: 0.8038990825688074, Time Cost: 0.071044921875
    
    100%|██████████| 1053/1053 [00:10<00:00, 102.29it/s, epoch=3/12, loss=0.375, lr=4.29e-6, metric_score=0.857, time=42.1]
    
    Epoch: 3, Test Loss: 0.5274856047970908, Test Score: 0.805045871559633, Time Cost: 0.07017683982849121
    
    100%|██████████| 1053/1053 [00:10<00:00, 102.43it/s, epoch=4/12, loss=0.206, lr=1.5e-6, metric_score=0.905, time=52.5]
    
    Epoch: 4, Test Loss: 0.5224815202610833, Test Score: 0.8061926605504587, Time Cost: 0.06879305839538574
    
    100%|██████████| 1053/1053 [00:10<00:00, 100.41it/s, epoch=5/12, loss=0.27, lr=5.25e-7, metric_score=0.857, time=63]   
    
    Epoch: 5, Test Loss: 0.5265864687306541, Test Score: 0.8061926605504587, Time Cost: 0.07750082015991211
    
    100%|██████████| 1053/1053 [00:10<00:00, 99.91it/s, epoch=6/12, loss=0.168, lr=1.84e-7, metric_score=0.952, time=73.7] 
    
    Epoch: 6, Test Loss: 0.5227903532130378, Test Score: 0.8061926605504587, Time Cost: 0.07305908203125
    
    100%|██████████| 1053/1053 [00:10<00:00, 100.13it/s, epoch=7/12, loss=0.474, lr=6.43e-8, metric_score=0.857, time=84.3]
    
    Epoch: 7, Test Loss: 0.5279938323157174, Test Score: 0.8061926605504587, Time Cost: 0.0726630687713623
    
    100%|██████████| 1053/1053 [00:10<00:00, 96.91it/s, epoch=8/12, loss=0.0884, lr=2.25e-8, metric_score=1, time=95.2]   
    
    Epoch: 8, Test Loss: 0.5263547939913613, Test Score: 0.8061926605504587, Time Cost: 0.07672500610351562
    
    100%|██████████| 1053/1053 [00:10<00:00, 98.68it/s, epoch=9/12, loss=0.179, lr=7.88e-9, metric_score=0.905, time=106] 
    
    Epoch: 9, Test Loss: 0.5326536872557232, Test Score: 0.8061926605504587, Time Cost: 0.07764220237731934
    
    100%|██████████| 1053/1053 [00:10<00:00, 98.53it/s, epoch=10/12, loss=0.221, lr=2.76e-9, metric_score=0.952, time=117]
    
    Epoch: 10, Test Loss: 0.5276471003890038, Test Score: 0.8061926605504587, Time Cost: 0.0757441520690918
    
    100%|██████████| 1053/1053 [00:10<00:00, 98.42it/s, epoch=11/12, loss=0.227, lr=9.65e-10, metric_score=0.857, time=127]
    
    Epoch: 11, Test Loss: 0.528856194445065, Test Score: 0.8061926605504587, Time Cost: 0.07407307624816895
    model checkpoint saving to ./ckpt/sst2_configs_checkpoint...
    
    evaluating result...
    accuracy 0.8061926605504587
    ```