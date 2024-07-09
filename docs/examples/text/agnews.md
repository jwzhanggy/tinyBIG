# AGNews Article Classification

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 9, 2024; latest Revision: July 9, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/agnews_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/agnews_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/agnews_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

In this example, we will build 2-layer PRN model with `identity_expansion`, `lorr_reconciliation` and `zero_remainder` 
component functions for the sentiment article classification based on the AGNews dataset.

The script code and model configuration files are provided as follows. 
We use `mps` as the device in the config file for this example.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'agnews_configs'
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
      random_seed: 5678
    
      data_configs:
        data_class: tinybig.data.agnews
        data_parameters:
          name: agnews
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
              lr: 5.0e-05
              weight_decay: 5.0e-05
          lr_scheduler_configs:
            lr_scheduler_class: torch.optim.lr_scheduler.ExponentialLR
            lr_scheduler_parameters:
              gamma: 0.95
          loss_configs:
            loss_class: torch.nn.CrossEntropyLoss
            loss_parameters:
              reduction: mean
    
      model_configs:
        model_class: tinybig.model.rpn
        model_parameters:
          name: reconciled_polynomial_network
          depth: 2
          depth_alloc: [1, 1]
          layer_configs:
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 25985
                n: 128
                width: 1
                fusion_strategy: average
                width_alloc: [1]
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
    
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 128
                n: 4
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
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                          enable_bias: True
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
    
    100%|██████████| 1875/1875 [00:19<00:00, 97.33it/s, epoch=0/12, loss=0.888, lr=5e-5, metric_score=0.734, time=19.3]
    
    Epoch: 0, Test Loss: 0.8839748997648224, Test Score: 0.7310526315789474, Time Cost: 0.6759779453277588
    
    100%|██████████| 1875/1875 [00:17<00:00, 105.91it/s, epoch=1/12, loss=0.58, lr=4.75e-5, metric_score=0.781, time=37.7] 
    
    Epoch: 1, Test Loss: 0.5829719530434168, Test Score: 0.7960526315789473, Time Cost: 0.6713669300079346
    
    100%|██████████| 1875/1875 [00:17<00:00, 105.29it/s, epoch=2/12, loss=0.419, lr=4.51e-5, metric_score=0.922, time=56.2]
    
    Epoch: 2, Test Loss: 0.44666625321412284, Test Score: 0.8853947368421052, Time Cost: 0.6344597339630127
    
    100%|██████████| 1875/1875 [00:18<00:00, 102.92it/s, epoch=3/12, loss=0.392, lr=4.29e-5, metric_score=0.906, time=75]  
    
    Epoch: 3, Test Loss: 0.37390605792278003, Test Score: 0.8994736842105263, Time Cost: 0.5626459121704102
    
    100%|██████████| 1875/1875 [00:19<00:00, 96.45it/s, epoch=4/12, loss=0.301, lr=4.07e-5, metric_score=0.906, time=95]  
    
    Epoch: 4, Test Loss: 0.335242745943931, Test Score: 0.9072368421052631, Time Cost: 0.6107730865478516
    
    100%|██████████| 1875/1875 [00:18<00:00, 100.69it/s, epoch=5/12, loss=0.453, lr=3.87e-5, metric_score=0.844, time=114] 
    
    Epoch: 5, Test Loss: 0.31101745504791994, Test Score: 0.9111842105263158, Time Cost: 0.5800421237945557
    
    100%|██████████| 1875/1875 [00:20<00:00, 92.28it/s, epoch=6/12, loss=0.255, lr=3.68e-5, metric_score=0.969, time=135]  
    
    Epoch: 6, Test Loss: 0.2947773579038492, Test Score: 0.9134210526315789, Time Cost: 0.58302903175354
    
    100%|██████████| 1875/1875 [00:19<00:00, 94.67it/s, epoch=7/12, loss=0.209, lr=3.49e-5, metric_score=0.906, time=156] 
    
    Epoch: 7, Test Loss: 0.2835594335523974, Test Score: 0.9152631578947369, Time Cost: 0.7760090827941895
    
    100%|██████████| 1875/1875 [00:20<00:00, 93.29it/s, epoch=8/12, loss=0.344, lr=3.32e-5, metric_score=0.906, time=176] 
    
    Epoch: 8, Test Loss: 0.27566457164137304, Test Score: 0.916578947368421, Time Cost: 0.6371800899505615
    
    100%|██████████| 1875/1875 [00:18<00:00, 99.63it/s, epoch=9/12, loss=0.191, lr=3.15e-5, metric_score=0.938, time=196]  
    
    Epoch: 9, Test Loss: 0.26999062634691473, Test Score: 0.9175, Time Cost: 0.6577751636505127
    
    100%|██████████| 1875/1875 [00:18<00:00, 99.30it/s, epoch=10/12, loss=0.263, lr=2.99e-5, metric_score=0.922, time=215]  
    
    Epoch: 10, Test Loss: 0.2657905873380789, Test Score: 0.9177631578947368, Time Cost: 0.6916248798370361
    
    100%|██████████| 1875/1875 [00:20<00:00, 91.06it/s, epoch=11/12, loss=0.3, lr=2.84e-5, metric_score=0.922, time=237]   
    
    Epoch: 11, Test Loss: 0.2626152299970639, Test Score: 0.9188157894736843, Time Cost: 0.6519889831542969
    model checkpoint saving to ./ckpt/agnews_configs_checkpoint...
    
    evaluating result...
    accuracy 0.9188157894736843
    
    ```
