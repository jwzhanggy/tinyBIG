# CIFAR10 Object Recognition

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 8, 2024; latest Revision: July 8, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/cifar10_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/cifar10_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/cifar10_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

-------------------------

In this example, we will build a 3-layer {{our}} model with `identity_expansion`, `identity_reconciliation` and `zero_remainder`
functions for the object detection task from the CIFAR10 dataset.

We will use `mps` as the device in the config file.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'cifar10_configs'
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
      device: &device mps
      random_seed: 1234
    
      data_configs:
        data_class: tinybig.data.cifar10
        data_parameters:
          name: mnist
          train_batch_size: 100
          test_batch_size: 64
    
      learner_configs:
        learner_class: tinybig.learner.backward_learner
        learner_parameters:
          name: error_backward_propagation
          n_epochs: 11
          optimizer_configs:
            optimizer_class: torch.optim.AdamW
            optimizer_parameters:
              lr: 1.0e-03
              weight_decay: 1.0e-05
          lr_scheduler_configs:
            lr_scheduler_class: torch.optim.lr_scheduler.ExponentialLR
            lr_scheduler_parameters:
              gamma: 0.65
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
                m: 3072
                n: 512
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
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
                          require_parameters: False
                          enable_bias: False
                      output_process_function_configs:
                        - function_class: torch.nn.GELU
                        - function_class: torch.nn.BatchNorm1d
                          function_parameters:
                            num_features: 512
                            device: *device
    
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 512
                n: 256
                width:
                fusion_strategy: average
                width_alloc: [ 1 ]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l:
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.identity_expansion
                        data_transformation_parameters:
                          name: identity_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
                      output_process_function_configs:
                        - function_class: torch.nn.GELU
                        - function_class: torch.nn.BatchNorm1d
                          function_parameters:
                            num_features: 256
                            device: *device
    
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 256
                n: 10
                width:
                fusion_strategy: average
                width_alloc: [ 1 ]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l:
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.identity_expansion
                        data_transformation_parameters:
                          name: identity_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
    
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
    
    100%|██████████| 500/500 [00:07<00:00, 65.77it/s, epoch=0/11, loss=1.59, lr=0.001, metric_score=0.42, time=7.62]
    
    Epoch: 0, Test Loss: 1.4929834885202395, Test Score: 0.4674, Time Cost: 1.3373339176177979
    
    100%|██████████| 500/500 [00:07<00:00, 66.27it/s, epoch=1/11, loss=0.999, lr=0.00065, metric_score=0.64, time=16.5]
    
    Epoch: 1, Test Loss: 1.384826884907522, Test Score: 0.5077, Time Cost: 1.2460088729858398
    
    100%|██████████| 500/500 [00:07<00:00, 65.83it/s, epoch=2/11, loss=1.34, lr=0.000423, metric_score=0.49, time=25.3] 
    
    Epoch: 2, Test Loss: 1.338048208671011, Test Score: 0.5282, Time Cost: 1.1734652519226074
    
    100%|██████████| 500/500 [00:07<00:00, 66.01it/s, epoch=3/11, loss=1.07, lr=0.000275, metric_score=0.65, time=34.1] 
    
    Epoch: 3, Test Loss: 1.2938201624876375, Test Score: 0.544, Time Cost: 1.167226791381836
    
    100%|██████████| 500/500 [00:07<00:00, 65.93it/s, epoch=4/11, loss=1.04, lr=0.000179, metric_score=0.65, time=42.8] 
    
    Epoch: 4, Test Loss: 1.2855375940632667, Test Score: 0.5568, Time Cost: 1.1595079898834229
    
    100%|██████████| 500/500 [00:07<00:00, 66.80it/s, epoch=5/11, loss=0.789, lr=0.000116, metric_score=0.73, time=51.5]
    
    Epoch: 5, Test Loss: 1.283127313586557, Test Score: 0.5584, Time Cost: 1.1664459705352783
    
    100%|██████████| 500/500 [00:07<00:00, 66.52it/s, epoch=6/11, loss=0.937, lr=7.54e-5, metric_score=0.67, time=60.2]
    
    Epoch: 6, Test Loss: 1.289850783575872, Test Score: 0.5549, Time Cost: 1.168416976928711
    
    100%|██████████| 500/500 [00:07<00:00, 67.04it/s, epoch=7/11, loss=0.837, lr=4.9e-5, metric_score=0.73, time=68.8]
    
    Epoch: 7, Test Loss: 1.299753411939949, Test Score: 0.5619, Time Cost: 1.2394850254058838
    
    100%|██████████| 500/500 [00:07<00:00, 66.79it/s, epoch=8/11, loss=0.801, lr=3.19e-5, metric_score=0.69, time=77.5]
    
    Epoch: 8, Test Loss: 1.3010211503429778, Test Score: 0.5641, Time Cost: 1.17451810836792
    
    100%|██████████| 500/500 [00:07<00:00, 67.09it/s, epoch=9/11, loss=0.788, lr=2.07e-5, metric_score=0.69, time=86.2]
    
    Epoch: 9, Test Loss: 1.3110735256960437, Test Score: 0.5632, Time Cost: 1.1685810089111328
    
    100%|██████████| 500/500 [00:07<00:00, 66.82it/s, epoch=10/11, loss=0.705, lr=1.35e-5, metric_score=0.79, time=94.8]
    
    Epoch: 10, Test Loss: 1.310013646153128, Test Score: 0.5672, Time Cost: 1.1657207012176514
    model checkpoint saving to ./ckpt/cifar10_configs_checkpoint...
    
    evaluating result...
    accuracy 0.5672
    ```