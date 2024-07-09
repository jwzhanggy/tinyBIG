# MNIST with Dual LPHM Reconciliation

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 8, 2024; latest Revision: July 8, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/mnist_dual_lphm_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/mnist_dual_lphm_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/mnist_dual_lphm_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

-------------------------

As introduced in the Quickstart tutorial, Dual LPHM reconciliation function can reduce the number of learnable parameters
in the RPN model a lot.

In this example, we will re-design the 3-layer {{our}} model built in the previous example by replacing the `identity_reconciliation`
with the `dual_lphm_reconciliation` function for the MNIST image data classification.

It will dramatically reduce the learnable parameter numbers from `39696000` to `9330`, which will greatly save the memory
space required for storing the parameters and their gradients in learning.

According to the experimental evaluation, {{our}} with `dual_lphm_reconciliation` achieves a descent testing accuracy about `0.9810`.
The testing can actually be further improved by tuning the rank parameter `r` of the `dual_lphm_reconciliation` function.

We will still use `mps` as the device in the config file.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'mnist_dual_lphm_configs'
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
        data_class: tinybig.data.mnist
        data_parameters:
          name: mnist
          train_batch_size: 64
          test_batch_size: 64
    
      learner_configs:
        learner_class: tinybig.learner.backward_learner
        learner_parameters:
          name: error_backward_propagation
          n_epochs: 20
          optimizer_configs:
            optimizer_class: torch.optim.AdamW
            optimizer_parameters:
              lr: 2.0e-03
              weight_decay: 2.0e-04
          lr_scheduler_configs:
            lr_scheduler_class: torch.optim.lr_scheduler.ExponentialLR
            lr_scheduler_parameters:
              gamma: 0.9
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
                m: 784
                n: 64
                width: 1
                fusion_strategy: average
                width_alloc: [1]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l: null
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.taylor_expansion
                        data_transformation_parameters:
                          name: taylor_expansion
                          d: 2
                        postprocess_function_configs:
                          - function_class: torch.nn.LayerNorm
                            function_parameters:
                              normalized_shape: 615440
                              device: *device
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.dual_lphm_reconciliation
                        parameter_fabrication_parameters:
                          name: dual_lphm_reconciliation
                          p: 8
                          q: 784
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
                m: 64
                n: 64
                width:
                fusion_strategy: average
                width_alloc: [ 1 ]
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l:
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.taylor_expansion
                        data_transformation_parameters:
                          name: taylor_expansion
                          d: 2
                          postprocess_function_configs:
                            - function_class: torch.nn.LayerNorm
                              function_parameters:
                                normalized_shape: 4160
                                device: *device
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.dual_lphm_reconciliation
                        parameter_fabrication_parameters:
                          name: dual_lphm_reconciliation
                          p: 8
                          q: 64
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
                m: 64
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
                        data_transformation_class: tinybig.expansion.taylor_expansion
                        data_transformation_parameters:
                          name: taylor_expansion
                          d: 2
                          postprocess_function_configs:
                            - function_class: torch.nn.LayerNorm
                              function_parameters:
                                normalized_shape: 4160
                                device: *device
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.dual_lphm_reconciliation
                        parameter_fabrication_parameters:
                          name: dual_lphm_reconciliation
                          p: 2
                          q: 64
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

    100%|██████████| 938/938 [00:32<00:00, 29.12it/s, epoch=0/20, loss=0.154, lr=0.002, metric_score=0.906, time=32.2] 
    
    Epoch: 0, Test Loss: 0.1287281839414292, Test Score: 0.9614, Time Cost: 3.4373199939727783
    
    100%|██████████| 938/938 [00:32<00:00, 29.25it/s, epoch=1/20, loss=0.103, lr=0.0018, metric_score=0.969, time=67.7] 
    
    Epoch: 1, Test Loss: 0.10673198028865039, Test Score: 0.969, Time Cost: 3.2739510536193848
    
    100%|██████████| 938/938 [00:31<00:00, 30.01it/s, epoch=2/20, loss=0.0153, lr=0.00162, metric_score=1, time=102]     
    
    Epoch: 2, Test Loss: 0.09450612840640127, Test Score: 0.9715, Time Cost: 3.2066781520843506
    
    100%|██████████| 938/938 [00:31<00:00, 30.25it/s, epoch=3/20, loss=0.149, lr=0.00146, metric_score=0.938, time=136] 
    
    Epoch: 3, Test Loss: 0.09049446262555659, Test Score: 0.9724, Time Cost: 3.3913190364837646
    
    100%|██████████| 938/938 [00:31<00:00, 29.56it/s, epoch=4/20, loss=0.0124, lr=0.00131, metric_score=1, time=172]    
    
    Epoch: 4, Test Loss: 0.08241389291420294, Test Score: 0.9754, Time Cost: 3.3962209224700928
    
    100%|██████████| 938/938 [00:32<00:00, 29.08it/s, epoch=5/20, loss=0.357, lr=0.00118, metric_score=0.875, time=207] 
    
    Epoch: 5, Test Loss: 0.07358021313060598, Test Score: 0.9781, Time Cost: 3.2792601585388184
    
    100%|██████████| 938/938 [00:32<00:00, 28.77it/s, epoch=6/20, loss=0.0193, lr=0.00106, metric_score=1, time=243]    
    
    Epoch: 6, Test Loss: 0.07322978683759215, Test Score: 0.9792, Time Cost: 3.478147029876709
    
    100%|██████████| 938/938 [00:32<00:00, 29.27it/s, epoch=7/20, loss=0.012, lr=0.000957, metric_score=1, time=279]     
    
    Epoch: 7, Test Loss: 0.08070176002194299, Test Score: 0.9786, Time Cost: 3.3920912742614746
    
    100%|██████████| 938/938 [00:31<00:00, 29.89it/s, epoch=8/20, loss=0.0161, lr=0.000861, metric_score=1, time=313]    
    
    Epoch: 8, Test Loss: 0.07754188040375211, Test Score: 0.9802, Time Cost: 3.2393269538879395
    
    100%|██████████| 938/938 [00:30<00:00, 30.34it/s, epoch=9/20, loss=0.0198, lr=0.000775, metric_score=1, time=348]    
    
    Epoch: 9, Test Loss: 0.07705280321200117, Test Score: 0.9804, Time Cost: 3.2612552642822266
    
    100%|██████████| 938/938 [00:31<00:00, 30.11it/s, epoch=10/20, loss=0.00647, lr=0.000697, metric_score=1, time=382]   
    
    Epoch: 10, Test Loss: 0.08428076295295929, Test Score: 0.9782, Time Cost: 3.4526174068450928
    
    100%|██████████| 938/938 [00:31<00:00, 29.53it/s, epoch=11/20, loss=0.283, lr=0.000628, metric_score=0.969, time=417] 
    
    Epoch: 11, Test Loss: 0.07899063697800701, Test Score: 0.9814, Time Cost: 3.278550148010254
    
    100%|██████████| 938/938 [00:31<00:00, 30.20it/s, epoch=12/20, loss=0.00808, lr=0.000565, metric_score=1, time=452]   
    
    Epoch: 12, Test Loss: 0.08305486166922214, Test Score: 0.9798, Time Cost: 3.269629955291748
    
    100%|██████████| 938/938 [00:31<00:00, 29.41it/s, epoch=13/20, loss=0.00124, lr=0.000508, metric_score=1, time=487]   
    
    Epoch: 13, Test Loss: 0.0823607053136387, Test Score: 0.9806, Time Cost: 3.2996010780334473
    
    100%|██████████| 938/938 [00:31<00:00, 29.90it/s, epoch=14/20, loss=0.0641, lr=0.000458, metric_score=0.969, time=521]
    
    Epoch: 14, Test Loss: 0.08280810932788232, Test Score: 0.981, Time Cost: 3.3404059410095215
    
    100%|██████████| 938/938 [00:32<00:00, 28.97it/s, epoch=15/20, loss=0.00654, lr=0.000412, metric_score=1, time=557]   
    
    Epoch: 15, Test Loss: 0.0896499605672028, Test Score: 0.9809, Time Cost: 3.358131170272827
    
    100%|██████████| 938/938 [00:31<00:00, 29.48it/s, epoch=16/20, loss=0.000128, lr=0.000371, metric_score=1, time=592]  
    
    Epoch: 16, Test Loss: 0.08846969538192688, Test Score: 0.9812, Time Cost: 3.2872140407562256
    
    100%|██████████| 938/938 [00:30<00:00, 30.36it/s, epoch=17/20, loss=0.002, lr=0.000334, metric_score=1, time=627]     
    
    Epoch: 17, Test Loss: 0.09712753198534886, Test Score: 0.9805, Time Cost: 3.468175172805786
    
    100%|██████████| 938/938 [00:31<00:00, 30.07it/s, epoch=18/20, loss=0.00103, lr=0.0003, metric_score=1, time=661]   
    
    Epoch: 18, Test Loss: 0.10845135996438492, Test Score: 0.9795, Time Cost: 3.36928391456604
    
    100%|██████████| 938/938 [00:30<00:00, 30.94it/s, epoch=19/20, loss=0.00051, lr=0.00027, metric_score=1, time=695]   
    
    Epoch: 19, Test Loss: 0.1052672725711357, Test Score: 0.9807, Time Cost: 3.479506254196167
    model checkpoint saving to ./ckpt/mnist_dual_lphm_configs_checkpoint...
    
    evaluating result...
    accuracy 0.9807
    ```
