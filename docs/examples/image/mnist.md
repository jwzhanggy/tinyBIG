# MNIST Image Classification

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 8, 2024; latest Revision: July 8, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/mnist_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/mnist_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/mnist_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

-------------------------

In this example, we will build a 3-layer {{our}} model with `taylor_expansion`, `identity_reconciliation` and `zero_remainder`
functions to classify the MNIST dataset of hand-written digit images.

We will use `mps` as the device in the config file, and you can change it according to your machine before running the script code.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'mnist_configs'
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
          n_epochs: 25
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
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
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
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
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
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
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

    100%|██████████| 938/938 [00:36<00:00, 25.44it/s, epoch=0/25, loss=0.0452, lr=0.002, metric_score=1, time=36.9]    
    
    Epoch: 0, Test Loss: 0.14725748572838165, Test Score: 0.9555, Time Cost: 3.299213171005249
    
    100%|██████████| 938/938 [00:35<00:00, 26.08it/s, epoch=1/25, loss=0.0666, lr=0.0018, metric_score=0.969, time=76.2]
    
    Epoch: 1, Test Loss: 0.08832730701012288, Test Score: 0.9717, Time Cost: 3.023775100708008
    
    100%|██████████| 938/938 [00:36<00:00, 25.47it/s, epoch=2/25, loss=0.0129, lr=0.00162, metric_score=1, time=116]     
    
    Epoch: 2, Test Loss: 0.08240678668799617, Test Score: 0.9765, Time Cost: 3.0006258487701416
    
    100%|██████████| 938/938 [00:36<00:00, 25.85it/s, epoch=3/25, loss=0.000843, lr=0.00146, metric_score=1, time=155]  
    
    Epoch: 3, Test Loss: 0.09966024028365429, Test Score: 0.9731, Time Cost: 3.0690932273864746
    
    100%|██████████| 938/938 [00:35<00:00, 26.38it/s, epoch=4/25, loss=0.000343, lr=0.00131, metric_score=1, time=194]  
    
    Epoch: 4, Test Loss: 0.08925511088793404, Test Score: 0.9739, Time Cost: 3.025567054748535
    
    100%|██████████| 938/938 [00:35<00:00, 26.17it/s, epoch=5/25, loss=0.0211, lr=0.00118, metric_score=1, time=233]    
    
    Epoch: 5, Test Loss: 0.11491756975460037, Test Score: 0.9699, Time Cost: 3.1549229621887207
    
    100%|██████████| 938/938 [00:35<00:00, 26.25it/s, epoch=6/25, loss=0.129, lr=0.00106, metric_score=0.969, time=272] 
    
    Epoch: 6, Test Loss: 0.09543848116054737, Test Score: 0.9765, Time Cost: 3.0333187580108643
    
    100%|██████████| 938/938 [00:35<00:00, 26.34it/s, epoch=7/25, loss=0.0192, lr=0.000957, metric_score=1, time=310]    
    
    Epoch: 7, Test Loss: 0.06982691217252265, Test Score: 0.9811, Time Cost: 3.0253307819366455
    
    100%|██████████| 938/938 [00:35<00:00, 26.46it/s, epoch=8/25, loss=0.000674, lr=0.000861, metric_score=1, time=349]  
    
    Epoch: 8, Test Loss: 0.10375890898652708, Test Score: 0.9732, Time Cost: 3.0132219791412354
    
    100%|██████████| 938/938 [00:35<00:00, 26.44it/s, epoch=9/25, loss=0.00115, lr=0.000775, metric_score=1, time=387]   
    
    Epoch: 9, Test Loss: 0.08423868822431006, Test Score: 0.979, Time Cost: 3.0301530361175537
    
    100%|██████████| 938/938 [00:35<00:00, 26.30it/s, epoch=10/25, loss=0.0073, lr=0.000697, metric_score=1, time=426]    
    
    Epoch: 10, Test Loss: 0.09018090074097593, Test Score: 0.9792, Time Cost: 3.027726173400879
    
    100%|██████████| 938/938 [00:35<00:00, 26.46it/s, epoch=11/25, loss=0.00995, lr=0.000628, metric_score=1, time=465]   
    
    Epoch: 11, Test Loss: 0.09117337604856153, Test Score: 0.978, Time Cost: 3.0312867164611816
    
    100%|██████████| 938/938 [00:35<00:00, 26.36it/s, epoch=12/25, loss=0.000759, lr=0.000565, metric_score=1, time=503]  
    
    Epoch: 12, Test Loss: 0.11916581861087498, Test Score: 0.9772, Time Cost: 3.162600040435791
    
    100%|██████████| 938/938 [00:35<00:00, 26.28it/s, epoch=13/25, loss=0.102, lr=0.000508, metric_score=0.969, time=542] 
    
    Epoch: 13, Test Loss: 0.09118759378719253, Test Score: 0.9824, Time Cost: 2.988072156906128
    
    100%|██████████| 938/938 [00:35<00:00, 26.11it/s, epoch=14/25, loss=0.0461, lr=0.000458, metric_score=0.969, time=581]
    
    Epoch: 14, Test Loss: 0.0869425951682757, Test Score: 0.9803, Time Cost: 3.0128040313720703
    
    100%|██████████| 938/938 [00:35<00:00, 26.35it/s, epoch=15/25, loss=6.94e-6, lr=0.000412, metric_score=1, time=620]   
    
    Epoch: 15, Test Loss: 0.08488982132411006, Test Score: 0.9826, Time Cost: 2.9784598350524902
    
    100%|██████████| 938/938 [00:35<00:00, 26.29it/s, epoch=16/25, loss=0.000103, lr=0.000371, metric_score=1, time=658]  
    
    Epoch: 16, Test Loss: 0.08816149920134123, Test Score: 0.9823, Time Cost: 2.9910941123962402
    
    100%|██████████| 938/938 [00:35<00:00, 26.33it/s, epoch=17/25, loss=4.12e-5, lr=0.000334, metric_score=1, time=697]   
    
    Epoch: 17, Test Loss: 0.10713126366508123, Test Score: 0.9829, Time Cost: 3.158099889755249
    
    100%|██████████| 938/938 [00:35<00:00, 26.09it/s, epoch=18/25, loss=0.00225, lr=0.0003, metric_score=1, time=736]   
    
    Epoch: 18, Test Loss: 0.09688288162248873, Test Score: 0.9829, Time Cost: 3.0078930854797363
    
    100%|██████████| 938/938 [00:35<00:00, 26.15it/s, epoch=19/25, loss=3.83e-6, lr=0.00027, metric_score=1, time=775]   
    
    Epoch: 19, Test Loss: 0.11367125700611343, Test Score: 0.9831, Time Cost: 2.995252847671509
    
    100%|██████████| 938/938 [00:35<00:00, 26.37it/s, epoch=20/25, loss=2.62e-5, lr=0.000243, metric_score=1, time=813]   
    
    Epoch: 20, Test Loss: 0.11589900395485465, Test Score: 0.9826, Time Cost: 2.9824440479278564
    
    100%|██████████| 938/938 [00:35<00:00, 26.38it/s, epoch=21/25, loss=5.81e-5, lr=0.000219, metric_score=1, time=852]   
    
    Epoch: 21, Test Loss: 0.10221088574088256, Test Score: 0.9838, Time Cost: 2.989346742630005
    
    100%|██████████| 938/938 [00:35<00:00, 26.32it/s, epoch=22/25, loss=3.58e-6, lr=0.000197, metric_score=1, time=891]   
    
    Epoch: 22, Test Loss: 0.11218179630007304, Test Score: 0.9842, Time Cost: 3.181006908416748
    
    100%|██████████| 938/938 [00:35<00:00, 26.31it/s, epoch=23/25, loss=0.00181, lr=0.000177, metric_score=1, time=929]   
    
    Epoch: 23, Test Loss: 0.10169062788332937, Test Score: 0.9857, Time Cost: 3.075958013534546
    
    100%|██████████| 938/938 [00:36<00:00, 26.04it/s, epoch=24/25, loss=1.94e-5, lr=0.00016, metric_score=1, time=969]   
    
    Epoch: 24, Test Loss: 0.10714568164599787, Test Score: 0.9855, Time Cost: 3.008065700531006
    model checkpoint saving to ./ckpt/mnist_configs_checkpoint...
    
    evaluating result...
    accuracy 0.9855
    ```