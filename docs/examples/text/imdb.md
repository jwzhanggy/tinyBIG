# IMDB Review Polarity Classification

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 9, 2024; latest Revision: July 9, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/imdb_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/imdb_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/imdb_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

For the IMDB dataset that KAN fails as introduced before, in this example, we will build a 3-layer RPN model with 
`identity_expansion`, `lorr_reconciliation` and `zero_remainder` to identify the polarity of the review comments.

The script code and model configuration files are provided as follows. 
We use `mps` as the device in the config file for this example.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'imdb_configs'
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
        data_class: tinybig.data.imdb
        data_parameters:
          name: imdb
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
              lr: 1.0e-04
              weight_decay: 1.0e-05
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
                m: 26964
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
                          r: 2
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
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.identity_expansion
                        data_transformation_parameters:
                          name: identity_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.lorr_reconciliation
                        parameter_fabrication_parameters:
                          name: lorr_reconciliation
                          r: 2
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
                          r: 2
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
    
    100%|██████████| 391/391 [00:05<00:00, 74.30it/s, epoch=0/20, loss=0.672, lr=0.0001, metric_score=0.85, time=5.3]  
    
    Epoch: 0, Test Loss: 0.6756334919149004, Test Score: 0.82976, Time Cost: 2.7346367835998535
    
    100%|██████████| 391/391 [00:04<00:00, 90.30it/s, epoch=1/20, loss=0.409, lr=9e-5, metric_score=0.825, time=12.4]
    
    Epoch: 1, Test Loss: 0.4088795690432839, Test Score: 0.8542, Time Cost: 1.9856078624725342
    
    100%|██████████| 391/391 [00:04<00:00, 89.42it/s, epoch=2/20, loss=0.143, lr=8.1e-5, metric_score=0.95, time=18.7] 
    
    Epoch: 2, Test Loss: 0.3103284809709815, Test Score: 0.87028, Time Cost: 1.9805498123168945
    
    100%|██████████| 391/391 [00:04<00:00, 94.39it/s, epoch=3/20, loss=0.148, lr=7.29e-5, metric_score=0.95, time=24.8]  
    
    Epoch: 3, Test Loss: 0.290586542500102, Test Score: 0.8784, Time Cost: 1.875448226928711
    
    100%|██████████| 391/391 [00:04<00:00, 92.70it/s, epoch=4/20, loss=0.103, lr=6.56e-5, metric_score=0.975, time=30.9] 
    
    Epoch: 4, Test Loss: 0.28517240074360767, Test Score: 0.88284, Time Cost: 1.8985819816589355
    
    100%|██████████| 391/391 [00:04<00:00, 90.78it/s, epoch=5/20, loss=0.175, lr=5.9e-5, metric_score=0.925, time=37.2] 
    
    Epoch: 5, Test Loss: 0.28596944819249764, Test Score: 0.88408, Time Cost: 2.0821990966796875
    
    100%|██████████| 391/391 [00:04<00:00, 91.77it/s, epoch=6/20, loss=0.169, lr=5.31e-5, metric_score=0.925, time=43.5] 
    
    Epoch: 6, Test Loss: 0.28997844215625385, Test Score: 0.88476, Time Cost: 1.9180989265441895
    
    100%|██████████| 391/391 [00:04<00:00, 92.13it/s, epoch=7/20, loss=0.118, lr=4.78e-5, metric_score=0.95, time=49.7]  
    
    Epoch: 7, Test Loss: 0.2956651762375594, Test Score: 0.88436, Time Cost: 1.8994801044464111
    
    100%|██████████| 391/391 [00:04<00:00, 92.69it/s, epoch=8/20, loss=0.0606, lr=4.3e-5, metric_score=0.975, time=55.8]
    
    Epoch: 8, Test Loss: 0.3034781050079924, Test Score: 0.88376, Time Cost: 1.8857190608978271
    
    100%|██████████| 391/391 [00:04<00:00, 92.65it/s, epoch=9/20, loss=0.194, lr=3.87e-5, metric_score=0.95, time=61.9]  
    
    Epoch: 9, Test Loss: 0.310135206374366, Test Score: 0.88284, Time Cost: 1.8619129657745361
    
    100%|██████████| 391/391 [00:04<00:00, 93.57it/s, epoch=10/20, loss=0.227, lr=3.49e-5, metric_score=0.95, time=67.9]  
    
    Epoch: 10, Test Loss: 0.31776188350165896, Test Score: 0.88304, Time Cost: 1.9146320819854736
    
    100%|██████████| 391/391 [00:04<00:00, 93.30it/s, epoch=11/20, loss=0.0866, lr=3.14e-5, metric_score=0.975, time=74]  
    
    Epoch: 11, Test Loss: 0.32553597102346626, Test Score: 0.88284, Time Cost: 1.8726470470428467
    
    100%|██████████| 391/391 [00:04<00:00, 93.70it/s, epoch=12/20, loss=0.0439, lr=2.82e-5, metric_score=1, time=80.1]    
    
    Epoch: 12, Test Loss: 0.3334499675675731, Test Score: 0.88264, Time Cost: 1.8681142330169678
    
    100%|██████████| 391/391 [00:04<00:00, 92.92it/s, epoch=13/20, loss=0.171, lr=2.54e-5, metric_score=0.95, time=86.2]  
    
    Epoch: 13, Test Loss: 0.3411501503032644, Test Score: 0.88284, Time Cost: 2.133592128753662
    
    100%|██████████| 391/391 [00:04<00:00, 93.63it/s, epoch=14/20, loss=0.0122, lr=2.29e-5, metric_score=1, time=92.5]    
    
    Epoch: 14, Test Loss: 0.348429038320356, Test Score: 0.88264, Time Cost: 1.8651437759399414
    
    100%|██████████| 391/391 [00:04<00:00, 93.33it/s, epoch=15/20, loss=0.0494, lr=2.06e-5, metric_score=0.975, time=98.5]
    
    Epoch: 15, Test Loss: 0.35523614200675274, Test Score: 0.88224, Time Cost: 1.8627548217773438
    
    100%|██████████| 391/391 [00:04<00:00, 90.99it/s, epoch=16/20, loss=0.0457, lr=1.85e-5, metric_score=0.975, time=105]
    
    Epoch: 16, Test Loss: 0.36199227931058925, Test Score: 0.88192, Time Cost: 2.1175999641418457
    
    100%|██████████| 391/391 [00:04<00:00, 89.08it/s, epoch=17/20, loss=0.0483, lr=1.67e-5, metric_score=0.975, time=111]
    
    Epoch: 17, Test Loss: 0.36797206425834494, Test Score: 0.88112, Time Cost: 1.9210257530212402
    
    100%|██████████| 391/391 [00:04<00:00, 90.39it/s, epoch=18/20, loss=0.0401, lr=1.5e-5, metric_score=1, time=117]    
    
    Epoch: 18, Test Loss: 0.3735352991067845, Test Score: 0.88036, Time Cost: 1.902376413345337
    
    100%|██████████| 391/391 [00:04<00:00, 92.86it/s, epoch=19/20, loss=0.0422, lr=1.35e-5, metric_score=1, time=124]    
    
    Epoch: 19, Test Loss: 0.378957481573686, Test Score: 0.88008, Time Cost: 1.875330924987793
    model checkpoint saving to ./ckpt/imdb_configs_checkpoint...
    
    evaluating result...
    accuracy 0.88008
    ```