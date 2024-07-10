# Naive Probabilistic Iris Species Inference

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 9, 2024; latest Revision: July 9, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/iris_example.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/iris_configs.yaml"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/iris_example.py"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a>

</span>
</div>

-----------------------------

In this example, we will build a 1-layer RPN model with `naive_laplace_expansion`, `identity_reconciliation` and `linear_remainder`
for inferring the species labels of the Iris dataset.

Currently, many probabilistic distribution functions provided by `pytorch` may need to run on `cpu`.
For the probabilistic expansion functions implemented in `tinybig`, they will automatically move the inputs to `cpu` before the expansion, 
and then transfer back to the original device when returning the expansion outputs.

Therefore, we can still use `mps` as the device for the model config file provided below.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    from tinybig.metric import accuracy
    
    print('loading configs...')
    config_file_name = 'iris_configs'
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
        data_class: tinybig.data.iris
        data_parameters:
          name: iris
          train_batch_size: 1000
          test_batch_size: 1000
    
      learner_configs:
        learner_class: tinybig.learner.backward_learner
        learner_parameters:
          name: error_backward_propagation
          n_epochs: 1000
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
                        data_transformation_class: tinybig.expansion.naive_laplace_expansion
                        data_transformation_parameters:
                          name: naive_laplace_expansion
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
    Epoch: 0, Test Loss: 3.3385839462280273, Test Score: 3.3385838587504746, Time Cost: 0.002852916717529297
    Epoch: 100, Test Loss: 0.20029181241989136, Test Score: 0.2002918031480495, Time Cost: 0.002424001693725586
    Epoch: 200, Test Loss: 0.058508310467004776, Test Score: 0.05850831156387198, Time Cost: 0.002733945846557617
    Epoch: 300, Test Loss: 0.05215996503829956, Test Score: 0.052159959597903045, Time Cost: 0.0025398731231689453
    Epoch: 400, Test Loss: 0.05397547408938408, Test Score: 0.05397546970579003, Time Cost: 0.0028302669525146484
    Epoch: 500, Test Loss: 0.05306984856724739, Test Score: 0.053069849117935254, Time Cost: 0.0022029876708984375
    Epoch: 600, Test Loss: 0.051849350333213806, Test Score: 0.051849349241034834, Time Cost: 0.0020189285278320312
    Epoch: 700, Test Loss: 0.05095481127500534, Test Score: 0.050954806913829656, Time Cost: 0.0020186901092529297
    Epoch: 800, Test Loss: 0.050374239683151245, Test Score: 0.050374236378489, Time Cost: 0.0044133663177490234
    Epoch: 900, Test Loss: 0.05000824108719826, Test Score: 0.050008237337228666, Time Cost: 0.0022449493408203125
    model checkpoint saving to ./ckpt/iris_configs_checkpoint...

    evaluating result...
    mse 0.04977021345255972

    evaluating rounded prediction labels...
    accuracy 1.0
    ```

