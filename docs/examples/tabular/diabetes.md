# Combinatorial Probabilistic Diabetes Diagnosis

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 9, 2024; latest Revision: July 9, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/diabetes_example.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/diabetes_configs.yaml"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/diabetes_example.py"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a>

</span>
</div>

-----------------------------

In this example, we will build a 1-layer RPN model with `combinatorial_normal_expansion`, `identity_reconciliation` and `linear_remainder`
for diagnosing the diabetes disease based on the Pima Indians Diabetes dataset.

We use `mps` as the device for the model config file provided below.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    from tinybig.metric import accuracy
    
    print('loading configs...')
    config_file_name = 'diabetes_configs'
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
        data_class: tinybig.data.diabetes
        data_parameters:
          name: diabetes
          train_batch_size: 1000
          test_batch_size: 1000
    
      learner_configs:
        learner_class: tinybig.learner.backward_learner
        learner_parameters:
          name: error_backward_propagation
          n_epochs: 3000
          optimizer_configs:
            optimizer_class: torch.optim.AdamW
            optimizer_parameters:
              lr: 5.0e-4
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
                m: 8
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
    Epoch: 0, Test Loss: 0.5340700745582581, Test Score: 0.5340701268297842, Time Cost: 0.006468772888183594
    Epoch: 100, Test Loss: 0.29197242856025696, Test Score: 0.2919724107636688, Time Cost: 0.0055277347564697266
    Epoch: 200, Test Loss: 0.2766035795211792, Test Score: 0.27660356945729825, Time Cost: 0.005342245101928711
    Epoch: 300, Test Loss: 0.26146456599235535, Test Score: 0.2614645354786587, Time Cost: 0.005430936813354492
    Epoch: 400, Test Loss: 0.24704204499721527, Test Score: 0.24704204207501665, Time Cost: 0.005308866500854492
    Epoch: 500, Test Loss: 0.23358845710754395, Test Score: 0.233588461016266, Time Cost: 0.005259990692138672
    Epoch: 600, Test Loss: 0.22124238312244415, Test Score: 0.22124238750083064, Time Cost: 0.005246162414550781
    Epoch: 700, Test Loss: 0.21007299423217773, Test Score: 0.21007298251503007, Time Cost: 0.0055370330810546875
    Epoch: 800, Test Loss: 0.2001015841960907, Test Score: 0.2001015802964822, Time Cost: 0.005261659622192383
    Epoch: 900, Test Loss: 0.1913149207830429, Test Score: 0.19131491785419621, Time Cost: 0.0054361820220947266
    Epoch: 1000, Test Loss: 0.1836737096309662, Test Score: 0.1836737053254299, Time Cost: 0.005302906036376953
    Epoch: 1100, Test Loss: 0.17711907625198364, Test Score: 0.17711907725023052, Time Cost: 0.005380153656005859
    Epoch: 1200, Test Loss: 0.17157748341560364, Test Score: 0.1715774714107086, Time Cost: 0.0056056976318359375
    Epoch: 1300, Test Loss: 0.16696523129940033, Test Score: 0.16696524386354172, Time Cost: 0.0053501129150390625
    Epoch: 1400, Test Loss: 0.16319161653518677, Test Score: 0.16319160136611355, Time Cost: 0.0054242610931396484
    Epoch: 1500, Test Loss: 0.16016244888305664, Test Score: 0.16016245362139456, Time Cost: 0.0053369998931884766
    Epoch: 1600, Test Loss: 0.15778301656246185, Test Score: 0.15778301431480563, Time Cost: 0.005410194396972656
    Epoch: 1700, Test Loss: 0.15595999360084534, Test Score: 0.1559599993896924, Time Cost: 0.005301952362060547
    Epoch: 1800, Test Loss: 0.15460419654846191, Test Score: 0.15460419713485304, Time Cost: 0.00522303581237793
    Epoch: 1900, Test Loss: 0.15363219380378723, Test Score: 0.153632188729002, Time Cost: 0.005263090133666992
    Epoch: 2000, Test Loss: 0.15296749770641327, Test Score: 0.15296750178082663, Time Cost: 0.005364894866943359
    Epoch: 2100, Test Loss: 0.15254199504852295, Test Score: 0.15254199940270272, Time Cost: 0.0052661895751953125
    Epoch: 2200, Test Loss: 0.15229643881320953, Test Score: 0.15229643199782647, Time Cost: 0.005544900894165039
    Epoch: 2300, Test Loss: 0.15218046307563782, Test Score: 0.1521804629655372, Time Cost: 0.005230903625488281
    Epoch: 2400, Test Loss: 0.15215271711349487, Test Score: 0.15215270496868652, Time Cost: 0.005433797836303711
    Epoch: 2500, Test Loss: 0.15218019485473633, Test Score: 0.15218018954948764, Time Cost: 0.005357980728149414
    Epoch: 2600, Test Loss: 0.15223759412765503, Test Score: 0.1522375949802569, Time Cost: 0.005576133728027344
    Epoch: 2700, Test Loss: 0.15230637788772583, Test Score: 0.15230638071102204, Time Cost: 0.005346059799194336
    Epoch: 2800, Test Loss: 0.15237337350845337, Test Score: 0.15237337481388422, Time Cost: 0.0055010318756103516
    Epoch: 2900, Test Loss: 0.15243025124073029, Test Score: 0.15243023843696565, Time Cost: 0.005589008331298828
    model checkpoint saving to ./ckpt/diabetes_configs_checkpoint...
    
    evaluating result...
    mse 0.15247175085902862
    
    evaluating rounded prediction labels...
    accuracy 0.8051948051948052
    ```