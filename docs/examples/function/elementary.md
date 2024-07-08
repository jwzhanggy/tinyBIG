# Elementary function approximation task
([elemetary_example.ipynb](../notes/elementary_example.ipynb), [elementary_configs.yaml](../configs/elementary_configs.yaml))

In this example, we will build a {{our}} model for the elementary function approximation task.

As mentioned before, we assume you have worked on the quickstart tutorial and the other tutorials provided at the website already.
Therefore, we will not introduce the technical details for the model components in this example anymore.

Since this is the first deep function learning task example, we will briefly describe the configurations of the {{our}} 
model, data, learner and evaluation metric as follows, which will be used to design the `config.yaml` file later.

## Configuration Descriptions

* **Data**: `elementary_function` with ID E.14, i.e.,
$$
    f(x, y) = \arcsin(x+y), 
$$
where $x, y \in (0, 0.5)$. Based on the formula, `2,000` input-output pairs are generated, 
which are further partitioned into the training and testing sets according to the `50%:50%` ratio.

* **Model**: `rpn_model` with 2 layers: [2, 2, 1], (i.e., input dimension: 2, middle layer dimension: 2, output dimension: 1).
    * Layer 1: `rpn_layer` with 1 head, 1 channel. Component functions: `extended_expansion` (of `bspline_expansion` and `taylor_expansion`), `lorr_reconciliation`, and `zero_remainder`.
    * Layer 2: `rpn_layer` with 1 head, 1 channel. Component functions: `extended_expansion` (of `bspline_expansion` and `taylor_expansion`), `lorr_reconciliation`, and `zero_remainder`.
  
* **Learner**: `backward_learner` with the following configurations
    * loss: `torch.nn.MSELoss`
    * optimizer: `torch.optim.AdamW` with `lr`: `3.0e-03` and `weight_decay`: `1.0e-04`.
    * lr_scheduler: `torch.optim.lr_scheduler.ExponentialLR` with `gamma`: `0.999`.
* **Output**: `rpn_output`
* **Metric**: `mse`

## Python Code and Model Configurations

Based on the above task description, we can compose the `config.yaml` file to design the configurations of the 
modules, components and models used for the task, which is provided as follows.

By running the following `script.py` on the `configs.yaml` file, we will train a {{our}} model to address the 
continuous function approximation task. Both the code and configuration files can be downloaded via the links
provided at the very beginning of this page.

=== "elementary_example.ipynb"
    ``` python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'elementary_configs'
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

    ```
=== "elementary_configs.yaml"
    ``` yaml linenums="1"
    configurations:
      device: mps
      random_seed: 123
    
      data_configs:
        data_class: tinybig.data.elementary_function
        data_parameters:
          name: elementary_function
          train_batch_size: 100
          test_batch_size: 100
          equation_index: 14
    
      learner_configs:
        learner_class: tinybig.learner.backward_learner
        learner_parameters:
          name: error_backward_propagation
          n_epochs: 2000
          optimizer_configs:
            optimizer_class: torch.optim.AdamW
            optimizer_parameters:
              lr: 3.0e-03
              weight_decay: 1.0e-04
          lr_scheduler_configs:
            lr_scheduler_class: torch.optim.lr_scheduler.ExponentialLR
            lr_scheduler_parameters:
              gamma: 0.999
          loss_configs:
            loss_class: torch.nn.MSELoss
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
                m: 2
                n: 2
                width: 1
                width_alloc: [1]
                fusion_strategy: average
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l: null
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.extended_expansion
                        data_transformation_parameters:
                          name: extended_expansion
                          expansion_function_configs:
                            - expansion_class: tinybig.expansion.bspline_expansion
                              expansion_parameters:
                                name: bspline_expansion
                            - expansion_class: tinybig.expansion.taylor_expansion
                              expansion_parameters:
                                name: taylor_expansion
                                d: 2
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.lorr_reconciliation
                        parameter_fabrication_parameters:
                          name: lorr_reconciliation
                          r: 1
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
                          require_parameters: False
                          enable_bias: False
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 2
                n: 1
                width: 1
                width_alloc: [ 1 ]
                fusion_strategy: average
                head_configs:
                  - head_class: tinybig.module.rpn_head
                    head_parameters:
                      l: null
                      channel_num: 1
                      data_transformation_configs:
                        data_transformation_class: tinybig.expansion.extended_expansion
                        data_transformation_parameters:
                          name: extended_expansion
                          expansion_function_configs:
                            - expansion_class: tinybig.expansion.bspline_expansion
                              expansion_parameters:
                                name: bspline_expansion
                            - expansion_class: tinybig.expansion.taylor_expansion
                              expansion_parameters:
                                name: taylor_expansion
                                d: 2
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.lorr_reconciliation
                        parameter_fabrication_parameters:
                          name: lorr_reconciliation
                          r: 1
                      remainder_configs:
                        remainder_class: tinybig.remainder.zero_remainder
                        remainder_parameters:
                          name: zero_remainder
                          require_parameters: False
                          enable_bias: False
    
      metric_configs:
        metric_class: tinybig.metric.mse
        metric_parameters:
          name: mse
    
      result_configs:
        result_class: tinybig.output.rpn_output
        result_parameters:
          name: rpn_output
    ```

??? quote "model training records and evaluation outputs"
    ```
    loading configs...
    {'configurations': {'device': 'mps', 'random_seed': 123, 'data_configs': {'data_class': 'tinybig.data.elementary_function', 'data_parameters': {'name': 'elementary_function', 'train_batch_size': 100, 'test_batch_size': 100, 'equation_index': 14}}, 'learner_configs': {'learner_class': 'tinybig.learner.backward_learner', 'learner_parameters': {'name': 'error_backward_propagation', 'n_epochs': 2000, 'optimizer_configs': {'optimizer_class': 'torch.optim.AdamW', 'optimizer_parameters': {'lr': 0.003, 'weight_decay': 0.0001}}, 'lr_scheduler_configs': {'lr_scheduler_class': 'torch.optim.lr_scheduler.ExponentialLR', 'lr_scheduler_parameters': {'gamma': 0.999}}, 'loss_configs': {'loss_class': 'torch.nn.MSELoss', 'loss_parameters': {'reduction': 'mean'}}}}, 'model_configs': {'model_class': 'tinybig.model.rpn', 'model_parameters': {'name': 'reconciled_polynomial_network', 'depth': 2, 'depth_alloc': [1, 1], 'layer_configs': [{'layer_class': 'tinybig.module.rpn_layer', 'layer_parameters': {'name': 'rpn_layer', 'm': 2, 'n': 2, 'width': 1, 'width_alloc': [1], 'fusion_strategy': 'average', 'head_configs': [{'head_class': 'tinybig.module.rpn_head', 'head_parameters': {'l': None, 'channel_num': 1, 'data_transformation_configs': {'data_transformation_class': 'tinybig.expansion.extended_expansion', 'data_transformation_parameters': {'name': 'extended_expansion', 'expansion_function_configs': [{'expansion_class': 'tinybig.expansion.bspline_expansion', 'expansion_parameters': {'name': 'bspline_expansion'}}, {'expansion_class': 'tinybig.expansion.taylor_expansion', 'expansion_parameters': {'name': 'taylor_expansion', 'd': 2}}]}}, 'parameter_fabrication_configs': {'parameter_fabrication_class': 'tinybig.reconciliation.lorr_reconciliation', 'parameter_fabrication_parameters': {'name': 'lorr_reconciliation', 'r': 1}}, 'remainder_configs': {'remainder_class': 'tinybig.remainder.zero_remainder', 'remainder_parameters': {'name': 'zero_remainder', 'require_parameters': False, 'enable_bias': False}}}}]}}, {'layer_class': 'tinybig.module.rpn_layer', 'layer_parameters': {'name': 'rpn_layer', 'm': 2, 'n': 1, 'width': 1, 'width_alloc': [1], 'fusion_strategy': 'average', 'head_configs': [{'head_class': 'tinybig.module.rpn_head', 'head_parameters': {'l': None, 'channel_num': 1, 'data_transformation_configs': {'data_transformation_class': 'tinybig.expansion.extended_expansion', 'data_transformation_parameters': {'name': 'extended_expansion', 'expansion_function_configs': [{'expansion_class': 'tinybig.expansion.bspline_expansion', 'expansion_parameters': {'name': 'bspline_expansion'}}, {'expansion_class': 'tinybig.expansion.taylor_expansion', 'expansion_parameters': {'name': 'taylor_expansion', 'd': 2}}]}}, 'parameter_fabrication_configs': {'parameter_fabrication_class': 'tinybig.reconciliation.lorr_reconciliation', 'parameter_fabrication_parameters': {'name': 'lorr_reconciliation', 'r': 1}}, 'remainder_configs': {'remainder_class': 'tinybig.remainder.zero_remainder', 'remainder_parameters': {'name': 'zero_remainder', 'require_parameters': False, 'enable_bias': False}}}}]}}]}}, 'metric_configs': {'metric_class': 'tinybig.metric.mse', 'metric_parameters': {'name': 'mse'}}, 'result_configs': {'result_class': 'tinybig.output.rpn_output', 'result_parameters': {'name': 'rpn_output'}}}}

    setting up environments...
    device:  mps ; random_seed:  123

    instantiating objects from config...
    parameter num:  47

    loading dataset...
    E.14,14,f,arcsinh(x+y),2,x,0,0.5,y,0,0.5

    training model...
    Epoch: 0, Test Loss: 0.2524416998028755, Test Score: 0.2524417051846299, Time Cost: 0.06097102165222168
    Epoch: 100, Test Loss: 0.00011527910683071241, Test Score: 0.00011527910800508923, Time Cost: 0.06091189384460449
    Epoch: 200, Test Loss: 2.391126545262523e-05, Test Score: 2.391126489365892e-05, Time Cost: 0.061573028564453125
    Epoch: 300, Test Loss: 6.572608162969118e-06, Test Score: 6.572608435869078e-06, Time Cost: 0.06058096885681152
    Epoch: 400, Test Loss: 3.5195622672290485e-06, Test Score: 3.519562341415268e-06, Time Cost: 0.06185793876647949
    Epoch: 500, Test Loss: 2.334099656309263e-06, Test Score: 2.3340997543815286e-06, Time Cost: 0.06178903579711914
    Epoch: 600, Test Loss: 1.6119351073484723e-06, Test Score: 1.6119351278171615e-06, Time Cost: 0.06792116165161133
    Epoch: 700, Test Loss: 1.163999343134492e-06, Test Score: 1.1639993293543274e-06, Time Cost: 0.06416893005371094
    Epoch: 800, Test Loss: 7.2217666229335e-07, Test Score: 7.22176668815331e-07, Time Cost: 0.07884573936462402
    Epoch: 900, Test Loss: 4.94887612489947e-07, Test Score: 4.948876104610631e-07, Time Cost: 0.06533408164978027
    Epoch: 1000, Test Loss: 3.025931547995242e-07, Test Score: 3.025931632419966e-07, Time Cost: 0.06408810615539551
    Epoch: 1100, Test Loss: 1.584212050431688e-07, Test Score: 1.5842121761223267e-07, Time Cost: 0.06264305114746094
    Epoch: 1200, Test Loss: 8.785054852467056e-08, Test Score: 8.785055151103172e-08, Time Cost: 0.06547284126281738
    Epoch: 1300, Test Loss: 4.33750560802082e-08, Test Score: 4.337505689102142e-08, Time Cost: 0.08099198341369629
    Epoch: 1400, Test Loss: 1.849457365032947e-08, Test Score: 1.8494574095504617e-08, Time Cost: 0.0652158260345459
    Epoch: 1500, Test Loss: 1.5995996971440718e-08, Test Score: 1.599599760880588e-08, Time Cost: 0.06752681732177734
    Epoch: 1600, Test Loss: 9.078116569583016e-09, Test Score: 9.0781169684237e-09, Time Cost: 0.06858992576599121
    Epoch: 1700, Test Loss: 8.637106319042687e-08, Test Score: 8.637106692820431e-08, Time Cost: 0.06311798095703125
    Epoch: 1800, Test Loss: 2.759304884580871e-09, Test Score: 2.7593048902881114e-09, Time Cost: 0.06516385078430176
    Epoch: 1900, Test Loss: 1.2575214647370103e-08, Test Score: 1.2575215319624022e-08, Time Cost: 0.07782721519470215
    
    model checkpoint saving to ./ckpt/elementary_configs_checkpoint...

    testing model...

    evaluating result...
    mse 1.5470938963708936e-09

    saving result...
    ```

In the above python code, we disable the `tqdm` progress bars and also only display testing scores every 100 epochs 
during training. If you want to check the `tqdm` and more testing information, you can change the above `train` function as follows:
```python
training_records = learner_obj.train(model=model_obj, data_loader=data_loader, device=device, metric=metric_obj)
```