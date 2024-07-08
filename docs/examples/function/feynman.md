# Feynman Function Approximation

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 8, 2024; latest Revision: July 8, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/feynman_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/feynman_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/feynman_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

-------------------------

In this example, we will build a 3-layer {{our}} model with `extended_expansion`, `lorr_reconciliation` and `linear_remainder`
functions to approximate the `feynman_function` dataset with formula ID I.9.18 (or with numerical `equation_index` 4).

This formula describes Newton's law of universal gravitation with $9$ arguments inputs, it is illustrated as follows:
$$
    f(G, m_1, m_2, x_1, y_1, z_1, x_2, y_2, z_2) = \frac{G \cdot m_1 \cdot m_2}{(x_2 - x_1)^2+(y_2 - y_1)^2+(z_2 - z_1)^2},
$$
where $G, m_1, m_2, x_2, y_2, z_2 \in [1, 2]$ and $x_1, y_1, z_1 \in [3, 4]$.

-------------------------

## Python Code and Model Configurations

=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'feynman_configs'
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

=== "model configs"
    ```yaml linenums="1"
    # three-layer rpn for Newton's law of universal gravitation function approximation
    configurations:
      device: cpu
      random_seed: 4567
    
      data_configs:
        data_class: tinybig.data.feynman_function
        data_parameters:
          name: elementary_function
          train_batch_size: 100
          test_batch_size: 100
          equation_index: 4
    
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
          depth: 3
          depth_alloc: [1, 1, 1]
          layer_configs:
            - layer_class: tinybig.module.rpn_layer
              layer_parameters:
                name: rpn_layer
                m: 9
                n: 2
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
                                t: 10
                                d: 4
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
                        remainder_class: tinybig.remainder.linear_remainder
                        remainder_parameters:
                          name: linear_remainder
                          require_parameters: True
                          enable_bias: False
                          activation_functions: torch.nn.functional.silu
    
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
                                t: 10
                                d: 4
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
                        remainder_class: tinybig.remainder.linear_remainder
                        remainder_parameters:
                          name: linear_remainder
                          require_parameters: True
                          enable_bias: False
                          activation_functions: torch.nn.functional.silu
    
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
                                t: 10
                                d: 4
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
                        remainder_class: tinybig.remainder.linear_remainder
                        remainder_parameters:
                          name: linear_remainder
                          require_parameters: True
                          enable_bias: False
                          activation_functions: torch.nn.functional.silu
    
      metric_configs:
        metric_class: tinybig.metric.mse
        metric_parameters:
          name: mse
    
      result_configs:
        result_class: tinybig.output.rpn_output
        result_parameters:
          name: rpn_output
    ```

???+ quote "rpn with nested expansion testing and evaluation outputs"
    ```shell
    evaluating result...
    mse 5.7441863398028705e-05
    ```