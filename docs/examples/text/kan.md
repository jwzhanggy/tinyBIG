# Failure of KAN on Sparse Data

<div style="display: flex; justify-content: space-between;">
<span style="text-align: left;">
    Author: Jiawei Zhang <br>
    (Released: July 8, 2024; latest Revision: July 8, 2024.)<br>
</span>
<span style="text-align: right;">

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/kan_failure_example.ipynb">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/configs/kan_failure_configs.yaml">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;">
    </a>

    <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/kan_failure_example.py">
    <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;">
    </a>

</span>
</div>

-------------------------

In this example, we will investigate the failure case reported in the RPN paper `[1]` about the recent KAN 
(Kolmogorov–Arnold Networks) model proposed in `[2]` on handling sparse data.

According to `[1]`, the KAN model can be represented with RPN by using `bspline_expansion`, `identity_reconciliation`,
and `linear_remainder` as the component functions. Here, we will investigate to apply the KAN model for classifying the IMDB
dataset, where each document is vectorized by `sklearn.TfidfVectorizer` into an extremely sparse vector.

Below, we will provide the python code and model configuration, and illustrate the training records, together with
the evaluation performance on the testing set.

-------------------------

## Python Code and Model Configurations
=== "python script"
    ```python linenums="1"
    from tinybig.config import rpn_config
    from tinybig.util import set_random_seed
    
    print('loading configs...')
    config_file_name = 'kan_failure_configs'
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
                        data_transformation_class: tinybig.expansion.bspline_expansion
                        data_transformation_parameters:
                          name: bspline_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                      remainder_configs:
                        remainder_class: tinybig.remainder.linear_remainder
                        remainder_parameters:
                          name: linear_remainder
                          require_parameters: True
                          enable_bias: False
                          activation_function_configs:
                            function_class: torch.nn.SiLU
    
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
                        data_transformation_class: tinybig.expansion.bspline_expansion
                        data_transformation_parameters:
                          name: bspline_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                      remainder_configs:
                        remainder_class: tinybig.remainder.linear_remainder
                        remainder_parameters:
                          name: linear_remainder
                          require_parameters: True
                          enable_bias: False
                          activation_function_configs:
                            function_class: torch.nn.SiLU
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
                        data_transformation_class: tinybig.expansion.bspline_expansion
                        data_transformation_parameters:
                          name: bspline_expansion
                      parameter_fabrication_configs:
                        parameter_fabrication_class: tinybig.reconciliation.identity_reconciliation
                        parameter_fabrication_parameters:
                          name: identity_reconciliation
                      remainder_configs:
                        remainder_class: tinybig.remainder.linear_remainder
                        remainder_parameters:
                          name: linear_remainder
                          require_parameters: True
                          enable_bias: False
                          activation_function_configs:
                            function_class: torch.nn.SiLU
    
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

    100%|██████████| 391/391 [00:16<00:00, 23.50it/s, epoch=0/20, loss=0.692, lr=0.0001, metric_score=0.575, time=16.7]
    
    Epoch: 0, Test Loss: 0.6931786558512226, Test Score: 0.5, Time Cost: 9.628262281417847
    
    100%|██████████| 391/391 [00:15<00:00, 25.41it/s, epoch=1/20, loss=0.693, lr=9e-5, metric_score=0.45, time=41.7] 
    
    Epoch: 1, Test Loss: 0.6931470397793119, Test Score: 0.5, Time Cost: 9.534846305847168
    
    100%|██████████| 391/391 [00:15<00:00, 25.09it/s, epoch=2/20, loss=0.693, lr=8.1e-5, metric_score=0.525, time=66.9]
    
    Epoch: 2, Test Loss: 0.6931509616429848, Test Score: 0.5, Time Cost: 9.546906232833862
    
    100%|██████████| 391/391 [00:15<00:00, 25.03it/s, epoch=3/20, loss=0.693, lr=7.29e-5, metric_score=0.625, time=92]  
    
    Epoch: 3, Test Loss: 0.6931469051734261, Test Score: 0.5, Time Cost: 9.616436958312988
    
    100%|██████████| 391/391 [00:15<00:00, 24.98it/s, epoch=4/20, loss=0.694, lr=6.56e-5, metric_score=0.4, time=117]  
    
    Epoch: 4, Test Loss: 0.6931621979569536, Test Score: 0.5, Time Cost: 9.66498613357544
    
    100%|██████████| 391/391 [00:15<00:00, 24.56it/s, epoch=5/20, loss=0.693, lr=5.9e-5, metric_score=0.525, time=143]
    
    Epoch: 5, Test Loss: 0.6931535824180564, Test Score: 0.5, Time Cost: 9.572320938110352
    
    100%|██████████| 391/391 [00:15<00:00, 25.24it/s, epoch=6/20, loss=0.693, lr=5.31e-5, metric_score=0.5, time=168]  
    
    Epoch: 6, Test Loss: 0.6931597494408298, Test Score: 0.5, Time Cost: 9.538740873336792
    
    100%|██████████| 391/391 [00:16<00:00, 24.40it/s, epoch=7/20, loss=0.693, lr=4.78e-5, metric_score=0.5, time=194]  
    
    Epoch: 7, Test Loss: 0.6931724850174106, Test Score: 0.5, Time Cost: 9.450575828552246
    
    100%|██████████| 391/391 [00:15<00:00, 24.71it/s, epoch=8/20, loss=0.692, lr=4.3e-5, metric_score=0.6, time=219]  
    
    Epoch: 8, Test Loss: 0.6931692090485712, Test Score: 0.5, Time Cost: 9.580528259277344
    
    100%|██████████| 391/391 [00:15<00:00, 24.59it/s, epoch=9/20, loss=0.691, lr=3.87e-5, metric_score=0.65, time=244] 
    
    Epoch: 9, Test Loss: 0.6931905321148045, Test Score: 0.5, Time Cost: 9.535048961639404
    
    100%|██████████| 391/391 [00:15<00:00, 25.28it/s, epoch=10/20, loss=0.693, lr=3.49e-5, metric_score=0.35, time=269] 
    
    Epoch: 10, Test Loss: 0.693147035358507, Test Score: 0.5, Time Cost: 9.524089813232422
    
    100%|██████████| 391/391 [00:15<00:00, 25.39it/s, epoch=11/20, loss=0.693, lr=3.14e-5, metric_score=0.5, time=294]  
    
    Epoch: 11, Test Loss: 0.6931473204241995, Test Score: 0.5, Time Cost: 9.606273174285889
    
    100%|██████████| 391/391 [00:15<00:00, 25.46it/s, epoch=12/20, loss=0.693, lr=2.82e-5, metric_score=0.475, time=319]
    
    Epoch: 12, Test Loss: 0.6931546899058935, Test Score: 0.5, Time Cost: 9.354511260986328
    
    100%|██████████| 391/391 [00:15<00:00, 25.24it/s, epoch=13/20, loss=0.693, lr=2.54e-5, metric_score=0.55, time=344] 
    
    Epoch: 13, Test Loss: 0.6931510639312627, Test Score: 0.5, Time Cost: 9.555925846099854
    
    100%|██████████| 391/391 [00:15<00:00, 24.82it/s, epoch=14/20, loss=0.693, lr=2.29e-5, metric_score=0.6, time=369]  
    
    Epoch: 14, Test Loss: 0.6931466630962498, Test Score: 0.5, Time Cost: 9.458446025848389
    
    100%|██████████| 391/391 [00:15<00:00, 25.35it/s, epoch=15/20, loss=0.693, lr=2.06e-5, metric_score=0.575, time=394]
    
    Epoch: 15, Test Loss: 0.6931504569090235, Test Score: 0.5, Time Cost: 9.424824714660645
    
    100%|██████████| 391/391 [00:15<00:00, 24.93it/s, epoch=16/20, loss=0.693, lr=1.85e-5, metric_score=0.525, time=419]
    
    Epoch: 16, Test Loss: 0.693146731999829, Test Score: 0.5, Time Cost: 9.512600183486938
    
    100%|██████████| 391/391 [00:15<00:00, 25.11it/s, epoch=17/20, loss=0.693, lr=1.67e-5, metric_score=0.375, time=444]
    
    Epoch: 17, Test Loss: 0.693148196353327, Test Score: 0.5, Time Cost: 9.611510753631592
    
    100%|██████████| 391/391 [00:15<00:00, 24.74it/s, epoch=18/20, loss=0.693, lr=1.5e-5, metric_score=0.6, time=470]  
    
    Epoch: 18, Test Loss: 0.6931469484668253, Test Score: 0.5, Time Cost: 9.608121156692505
    
    100%|██████████| 391/391 [00:16<00:00, 24.14it/s, epoch=19/20, loss=0.693, lr=1.35e-5, metric_score=0.5, time=496]  
    
    Epoch: 19, Test Loss: 0.6931468187390691, Test Score: 0.5, Time Cost: 9.569139957427979
    model checkpoint saving to ./ckpt/kan_failure_configs_checkpoint...

    evaluating result...
    accuracy 0.5

    ```
---------------------

## Observations

The above training records and testing scores are consistent with the problems on KAN as reported in the RPN paper `[1]`.
They both indicate that KAN cannot be trained with the sparse data vectorized with `sklearn.TfidfVectorizer`,
and the model is just doing the random guess when classifying the documents.

These observations reveal major deficiencies in KAN's model design not discovered nor reported in the previous KAN paper
`[2]`, which may pose challenges for it in replacing MLP as a new base model for more complex learning scenarios.

**Reference**

`[1] Jiawei Zhang. RPN: Reconciled Polynomial Network Towards Unifying PGMs, Kernel SVMs, MLP and KAN. arXiv 2407.04819.`

`[2] Ziming Liu, et al. KAN: Kolmogorov-Arnold Networks. arXiv 2404.19756.`