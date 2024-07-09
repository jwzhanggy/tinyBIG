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
```

=== "model configs"
```yaml linenums="1"
```

## Observations

The above training records and testing scores are consistent with the problems on KAN as reported in the RPN paper `[1]`.
They both indicate that KAN cannot be trained with the sparse data vectorized with `sklearn.TfidfVectorizer`,
and the model is just doing the random guess when classifying the documents.

These observations reveal major deficiencies in KAN's model design not discovered nor reported in the previous KAN paper
`[2]`, which may pose challenges for it in replacing MLP as a new base model for more complex learning scenarios.

**Reference**

`[1] Jiawei Zhang. RPN: Reconciled Polynomial Network Towards Unifying PGMs, Kernel SVMs, MLP and KAN. arXiv 2407.04819.`
`[2] Ziming Liu, et al. KAN: Kolmogorov-Arnold Networks. arXiv 2404.19756.`