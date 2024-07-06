# Tutorial on Data Expansion Functions

Author: Jiawei Zhang <br>
(Released: July 4, 2024; 1st Revision: July 10, 2024.)

In this tutorial, you will learn

* what is data expansion,
* how to do data expansion,
* visualize expansion outputs,
* how to extend and nest expansions,
* and the optional processing functions. 

This tutorial is prepared based on the recent RPN paper [1], readers are also 
recommended to read that paper first before trying with the {{toolkit}} toolkit.

## 1. What is data expansion function?

As introduced in [1], **data expansion function** is one component function used 
in the RPN model for expanding the input data vector to a high-dimensional space.



\begin{equation}
    \kappa: R^m \to R^D,
\end{equation}

where


## 2. Examples of data expansion functions?

## 3. Taylor's expansion on discrete images

### 3.1 Import package and setup environment

### 3.2 Download image data and create dataloader

### 3.3 Image expansion and visualization

## 4. Bspline expansion on continuous functions

### 4.1 Function dataset creation and loading

### 4.2 Function expansion and visualization

## 5. RBF expansion on tabular data

### 5.1 Tabular dataset loading

### 5.2 Tabular data expansion and visualization

## 6. Extended and Nested Expansions

### 6.1 Extended expansion

### 6.2 Nested expansions

## 7. Optional processing functions for expansions

### 7.1 Pre-processing functions

### 7.2 Post-processing functions

## Expansion Function Instantiation with Configs

## 8. Conclusion

## References

[1] Jiawei Zhang. RPN: Reconciled Polynomial Network. Towards Unifying PGMs, Kernel SVMs, MLP and KAN.



=== "rpn.py"
    ```py title="bubble_sort.py" linenums="1" 
    import numpy as np
    import pandas as pd 
    ```

=== "config.yaml"
    ``` yaml title="config.yaml" linenums="1"
    theme:
      features:
        - content.code.annotate
    ```

``` yaml
theme:
      features:
        - content.code.annotate # (1)!
```

1.  Look ma, less line noise!