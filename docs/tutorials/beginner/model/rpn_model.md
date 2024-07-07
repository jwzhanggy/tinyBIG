To model the underlying data distribution mapping $f: R^m \to R^n$, the {{our}} model disentangle the input data from 
model parameters into three component functions:

* **Data Expansion Function**: $\kappa: R^m \to R^D$.
* **Parameter Reconciliatoin Function**: $\psi: R^l \to R^{n \times D}$.
* **Remainder Function** $\pi: R^m \to R^n$.

So, the underlying mapping $f$ can be approximated by {{our}} as the inner product of the expansion function with
the reconciliation function, subsequentlly summed with the remainder function:
$$
g(\mathbf{x} | \mathbf{w}) = \left \langle \kappa(\mathbf{x}), \psi(\mathbf{w}) \right \rangle + \pi(\mathbf{x}).
$$