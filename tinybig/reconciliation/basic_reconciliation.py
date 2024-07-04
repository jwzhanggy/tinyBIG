# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Basic parameter reconciliation functions.

This module contains the basic parameter reconciliation functions,
including constant_reconciliation, constant_eye_reconciliation, identity_reconciliation,
masking_reconciliation, and duplicated_padding_reconciliation.
"""

import torch

from tinybig.reconciliation import fabrication


#####################
# Basic reconciliation #
#####################

class constant_reconciliation(fabrication):
    r"""
    The constant parameter reconciliation function.

    It performs the constant parameter reconciliation, and returns the constant reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    As the simplest parameter reconciliation function, the **constant parameter reconciliation** projects
    any input parameters to constants (e.g., zeros or ones) as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w} | c) = c \cdot \mathbf{1}^{n \times D} = \mathbf{C} \in {R}^{n \times D},
        \end{equation}
    $$
    where the output matrix $\mathbf{C}$ of size $n \times D$ is filled with the provided constant $c$.

    For constant parameter reconciliation, the input parameter $\mathbf{w}$ is not required,
    which together with its dimension hyper-parameter $l$ can both be set to \textit{none} in implementation.

    If the output constant $\mathbf{C} = \mathbf{0}$ or $\mathbf{C} = \mathbf{1}$,
    we can also name the functions as **zero reconciliation** and **one reconciliation**, respectively.
    Constant parameter reconciliation functions can accommodate outputs according to requirements.

    Constant reconciliation contributes almost nothing to model learning since it involves no parameters,
    but it provides RPN with substantial flexibility in representing and designing many models.

    Attributes
    ----------
    name: str, default = 'constant_reconciliation'
        Name of the reconciliation function.
    c: float, default = 1.0
        The constant value of the reconciliation function.

    Methods
    ----------
    __init__
        It initializes the parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name: str = 'constant_c_reconciliation', c: float = 1.0, *args, **kwargs):
        """
        The initialization method of the constant parameter reconciliation function.

        It initializes a constant parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.
        Since the constant parameter reconciliation doesn't require parameters, it will
        set the "require_parameters" as False in the initialization.

        Parameters
        ----------
        name: str, default = 'constant_c_reconciliation'
            Name of the constant parameter reconciliation function.
        c: float, default = 1.0
            The constant value of the reconciliation function

        Returns
        ----------
        object
            The constant parameter reconciliation function object.
        """
        super().__init__(name=name, require_parameters=False, *args, **kwargs)
        self.c = c

    def calculate_l(self, n: int, D: int):
        """
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., l, of the parameter reconciliation function
        based on the intermediate and output space dimensions, n and D.
        For constant parameter reconciliation, it doesn't require any learnable parameters, and this function
        will return the parameter number as 0 by default.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        int
            The number of required learnable parameters.
        """
        return 0

    def forward(self, n: int, D: int, w: torch.nn.Parameter = None, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the constant parameter reconciliation operation to the input parameter of length l,
        and returns the reconciled parameter matrix of shape (n, D) as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w} | c) = c \cdot \mathbf{1}^{n \times D} = \mathbf{C} \in {R}^{n \times D}.
            \end{equation}
        $$

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.
        w: torch.nn.Parameter, default = None
            The learnable parameters of the model.
            For constant reconciliation, it is assigned with a default value None.
        device: str, default = 'cpu'
            Device to perform the parameter reconciliation.

        Returns
        ----------
        torch.Tensor
            The reconciled parameter matrix of shape (n, D).
        """
        return self.c * torch.ones(n, D).to(device)


class zero_reconciliation(constant_reconciliation):
    r"""
    The zero parameter reconciliation function.

    It performs the zero parameter reconciliation, and returns the zero reconciled parameter matrix of shape (n, D).
    This class inherits from the constant_reconciliation class defined above.

    ...

    Notes
    ----------
    As a special case of the constant parameter reconciliation function, **zero reconciliation** projects
    any input parameters to zero matrix of shape (n, D) as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{0} \in {R}^{n \times D},
        \end{equation}
    $$
    where the output matrix $\mathbf{0}$ of size $n \times D$ is filled with zeros.
    """
    def __init__(self, name: str = 'zero_reconciliation', *args, **kwargs):
        r"""
        The initialization method of the zero parameter reconciliation function.

        It initializes a zero parameter reconciliation function object.
        This method will call the initialization method of the constant_reconciliation
        class with constant parameter $c=0.0$.

        Parameters
        ----------
        name: str, default = 'zero_reconciliation'
            Name of the parameter reconciliation function.

        Returns
        ----------
        object
            The zero parameter reconciliation function object.
        """
        super().__init__(name=name, c=0.0, *args, **kwargs)


class one_reconciliation(constant_reconciliation):
    r"""
    The one parameter reconciliation function.

    It performs the one parameter reconciliation, and returns the one reconciled parameter matrix of shape (n, D).
    This class inherits from the constant_reconciliation class defined above.

    ...

    Notes
    ----------
    As a special case of the constant parameter reconciliation function, **one reconciliation** projects
    any input parameters to one matrix of shape (n, D) as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{1} \in {R}^{n \times D},
        \end{equation}
    $$
    where the output matrix $\mathbf{1}$ of size $n \times D$ is filled with ones.
    """
    def __init__(self, name: str = 'one_reconciliation', *args, **kwargs):
        r"""
        The initialization method of the one parameter reconciliation function.

        It initializes an one parameter reconciliation function object.
        This method will call the initialization method of the constant_reconciliation
        class with constant parameter $c=1.0$.

        Parameters
        ----------
        name: str, default = 'one_reconciliation'
            Name of the parameter reconciliation function.

        Returns
        ----------
        object
            The one parameter reconciliation function object.
        """
        super().__init__(name=name, c=1.0, *args, **kwargs)


class constant_eye_reconciliation(fabrication):
    r"""
    The constant eye parameter reconciliation function.

    It performs the constant eye parameter reconciliation, and returns the constant eye parameter matrix of shape (n, D).
    It is a special case of the constant_reconciliation function defined above.
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    As a special case of the constant_reconciliation function defined above,
    the **constant eye parameter reconciliation** projects any input parameters to constant eye matrix as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{I}^{n \times D} \in {R}^{n \times D},
        \end{equation}
    $$
    where the output matrix $\mathbf{I}$ of size $n \times D$ is returned as an eye matrix.

    For constant eye parameter reconciliation, the input parameter $\mathbf{w}$ is not required,
    which together with its dimension hyper-parameter $l$ can both be set to \textit{none} in implementation.

    Similar as the above constant parameter reconciliation function, the constant eye reconciliation
    contributes almost nothing to model learning since it involves no learnable parameters, but it provides
    RPN with substantial flexibility in representing and designing many models.

    Attributes
    ----------
    name: str, default = 'constant_eye_reconciliation'
        Name of the reconciliation function

    Methods
    ----------
    __init__
        It initializes the parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='constant_eye_reconciliation', *args, **kwargs):
        """
        The initialization method of the constant eye parameter reconciliation function.

        It initializes a constant eye parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.
        Since the constant eye parameter reconciliation doesn't require parameters, it will
        set the "require_parameters" as False in the initialization.

        Parameters
        ----------
        name: str, default = 'constant_eye_reconciliation'
            Name of the constant eye parameter reconciliation function.

        Returns
        ----------
        object
            The constant eye parameter reconciliation function object.
        """
        super().__init__(name=name, require_parameters=False, *args, **kwargs)

    def calculate_l(self, n: int, D: int):
        """
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., l, of the parameter reconciliation function
        based on the intermediate and output space dimensions, n and D.
        For constant eye parameter reconciliation, it doesn't require any learnable parameters, and this function
        will return the parameter number as 0 by default.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        int
            The number of required learnable parameters.
        """
        return 0

    def forward(self, n: int, D: int, w: torch.nn.Parameter = None, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the constant eye parameter reconciliation operation to the input parameter of length l,
        and returns the reconciled parameter matrix of shape (n, D) as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \mathbf{I}^{n \times D} \in {R}^{n \times D},
            \end{equation}
        $$
        where the output matrix $\mathbf{I}$ of size $n \times D$ is returned as an eye matrix.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.
        w: torch.nn.Parameter, default = None
            The learnable parameters of the model.
            For constant eye reconciliation, it is assigned with a default value None.
        device: str, default = 'cpu'
            Device to perform the parameter reconciliation.

        Returns
        ----------
        torch.Tensor
            The reconciled parameter matrix of shape (n, D).
        """
        return torch.eye(n=n, m=D).to(device)


class identity_reconciliation(fabrication):
    r"""
    The identity parameter reconciliation function.

    It performs the identity parameter reconciliation, and returns the identity reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    The **identity parameter reconciliation** projects any input parameters to themselves as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \text{reshape}(\mathbf{w}) = \mathbf{W} \in {R}^{n \times D},
        \end{equation}
    $$
    where the function will reshape the parameters from vector $\mathbf{w}$ of length $l$ to the matrix
    $\mathbf{W}$ of size $n \times D$.

    For the identity parameter reconciliation function, the required number of parameter length can be denoted as
    $$
        \begin{equation}
            l = n \times D,
        \end{equation}
    $$
    where $n$ and $D$ denote the output space and expansion space dimensions, respectively.

    Identity parameter reconciliation is straightforward and may work well for some expansion functions whose
    output dimension $D$ is not very large.
    However, when used with expansion functions that produce a large output dimension
    (such as the high-order Taylor's polynomial expansions), the identity parameter reconciliation
    function may fail due to the "curse of dimensionality" issues.

    Attributes
    ----------
    name: str, default = 'identity_reconciliation'
        Name of the parameter reconciliation function

    Methods
    ----------
    __init__
        It initializes the parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='identity_reconciliation', *args, **kwargs):
        """
        The initialization method of the identity parameter reconciliation function.

        It initializes an identity parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'identity_reconciliation'
            Name of the identity parameter reconciliation function.

        Returns
        ----------
        object
            The identity parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_l(self, n: int, D: int):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., l, of the parameter reconciliation function
        based on the intermediate and output space dimensions, n and D, which can be represented as follows:
        $$
            \begin{equation}
                l = n \times D.
            \end{equation}
        $$

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        int
            The number of required learnable parameters.
        """
        return n*D

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device: str = 'cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the identity parameter reconciliation operation to the input parameter of length l,
        and returns the reconciled parameter matrix of shape (n, D) as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \text{reshape}(\mathbf{w}) = \mathbf{W} \in {R}^{n \times D},
            \end{equation}
        $$

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.
        w: torch.nn.Parameter, default = None
            The learnable parameters of the model.
        device: str, default = 'cpu'
            Device to perform the parameter reconciliation.

        Returns
        ----------
        torch.Tensor
            The reconciled parameter matrix of shape (n, D).
        """
        return w.view(n, D).to(device)


class masking_reconciliation(fabrication):
    r"""
    The masking parameter reconciliation function.

    It performs the masking parameter reconciliation, and returns the masked parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    To mitigate the identified limitation of identity parameter reconciliation function, the masking parameter
    reconciliation function curtail the count of learnable parameters in $\mathbf{W}$ to a reduced number of $l$
    via a randomly generated masking matrix $\mathbf{M}$ as follows:
    $$
        \begin{equation}
            \psi({\mathbf{w}}) = (\mathbf{M} \odot \text{reshape}(\mathbf{w})) = (\mathbf{M} \odot \mathbf{W}) \in {R}^{n \times D},
        \end{equation}
    $$
    where the term $\mathbf{M} \in \{0, 1\}^{n \times D}$ denotes the binary masking matrix only with $l$
    non-zero entries and $\odot$ denotes the element-wise product operator.

    To facilitate practical implementation, instead of pre-define the parameter dimension $l$, the masking reconciliation
    function uses the masking ratio $p$ as a parameter of the masking based reconciliation function instead.
    This parameter, in conjunction with the output dimensions $n \times D$, computes the requisite parameter vector
    dimension, shown as follows:
    $$
        \begin{equation}
            l = p \times n \times D,
        \end{equation}
    $$
    where the masking ratio takes value from $p \in [0, 1]$. For masking_ratio p=1.0: all parameters are used;
    while masking_ratio p=0.0: no parameters will be used.


    Attributes
    ----------
    name: str, default = 'masking_reconciliation'
        Name of the parameter reconciliation function
    p: float, default = 0.5
        The masking ratio of elements in the parameter matrix, which denotes the percentage of used parameter,
        e.g., masking_ratio p=1.0: all parameters are used; masking_ratio p=0.0: no parameters will be used.
    fixed_mask: bool, default = True
        Whether the masking matrix is fixed for all inputs or not.

    Methods
    ----------
    __init__
        It initializes the parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='masking_reconciliation', p=0.5, fixed_mask: bool = True, *args, **kwargs):
        """
        The initialization method of the masking parameter reconciliation function.

        It initializes a masking parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'masking_reconciliation'
            Name of the parameter reconciliation function.
        p: float, default = 0.5
            The masking ratio of elements in the parameter matrix, which denotes the percentage of used parameter,
            e.g., masking_ratio p=1.0: all parameters are used; masking_ratio p=0.0: no parameters will be used.
        fixed_mask: bool, default = True
            Whether the masking matrix is fixed for all inputs or not.

        Returns
        ----------
        object
            The masking parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.mask_matrix = None
        self.fixed_mask = fixed_mask

    def calculate_l(self, n: int, D: int):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., $l$, of the parameter reconciliation function
        based on the intermediate and output space dimensions, $n$ and $D$, and masking ratio parameter $p$,
        which can be represented as follows:
        $$
            \begin{equation}
                l = p \times n \times D.
            \end{equation}
        $$

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        int
            The number of required learnable parameters.
        """
        return n * D

    def generate_masking_matrix(self, n, D):
        """
        The masking matrix generation method.

        It generates the masking matrix of shape (n, D) subject to the masking ratio parameter $p$.
        The method first randomly generates a matrix of shape (n, D) and then compares it with parameter $p$
        to define the binary masking matrix.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        torch.Tensor
            The binary masking matrix of shape (n, D).
        """
        self.mask_matrix = torch.rand(n, D) < self.p

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the masking parameter reconciliation operation to the input parameter vector,
        and returns the reconciled parameter matrix of shape (n, D) subject to the masking ratio $p$ as follows:
        $$
            \begin{equation}
                \psi({\mathbf{w}}) = (\mathbf{M} \odot \text{reshape}(\mathbf{w})) = (\mathbf{M} \odot \mathbf{W}) \in {R}^{n \times D},
            \end{equation}
        $$

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.
        w: torch.nn.Parameter, default = None
            The learnable parameters of the model.
        device: str, default = 'cpu'
            Device to perform the parameter reconciliation.

        Returns
        ----------
        torch.Tensor
            The reconciled parameter matrix of shape (n, D).
        """
        if not self.fixed_mask:
            self.generate_masking_matrix(n=n, D=D)
        else:
            if self.mask_matrix is None:
                self.generate_masking_matrix(n=n, D=D)
        return w.view(n, D) * self.mask_matrix.to(device)


class duplicated_padding_reconciliation(fabrication):
    r"""
    The duplicated padding based parameter reconciliation function.

    It performs the duplicated padding based parameter reconciliation, and returns the reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    Specifically, for the parameter vector $\mathbf{w} \in {R}^{l}$ of length $l$,
    it can be reshaped into a matrix $\mathbf{W}$ comprising $s$ rows and $t$ columns, where $l = s \times t$.
    Through the multiplication of $\mathbf{W}$ with a constant matrix $\mathbf{C} \in {R}^{p \times q}$
    populated with the constant value of ones, the **duplicated padding based parameter reconciliation** function
    can be defined as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{C} \otimes \mathbf{W} =  \begin{bmatrix}
                                                                C_{1,1} \mathbf{W} & C_{1,2} \mathbf{W}      & \cdots & C_{1,q} \mathbf{W}      \\\\
                                                                C_{2,1} \mathbf{W} & C_{2,2} \mathbf{W}      & \cdots & C_{2,q} \mathbf{W}      \\\\
                                                                \vdots & \vdots & \ddots & \vdots \\\\
                                                                C_{p,1} \mathbf{W} & C_{p,2} \mathbf{W}      & \cdots & C_{p,q} \mathbf{W}      \\\\
                                                                \end{bmatrix} \in {R}^{ps \times qt},
        \end{equation}
    $$
    where $\mathbf{W} = \text{reshape}(\mathbf{w})$ and $\otimes$ denotes the Kronecker product operator.
    The output dimensions should meet the constraints that $p \times s = n$ and $q \times t = D$, where renders
    $s = \frac{n}{p}$ and $t = \frac{D}{q}$.

    For the duplicated padding based parameter reconciliation, the number of required parameter $l$ is defined as
    $$
        \begin{equation}
            l= s \times t = \frac{n \times D}{pq},
        \end{equation}
    $$
    where $p$ and $q$ are the duplication numbers in the row and column, respectively.

    Attributes
    ----------
    name: str, default = 'duplicated_padding_reconciliation'
        Name of the parameter reconciliation function
    p: int, default = 2
        Duplication times in the rows.
    q: int, default = None
        Duplication times in the columns.
        If q is not provided with initial values, it will be assigned with value p as well by default.

    Methods
    ----------
    __init__
        It initializes the parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='duplicated_padding_reconciliation', p=2, q=None, *args, **kwargs):
        """
        The initialization method of the duplicated padding based parameter reconciliation function.

        It initializes a duplicated padding based parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'duplicated_padding_reconciliation'
            Name of the parameter reconciliation function
        p: int, default = 2
            Duplication times in the rows.
        q: int, default = None
            Duplication times in the columns.
            If q is not provided with initial values, it will be assigned with value p by default.

        Returns
        ----------
        object
            The masking parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p

    def calculate_l(self, n: int, D: int):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., $l$, of the parameter reconciliation function
        based on the intermediate and output space dimensions, $n$ and $D$, and duplication parameters $p$ and $q$,
        which can be represented as follows:
        $$
            \begin{equation}
                l= s \times t = \frac{n \times D}{pq}.
            \end{equation}
        $$

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        int
            The number of required learnable parameters.
        """
        s, t = int(n/self.p), int(D/self.q)
        assert (self.p * self.q * s * t == n * D)
        return s * t

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the duplicated padding based parameter reconciliation operation to the input parameter vector,
        and returns the reconciled parameter matrix of shape (n, D) subject to duplication parameters $p$ and $q$ as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \mathbf{C} \otimes \mathbf{W} =  \begin{bmatrix}
                                                                    C_{1,1} \mathbf{W} & C_{1,2} \mathbf{W}      & \cdots & C_{1,q} \mathbf{W}      \\\\
                                                                    C_{2,1} \mathbf{W} & C_{2,2} \mathbf{W}      & \cdots & C_{2,q} \mathbf{W}      \\\\
                                                                    \vdots & \vdots & \ddots & \vdots \\\\
                                                                    C_{p,1} \mathbf{W} & C_{p,2} \mathbf{W}      & \cdots & C_{p,q} \mathbf{W}      \\\\
                                                                    \end{bmatrix} \in {R}^{n \times D},
            \end{equation}
        $$
        where $\mathbf{W} = \text{reshape}(\mathbf{w}) \in R^{s \times t}$ and $\otimes$ denotes the Kronecker product operator.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.
        w: torch.nn.Parameter, default = None
            The learnable parameters of the model.
        device: str, default = 'cpu'
            Device to perform the parameter reconciliation.

        Returns
        ----------
        torch.Tensor
            The reconciled parameter matrix of shape (n, D).
        """
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        s, t = int(n / self.p), int(D / self.q)
        A = torch.ones(self.p, self.q).view(1, -1).to(device)
        return torch.einsum('pq,st->psqt', A, w).view(self.p*s, self.q*t).to(device)