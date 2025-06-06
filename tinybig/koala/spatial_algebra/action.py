# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Force Spatial Vector Classes #
################################

import torch
from typing import List, Union, Optional

from tinybig.koala.spatial_algebra.space import space


class action(space):
    """
    Abstract base class for all force-related vectors.

    Provides common functionalities such as initialization, device management,
    data type conversion, and arithmetic operations.
    """

    def __init__(
        self,
        *coords: Union[float, List[float], torch.Tensor],
        name: str = 'action',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        dim: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize a action spatial vector.

        Parameters
        ----------
        *coords : float, list, or torch.Tensor
            The coordinate values of the force vector.
            - For 2D: (Fx, Fy) or [Fx, Fy] or torch.Tensor([Fx, Fy]).
            - For 3D: (Fx, Fy, Fz) or [Fx, Fy, Fz] or torch.Tensor([Fx, Fy, Fz]).
        device : str, optional
            The device on which the tensor will be allocated (default is 'cpu').
        dtype : torch.dtype, optional
            The data type of the underlying tensor (default is torch.float32).
        dim : int, optional
            The dimension of the force vector. If not specified, it will be inferred
            from the length of `coords`. Must be either 2 or 3.

        Raises
        ------
        ValueError
            If the number of coordinates doesn't match the specified or inferred dimension,
            or if the dimension is not 2 or 3.
        TypeError
            If the coordinates are not provided as floats, lists, or torch.Tensors.
        """
        super().__init__(name=name, dim=dim, device=device, dtype=dtype)

        if len(coords) == 1 and isinstance(coords[0], (list, torch.Tensor)):
            # Single argument provided as list or tensor
            coords_input = coords[0]
            if isinstance(coords_input, list):
                coords_tensor = torch.tensor(coords_input, device=device, dtype=dtype)
            elif isinstance(coords_input, torch.Tensor):
                coords_tensor = coords_input.to(device=device, dtype=dtype)
            else:
                raise TypeError("Single argument must be a list or a torch.Tensor")
        else:
            # Multiple float arguments provided
            coords_tensor = torch.tensor(coords, device=device, dtype=dtype)

        # Infer dimension if not provided
        if dim is not None:
            if len(coords_tensor) != dim:
                raise ValueError(f"Number of coordinates must match dim={dim}")
        else:
            dim = len(coords_tensor)

        if dim not in (2, 3):
            raise ValueError("Only 2D or 3D force vectors are supported.")

        self.v = coords_tensor

    def to_list(self) -> List[float]:
        """
        Returns the force vector coordinates as a list.

        Returns
        -------
        List[float]
            The force vector coordinates.
        """
        return self.v.tolist()

    def to_tensor(self) -> torch.Tensor:
        """
        Returns the force vector coordinates as a tensor.

        Returns
        -------
        torch.Tensor
            The force vector coordinates.
        """
        return self.v.clone()

    def to_device(self, device: str) -> "action":
        """
        Moves the force vector tensor to the specified device.

        Parameters
        ----------
        device : str
            The target device ('cpu' or 'cuda').

        Returns
        -------
        ForceVector
            A new ForceVector instance on the specified device.
        """
        return self.__class__(
            *self.v.tolist(),
            device=device,
            dtype=self.dtype
        )

    def to_dtype(self, dtype: torch.dtype) -> "action":
        """
        Changes the data type of the force vector tensor.

        Parameters
        ----------
        dtype : torch.dtype
            The target data type (e.g., torch.float32, torch.float32).

        Returns
        -------
        ForceVector
            A new ForceVector instance with the specified data type.
        """
        return self.__class__(
            *self.v.tolist(),
            device=self.device,
            dtype=dtype
        )

    def __add__(self, other: "action") -> "action":
        """
        Adds two ForceVector instances element-wise.

        Parameters
        ----------
        other : ForceVector
            The ForceVector to add.

        Returns
        -------
        ForceVector
            The resulting ForceVector after addition.

        Raises
        ------
        ValueError
            If the dimensions of the force vectors do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot add force vectors with different dimensions.")
        new_v = self.v + other.v
        return self.__class__(*new_v.tolist(), device=self.device, dtype=self.dtype)

    def __sub__(self, other: "action") -> "action":
        """
        Subtracts two ForceVector instances element-wise.

        Parameters
        ----------
        other : ForceVector
            The ForceVector to subtract.

        Returns
        -------
        ForceVector
            The resulting ForceVector after subtraction.

        Raises
        ------
        ValueError
            If the dimensions of the force vectors do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot subtract force vectors with different dimensions.")
        new_v = self.v - other.v
        return self.__class__(*new_v.tolist(), device=self.device, dtype=self.dtype)

    def __mul__(self, scalar: float) -> "action":
        """
        Multiplies the ForceVector by a scalar.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        ForceVector
            The resulting ForceVector after scalar multiplication.
        """
        new_v = self.v * scalar
        return self.__class__(*new_v.tolist(), device=self.device, dtype=self.dtype)

    def __rmul__(self, scalar: float) -> "action":
        """
        Enables scalar multiplication from the left.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        ForceVector
            The resulting ForceVector after scalar multiplication.
        """
        return self.__mul__(scalar)

    def __repr__(self):
        return f"{self.__class__.__name__}(v={self.v.tolist()})"


class force_2d(action):
    """
    Represents a 2D force vector.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The force components.
        - As two floats: (Fx, Fy).
        - As a list: [Fx, Fy].
        - As a torch.Tensor: tensor containing [Fx, Fy].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of force components does not match 2.
    TypeError
        If the force components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a force_2d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The force components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=2, **kwargs)


class force_3d(action):
    """
    Represents a 3D force vector.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The force components.
        - As three floats: (Fx, Fy, Fz).
        - As a list: [Fx, Fy, Fz].
        - As a torch.Tensor: tensor containing [Fx, Fy, Fz].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of force components does not match 3.
    TypeError
        If the force components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a force_3d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The force components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=3, **kwargs)



class torque_2d(action):
    """
    Represents a 2D torque vector.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The force components.
        - As two floats: (Fx, Fy).
        - As a list: [Fx, Fy].
        - As a torch.Tensor: tensor containing [Fx, Fy].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of torque components does not match 2.
    TypeError
        If the torque components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a torque_2d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The torque components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=2, **kwargs)


class torque_3d(action):
    """
    Represents a 3D torque vector.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The force components.
        - As three floats: (Fx, Fy, Fz).
        - As a list: [Fx, Fy, Fz].
        - As a torch.Tensor: tensor containing [Fx, Fy, Fz].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of force components does not match 3.
    TypeError
        If the force components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a force_3d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The force components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=3, **kwargs)


class momentum_2d(action):
    """
    Represents a 2D momentum vector.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The momentum components.
        - As two floats: (Px, Py).
        - As a list: [Px, Py].
        - As a torch.Tensor: tensor containing [Px, Py].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of momentum components does not match 2.
    TypeError
        If the momentum components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a momentum_2d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The momentum components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=2, **kwargs)


class momentum_3d(action):
    """
    Represents a 3D momentum vector.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The momentum components.
        - As three floats: (Px, Py, Pz).
        - As a list: [Px, Py, Pz].
        - As a torch.Tensor: tensor containing [Px, Py, Pz].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of momentum components does not match 3.
    TypeError
        If the momentum components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a momentum_3d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The momentum components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=3, **kwargs)


class wrench(action):
    """
    Represents a spatial wrench consisting of force and torque (moment).
    """

    def __init__(
        self,
        force: Union[force_2d, force_3d],
        torque: Union[torque_2d, torque_3d],
        name: str = 'wrench',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(name=name, device=device, dtype=dtype)

        if not isinstance(force, (force_2d, force_3d)):
            raise TypeError("force must be an instance of force_2d or force_3d.")
        if not isinstance(torque, (torque_2d, torque_3d)):
            raise TypeError("torque must be an instance of Torque2D or Torque3D.")

        if force.to_tensor().shape[0] != torque.to_tensor().shape[0]:
            raise ValueError("Dimensions of force and torque must match.")

        self.force = force
        self.torque = torque


class wrench_2d(wrench):
    def __init__(
        self,
        fx: float, fy: float,
        tau_x: float, tau_y: float,
        name: str = 'wrench_2d',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        force = force_2d(fx, fy, device=device, dtype=dtype)
        torque = torque_2d(tau_x, tau_y, device=device, dtype=dtype)
        super().__init__(name=name, force=force, torque=torque, device=device, dtype=dtype)


class wrench_3d(wrench):
    def __init__(
        self,
        fx: float, fy: float, fz: float,
        tau_x: float, tau_y: float, tau_z: float,
        name: str = 'wrench_3d', device='cpu', dtype=torch.float32
    ):
        force = force_3d(fx, fy, fz, device=device, dtype=dtype)
        torque = torque_3d(tau_x, tau_y, tau_z, device=device, dtype=dtype)
        super().__init__(name=name, force=force, torque=torque, device=device, dtype=dtype)



if __name__ == '__main__':
    # -------------------
    # Force Example
    # -------------------
    print("=== Force Example ===")

    # Initialize 2D Force
    force_2d = force_2d(10.0, 5.0)
    print("force_2d:", force_2d)

    # Initialize 3D Force
    force_3d = force_3d([3.0, 4.0, 5.0])
    print("force_3d:", force_3d)

    # Add two forces
    force_2d_added = force_2d + force_2d(2.0, 3.0)
    print("force_2d Added:", force_2d_added)

    # Scalar multiplication
    force_3d_scaled = 2 * force_3d
    print("force_3d Scaled:", force_3d_scaled)

    # -------------------
    # Momentum Example
    # -------------------
    print("\n=== Momentum Example ===")

    # Initialize 2D Momentum
    momentum_2d = momentum_2d(20.0, 10.0)
    print("momentum_2d:", momentum_2d)

    # Initialize 3D Momentum
    momentum_3d = momentum_3d([6.0, 8.0, 10.0])
    print("momentum_3d:", momentum_3d)

    # Subtract two momenta
    momentum_2d_subtracted = momentum_2d - momentum_2d(5.0, 5.0)
    print("momentum_2d Subtracted:", momentum_2d_subtracted)

    # Scalar multiplication
    momentum_3d_scaled = momentum_3d * 0.5
    print("momentum_3d Scaled:", momentum_3d_scaled)

    # -------------------
    # Conversion Methods Example
    # -------------------
    print("\n=== Conversion Methods Example ===")

    # Convert force_2d to list and tensor
    force_2d_list = force_2d.to_list()
    force_2d_tensor = force_2d.to_tensor()
    print("force_2d as list:", force_2d_list)
    print("force_2d as tensor:", force_2d_tensor)

    # Convert momentum_3d to list and tensor
    momentum_3d_list = momentum_3d.to_list()
    momentum_3d_tensor = momentum_3d.to_tensor()
    print("momentum_3d as list:", momentum_3d_list)
    print("momentum_3d as tensor:", momentum_3d_tensor)

    # Move Acceleration3D to GPU if available
    if torch.cuda.is_available():
        force_3d_gpu = force_3d.to_device('cuda')
        print("force_3d on GPU:", force_3d_gpu)
    elif torch.backends.mps.is_available():
        force_3d = force_3d.to_dtype(torch.float32)
        force_3d_mps = force_3d.to_device('mps')
        print("force_3d on MPS:", force_3d_mps)
    else:
        print("CUDA or MPS is not available. Skipping GPU example.")