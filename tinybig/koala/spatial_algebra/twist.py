# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############
# Twist Class #
###############

import torch
from typing import List, Union, Optional

from tinybig.koala.spatial_algebra.space import space
from tinybig.koala.spatial_algebra.motion import velocity_2d, velocity_3d, angular_velocity_2d, angular_velocity_3d


class twist(space):
    """
    Abstract base class for all twist vectors.

    Combines linear and angular velocities into a single entity.
    """

    def __init__(
        self,
        linear_velocity: Union[float, List[float], torch.Tensor],
        angular_velocity: Union[float, List[float], torch.Tensor],
        name: str = 'twist',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64,
        dim: Optional[int] = None,
        *args, **kwargs
    ):
        """
        Initialize a Twist.

        Parameters
        ----------
        linear_velocity : float, list, or torch.Tensor
            The linear velocity components.
        angular_velocity : float, list, or torch.Tensor
            The angular velocity components.
        device : str, optional
            The device on which the tensors will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensors.
        dim : int, optional
            The dimension of the twist. Must be 2 or 3.
        """
        super().__init__(name=name, dim=dim, device=device, dtype=dtype)

        if dim is None:
            # Infer dimension based on input
            if isinstance(linear_velocity, (list, torch.Tensor)):
                linear_len = len(linear_velocity)
            else:
                linear_len = 1  # Assume scalar for 2D

            if isinstance(angular_velocity, (list, torch.Tensor)):
                angular_len = len(angular_velocity)
            else:
                angular_len = 1  # Assume scalar for 2D

            if linear_len == 1 and angular_len == 1:
                dim = 2
            elif linear_len == 3 and angular_len == 3:
                dim = 3
            else:
                raise ValueError("Mismatch between linear and angular velocity dimensions.")

        if dim not in (2, 3):
            raise ValueError("Twist dimension must be 2 or 3.")

        # Initialize linear velocity
        if dim == 2:
            self.linear = velocity_2d(linear_velocity, device=device, dtype=dtype)
            self.angular = angular_velocity_2d(angular_velocity, device=device, dtype=dtype)
        else:
            self.linear = velocity_3d(linear_velocity, device=device, dtype=dtype)
            self.angular = angular_velocity_3d(angular_velocity, device=device, dtype=dtype)

    def to_list(self) -> List[float]:
        """
        Returns the twist as a combined list of linear and angular velocities.

        Returns
        -------
        List[float]
            The twist as [vx, vy, (vz), wx, wy, (wz)].
        """
        return self.linear.to_list() + self.angular.to_list()

    def to_tensor(self) -> torch.Tensor:
        """
        Returns the twist as a combined tensor of linear and angular velocities.

        Returns
        -------
        torch.Tensor
            The twist as [vx, vy, (vz), wx, wy, (wz)].
        """
        return torch.cat((self.linear.to_tensor(), self.angular.to_tensor()))

    def to_device(self, device: str) -> "twist":
        """
        Moves the twist tensors to the specified device.

        Parameters
        ----------
        device : str
            The target device ('cpu' or 'cuda').

        Returns
        -------
        Twist
            A new Twist instance on the specified device.
        """
        return self.__class__(
            linear_velocity=self.linear.to_device(device).to_list(),
            angular_velocity=self.angular.to_device(device).to_list(),
            device=device,
            dtype=self.dtype,
            dim=self.dim
        )

    def to_dtype(self, dtype: torch.dtype) -> "twist":
        """
        Changes the data type of the twist tensors.

        Parameters
        ----------
        dtype : torch.dtype
            The target data type (e.g., torch.float32, torch.float64).

        Returns
        -------
        Twist
            A new Twist instance with the specified data type.
        """
        return self.__class__(
            linear_velocity=self.linear.to_dtype(dtype).to_list(),
            angular_velocity=self.angular.to_dtype(dtype).to_list(),
            device=self.device,
            dtype=dtype,
            dim=self.dim
        )

    def __add__(self, other: "twist") -> "twist":
        """
        Adds two Twists element-wise.

        Parameters
        ----------
        other : Twist
            The Twist to add.

        Returns
        -------
        Twist
            The resulting Twist after addition.

        Raises
        ------
        ValueError
            If the dimensions of the twists do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot add twists with different dimensions.")
        new_linear = self.linear + other.linear
        new_angular = self.angular + other.angular
        return self.__class__(
            linear_velocity=new_linear.to_list(),
            angular_velocity=new_angular.to_list(),
            device=self.device,
            dtype=self.dtype,
            dim=self.dim
        )

    def __sub__(self, other: "twist") -> "twist":
        """
        Subtracts two Twists element-wise.

        Parameters
        ----------
        other : Twist
            The Twist to subtract.

        Returns
        -------
        Twist
            The resulting Twist after subtraction.

        Raises
        ------
        ValueError
            If the dimensions of the twists do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot subtract twists with different dimensions.")
        new_linear = self.linear - other.linear
        new_angular = self.angular - other.angular
        return self.__class__(
            linear_velocity=new_linear.to_list(),
            angular_velocity=new_angular.to_list(),
            device=self.device,
            dtype=self.dtype,
            dim=self.dim
        )

    def __mul__(self, scalar: float) -> "twist":
        """
        Multiplies the Twist by a scalar.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        Twist
            The resulting Twist after scalar multiplication.
        """
        new_linear = self.linear * scalar
        new_angular = self.angular * scalar
        return self.__class__(
            linear_velocity=new_linear.to_list(),
            angular_velocity=new_angular.to_list(),
            device=self.device,
            dtype=self.dtype,
            dim=self.dim
        )

    def __rmul__(self, scalar: float) -> "twist":
        """
        Enables scalar multiplication from the left.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        Twist
            The resulting Twist after scalar multiplication.
        """
        return self.__mul__(scalar)

    def __repr__(self):
        return f"{self.__class__.__name__}(linear_velocity={self.linear.to_list()}, angular_velocity={self.angular.to_list()})"


class twist_2d(twist):
    """
    Represents a 2D twist (linear and angular velocities).

    Parameters
    ----------
    linear_velocity : float, list, or torch.Tensor
        The linear velocity components.
        - As two floats: (vx, vy).
        - As a list: [vx, vy].
        - As a torch.Tensor: tensor containing [vx, vy].
    angular_velocity : float, list, or torch.Tensor
        The angular velocity component.
        - As a single float: wz.
        - As a list: [wz].
        - As a torch.Tensor: tensor containing [wz].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the tensors (default is torch.float64).
    """

    def __init__(
        self,
        linear_velocity: Union[float, List[float], torch.Tensor],
        angular_velocity: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64,
        **kwargs
    ):
        """
        Initialize a Twist2D instance.

        Parameters
        ----------
        linear_velocity : float, list, or torch.Tensor
            The linear velocity components.
        angular_velocity : float, list, or torch.Tensor
            The angular velocity component.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensors.
        """
        super().__init__(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            device=device,
            dtype=dtype,
            dim=2,
            **kwargs
        )


class twist_3d(twist):
    """
    Represents a 3D twist (linear and angular velocities).

    Parameters
    ----------
    linear_velocity : float, list, or torch.Tensor
        The linear velocity components.
        - As three floats: (vx, vy, vz).
        - As a list: [vx, vy, vz].
        - As a torch.Tensor: tensor containing [vx, vy, vz].
    angular_velocity : float, list, or torch.Tensor
        The angular velocity components.
        - As three floats: (wx, wy, wz).
        - As a list: [wx, wy, wz].
        - As a torch.Tensor: tensor containing [wx, wy, wz].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the tensors (default is torch.float64).
    """

    def __init__(
        self,
        linear_velocity: Union[float, List[float], torch.Tensor],
        angular_velocity: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64,
        **kwargs
    ):
        """
        Initialize a Twist3D instance.

        Parameters
        ----------
        linear_velocity : float, list, or torch.Tensor
            The linear velocity components.
        angular_velocity : float, list, or torch.Tensor
            The angular velocity components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensors.
        """
        super().__init__(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            device=device,
            dtype=dtype,
            dim=3,
            **kwargs
        )
