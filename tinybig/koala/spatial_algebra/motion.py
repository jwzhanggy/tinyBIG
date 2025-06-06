# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Motion Spatial Vector Classes #
#################################

import torch
from typing import List, Union, Optional

from tinybig.koala.spatial_algebra.space import space


class motion(space):
    """
    Abstract base class for all motion vectors.

    Provides common functionalities such as initialization, device management,
    data type conversion, and arithmetic operations.
    """

    def __init__(
        self,
        *coords: Union[float, List[float], torch.Tensor],
        name: str = 'motion',
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        dim: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize a motion.

        Parameters
        ----------
        *coords : float, list, or torch.Tensor
            The coordinate values of the motion vector.
            - For 2D: (x, y) or [x, y] or torch.Tensor([x, y]).
            - For 3D: (x, y, z) or [x, y, z] or torch.Tensor([x, y, z]).
        device : str, optional
            The device on which the tensor will be allocated (default is 'cpu').
        dtype : torch.dtype, optional
            The data type of the underlying tensor (default is torch.float32).
        dim : int, optional
            The dimension of the motion vector. If not specified, it will be inferred
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
            raise ValueError("Only 2D or 3D motion vectors are supported.")

        self.v = coords_tensor

    def to_list(self) -> List[float]:
        """
        Returns the motion vector coordinates as a list.

        Returns
        -------
        List[float]
            The motion vector coordinates.
        """
        return self.v.tolist()

    def to_tensor(self) -> torch.Tensor:
        """
        Returns the motion vector coordinates as a tensor.

        Returns
        -------
        torch.Tensor
            The motion vector coordinates.
        """
        return self.v.clone()

    def to_device(self, device: str) -> "motion":
        """
        Moves the motion vector tensor to the specified device.

        Parameters
        ----------
        device : str
            The target device ('cpu' or 'cuda').

        Returns
        -------
        motion
            A new motion instance on the specified device.
        """
        return self.__class__(
            *self.v.tolist(),
            device=device,
            dtype=self.dtype
        )

    def to_dtype(self, dtype: torch.dtype) -> "motion":
        """
        Changes the data type of the motion vector tensor.

        Parameters
        ----------
        dtype : torch.dtype
            The target data type (e.g., torch.float32, torch.float32).

        Returns
        -------
        motion
            A new motion instance with the specified data type.
        """
        return self.__class__(
            *self.v.tolist(),
            device=self.device,
            dtype=dtype
        )

    def __add__(self, other: "motion") -> "motion":
        """
        Adds two motion instances element-wise.

        Parameters
        ----------
        other : motion
            The motion to add.

        Returns
        -------
        motion
            The resulting motion after addition.

        Raises
        ------
        ValueError
            If the dimensions of the motion vectors do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot add motion vectors with different dimensions.")
        new_v = self.v + other.v
        return self.__class__(*new_v.tolist(), device=self.device, dtype=self.dtype)

    def __sub__(self, other: "motion") -> "motion":
        """
        Subtracts two motion instances element-wise.

        Parameters
        ----------
        other : motion
            The motion to subtract.

        Returns
        -------
        motion
            The resulting motion after subtraction.

        Raises
        ------
        ValueError
            If the dimensions of the motion vectors do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot subtract motion vectors with different dimensions.")
        new_v = self.v - other.v
        return self.__class__(*new_v.tolist(), device=self.device, dtype=self.dtype)

    def __mul__(self, scalar: float) -> "motion":
        """
        Multiplies the motion by a scalar.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        motion
            The resulting motion after scalar multiplication.
        """
        new_v = self.v * scalar
        return self.__class__(*new_v.tolist(), device=self.device, dtype=self.dtype)

    def __rmul__(self, scalar: float) -> "motion":
        """
        Enables scalar multiplication from the left.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        motion
            The resulting motion after scalar multiplication.
        """
        return self.__mul__(scalar)

    def __repr__(self):
        return f"{self.__class__.__name__}(v={self.v.tolist()})"


class velocity_2d(motion):
    """
    Represents 2D linear velocity.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The velocity components.
        - As two floats: (vx, vy).
        - As a list: [vx, vy].
        - As a torch.Tensor: tensor containing [vx, vy].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of velocity components does not match 2.
    TypeError
        If the velocity components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a velocity_2d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The velocity components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=2, **kwargs)


class velocity_3d(motion):
    """
    Represents 3D linear velocity.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The velocity components.
        - As three floats: (vx, vy, vz).
        - As a list: [vx, vy, vz].
        - As a torch.Tensor: tensor containing [vx, vy, vz].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of velocity components does not match 3.
    TypeError
        If the velocity components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a velocity_3d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The velocity components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=3, **kwargs)


class acceleration_2d(motion):
    """
    Represents 2D linear acceleration.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The acceleration components.
        - As two floats: (ax, ay).
        - As a list: [ax, ay].
        - As a torch.Tensor: tensor containing [ax, ay].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of acceleration components does not match 2.
    TypeError
        If the acceleration components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize an acceleration_2d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The acceleration components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=2, **kwargs)


class acceleration_3d(motion):
    """
    Represents 3D linear acceleration.

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The acceleration components.
        - As three floats: (ax, ay, az).
        - As a list: [ax, ay, az].
        - As a torch.Tensor: tensor containing [ax, ay, az].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of acceleration components does not match 3.
    TypeError
        If the acceleration components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize an acceleration_3d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The acceleration components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=3, **kwargs)


class angular_velocity_2d(motion):
    """
    Represents 2D angular velocity (scalar), internally stored as a 2D vector.

    Parameters
    ----------
    angle_rate : float, list, or torch.Tensor
        The angular velocity component.
        - As a single float: theta_rate.
        - As a list: [theta_rate].
        - As a torch.Tensor: tensor containing [theta_rate].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If more than one angular velocity component is provided.
    TypeError
        If the angular velocity is not provided as a float, list, or torch.Tensor.
    """

    def __init__(
        self,
        angle_rate: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize an angular_velocity_2d instance.

        Internally represents angular velocity as a 2D vector [0, ωz].

        Parameters
        ----------
        angle_rate : float, list, or torch.Tensor
            The angular velocity component.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        # Extract scalar angular velocity
        if isinstance(angle_rate, list):
            if len(angle_rate) != 1:
                raise ValueError("angular_velocity_2d requires a single angular velocity component.")
            angle_scalar = angle_rate[0]
        elif isinstance(angle_rate, torch.Tensor):
            if angle_rate.numel() != 1:
                raise ValueError("angular_velocity_2d requires a single angular velocity component.")
            angle_scalar = angle_rate.item()
        elif isinstance(angle_rate, (float, int)):
            angle_scalar = float(angle_rate)
        else:
            raise TypeError("angle_rate must be a float, list, or torch.Tensor.")

        # Represent as 2D vector [0, ωz]
        angle_vector = [0.0, angle_scalar]

        super().__init__(
            angle_vector,
            device=device,
            dtype=dtype,
            dim=2,  # Set dim=2 to comply with motion
            **kwargs
        )
        self.theta_rate = angle_scalar  # Scalar value

    def to_list(self) -> List[float]:
        """
        Returns the angular velocity as a list.

        Returns
        -------
        List[float]
            The angular velocity [0.0, ωz].
        """
        return [0.0, self.theta_rate]

    def to_tensor(self) -> torch.Tensor:
        """
        Returns the angular velocity as a tensor.

        Returns
        -------
        torch.Tensor
            The angular velocity tensor [0.0, ωz].
        """
        return torch.tensor([0.0, self.theta_rate], device=self.device, dtype=self.dtype)

    def get_scalar(self) -> float:
        """
        Retrieves the scalar angular velocity (ωz).

        Returns
        -------
        float
            The scalar angular velocity.
        """
        return self.theta_rate

    def __mul__(self, scalar: float) -> "angular_velocity_2d":
        """
        Multiplies the angular_velocity_2d by a scalar.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        angular_velocity_2d
            The resulting angular_velocity_2d after multiplication.
        """
        return angular_velocity_2d(self.theta_rate * scalar, device=self.device, dtype=self.dtype)

    def __rmul__(self, scalar: float) -> "angular_velocity_2d":
        """
        Enables scalar multiplication from the left.

        Parameters
        ----------
        scalar : float
            The scalar to multiply with.

        Returns
        -------
        angular_velocity_2d
            The resulting angular_velocity_2d after multiplication.
        """
        return self.__mul__(scalar)

    def __add__(self, other: "angular_velocity_2d") -> "angular_velocity_2d":
        """
        Adds two angular_velocity_2d instances.

        Parameters
        ----------
        other : angular_velocity_2d
            The angular_velocity_2d to add.

        Returns
        -------
        angular_velocity_2d
            The resulting angular_velocity_2d after addition.
        """
        if not isinstance(other, angular_velocity_2d):
            raise TypeError("Can only add angular_velocity_2d with angular_velocity_2d.")
        return angular_velocity_2d(self.theta_rate + other.theta_rate, device=self.device, dtype=self.dtype)

    def __sub__(self, other: "angular_velocity_2d") -> "angular_velocity_2d":
        """
        Subtracts two angular_velocity_2d instances.

        Parameters
        ----------
        other : angular_velocity_2d
            The angular_velocity_2d to subtract.

        Returns
        -------
        angular_velocity_2d
            The resulting angular_velocity_2d after subtraction.
        """
        if not isinstance(other, angular_velocity_2d):
            raise TypeError("Can only subtract angular_velocity_2d with angular_velocity_2d.")
        return angular_velocity_2d(self.theta_rate - other.theta_rate, device=self.device, dtype=self.dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(theta_rate={self.theta_rate:.4f} rad/s)"


class angular_velocity_3d(motion):
    """
    Represents 3D angular velocity (vector).

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The angular velocity components.
        - As three floats: (wx, wy, wz).
        - As a list: [wx, wy, wz].
        - As a torch.Tensor: tensor containing [wx, wy, wz].
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of angular velocity components does not match 3.
    TypeError
        If the angular velocity components are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *args: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize an angular_velocity_3d instance.

        Parameters
        ----------
        *args : float, list, or torch.Tensor
            The angular velocity components.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        """
        super().__init__(*args, device=device, dtype=dtype, dim=3, **kwargs)


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Velocity Example
    print("=== Velocity Example ===")

    # Initialize 2D Velocity
    vel2d = velocity_2d(3.0, 4.0)
    print("velocity_2d:", vel2d)

    # Initialize 3D Velocity
    vel3d = velocity_3d([1.0, 2.0, 3.0])
    print("velocity_3d:", vel3d)

    # Add two velocities
    vel2d_added = vel2d + velocity_2d(1.0, 1.0)
    print("velocity_2d Added:", vel2d_added)

    # Scalar multiplication
    vel3d_scaled = 2 * vel3d
    print("velocity_3d Scaled:", vel3d_scaled)

    # Acceleration Example
    print("\n=== Acceleration Example ===")

    # Initialize 2D Acceleration
    acc2d = acceleration_2d(0.5, 0.5)
    print("acceleration_2d:", acc2d)

    # Initialize 3D Acceleration
    acc3d = acceleration_3d(0.1, 0.2, 0.3)
    print("acceleration_3d:", acc3d)

    # Subtract two accelerations
    acc2d_subtracted = acc2d - acceleration_2d(0.2, 0.2)
    print("acceleration_2d Subtracted:", acc2d_subtracted)

    # Scalar multiplication
    acc3d_scaled = acc3d * 3
    print("acceleration_3d Scaled:", acc3d_scaled)

    # Angular Velocity Example
    print("\n=== Angular Velocity Example ===")

    # Initialize 2D Angular Velocity
    ang_vel2d = angular_velocity_2d(1.5708)  # 90 degrees in radians
    print("angular_velocity_2d:", ang_vel2d)

    # Initialize 3D Angular Velocity
    ang_vel3d = angular_velocity_3d([0.1, 0.2, 0.3])
    print("angular_velocity_3d:", ang_vel3d)

    # Invert Angular Velocity
    ang_vel2d_inv = ang_vel2d * -1
    print("angular_velocity_2d Inverted:", ang_vel2d_inv)

    # Add two 3D Angular Velocities
    ang_vel3d_added = ang_vel3d + angular_velocity_3d([0.4, 0.5, 0.6])
    print("angular_velocity_3d Added:", ang_vel3d_added)

    # Scalar multiplication
    ang_vel3d_scaled = 0.5 * ang_vel3d
    print("angular_velocity_3d Scaled:", ang_vel3d_scaled)

    # Conversion Methods Example
    print("\n=== Conversion Methods Example ===")

    # Convert velocity_2d to list and tensor
    vel2d_list = vel2d.to_list()
    vel2d_tensor = vel2d.to_tensor()
    print("velocity_2d as list:", vel2d_list)
    print("velocity_2d as tensor:", vel2d_tensor)

    # Convert angular_velocity_2d to list and tensor
    ang_vel2d_list = ang_vel2d.to_list()
    ang_vel2d_tensor = ang_vel2d.to_tensor()
    print("angular_velocity_2d as list:", ang_vel2d_list)
    print("angular_velocity_2d as tensor:", ang_vel2d_tensor)

    # Move acceleration_3d to GPU if available
    if torch.cuda.is_available():
        acc3d_gpu = acc3d.to_device('cuda')
        print("acceleration_3d on GPU:", acc3d_gpu)
    elif torch.backends.mps.is_available():
        acc3d = acc3d.to_dtype(torch.float32)
        acc3d_mps = acc3d.to_device('mps')
        print("acceleration_3d on MPS:", acc3d_mps)
    else:
        print("CUDA or MPS is not available. Skipping GPU example.")
