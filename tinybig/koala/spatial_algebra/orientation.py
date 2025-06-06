# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
# Orientation Class #
#####################

import torch
import math
from abc import abstractmethod
from typing import Union, List

from tinybig.koala.spatial_algebra.space import space
from tinybig.koala.spatial_algebra.position import position_2d, position_3d


class orientation(space):
    """
    Abstract Base Class for Orientation.

    This class defines a common interface for different dimensional orientations.
    Subclasses must implement the abstract methods to handle specific dimensionalities.
    """

    def __init__(
        self,
        name: str = 'orientation',
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(name=name, device=device, dtype=dtype)

    @abstractmethod
    def to_rotation_matrix(self) -> torch.Tensor:
        """
        Converts the orientation to a rotation matrix.

        Returns
        -------
        torch.Tensor
            A rotation matrix representing the orientation.
        """
        pass

    @abstractmethod
    def inverse(self) -> "orientation":
        """
        Returns the inverse of the orientation.

        Returns
        -------
        Orientation
            The inverse orientation.
        """
        pass

    @abstractmethod
    def compose(self, other: "orientation") -> "orientation":
        """
        Composes this orientation with another.

        Parameters
        ----------
        other : Orientation
            The other orientation to compose with.

        Returns
        -------
        Orientation
            The composed orientation.
        """
        pass

    @abstractmethod
    def to_list(self) -> List[float]:
        """
        Converts the orientation to a list representation.

        Returns
        -------
        List[float]
            A list representing the orientation.
        """
        pass

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        """
        Returns the orientation as a tensor.

        Returns
        -------
        torch.Tensor
            A tensor representing the orientation.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, dtype={self.dtype})"

    @classmethod
    @abstractmethod
    def from_rotation_matrix(cls, R: torch.Tensor) -> "orientation":
        """
        Creates an Orientation instance from a rotation matrix.

        Parameters
        ----------
        R : torch.Tensor
            A rotation matrix.

        Returns
        -------
        Orientation
            The corresponding Orientation instance.
        """
        pass


class orientation_2d(orientation):
    """
    Represents a 2D orientation using a single angle (theta).

    Provides methods to convert to and from quaternions and rotation matrices.
    Supports input angles in radians or degrees.
    """

    def __init__(
        self,
        name: str = "orientation_2d",
        angle: Union[float, List[float], torch.Tensor] = None,
        unit: str = "rad",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initializes the Orientation2D instance with a single angle.

        Parameters
        ----------
        angle : float, list, or torch.Tensor
            The rotation angle.
            - As a float: represents the angle in radians or degrees.
            - As a list: [theta].
            - As a torch.Tensor: tensor containing [theta].
        unit : str, optional
            The unit of the angle ('rad' for radians or 'deg' for degrees).
            Default is 'rad'.
        device : str, optional
            The device on which the tensor will be allocated.
            Default is 'cpu'.
        dtype : torch.dtype, optional
            The data type of the tensor. Default is torch.float32.

        Raises
        ------
        ValueError
            If the angle is not a scalar or list with one element.
        TypeError
            If the angle is not a float, list, or torch.Tensor.
        """
        super().__init__(name=name, device=device, dtype=dtype)

        if angle is None:
            raise ValueError("angle cannot be None")

        if isinstance(angle, list):
            if len(angle) != 1:
                raise ValueError("Orientation2D requires a single angle.")
            angle_tensor = torch.tensor(angle, device=device, dtype=dtype)
        elif isinstance(angle, torch.Tensor):
            if angle.numel() == 1:
                angle_tensor = angle.to(device=device, dtype=dtype)
            else:
                raise ValueError("Orientation2D requires a single angle.")
        elif isinstance(angle, (float, int)):
            angle_tensor = torch.tensor([angle], device=device, dtype=dtype)
        else:
            raise TypeError("angle must be a float, list, or torch.Tensor.")

        if unit.lower() == "deg":
            angle_tensor = torch.deg2rad(angle_tensor)
        elif unit.lower() == "rad":
            pass  # Already in radians
        else:
            raise ValueError("unit must be either 'rad' or 'deg'")

        # Handle both 0-dim and 1-dim tensors
        if angle_tensor.dim() == 0:
            self.theta = angle_tensor
        elif angle_tensor.dim() == 1 and angle_tensor.numel() == 1:
            self.theta = angle_tensor[0]
        else:
            raise ValueError("Orientation2D requires a single angle.")

        self.device = device
        self.dtype = dtype

    def to_quaternion(self) -> torch.Tensor:
        """
        Converts the 2D orientation to a quaternion.

        Returns
        -------
        torch.Tensor
            A quaternion [w, x, y, z].
            For 2D, y and z components are zero.
        """
        half_theta = self.theta / 2.0
        w = torch.cos(half_theta)
        x = torch.sin(half_theta)
        y = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        z = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return torch.stack([w, x, y, z])

    @classmethod
    def from_quaternion(cls, quaternion: Union[List[float], torch.Tensor], device: str = "cpu",
                        dtype: torch.dtype = torch.float32) -> "orientation_2d":
        """
        Creates an Orientation2D instance from a quaternion.

        Parameters
        ----------
        quaternion : list or torch.Tensor
            A quaternion [w, x, y, z]. For 2D, y and z should be zero.
        device : str, optional
            The device of the tensors (default is "cpu").
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Returns
        -------
        Orientation2D
            The corresponding Orientation2D instance.

        Raises
        ------
        ValueError
            If the quaternion does not represent a valid 2D rotation.
        """
        if isinstance(quaternion, list):
            if len(quaternion) != 4:
                raise ValueError("Quaternion must have four components [w, x, y, z].")
            quaternion = torch.tensor(quaternion, device=device, dtype=dtype)
        elif isinstance(quaternion, torch.Tensor):
            if quaternion.numel() != 4:
                raise ValueError("Quaternion must have four components [w, x, y, z].")
            quaternion = quaternion.to(device=device, dtype=dtype)
        else:
            raise TypeError("quaternion must be a list or a torch.Tensor.")

        # Normalize the quaternion
        quaternion = quaternion / quaternion.norm()

        w, x, y, z = quaternion
        if not (torch.isclose(y, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-6) and
                torch.isclose(z, torch.tensor(0.0, device=device, dtype=dtype), atol=1e-6)):
            raise ValueError("Quaternion for Orientation2D must have y and z components equal to zero.")

        theta = 2 * torch.acos(w)
        # Adjust theta to be in the range [-pi, pi]
        theta = torch.where(theta > math.pi, theta - 2 * math.pi, theta)
        theta = torch.where(theta < -math.pi, theta + 2 * math.pi, theta)

        return cls(angle=theta, unit="rad", device=device, dtype=dtype)

    def to_rotation_matrix(self) -> torch.Tensor:
        """
        Converts the 2D orientation to a 2x2 rotation matrix.

        Returns
        -------
        torch.Tensor
            A 2x2 rotation matrix.
        """
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        R = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], device=self.device, dtype=self.dtype)
        return R

    @classmethod
    def from_rotation_matrix(cls, R: torch.Tensor, device: str = "cpu",
                             dtype: torch.dtype = torch.float32) -> "orientation_2d":
        """
        Creates an Orientation2D instance from a 2x2 rotation matrix.

        Parameters
        ----------
        R : torch.Tensor
            A 2x2 rotation matrix.
        device : str, optional
            The device of the tensors (default is "cpu").
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Returns
        -------
        Orientation2D
            The corresponding Orientation2D instance.

        Raises
        ------
        ValueError
            If R is not a valid 2x2 rotation matrix.
        """
        if not isinstance(R, torch.Tensor):
            raise TypeError("R must be a torch.Tensor.")
        if R.shape != (2, 2):
            raise ValueError("Rotation matrix R must be 2x2.")

        # Validate orthogonality and determinant
        should_be_identity = torch.mm(R, R.t())
        identity = torch.eye(2, device=device, dtype=dtype)
        if not torch.allclose(should_be_identity, identity, atol=1e-6):
            raise ValueError("Rotation matrix R must be orthogonal.")
        det = torch.det(R)
        if not torch.isclose(det, torch.tensor(1.0, device=device, dtype=dtype), atol=1e-6):
            raise ValueError("Rotation matrix R must have a determinant of +1.")

        theta = torch.atan2(R[1, 0], R[0, 0])
        return cls(angle=theta, unit="rad", device=device, dtype=dtype)

    def inverse(self) -> "orientation_2d":
        """
        Computes the inverse (negative) of the current orientation.

        Returns
        -------
        Orientation2D
            The inverse orientation.
        """
        return orientation_2d(angle=-self.theta, unit="rad", device=self.device, dtype=self.dtype)

    def compose(self, other: "orientation_2d") -> "orientation_2d":
        """
        Composes this orientation with another orientation.

        Parameters
        ----------
        other : Orientation2D
            The other orientation to compose with.

        Returns
        -------
        Orientation2D
            The composed orientation.
        """
        composed_theta = self.theta + other.theta
        # Normalize to [-pi, pi]
        composed_theta = (composed_theta + math.pi) % (2 * math.pi) - math.pi
        return orientation_2d(angle=composed_theta, unit="rad", device=self.device, dtype=self.dtype)

    def apply_rotation(self, position: "position_2d") -> "position_2d":
        """
        Applies the orientation rotation to a Position2D object.

        Parameters
        ----------
        position : Position2D
            The position to rotate.

        Returns
        -------
        Position2D
            The rotated position.
        """
        if not isinstance(position, position_2d):
            raise TypeError("Input must be a Position2D object.")
        R = self.to_rotation_matrix()
        rotated_p = torch.mv(R, position.p)
        return position_2d(rotated_p)

    def to_euler_angles(self, unit: str = "rad") -> torch.Tensor:
        """
        Retrieves the orientation angle (theta) in the specified unit.

        Parameters
        ----------
        unit : str, optional
            The unit of the angle to retrieve ('rad' for radians or 'deg' for degrees).
            Default is 'rad'.

        Returns
        -------
        torch.Tensor
            The orientation angle in the specified unit.

        Raises
        ------
        ValueError
            If the unit is not 'rad' or 'deg'.
        """
        if unit.lower() == "deg":
            return torch.rad2deg(torch.tensor([self.theta], device=self.device, dtype=self.dtype))
        elif unit.lower() == "rad":
            return torch.tensor([self.theta], device=self.device, dtype=self.dtype)
        else:
            raise ValueError("unit must be either 'rad' or 'deg'")

    def to_list(self) -> List[float]:
        """
        Returns the angle as a list.

        Returns
        -------
        List[float]
            The angle in radians.
        """
        return [self.theta.item()]

    def to_tensor(self) -> torch.Tensor:
        """
        Returns the angle as a tensor.

        Returns
        -------
        torch.Tensor
            The angle in radians.
        """
        return torch.tensor([self.theta], device=self.device, dtype=self.dtype)

    def __repr__(self):
        return f"Orientation2D(theta={self.theta.item():.4f} rad)"


class orientation_3d:
    """
    Represents a 3D orientation using Euler angles (roll, pitch, yaw).
    Provides methods to convert to and from quaternions and rotation matrices.
    Supports input angles in radians or degrees.
    """

    def __init__(
        self,
        euler_angles: Union[List[float], torch.Tensor],
        unit: str = "rad",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initializes the orientation_3d instance with Euler angles.

        Parameters
        ----------
        euler_angles : List[float] or torch.Tensor
            A list or tensor of Euler angles [roll, pitch, yaw].
        unit : str, optional
            The unit of the Euler angles ('rad' for radians or 'deg' for degrees).
            Default is 'rad'.
        device : str, optional
            The device of the tensors (default is "cpu").
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Raises
        ------
        TypeError
            If euler_angles is not a list or torch.Tensor.
        ValueError
            If euler_angles does not contain exactly 3 elements.
            If unit is not 'rad' or 'deg'.
        """
        if isinstance(euler_angles, list):
            euler_angles = torch.tensor(euler_angles, device=device, dtype=dtype)
        elif isinstance(euler_angles, torch.Tensor):
            euler_angles = euler_angles.to(device=device, dtype=dtype)
        else:
            raise TypeError("euler_angles must be a list or a torch.Tensor")

        if euler_angles.shape != (3,):
            raise ValueError("euler_angles must be a 3-element list or tensor")

        if unit.lower() == "deg":
            euler_angles = torch.deg2rad(euler_angles)
        elif unit.lower() == "rad":
            pass  # Already in radians
        else:
            raise ValueError("unit must be either 'rad' or 'deg'")

        self.roll, self.pitch, self.yaw = euler_angles
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_quaternion(cls, quaternion: Union[List[float], torch.Tensor], device: str = "cpu",
                        dtype: torch.dtype = torch.float32) -> "orientation_3d":
        """
        Creates an orientation_3d instance from a quaternion.

        Parameters
        ----------
        quaternion : List[float] or torch.Tensor
            A quaternion [w, x, y, z].
        device : str, optional
            The device of the tensors (default is "cpu").
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Returns
        -------
        orientation_3d
            The corresponding orientation_3d instance.

        Raises
        ------
        TypeError
            If quaternion is not a list or torch.Tensor.
        ValueError
            If quaternion does not contain exactly 4 elements.
        """
        if isinstance(quaternion, list):
            quaternion = torch.tensor(quaternion, device=device, dtype=dtype)
        elif isinstance(quaternion, torch.Tensor):
            quaternion = quaternion.to(device=device, dtype=dtype)
        else:
            raise TypeError("quaternion must be a list or a torch.Tensor")

        if quaternion.shape != (4,):
            raise ValueError("quaternion must be a 4-element list or tensor [w, x, y, z]")

        quaternion = quaternion / quaternion.norm()  # Ensure unit quaternion
        roll, pitch, yaw = cls.quaternion_to_euler(quaternion)
        return cls(euler_angles=[roll.item(), pitch.item(), yaw.item()], unit="rad", device=device, dtype=dtype)

    @classmethod
    def from_rotation_matrix(cls, R: torch.Tensor, device: str = "cpu",
                             dtype: torch.dtype = torch.float32) -> "orientation_3d":
        """
        Creates an orientation_3d instance from a rotation matrix.

        Parameters
        ----------
        R : torch.Tensor
            A 3x3 rotation matrix.
        device : str, optional
            The device of the tensors (default is "cpu").
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Returns
        -------
        orientation_3d
            The corresponding orientation_3d instance.

        Raises
        ------
        TypeError
            If R is not a torch.Tensor.
        ValueError
            If R is not a 3x3 matrix.
        """
        if not isinstance(R, torch.Tensor):
            raise TypeError("R must be a torch.Tensor")
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix R must be 3x3")
        # Optional: Validate orthogonality and determinant
        # For simplicity, we'll assume R is a valid rotation matrix

        quaternion = cls.rotation_matrix_to_quaternion(R)
        return cls.from_quaternion(quaternion, device=device, dtype=dtype)

    def to_quaternion(self) -> torch.Tensor:
        """
        Converts the Euler angles to a quaternion.

        Returns
        -------
        torch.Tensor
            A quaternion [w, x, y, z].
        """
        return self.euler_to_quaternion(
            torch.tensor([self.roll, self.pitch, self.yaw], device=self.device, dtype=self.dtype))

    @staticmethod
    def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
        """
        Converts Euler angles to a quaternion.

        Parameters
        ----------
        euler : torch.Tensor
            A tensor of Euler angles [roll, pitch, yaw] in radians.

        Returns
        -------
        torch.Tensor
            A quaternion [w, x, y, z].
        """
        roll, pitch, yaw = euler
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z])

    @staticmethod
    def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
        """
        Converts a quaternion to Euler angles.

        Parameters
        ----------
        q : torch.Tensor
            A quaternion [w, x, y, z].

        Returns
        -------
        torch.Tensor
            Euler angles [roll, pitch, yaw] in radians.
        """
        w, x, y, z = q
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * (math.pi / 2),
            torch.asin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw])

    @staticmethod
    def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """
        Converts a rotation matrix to a quaternion.

        Parameters
        ----------
        R : torch.Tensor
            A 3x3 rotation matrix.

        Returns
        -------
        torch.Tensor
            A quaternion [w, x, y, z].
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        quat = torch.tensor([w, x, y, z], device=R.device, dtype=R.dtype)
        quat = quat / quat.norm()
        return quat

    def to_rotation_matrix(self) -> torch.Tensor:
        """
        Converts the Euler angles to a 3x3 rotation matrix.

        Returns
        -------
        torch.Tensor
            A 3x3 rotation matrix.
        """
        return self.quaternion_to_rotation_matrix(self.to_quaternion())

    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """
        Converts a quaternion to a rotation matrix.

        Parameters
        ----------
        q : torch.Tensor
            A quaternion [w, x, y, z].

        Returns
        -------
        torch.Tensor
            A 3x3 rotation matrix.
        """
        w, x, y, z = q
        R = torch.empty((3, 3), device=q.device, dtype=q.dtype)
        R[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        R[0, 1] = 2 * (x * y - z * w)
        R[0, 2] = 2 * (x * z + y * w)

        R[1, 0] = 2 * (x * y + z * w)
        R[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        R[1, 2] = 2 * (y * z - x * w)

        R[2, 0] = 2 * (x * z - y * w)
        R[2, 1] = 2 * (y * z + x * w)
        R[2, 2] = 1 - 2 * (x ** 2 + y ** 2)
        return R

    def inverse(self) -> "orientation_3d":
        """
        Computes the inverse (negative) of the current orientation.

        Returns
        -------
        orientation_3d
            The inverse orientation.
        """
        # In Euler angles, inverse rotation is achieved by negating the angles
        inverse_angles = torch.tensor([-self.roll, -self.pitch, -self.yaw], device=self.device, dtype=self.dtype)
        return orientation_3d(euler_angles=inverse_angles, unit="rad", device=self.device, dtype=self.dtype)

    def compose(self, other: "orientation_3d") -> "orientation_3d":
        """
        Composes this orientation with another orientation.

        Parameters
        ----------
        other : orientation_3d
            The other orientation to compose with.

        Returns
        -------
        orientation_3d
            The composed orientation.
        """
        # Convert both to quaternions
        q1 = self.to_quaternion()
        q2 = other.to_quaternion()
        # Multiply quaternions
        composed_quat = orientation_3d.quaternion_multiply(q1, q2)
        composed_quat = composed_quat / composed_quat.norm()
        # Convert back to Euler angles
        roll, pitch, yaw = orientation_3d.quaternion_to_euler(composed_quat)
        return orientation_3d(euler_angles=[roll.item(), pitch.item(), yaw.item()], unit="rad", device=self.device,
                             dtype=self.dtype)

    @staticmethod
    def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiplies two quaternions.

        Parameters
        ----------
        q1 : torch.Tensor
            First quaternion [w1, x1, y1, z1].
        q2 : torch.Tensor
            Second quaternion [w2, x2, y2, z2].

        Returns
        -------
        torch.Tensor
            The product quaternion [w, x, y, z].
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z])

    def apply_rotation(self, position: "position_3d") -> "position_3d":
        """
        Applies the orientation rotation to a Position3D object.

        Parameters
        ----------
        position : Position3D
            The position to rotate.

        Returns
        -------
        Position3D
            The rotated position.
        """
        if not isinstance(position, position_3d):
            raise TypeError("Input must be a Position3D object")
        rotated_vector = self.rotate_vector(position.p)
        return position_3d(*rotated_vector.tolist(), device=self.device, dtype=self.dtype)

    def rotate_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Rotates a 3D vector using the current orientation.

        Parameters
        ----------
        vector : torch.Tensor
            A 3-element tensor representing the vector [x, y, z].

        Returns
        -------
        torch.Tensor
            The rotated vector [x', y', z'].
        """
        q = self.to_quaternion()
        q_conj = orientation_3d.quaternion_conjugate(q)
        v = torch.cat([torch.tensor([0.0], device=self.device, dtype=self.dtype), vector])
        qv = orientation_3d.quaternion_multiply(q, v)
        rotated_v = orientation_3d.quaternion_multiply(qv, q_conj)
        return rotated_v[1:]

    @staticmethod
    def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
        """
        Computes the conjugate of a quaternion.

        Parameters
        ----------
        q : torch.Tensor
            A quaternion [w, x, y, z].

        Returns
        -------
        torch.Tensor
            The conjugate quaternion [w, -x, -y, -z].
        """
        w, x, y, z = q
        return torch.tensor([w, -x, -y, -z], device=q.device, dtype=q.dtype)

    def to_list(self) -> List[float]:
        """
        Returns the Euler angles as a list [roll, pitch, yaw].

        Returns
        -------
        List[float]
            The Euler angles in radians.
        """
        return [self.roll.item(), self.pitch.item(), self.yaw.item()]

    def to_tensor(self) -> torch.Tensor:
        """
        Returns the Euler angles as a tensor [roll, pitch, yaw].

        Returns
        -------
        torch.Tensor
            The Euler angles in radians.
        """
        return torch.tensor([self.roll, self.pitch, self.yaw], device=self.device, dtype=self.dtype).clone()

    def to_euler_angles(self, unit: str = "rad") -> torch.Tensor:
        """
        Returns the Euler angles in the specified unit.

        Parameters
        ----------
        unit : str, optional
            The unit of the output angles ("rad" or "deg").

        Returns
        -------
        torch.Tensor
            Euler angles [roll, pitch, yaw] in the specified unit.

        Raises
        ------
        NotImplementedError
            If an unsupported unit is specified.
        """
        euler = torch.tensor([self.roll, self.pitch, self.yaw], device=self.device, dtype=self.dtype)
        if unit.lower() == "rad":
            return euler
        elif unit.lower() == "deg":
            return torch.rad2deg(euler)
        else:
            raise NotImplementedError("Only 'rad' and 'deg' units are supported")

    def normalize(self) -> "orientation_3d":
        """
        Normalizes the orientation (ensures quaternion is unit length).

        Returns
        -------
        orientation_3d
            The normalized orientation.
        """
        # Since Euler angles are angle-based, normalization isn't directly applicable.
        # However, ensuring the underlying quaternion is normalized is handled in conversion methods.
        return self

    def slerp(self, other: "orientation_3d", t: float) -> "orientation_3d":
        """
        Performs Spherical Linear Interpolation (SLERP) between two orientations.

        Parameters
        ----------
        other : orientation_3d
            The target orientation.
        t : float
            Interpolation parameter between 0 and 1.

        Returns
        -------
        orientation_3d
            The interpolated orientation.

        Raises
        ------
        ValueError
            If t is not in the range [0, 1].
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("Interpolation parameter t must be between 0 and 1")

        q1 = self.to_quaternion()
        q2 = other.to_quaternion()

        interpolated_quat = orientation_3d.slerp_quaternions(q1, q2, t)
        interpolated_quat = interpolated_quat / interpolated_quat.norm()

        roll, pitch, yaw = orientation_3d.quaternion_to_euler(interpolated_quat)
        return orientation_3d(euler_angles=[roll.item(), pitch.item(), yaw.item()], unit="rad", device=self.device,
                             dtype=self.dtype)

    @staticmethod
    def slerp_quaternions(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
        """
        Performs Spherical Linear Interpolation (SLERP) between two quaternions.

        Parameters
        ----------
        q1 : torch.Tensor
            Starting quaternion [w1, x1, y1, z1].
        q2 : torch.Tensor
            Ending quaternion [w2, x2, y2, z2].
        t : float
            Interpolation parameter between 0 and 1.

        Returns
        -------
        torch.Tensor
            The interpolated quaternion [w, x, y, z].
        """
        dot = torch.dot(q1, q2)

        # If the dot product is negative, SLERP won't take the shorter path.
        # Fix by reversing one quaternion.
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            # Quaternions are very close, use linear interpolation
            result = q1 + t * (q2 - q1)
            return result / result.norm()

        # Compute the angle between the quaternions
        theta_0 = torch.acos(dot)
        theta = theta_0 * t

        sin_theta = torch.sin(theta)
        sin_theta_0 = torch.sin(theta_0)

        s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return (s1 * q1) + (s2 * q2)

    def delta(self, other: "orientation_3d") -> torch.Tensor:
        """
        Computes the differential rotation vector between this orientation and another.

        Parameters
        ----------
        other : orientation_3d
            The target orientation.

        Returns
        -------
        torch.Tensor
            The rotation vector representing the difference [x, y, z].
        """
        relative_orientation = self.inverse().compose(other)
        return relative_orientation.log()

    def log(self) -> torch.Tensor:
        """
        Computes the logarithm map of the quaternion, returning the rotation vector.

        Returns
        -------
        torch.Tensor
            A 3-element tensor representing the rotation vector [x, y, z].
        """
        q = self.to_quaternion()
        return orientation_3d.quaternion_log_map(q)

    @staticmethod
    def quaternion_log_map(q: torch.Tensor) -> torch.Tensor:
        """
        Computes the logarithm map of a quaternion.

        Parameters
        ----------
        q : torch.Tensor
            A unit quaternion [w, x, y, z].

        Returns
        -------
        torch.Tensor
            The rotation vector [x, y, z].
        """
        w, x, y, z = q
        norm_v = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = 2 * torch.acos(w)
        if norm_v < 1e-8:
            return torch.zeros(3, device=q.device, dtype=q.dtype)
        else:
            return (theta / norm_v) * torch.tensor([x, y, z], device=q.device, dtype=q.dtype)

    @classmethod
    def exp(cls, rotation_vector: torch.Tensor, device: str = "cpu",
            dtype: torch.dtype = torch.float32) -> "orientation_3d":
        """
        Computes the exponential map, converting a rotation vector to Euler angles.

        Parameters
        ----------
        rotation_vector : torch.Tensor
            A 3-element tensor representing the rotation vector [x, y, z].
        device : str, optional
            The device of the tensors (default is "cpu").
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Returns
        -------
        orientation_3d
            The corresponding orientation_3d instance.
        """
        quaternion = cls.exp_quaternion(rotation_vector, device=device, dtype=dtype)
        roll, pitch, yaw = cls.quaternion_to_euler(quaternion)
        return cls(euler_angles=[roll.item(), pitch.item(), yaw.item()], unit="rad", device=device, dtype=dtype)

    @staticmethod
    def exp_quaternion(rotation_vector: torch.Tensor, device: str = "cpu",
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Computes the exponential map, converting a rotation vector to a quaternion.

        Parameters
        ----------
        rotation_vector : torch.Tensor
            A 3-element tensor representing the rotation vector [x, y, z].
        device : str, optional
            The device of the tensors (default is "cpu").
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Returns
        -------
        torch.Tensor
            The corresponding quaternion [w, x, y, z].
        """
        theta = torch.norm(rotation_vector)
        if theta < 1e-8:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        else:
            axis = rotation_vector / theta
            half_theta = theta / 2.0
            w = torch.cos(half_theta)
            xyz = torch.sin(half_theta) * axis
            return torch.cat([w.unsqueeze(0), xyz])

    def adjoint(self) -> torch.Tensor:
        """
        Computes the adjoint matrix of the current quaternion.

        Returns
        -------
        torch.Tensor
            A 4x4 adjoint matrix.
        """
        q = self.to_quaternion()
        return orientation_3d.quaternion_adjoint(q)

    @staticmethod
    def quaternion_adjoint(q: torch.Tensor) -> torch.Tensor:
        """
        Computes the adjoint matrix of a quaternion.

        Parameters
        ----------
        q : torch.Tensor
            A quaternion [w, x, y, z].

        Returns
        -------
        torch.Tensor
            A 4x4 adjoint matrix.
        """
        w, x, y, z = q
        adj = torch.tensor([
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w]
        ], device=q.device, dtype=q.dtype)
        return adj

    def jacobian(self) -> torch.Tensor:
        """
        Computes the Jacobian matrix for quaternion-based orientation.

        Returns
        -------
        torch.Tensor
            The Jacobian matrix, equivalent to the adjoint matrix.
        """
        return self.adjoint()

    def to_euler_angles_formatted(self, unit: str = "rad") -> str:
        """
        Returns a formatted string of the Euler angles in the specified unit.

        Parameters
        ----------
        unit : str, optional
            The unit of the output angles ("rad" or "deg").

        Returns
        -------
        str
            Formatted Euler angles.

        Raises
        ------
        NotImplementedError
            If an unsupported unit is specified.
        """
        angles = self.to_euler_angles(unit=unit)
        if unit.lower() == "rad":
            unit_symbol = "rad"
        elif unit.lower() == "deg":
            unit_symbol = "deg"
        else:
            unit_symbol = ""
        return f"orientation_3d(roll={angles[0].item():.4f} {unit_symbol}, pitch={angles[1].item():.4f} {unit_symbol}, yaw={angles[2].item():.4f} {unit_symbol})"

    def __repr__(self):
        return self.to_euler_angles_formatted(unit="rad")


if __name__ == "__main__":
    # Define Euler angles: 30° roll, 45° pitch, 60° yaw
    roll_deg = 30
    pitch_deg = 45
    yaw_deg = 60
    euler_angles_deg = [roll_deg, pitch_deg, yaw_deg]

    # Create orientation_3d instance with degrees
    orientation_deg = orientation_3d(euler_angles=euler_angles_deg, unit="deg")
    print("Orientation (Degrees):", orientation_deg)

    # Retrieve Euler angles in radians
    euler_rad = orientation_deg.to_euler_angles(unit="rad")
    print("Euler Angles (Radians):", euler_rad)

    # Retrieve Euler angles in degrees
    euler_deg = orientation_deg.to_euler_angles(unit="deg")
    print("Euler Angles (Degrees):", euler_deg)

    # Convert to quaternion
    quaternion = orientation_deg.to_quaternion()
    print("Quaternion:", quaternion)

    # Convert to rotation matrix
    rotation_matrix = orientation_deg.to_rotation_matrix()
    print("Rotation Matrix:\n", rotation_matrix)

    # Initialize from quaternion
    orientation_from_quat = orientation_3d.from_quaternion(quaternion)
    print("Orientation from Quaternion:", orientation_from_quat)

    # Define a position at (1, 0, 0)
    position = position_3d(1.0, 0.0, 0.0)
    print("Original Position:", position)

    # Apply rotation
    rotated_position = orientation_deg.apply_rotation(position)
    print("Rotated Position:", rotated_position)

    # Interpolate between two orientations
    # Define another orientation: 15° roll, 30° pitch, 45° yaw
    orientation2_deg = orientation_3d(euler_angles=[15, 30, 45], unit="deg")
    print("Second Orientation (Degrees):", orientation2_deg)

    # Perform SLERP with t=0.5
    interpolated_orientation = orientation_deg.slerp(orientation2_deg, t=0.5)
    print("Interpolated Orientation:", interpolated_orientation)

    # Retrieve interpolated Euler angles in degrees
    interpolated_euler_deg = interpolated_orientation.to_euler_angles(unit="deg")
    print("Interpolated Euler Angles (Degrees):", interpolated_euler_deg)