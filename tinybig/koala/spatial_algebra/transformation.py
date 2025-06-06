# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
# Orientation Class #
#####################

import math
import torch
from typing import List, Union

from tinybig.koala.spatial_algebra.space import space
from tinybig.koala.spatial_algebra.position import position, position_2d, position_3d
from tinybig.koala.spatial_algebra.orientation import orientation_2d, orientation_3d
from tinybig.koala.spatial_algebra.twist import twist, twist_2d, twist_3d


class transformation(space):
    """
    Represents a spatial transformation consisting of translation and rotation.

    This is an abstract base class and should not be instantiated directly.
    Use transformation2D or transformation3D subclasses instead.
    """

    def __init__(
        self,
        translation: Union["position_2d", "position_3d"],
        rotation: Union["orientation_2d", "orientation_3d"],
        name: str = 'transformation', dim: int = 3,
        device: str = 'cpu', dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the transformation instance with translation and rotation.

        Parameters
        ----------
        translation : position_2d or position_3d
            The translation component of the transformation.
        rotation : orientation_2d or orientation_3d
            The rotation component of the transformation.

        Raises
        ------
        TypeError
            If translation or rotation are not instances of position_2d/position_3d or orientation_2d/orientation_3d.
        ValueError
            If the dimensions of translation and rotation do not match.
        """
        super().__init__(name=name, dim=dim, device=device, dtype=dtype)

        if not isinstance(translation, (position_2d, position_3d)):
            raise TypeError("translation must be an instance of position_2d or position_3d.")
        if not isinstance(rotation, (orientation_2d, orientation_3d)):
            raise TypeError("rotation must be an instance of orientation_2d or orientation_3d.")

        # Check dimension consistency
        if isinstance(rotation, orientation_2d) and translation.dim != 2:
            raise ValueError("Dimension mismatch: position_2d with orientation_2d required.")
        if isinstance(rotation, orientation_3d) and translation.dim != 3:
            raise ValueError("Dimension mismatch: position_3d with orientation_3d required.")

        self.translation = translation
        self.rotation = rotation

    def compose(self, other: "transformation") -> "transformation":
        """
        Composes this transformation with another transformation.

        Parameters
        ----------
        other : transformation
            The other transformation to compose with.

        Returns
        -------
        transformation
            The composed transformation.

        Raises
        ------
        ValueError
            If the dimensions of the transformations do not match.
        """
        if self.translation.dim != other.translation.dim:
            raise ValueError("Cannot compose transformations with different dimensions.")

        # Compose rotations
        if isinstance(self.rotation, orientation_2d) and isinstance(other.rotation, orientation_2d):
            composed_rotation = self.rotation.compose(other.rotation)
        elif isinstance(self.rotation, orientation_3d) and isinstance(other.rotation, orientation_3d):
            composed_rotation = self.rotation.compose(other.rotation)
        else:
            raise ValueError("Cannot compose transformations with different rotation types.")

        # Rotate the other's translation by this rotation and add to this translation
        rotated_translation = self.rotation.apply_rotation(other.translation)
        composed_translation = self.translation + rotated_translation

        if isinstance(self, transformation2D):
            return transformation2D(composed_translation, composed_rotation)
        elif isinstance(self, transformation3D):
            return transformation3D(composed_translation, composed_rotation)
        else:
            raise TypeError("Unsupported transformation subclass.")

    def inverse(self) -> "transformation":
        """
        Computes the inverse of the transformation.

        Returns
        -------
        transformation
            The inverse transformation.
        """
        inverse_rotation = self.rotation.inverse()

        if isinstance(self.translation, position_2d):
            inverse_translation = inverse_rotation.apply_rotation(position_2d(*(-self.translation.p).tolist(), device=self.translation.device, dtype=self.translation.dtype))
        elif isinstance(self.translation, position_3d):
            inverse_translation = inverse_rotation.apply_rotation(position_3d(*(-self.translation.p).tolist(), device=self.translation.device, dtype=self.translation.dtype))
        else:
            raise TypeError("Unsupported Translation subclass.")

        if isinstance(self, transformation2D):
            return transformation2D(inverse_translation, inverse_rotation)
        elif isinstance(self, transformation3D):
            return transformation3D(inverse_translation, inverse_rotation)
        else:
            raise TypeError("Unsupported transformation subclass.")

    def apply(self, position: Union["position_2d", "position_3d"]) -> Union["Position", "position_2d", "position_3d"]:
        """
        Applies the transformation to a given position.

        Parameters
        ----------
        position : position_2d or position_3d
            The position to transform.

        Returns
        -------
        position_2d or position_3d
            The transformed position.

        Raises
        ------
        ValueError
            If the dimensions of the position do not match the transformation.
        """
        if self.translation.dim != position.dim:
            raise ValueError("Cannot apply transformation to position with different dimensions.")

        # Rotate the position and then translate
        rotated_position = self.rotation.apply_rotation(position)
        transformed_position = rotated_position + self.translation
        return transformed_position

    def to_matrix(self) -> torch.Tensor:
        """
        Converts the transformation to a homogeneous transformation matrix.

        Returns
        -------
        torch.Tensor
            A 3x3 (2D) or 4x4 (3D) homogeneous transformation matrix.
        """
        if isinstance(self, transformation2D):
            R = self.rotation.to_rotation_matrix()
            t = self.translation.p
            T = torch.eye(3, device=self.device, dtype=self.dtype)
            T[:2, :2] = R
            T[:2, 2] = t
            return T
        elif isinstance(self, transformation3D):
            R = self.rotation.to_rotation_matrix()
            t = self.translation.p
            T = torch.eye(4, device=self.device, dtype=self.dtype)
            T[:3, :3] = R
            T[:3, 3] = t
            return T
        else:
            raise TypeError("Unsupported transformation subclass.")

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> "transformation":
        """
        Creates a transformation instance from a homogeneous transformation matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            A 3x3 (2D) or 4x4 (3D) homogeneous transformation matrix.

        Returns
        -------
        transformation2D or transformation3D
            The corresponding transformation instance.

        Raises
        ------
        ValueError
            If the matrix dimensions are not 3x3 or 4x4.
        """
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("matrix must be a torch.Tensor.")

        if matrix.shape == (3, 3):
            # 2D transformation
            R = matrix[:2, :2]
            t = matrix[:2, 2]
            rotation = orientation_2d.from_rotation_matrix(R)
            translation = position_2d(t)
            return transformation2D(translation, rotation)
        elif matrix.shape == (4, 4):
            # 3D transformation
            R = matrix[:3, :3]
            t = matrix[:3, 3]
            rotation = orientation_3d.from_rotation_matrix(R)
            translation = position_3d(t)
            return transformation3D(translation, rotation)
        else:
            raise ValueError("transformation matrix must be either 3x3 (2D) or 4x4 (3D).")

    def __repr__(self):
        return f"transformation(translation={self.translation}, rotation={self.rotation})"


class transformation2D(transformation):
    """
    Represents a 2D spatial transformation consisting of translation and rotation.

    Inherits from the transformation base class.
    """

    def __init__(
        self,
        translation: Union["position_2d", List[float], torch.Tensor],
        rotation: Union["orientation_2d", float, torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initializes the transformation2D instance with translation and rotation.

        Parameters
        ----------
        translation : position_2d, list, or torch.Tensor
            The translation component.
            - As position_2d instance.
            - As a list: [x, y].
            - As a torch.Tensor: tensor containing [x, y].
        rotation : orientation_2d, float, list, or torch.Tensor
            The rotation component.
            - As orientation_2d instance.
            - As a float: theta in radians.
            - As a list: [theta].
            - As a torch.Tensor: tensor containing [theta].
        device : str, optional
            The device on which tensors will be allocated.
            Default is 'cpu'.
        dtype : torch.dtype, optional
            The data type of tensors.
            Default is torch.float32.

        Raises
        ------
        TypeError
            If translation or rotation inputs are of incorrect types.
        ValueError
            If the translation or rotation inputs have incorrect dimensions.
        """
        # Handle translation input
        if isinstance(translation, position_2d):
            translation_obj = translation
        else:
            translation_obj = position_2d(translation.as_tensor(), device=device, dtype=dtype)

        # Handle rotation input
        if isinstance(rotation, orientation_2d):
            rotation_obj = rotation
        elif isinstance(rotation, (float, int)):
            rotation_obj = orientation_2d(angle=rotation, unit="rad", device=device, dtype=dtype)
        elif isinstance(rotation, torch.Tensor):
            rotation_obj = orientation_2d(rotation, device=device, dtype=dtype)
        else:
            raise TypeError("rotation must be an instance of orientation_2d, float, list, or torch.Tensor.")

        super().__init__(translation_obj, rotation_obj)

    def __repr__(self):
        return f"transformation2D(translation={self.translation}, rotation={self.rotation})"


class transformation3D(transformation):
    """
    Represents a 3D spatial transformation consisting of translation and rotation.

    Inherits from the transformation base class.
    """

    def __init__(
        self,
        translation: Union["position_3d", List[float], torch.Tensor],
        rotation: Union["orientation_3d", List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initializes the transformation3D instance with translation and rotation.

        Parameters
        ----------
        translation : position_3d, list, or torch.Tensor
            The translation component.
            - As position_3d instance.
            - As a list: [x, y, z].
            - As a torch.Tensor: tensor containing [x, y, z].
        rotation : orientation_3d, list, or torch.Tensor
            The rotation component.
            - As orientation_3d instance.
            - As a list: [roll, pitch, yaw].
            - As a torch.Tensor: tensor containing [roll, pitch, yaw].
        device : str, optional
            The device on which tensors will be allocated.
            Default is 'cpu'.
        dtype : torch.dtype, optional
            The data type of tensors.
            Default is torch.float32.

        Raises
        ------
        TypeError
            If translation or rotation inputs are of incorrect types.
        ValueError
            If the translation or rotation inputs have incorrect dimensions.
        """
        # Handle translation input
        if isinstance(translation, position_3d):
            translation_obj = translation
        else:
            translation_obj = position_3d(translation.as_tensor(), device=device, dtype=dtype)

        # Handle rotation input
        if isinstance(rotation, orientation_3d):
            rotation_obj = rotation
        elif isinstance(rotation, list):
            if len(rotation) != 3:
                raise ValueError("orientation_3d requires three angles [roll, pitch, yaw].")
            rotation_obj = orientation_3d(euler_angles=rotation, unit="rad", device=device, dtype=dtype)
        elif isinstance(rotation, torch.Tensor):
            if rotation.numel() != 3:
                raise ValueError("orientation_3d requires three angles [roll, pitch, yaw].")
            rotation_obj = orientation_3d(euler_angles=rotation, unit="rad", device=device, dtype=dtype)
        else:
            raise TypeError("rotation must be an instance of orientation_3d, list, or torch.Tensor.")

        super().__init__(translation_obj, rotation_obj)

    def __repr__(self):
        return f"transformation3D(translation={self.translation}, rotation={self.rotation})"


if __name__ == "__main__":
    # -------------------
    # 3D transformation Example
    # -------------------
    print("=== 3D transformation Example ===")
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

    # Define a transformation3D: translation and rotation
    translation = position_3d(1.0, 2.0, 3.0)
    transformation = transformation3D(translation, orientation_deg)
    print("transformation3D:", transformation)

    # Apply transformation3D to a position_3d
    position_to_transform = position_3d(4.0, 5.0, 6.0)
    transformed_position = transformation.apply(position_to_transform)
    print("Transformed Position:", transformed_position)

    # Invert the transformation3D
    inverse_transformation = transformation.inverse()
    print("Inverse transformation3D:", inverse_transformation)

    # Compose two transformation3D instances
    # Define another transformation3D
    translation2 = position_3d(0.5, 1.0, 1.5)
    orientation2 = orientation_3d(euler_angles=[15, 30, 45], unit="deg")
    transformation2 = transformation3D(translation2, orientation2)
    print("transformation3D_2:", transformation2)

    # Compose transformations
    composed_transformation = transformation.compose(transformation2)
    print("Composed transformation3D:", composed_transformation)

    # Retrieve composed Euler angles in degrees
    composed_euler_deg = composed_transformation.rotation.to_euler_angles(unit="deg")
    print("Composed Euler Angles (Degrees):", composed_euler_deg)

    # -------------------
    # 2D transformation Example
    # -------------------
    print("\n=== 2D transformation Example ===")
    # Define angle: 45° rotation
    theta_deg = 45
    theta_rad = math.radians(theta_deg)

    # Create orientation_2d instance with degrees
    orientation_2d = orientation_2d(angle=theta_deg, unit="deg")
    print("orientation_2d (Degrees):", orientation_2d)
    # Output: orientation_2d (Degrees): orientation_2d(theta=0.7854 rad)

    # Retrieve angle in radians
    theta_retrieved_rad = orientation_2d.to_euler_angles(unit="rad")
    print("Theta (Radians):", theta_retrieved_rad)
    # Expected Output: Theta (Radians): tensor([0.7854], dtype=torch.float32)

    # Retrieve angle in degrees
    theta_retrieved_deg = orientation_2d.to_euler_angles(unit="deg")
    print("Theta (Degrees):", theta_retrieved_deg)
    # Expected Output: Theta (Degrees): tensor([45.], dtype=torch.float32)

    # Convert to quaternion
    quaternion2d = orientation_2d.to_quaternion()
    print("Quaternion2D:", quaternion2d)
    # Expected Output: Quaternion2D: tensor([0.7071, 0.7071, 0.0000, 0.0000], dtype=torch.float32)

    # Convert to rotation matrix
    rotation_matrix2d = orientation_2d.to_rotation_matrix()
    print("Rotation Matrix 2D:\n", rotation_matrix2d)
    # Expected Output:
    # Rotation Matrix 2D:
    #  tensor([[ 0.7071, -0.7071],
    #          [ 0.7071,  0.7071]], dtype=torch.float32)

    # Initialize from quaternion
    orientation_2d_from_quat = orientation_2d.from_quaternion(quaternion2d)
    print("orientation_2d from Quaternion:", orientation_2d_from_quat)
    # Expected Output: orientation_2d from Quaternion: orientation_2d(theta=0.7854 rad)

    # Define a position at (1, 0)
    position_2d = position_2d(1.0, 0.0)
    print("Original position_2d:", position_2d)
    # Expected Output: Original position_2d: position_2d(x=1.0000, y=0.0000)

    # Apply rotation
    rotated_position_2d = orientation_2d.apply_rotation(position_2d)
    print("Rotated position_2d:", rotated_position_2d)
    # Expected Output: Rotated position_2d: position_2d(x=0.7071, y=0.7071)

    # Define a transformation2D: translation and rotation
    translation2d = position_2d(2.0, 3.0)
    transformation2d = transformation2D(translation2d, orientation_2d)
    print("transformation2D:", transformation2d)
    # Expected Output: transformation2D(translation=position_2d(x=2.0000, y=3.0000), rotation=orientation_2d(theta=0.7854 rad))

    # Apply transformation2D to a position_2d
    position_to_transform2d = position_2d(4.0, 5.0)
    transformed_position_2d = transformation2d.apply(position_to_transform2d)
    print("Transformed position_2d:", transformed_position_2d)
    # Expected Output: Transformed position_2d: position_2d(x=4.7071, y=5.7071)

    # Invert the transformation2D
    inverse_transformation2d = transformation2d.inverse()
    print("Inverse transformation2D:", inverse_transformation2d)
    # Expected Output: Inverse transformation2D(translation=position_2d(x=-2.0000, y=-3.0000), rotation=orientation_2d(theta=-0.7854 rad))

    # Compose two transformation2D instances
    # Define another transformation2D
    translation2d_2 = position_2d(1.0, 1.0)
    orientation_2d_2 = orientation_2d(angle=30, unit="deg")
    transformation2d_2 = transformation2D(translation2d_2, orientation_2d_2)
    print("transformation2D_2:", transformation2d_2)
    # Expected Output: transformation2D_2(translation=position_2d(x=1.0000, y=1.0000), rotation=orientation_2d(theta=0.5236 rad))

    # Compose transformations
    composed_transformation2d = transformation2d.compose(transformation2d_2)
    print("Composed transformation2D:", composed_transformation2d)
    # Expected Output: Composed transformation2D(translation=position_2d(x=2.8660, y=4.2321), rotation=orientation_2d(theta=1.3090 rad))

    # Retrieve composed theta in degrees
    composed_theta_deg = composed_transformation2d.rotation.to_euler_angles(unit="deg")
    print("Composed Theta (Degrees):", composed_theta_deg)
    # Expected Output: Composed Theta (Degrees): tensor([75.0000], dtype=torch.float32)