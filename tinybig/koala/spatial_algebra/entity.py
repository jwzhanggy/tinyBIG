# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Spatial Entity Classes #
##########################

import torch
from typing import Union

from tinybig.koala.spatial_algebra.space import spatial_base
from tinybig.koala.spatial_algebra.position import position_2d, position_3d
from tinybig.koala.spatial_algebra.orientation import orientation_2d, orientation_3d
from tinybig.koala.spatial_algebra.motion import velocity_2d, velocity_3d, angular_velocity_2d, angular_velocity_3d
from tinybig.koala.spatial_algebra.action import force_2d, force_3d, torque_2d, torque_3d


class entity:
    """
    Abstract base class representing a physical entity with mass and spatial attributes.

    This class should not be instantiated directly. Use Entity2D or Entity3D subclasses instead.
    """

    def __init__(
        self,
        name: str = 'entity',
        mass: float = None,
        dim: int = 3,
        position: Union[position_2d, position_3d] = None,
        orientation: Union[orientation_2d, orientation_3d] = None,
        velocity: Union[velocity_2d, velocity_3d] = None,
        angular_velocity: Union[angular_velocity_2d, angular_velocity_3d] = None,
        device: str = 'cpu', dtype: torch.dtype = torch.float32,
        *args, **kwargs
    ):
        """
        Initializes the Entity instance.

        Parameters
        ----------
        name: str
            The name of the entity. Default is 'entity'.
        mass : float
            The mass of the entity. Default is None.
        position : Position2D or Position3D
            The spatial position of the entity. Default is None.
        orientation : Orientation2D or Orientation3D
            The spatial orientation of the entity. Default is None.
        velocity : Velocity2D or Velocity3D
            The linear velocity of the entity. Default is None.
        angular_velocity : AngularVelocity2D or AngularVelocity3D
            The angular velocity of the entity. Default is None.
        device : str
            The device of the entity. Default is 'cpu'.
        """
        if mass is not None and mass <= 0:
            raise ValueError("Mass must be a positive value.")

        self.name = name
        self.dim = dim
        self.device = device
        self.dtype = dtype

        self.mass = mass

        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self._force = self._initialize_force()
        self._torque = self._initialize_torque()

    def _initialize_force(self) -> Union[force_2d, force_3d]:
        """
        Initializes the force accumulator based on the entity's dimension.

        Returns
        -------
        Force2D or Force3D
            The initialized force.
        """
        if self.dim == 2:
            return force_2d(fx=0.0, fy=0.0)
        elif self.dim == 3:
            return force_3d(fx=0.0, fy=0.0, fz=0.0)
        else:
            raise ValueError("Unsupported dimension.")

    def _initialize_torque(self) -> Union[torque_2d, torque_3d]:
        """
        Initializes the torque accumulator based on the entity's dimension.

        Returns
        -------
        Torque2D or Torque3D
            The initialized torque.
        """
        if self.dim == 2:
            return torque_2d(tau=0.0)
        elif self.dim == 3:
            return torque_3d(tau_x=0.0, tau_y=0.0, tau_z=0.0)
        else:
            raise ValueError("Unsupported dimension.")

    def apply_force(self, force: Union[force_2d, force_3d]) -> None:
        """
        Applies an external force to the entity.

        Parameters
        ----------
        force : Force2D or Force3D
            The force to apply.
        """
        if self.dim == 2 and not isinstance(force, force_2d):
            raise TypeError("Expected Force2D for a 2D entity.")
        if self.dim == 3 and not isinstance(force, force_3d):
            raise TypeError("Expected Force3D for a 3D entity.")
        self.force += force

    def apply_torque(self, torque: Union[torque_2d, torque_3d]) -> None:
        """
        Applies an external torque to the entity.

        Parameters
        ----------
        torque : Torque2D or Torque3D
            The torque to apply.
        """
        if self.dim == 2 and not isinstance(torque, torque_2d):
            raise TypeError("Expected torque_2d for a 2D entity.")
        if self.dim == 3 and not isinstance(torque, torque_3d):
            raise TypeError("Expected torque_3d for a 3D entity.")
        self.torque += torque

    def compute_acceleration(self) -> Union[torch.Tensor, torch.Tensor]:
        """
        Computes the linear and angular acceleration based on applied forces and torques.

        Returns
        -------
        torch.Tensor
            Linear acceleration vector.
        torch.Tensor
            Angular acceleration vector.
        """
        if self.dim == 2:
            # For 2D: linear acceleration = force / mass
            linear_acc = self.force.to_tensor() / self.mass
            # Angular acceleration = torque / moment of inertia
            # Assuming moment of inertia I for a point mass: I = mass * r^2
            # For simplicity, assuming I = mass (unit moment of inertia)
            angular_acc = self.torque.to_tensor() / self.mass
            return linear_acc, angular_acc
        elif self.dim == 3:
            # For 3D: linear acceleration = force / mass
            linear_acc = self.force.to_tensor() / self.mass
            # Angular acceleration = torque / moment of inertia
            # Assuming moment of inertia I for a point mass: I = mass * r^2
            # For simplicity, assuming diagonal inertia tensor with Ixx=Iyy=Izz=mass
            angular_acc = self.torque.to_tensor() / self.mass
            return linear_acc, angular_acc
        else:
            raise ValueError("Unsupported dimension.")

    def update_state(self, dt: float) -> None:
        """
        Updates the entity's velocity and position based on applied forces and torques.

        Parameters
        ----------
        dt : float
            The time step over which to update the state.
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        linear_acc, angular_acc = self.compute_acceleration()

        # Update velocities
        self.velocity += linear_acc * dt
        self.angular_velocity += angular_acc * dt

        # Update positions
        self.position += self.velocity * dt
        self.orientation = self.orientation.compose(
            self.angular_velocity_to_orientation_change(angular_acc, dt)
        )

        # Reset forces and torques after update
        self.reset_force_and_torque()

    def angular_velocity_to_orientation_change(self, angular_acc: torch.Tensor, dt: float) -> Union[orientation_2d, orientation_3d]:
        """
        Converts angular acceleration to an orientation change over the time step.

        Parameters
        ----------
        angular_acc : torch.Tensor
            Angular acceleration vector.
        dt : float
            Time step.

        Returns
        -------
        Orientation2D or Orientation3D
            The orientation change.
        """
        if self.dim == 2:
            # For 2D: scalar angular acceleration
            delta_theta = angular_acc.item() * dt + 0.5 * self.angular_velocity.omega.item() * dt ** 2
            return orientation_2d(angle=delta_theta, unit="rad", device=self.orientation.theta.device,
                                 dtype=self.orientation.theta.dtype)
        elif self.dim == 3:
            # For 3D: vector angular acceleration
            delta_euler = (angular_acc * dt) + (0.5 * self.angular_velocity.to_tensor() * dt ** 2)
            return orientation_3d(
                euler_angles=delta_euler.tolist(),
                unit="rad",
                device=self.orientation.rotation_matrix.device,
                dtype=self.orientation.rotation_matrix.dtype
            )
        else:
            raise ValueError("Unsupported dimension.")

    def reset_force_and_torque(self) -> None:
        """
        Resets the accumulated forces and torques to zero.
        """
        self.force = self._initialize_force()
        self.torque = self._initialize_torque()

    def __repr__(self):
        """
        Returns the string representation of the Entity instance.

        Returns
        -------
        str
            String representation.
        """
        return (f"Entity(dim={self.dim}, mass={self.mass}, "
                f"position={self.position}, orientation={self.orientation}, "
                f"velocity={self.velocity}, angular_velocity={self.angular_velocity}, "
                f"force={self.force}, torque={self.torque})")


class entity_2d(entity):
    """
    Represents a 2D physical entity with mass and spatial attributes.
    """

    def __init__(
        self,
        name: str = 'entity2d',
        mass: float = None,
        position: position_2d = None,
        orientation: orientation_2d = None,
        velocity: velocity_2d = None,
        angular_velocity: angular_velocity_2d = None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
        Initializes the Entity2D instance.

        Parameters
        ----------
        name: str
            The name of the entity. Default is 'entity_2d'.
        mass : float
            The mass of the entity. Default is None.
        position : position_2d
            The spatial position of the entity. Default is None.
        orientation : orientation_2d
            The spatial orientation of the entity. Default is None.
        velocity : velocity_2d
            The linear velocity of the entity. Default is None.
        angular_velocity : angular_velocity_2d
            The angular velocity of the entity. Default is None.
        device : str
            The device of the entity. Default is 'cpu'.
        """
        super().__init__(
            name=name, mass=mass, position=position, orientation=orientation,
            velocity=velocity, angular_velocity=angular_velocity, device=device
        )


class entity_3d(entity):
    """
    Represents a 3D physical entity with mass and spatial attributes.
    """

    def __init__(
        self,
        name: str = 'entity3d',
        mass: float = None,
        position: position_3d = None,
        orientation: orientation_3d = None,
        velocity: velocity_3d = None,
        angular_velocity: angular_velocity_3d = None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
        Initializes the Entity3D instance.

        Parameters
        ----------
        name: str
            The name of the entity. Default is 'entity_3d'.
        mass : float
            The mass of the entity.
        position : position_3d
            The spatial position of the entity.
        orientation : orientation_3d
            The spatial orientation of the entity.
        velocity : velocity_3d
            The linear velocity of the entity.
        angular_velocity : angular_velocity_3d
            The angular velocity of the entity.
        device : str
            The device of the entity. Default is 'cpu'.
        """
        super().__init__(
            name=name, mass=mass, position=position, orientation=orientation,
            velocity=velocity, angular_velocity=angular_velocity, device=device
        )
