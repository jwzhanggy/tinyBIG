# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################
# Position Class #
##################

import torch
from typing import Optional, Union, List

from tinybig.koala.spatial_algebra.space import space


class position(space):
    """
    Represents a D-dimensional position using a PyTorch tensor.

    This class handles both 2D and 3D positions by specifying the `dim` parameter.
    When `dim=2`, the position is represented as (x, y).
    When `dim=3`, the position is represented as (x, y, z).

    Parameters
    ----------
    *coords : float, list, or torch.Tensor
        The coordinate values of the position.
        - For 2D: (x, y) or [x, y] or torch.Tensor([x, y]).
        - For 3D: (x, y, z) or [x, y, z] or torch.Tensor([x, y, z]).
    device : str, optional
        The device on which the tensor will be allocated (e.g., 'cpu' or 'cuda').
        Default is 'cpu'.
    dtype : torch.dtype, optional
        The data type of the underlying tensor. Default is torch.float32.
    dim : int, optional
        The dimension of the position. If not specified, it will be inferred from
        the length of `coords`. Must be either 2 or 3.

    Raises
    ------
    ValueError
        If the number of coordinates doesn't match the specified or inferred dimension,
        or if the dimension is not 2 or 3.
    TypeError
        If the coordinates are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *coords: Union[float, List[float], torch.Tensor],
        name: str = 'position', dim: Optional[int] = None,
        device: str = 'cpu', dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize a Position object.

        Parameters
        ----------
        *coords : float, list, or torch.Tensor
            The coordinate values of the position.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        dim : int, optional
            The dimension of the position.

        Raises
        ------
        ValueError
            If the number of coordinates doesn't match the specified or inferred dimension,
            or if the dimension is not 2 or 3.
        TypeError
            If the coordinates are not provided as floats, lists, or torch.Tensors.
        """
        super().__init__(name=name, dim=dim, device=device,dtype=dtype)

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
            raise ValueError("Only 2D or 3D positions are supported.")

        self.p = coords_tensor

    def __repr__(self) -> str:
        """
        Return a string representation of the Position object.
        """
        coord_str = ", ".join([f"{c.item()}" for c in self.p])
        return f"Position({coord_str}) in {self.dim}D"

    def to_homogeneous(self) -> torch.Tensor:
        """
        Convert the position to homogeneous coordinates.

        Returns
        -------
        torch.Tensor
            A (dim+1)-element tensor representing the position in homogeneous form.
            For 2D: [x, y, 1]
            For 3D: [x, y, z, 1]
        """
        ones = torch.ones(1, dtype=self.dtype, device=self.device)
        return torch.cat([self.p, ones], dim=0)

    def transform(self, T: torch.Tensor) -> "position":
        """
        Apply a homogeneous transformation to the position.

        Parameters
        ----------
        T : torch.Tensor
            A (dim+1)x(dim+1) homogeneous transformation matrix.
            For dim=2, T must be 3x3.
            For dim=3, T must be 4x4.

        Returns
        -------
        Position
            A new Position object representing the transformed position.

        Raises
        ------
        ValueError
            If the transformation matrix shape doesn't match the expected size for the dimension.
        """
        expected_shape = (self.dim + 1, self.dim + 1)
        if T.shape != expected_shape:
            raise ValueError(f"Transformation matrix must be {expected_shape}")

        p_homo = self.to_homogeneous()
        p_transformed = T @ p_homo
        new_coords = p_transformed[:self.dim] / p_transformed[-1]
        return self.__class__(
            *new_coords.tolist(),
            device=self.device,
            dtype=self.dtype,
            dim=self.dim
        )

    def get_dim(self):
        """
        Get the dimension of the position.

        Returns
        -------
        int
            The dimension of the position.
        """
        return self.dim

    def as_tensor(self) -> torch.Tensor:
        """
        Get the underlying position tensor.

        Returns
        -------
        torch.Tensor
            A tensor of shape (dim,) representing the position coordinates.
        """
        return self.p

    def get_position(self) -> torch.Tensor:
        """
        Get the position of the position on the device.

        Returns
        -------
        torch.Tensor
            The position tensor of the object.
        """
        return self.as_tensor()

    def from_tensor(self, p: torch.Tensor) -> None:
        """
        Create a Position object from a given tensor.

        Parameters
        ----------
        p : torch.Tensor
            A 1D tensor with shape (2), or (3), representing a position in 2D or 3D.

        Raises
        ------
        ValueError
            If the input tensor is not 1D or does not have length 2 or 3.
        """
        if p.ndim != 1:
            raise ValueError("Input tensor must be 1D")
        if p.shape[0] not in (2, 3):
            raise ValueError("Input tensor must be length 2 or 3")
        self.p = p

    def set_position(self, p: Union[List[float], torch.Tensor]) -> None:
        """
        Reset the internal position vector p using a list of floats or a torch.Tensor.

        Parameters
        ----------
        p : list of float or torch.Tensor
            The new coordinates for the position. Must match self.dim in length.

        Raises
        ------
        ValueError
            If the input does not match the dimension or is not of the correct shape.
        TypeError
            If the input is neither a list nor a torch.Tensor.
        """
        if isinstance(p, list):
            if len(p) != self.dim:
                raise ValueError(f"Input list length must be {self.dim}, got {len(p)}")
            p = torch.tensor(p, device=self.device, dtype=self.dtype)
        elif isinstance(p, torch.Tensor):
            if p.ndim != 1:
                raise ValueError("Input tensor must be 1D")
            if p.shape[0] != self.dim:
                raise ValueError(f"Input tensor length must be {self.dim}, got {p.shape[0]}")
        else:
            raise TypeError("Value must be either a list or a torch.Tensor")
        self.from_tensor(p)

    def __add__(self, other: "position") -> "position":
        """
        Adds two Position instances element-wise.

        Parameters
        ----------
        other : Position
            The Position to add.

        Returns
        -------
        Position
            The resulting Position after addition.

        Raises
        ------
        ValueError
            If the dimensions of the positions do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot add positions with different dimensions.")
        new_p = self.p + other.p
        return position(*new_p.tolist(), device=self.device, dtype=self.dtype, dim=self.dim)

    def __sub__(self, other: "position") -> "position":
        """
        Subtracts two Position instances element-wise.

        Parameters
        ----------
        other : Position
            The Position to subtract.

        Returns
        -------
        Position
            The resulting Position after subtraction.

        Raises
        ------
        ValueError
            If the dimensions of the positions do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot subtract positions with different dimensions.")
        new_p = self.p - other.p
        return position(*new_p.tolist(), device=self.device, dtype=self.dtype, dim=self.dim)

    def distance_to(self, other: "position") -> float:
        """
        Computes the Euclidean distance to another Position.

        Parameters
        ----------
        other : Position
            The Position to compute the distance to.

        Returns
        -------
        float
            The Euclidean distance.

        Raises
        ------
        ValueError
            If the dimensions of the positions do not match.
        """
        if self.dim != other.dim:
            raise ValueError("Cannot compute distance between positions with different dimensions.")
        return torch.norm(self.p - other.p).item()


class position_2d(position):
    """
    A subclass of Position specifically for 2D.

    This class fixes dim=2. The position is represented as (x, y).

    Parameters
    ----------
    *coord : float, list, or torch.Tensor
        The coordinate values of the position.
        - For 2D: (x, y) or [x, y] or torch.Tensor([x, y]).
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of coordinates doesn't match dim=2.
    TypeError
        If the coordinates are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *coord: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        dim: int = 2,
        **kwargs
    ):
        """
        Initialize a Position2D object with fixed dimension (dim=2).

        Parameters
        ----------
        coord : float, list, or torch.Tensor
            The coordinate values of the position.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        *args
            Additional arguments passed to the base class.
        **kwargs
            Additional keyword arguments passed to the base class.

        Raises
        ------
        ValueError
            If the number of coordinates doesn't match dim=2.
        TypeError
            If the coordinates are not provided as floats, lists, or torch.Tensors.
        """
        assert dim == 2, "Dimension must be 2"
        super().__init__(*coord, device=device, dtype=dtype, dim=dim, **kwargs)


class position_3d(position):
    """
    A subclass of Position specifically for 3D.

    This class fixes dim=3. The position is represented as (x, y, z).

    Parameters
    ----------
    *args : float, list, or torch.Tensor
        The coordinate values of the position.
        - For 3D: (x, y, z) or [x, y, z] or torch.Tensor([x, y, z]).
    device : str, optional
        The device on which the tensor will be allocated (default is 'cpu').
    dtype : torch.dtype, optional
        The data type of the underlying tensor (default is torch.float32).

    Raises
    ------
    ValueError
        If the number of coordinates doesn't match dim=3.
    TypeError
        If the coordinates are not provided as floats, lists, or torch.Tensors.
    """

    def __init__(
        self,
        *coords: Union[float, List[float], torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        dim: int = 3,
        **kwargs
    ):
        """
        Initialize a Position3D object with fixed dimension (dim=3).

        Parameters
        ----------
        *coords : float, list, or torch.Tensor
            The coordinate values of the position.
        device : str, optional
            The device on which the tensor will be allocated.
        dtype : torch.dtype, optional
            The data type of the tensor.
        *args
            Additional arguments passed to the base class.
        **kwargs
            Additional keyword arguments passed to the base class.

        Raises
        ------
        ValueError
            If the number of coordinates doesn't match dim=3.
        TypeError
            If the coordinates are not provided as floats, lists, or torch.Tensors.
        """
        assert dim == 3, "Dimension must be 3"
        super().__init__(*coords, device=device, dtype=dtype, dim=dim, **kwargs)


if __name__ == "__main__":
    pos2d = position_2d(1.0, 2.0)
    T2d = torch.tensor([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=pos2d.dtype, device=pos2d.device)
    transformed_2d = pos2d.transform(T2d)

    pos3d = position_3d(1.0, 2.0, 3.0)
    angle_3d = torch.deg2rad(torch.tensor(90.0, dtype=pos3d.dtype, device=pos3d.device))
    Rz_3d = torch.tensor([
        [torch.cos(angle_3d), -torch.sin(angle_3d), 0.0, 5.0],
        [torch.sin(angle_3d), torch.cos(angle_3d), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=pos3d.dtype, device=pos3d.device)
    transformed_3d = pos3d.transform(Rz_3d)
