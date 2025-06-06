# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Spatial Entity Classes #
##########################

import torch
from typing import Optional, Callable, List


class time:
    """
    Represents and manages the simulation or application's temporal aspects.

    Attributes
    ----------
    current_time : torch.Tensor
        The current simulation time.
    dt : torch.Tensor
        The time step for each simulation update.
    running : bool
        Indicates whether the simulation is currently running.
    observers : List[Callable[[float], None]]
        A list of callback functions to notify on time updates.
    """

    def __init__(self, name: str = 'time', t: float = 0.0, dt: float = 0.01, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        """
        Initializes the Time instance.

        Parameters
        ----------
        name : str
            The name of the time instance. Default is 'time'.
        t : float, optional
            The starting time of the simulation (default is 0.0).
        dt : float, optional
            The time step for each simulation update (default is 0.1).
        device : str, optional
            The device on which the tensors are allocated (default is 'cpu').
        dtype : torch.dtype, optional
            The data type of the tensors (default is torch.float32).

        Raises
        ------
        ValueError
            If dt is not positive.
        """
        self.name = name
        self.device = device
        self.dtype = dtype

        if dt <= 0:
            raise ValueError("Time step dt must be a positive value.")

        self.t = torch.tensor([t], device=device, dtype=dtype)
        self.dt = torch.tensor([dt], device=device, dtype=dtype)
        self.running = False
        self.observers: List[Callable[[float], None]] = []

    def start(self) -> None:
        """
        Starts the simulation time.

        Sets the running flag to True.
        """
        self.running = True

    def pause(self) -> None:
        """
        Pauses the simulation time.

        Sets the running flag to False.
        """
        self.running = False

    def reset(self, tau: float = 0.0) -> None:
        """
        Resets the simulation time to a specified initial time.

        Parameters
        ----------
        tau : float, optional
            The time to reset to (default is 0.0).
        """
        self.t = torch.tensor([tau], device=self.device, dtype=self.dtype)
        self.notify_observers()

    def step(self) -> None:
        """
        Advances the simulation time by one time step.

        If the simulation is running, increments the current time by dt.
        Notifies all registered observers about the time update.
        """
        if self.running:
            self.t += self.dt
            self.notify_observers()

    def set_dt(self, new_dt: float) -> None:
        """
        Sets a new time step for the simulation.

        Parameters
        ----------
        new_dt : float
            The new time step value.

        Raises
        ------
        ValueError
            If new_dt is not positive.
        """
        if new_dt <= 0:
            raise ValueError("Time step dt must be a positive value.")
        self.dt = torch.tensor([new_dt], device=self.device, dtype=self.dtype)

    def get_current_time(self) -> float:
        """
        Retrieves the current simulation time.

        Returns
        -------
        float
            The current time value.
        """
        return self.t.item()

    def get_dt(self) -> float:
        """
        Retrieves the current time step.

        Returns
        -------
        float
            The time step value.
        """
        return self.dt.item()

    def add_observer(self, callback: Callable[[float], None]) -> None:
        """
        Adds a callback function to be notified on time updates.

        Parameters
        ----------
        callback : Callable[[float], None]
            A function that takes the current time as an argument.
        """
        if callback not in self.observers:
            self.observers.append(callback)

    def remove_observer(self, callback: Callable[[float], None]) -> None:
        """
        Removes a previously added callback function.

        Parameters
        ----------
        callback : Callable[[float], None]
            The callback function to remove.
        """
        if callback in self.observers:
            self.observers.remove(callback)

    def notify_observers(self) -> None:
        """
        Notifies all registered observers about the current time update.

        Calls each observer with the current time as an argument.
        """
        current_time = self.get_current_time()
        for callback in self.observers:
            callback(current_time)

    def run(self, steps: Optional[int] = None) -> None:
        """
        Runs the simulation, advancing time in a loop.

        Parameters
        ----------
        steps : Optional[int], optional
            The number of steps to run. If None, runs indefinitely until paused.
            (default is None)

        Notes
        -----
        This method is blocking and should be run in a separate thread if used in an interactive application.
        """
        step_count = 0
        while self.running and (steps is None or step_count < steps):
            self.step()
            step_count += 1

    def __repr__(self):
        """
        Returns the string representation of the Time instance.

        Returns
        -------
        str
            String representation.
        """
        return f"Time(current_time={self.get_current_time()}, dt={self.get_dt()}, running={self.running})"
