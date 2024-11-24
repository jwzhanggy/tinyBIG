# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################
# Math Function Dataloader #
############################


import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from tinybig.data.base_data import dataloader, dataset
from tinybig.module.base_function import function


class function_dataloader(dataloader):
    """
    A dataloader class for handling mathematical functions and equations.

    This class extends the base `dataloader` class to provide functionality for loading equations,
    processing variables, and generating datasets based on mathematical formulas.

    Attributes
    ----------
    name: str
        The name of the dataloader instance.
    function_list: list
        A list of string representations of mathematical functions or equations.
    equation_index: int
        The index of the currently selected equation.

    Methods
    -------
    __init__(name='function_dataloader', function_list: list = [], equation_index: int = 0, ...)
        Initializes the function dataloader.
    load_equation(index: int = 0)
        Loads and processes an equation from the `function_list` by its index.
    load_all_equations()
        Loads and processes all equations in the `function_list`.
    generate_data(formula: str, variables: dict, num: int = 2000, value_range: list = (0, 1), ...)
        Generates synthetic data based on a given formula and variable ranges.
    load(equation_index: int = None, ...)
        Loads training and testing datasets for a specified equation.
    """
    def __init__(self, name='function_dataloader', function_list: list = [], equation_index: int = 0, train_batch_size=64, test_batch_size=64):
        """
        Initializes the function dataloader with a list of equations and configurations.

        Parameters
        ----------
        name: str, default = 'function_dataloader'
            The name of the dataloader instance.
        function_list: list, default = []
            A list of string representations of mathematical functions or equations.
        equation_index: int, default = 0
            The index of the currently selected equation.
        train_batch_size: int, default = 64
            The batch size for training data.
        test_batch_size: int, default = 64
            The batch size for testing data.

        Returns
        -------
        None
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        self.equation_index = equation_index
        self.function_list = function_list

    def load_equation(self, index: int = 0):
        """
        Loads and processes a specific equation from the function list.

        Parameters
        ----------
        index: int, default = 0
            The index of the equation to load. If None, the default `equation_index` is used.

        Returns
        -------
        tuple
            A tuple containing the processed equation dictionary and the original string representation.

        Raises
        ------
        AssertionError
            If the index is out of range of the `function_list`.
        """
        index = index if index is not None else self.equation_index

        if index is None:
            return self.load_all_equations()
        else:
            assert index in range(0, len(self.function_list))
            str_equation = self.function_list[index]
            processed_equation = {}
            equation_contents = str_equation.split(',')
            processed_equation['equ_file_name'] = equation_contents[0]
            processed_equation['equ_number'] = int(equation_contents[1])
            processed_equation['equ_output'] = equation_contents[2]
            processed_equation['equ_formula'] = equation_contents[3]
            processed_equation['equ_variable_num'] = int(equation_contents[4])
            processed_equation['equ_variables'] = {}
            for var_index in range(0, processed_equation['equ_variable_num']):
                var_name = equation_contents[5 + var_index * 3]
                if var_name is None or var_name == '':
                    break
                var_low = float(equation_contents[6 + var_index * 3])
                var_high = float(equation_contents[7 + var_index * 3])
                processed_equation['equ_variables'][var_index] = {
                    'var_name': var_name,
                    'var_low': var_low,
                    'var_high': var_high
                }
            return processed_equation, str_equation

    def load_all_equations(self):
        """
        Loads and processes all equations in the `function_list`.

        Returns
        -------
        dict
            A dictionary where each key is an equation index, and the value is a tuple containing the
            processed equation dictionary and the original string representation.
        """
        processed_equations = {}
        for index in range(0, len(self.function_list)):
            processed_equations[index] = self.load_equation(index=index)
        return processed_equations

    @staticmethod
    def generate_data(formula: str, variables: dict, num: int = 2000, value_range: list = (0, 1),
                      normalize_X: bool = False, normalize_y: bool = False, *args, **kwargs):
        """
        Generates synthetic data based on a formula and variable ranges.

        Parameters
        ----------
        formula: str
            The mathematical formula to generate data for.
        variables: dict
            A dictionary of variable names and their respective ranges.
        num: int, default = 2000
            The number of data samples to generate.
        value_range: list, default = [0, 1]
            The default range for variables if not specified in `variables`.
        normalize_X: bool, default = False
            Whether to normalize the input features.
        normalize_y: bool, default = False
            Whether to normalize the output values.

        Returns
        -------
        tuple
            A tuple containing input features `X` and output values `y` as tensors.

        Raises
        ------
        AssertionError
            If variable ranges are not properly specified.
        """
        var_name_list = []
        var_value_space = []
        for var in variables:
            var_name_list.append(variables[var]['var_name'])
            var_low = variables[var]['var_low'] if variables[var]['var_low'] != '' else value_range[0]
            var_high = variables[var]['var_high'] if variables[var]['var_high'] != '' else value_range[1]
            assert var_low is not None and var_high is not None
            var_value_space.append(var_low + (var_high - var_low) * torch.rand(num))

        X = []
        y = []
        variables = ' '.join(var_name_list)
        func = function.string_to_function(formula, variables)
        for var_values in zip(*var_value_space):
            X.append(var_values)
            y.append(func(*var_values))
        X = torch.Tensor(X)
        y = torch.Tensor(y)

        if normalize_X:
            X_mean = torch.mean(X, dim=0, keepdim=True)
            X_std = torch.std(X, dim=0, keepdim=True)
            X = (X - X_mean)/X_std
        if normalize_y:
            y_mean = torch.mean(y, dim=0, keepdim=True)
            y_std = torch.std(y, dim=0, keepdim=True)
            y = (y - y_mean) / y_std

        return X, y

    def load(self, equation_index: int = None, cache_dir='./data/', num=2000,
             train_percentage=0.5, random_state=1234, shuffle=False, *args, **kwargs):
        """
        Loads training and testing datasets for a specified equation.

        Parameters
        ----------
        equation_index: int, default = None
            The index of the equation to load. If None, the default `equation_index` is used.
        cache_dir: str, default = './data/'
            The directory for caching data.
        num: int, default = 2000
            The number of data samples to generate.
        train_percentage: float, default = 0.5
            The percentage of data to use for training.
        random_state: int, default = 1234
            The random seed for reproducibility.
        shuffle: bool, default = False
            Whether to shuffle the data before splitting.

        Returns
        -------
        dict
            A dictionary containing training and testing data loaders, and the equation string.

        Raises
        ------
        ValueError
            If the `equation_index` is invalid or out of range.
        """
        equation_index = equation_index if equation_index is not None else self.equation_index

        if type(equation_index) is not int or equation_index not in range(0, len(self.function_list)):
            raise ValueError('The equation_index needs to be an integer from 0 to {}, '
                             'its current value {} is out of range...'.format(len(self.function_list)-1, equation_index))

        processed_equation, str_equation = self.load_equation(index=equation_index)

        X, y = self.generate_data(
            formula=processed_equation['equ_formula'],
            variables=processed_equation['equ_variables'],
            num=num,
            normalize_X=False,
            normalize_y=False,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=int(train_percentage*len(X)),
            random_state=random_state, shuffle=shuffle
        )

        train_dataset = dataset(X_train, torch.unsqueeze(y_train, 1))
        test_dataset = dataset(X_test, torch.unsqueeze(y_test, 1))
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)
        return {'train_loader': train_loader, 'test_loader': test_loader, 'str_equation': str_equation}

# The list of elementary functions for continuous function fitting evaluation
Elementary_Function_Equations = (
    "E.0,0,f,x+y,2,x,0,1,y,0,1",
    "E.1,1,f,1/(x+y),2,x,0,1,y,0,1",
    "E.2,2,f,(x+y)**2,2,x,0,1,y,0,1",
    "E.3,3,f,exp(x+y),2,x,0,1,y,0,1",
    "E.4,4,f,ln(x+y),2,x,0,1,y,0,1",
    "E.5,5,f,sin(x+y),2,x,0,1,y,0,1",
    "E.6,6,f,cos(x+y),2,x,0,1,y,0,1",
    "E.7,7,f,tan(x+y),2,x,0,0.5,y,0,0.5",
    "E.8,8,f,arcsin(x+y),2,x,0,0.5,y,0,0.5",
    "E.9,9,f,arccos(x+y),2,x,0,0.5,y,0,0.5",
    "E.10,10,f,arctan(x+y),2,x,0,0.5,y,0,0.5",
    "E.11,11,f,sinh(x+y),2,x,0,1,y,0,1",
    "E.12,12,f,cosh(x+y),2,x,0,1,y,0,1",
    "E.13,13,f,tanh(x+y),2,x,0,1,y,0,1",
    "E.14,14,f,arcsinh(x+y),2,x,0,0.5,y,0,0.5",
    "E.15,15,f,arccosh(x+y),2,x,0.5,1.0,y,0.5,1.0",
    "E.16,16,f,arctanh(x+y),2,x,0,0.5,y,0,0.5",
)

# The list of composite functions for continuous function fitting evaluation
Composite_Function_Equations = (
    "C.0,0,f,(x+y)+1/(x+y),2,x,0,1,y,0,1",
    "C.1,1,f,(x+y)+(x+y)**2,2,x,0,1,y,0,1",
    "C.2,2,f,(x+y)**2+exp(x+y),2,x,0,1,y,0,1",
    "C.3,3,f,exp(x+y)+ln(x+y),2,x,0,1,y,0,1",
    "C.4,4,f,(x+y)**2+sin(x+y),2,x,0,1,y,0,1",
    "C.5,5,f,cos(x+y)+arccos(x+y),2,x,0,0.5,y,0,0.5",
    "C.6,6,f,exp(x+y)/(x+y),2,x,0,1,y,0,1",
    "C.7,7,f,(x+y)**2*ln(x+y),2,x,0,1,y,0,1",
    "C.8,8,f,(x+y)*sin(x+y),2,x,0,1,y,0,1",
    "C.9,9,f,exp(x+y)*ln(x+y),2,x,0,1,y,0,1",
    "C.10,10,f,sin(x+y)*sinh(x+y),2,x,0,1,y,0,1",
    "C.11,11,f,arccos(x+y)*arctanh(x+y),2,x,0,0.5,y,0,0.5",
    "C.12,12,f,exp((x+y)+exp(x+y)),2,x,0,0.5,y,0,0.5",
    "C.13,13,f,exp(sin(x+y)+cos(x+y)),2,x,0,1,y,0,1",
    "C.14,14,f,ln((x+y)**2+exp(x+y)),2,x,0.5,1,y,0.5,1",
    "C.15,15,f,tan(exp(x+y)+ln(x+y)),2,x,0,1,y,0,1",
    "C.16,16,f,1/(1+exp(-x-y)),2,x,0,1,y,0,1",
)


class elementary_function(function_dataloader):
    """
    A dataloader class for elementary functions.

    This class extends the `function_dataloader` to handle elementary mathematical functions.

    Attributes
    ----------
    name: str, default = 'elementary_function'
        The name of the dataloader instance.
    function_list: list, default = Elementary_Function_Equations
        The list of elementary function equations.

    Methods
    -------
    __init__(name='elementary_function', ...)
        Initializes the dataloader for elementary functions.
    """
    def __init__(self, name='elementary_function', function_list: list = Elementary_Function_Equations, *args, **kwargs):
        """
        A dataloader class for elementary functions.

        This class extends the `function_dataloader` to handle elementary mathematical functions.

        Attributes
        ----------
        name: str, default = 'elementary_function'
            The name of the dataloader instance.
        function_list: list, default = Elementary_Function_Equations
            The list of elementary function equations.

        Methods
        -------
        __init__(name='elementary_function', ...)
            Initializes the dataloader for elementary functions.
        """
        super().__init__(name=name, function_list=function_list, *args, **kwargs)


class composite_function(function_dataloader):
    """
    A dataloader class for composite functions.

    This class extends the `function_dataloader` to handle composite mathematical functions.

    Attributes
    ----------
    name: str, default = 'composite_function'
        The name of the dataloader instance.
    function_list: list, default = Composite_Function_Equations
        The list of composite function equations.

    Methods
    -------
    __init__(name='composite_function', ...)
        Initializes the dataloader for composite functions.
    """
    def __init__(self, name='composite_function', function_list: list = Composite_Function_Equations, *args, **kwargs):
        """
        Initializes the dataloader for composite functions.

        Parameters
        ----------
        name: str, default = 'composite_function'
            The name of the dataloader instance.
        function_list: list, default = Composite_Function_Equations
            The list of composite function equations.

        Returns
        -------
        None
        """
        super().__init__(name=name, function_list=function_list, *args, **kwargs)


