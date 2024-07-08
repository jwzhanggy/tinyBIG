"""
This module implements the utility functions and methods that will be used in the tinyBIG toolkit.

The function and methods in this module can be used for very diverse purposes, including random seed setup,
data processing, model initialization, model component processing, output processing, etc.

## Functions in this Module

This module contains the following categories of utility functions:

* Utility Functions for Project Initialization
* Utility Functions for RPN Initialization
* Utility Functions for Data Processing in Expansions
* Utility Functions for Math Function Dataset Creation
* Utility Functions for Data and Result Directory Creation
"""

from tinybig.util.util import (
    get_obj_from_str,
    process_function_list,
    register_function_parameters,
    special_function_process,
    func_x,
    str_func_x,
    process_num_alloc_configs,
    string_to_function,
    create_directory_if_not_exists,
    set_random_seed,
)
