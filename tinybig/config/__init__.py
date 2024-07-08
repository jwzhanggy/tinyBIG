"""
This module defines the configurations of the RPN model.

Specifically, this module will introduce a general model config class, based on which,
a detailed config class will be introduced for the RPN model.

## Object Configuration Template

For the RPN model with complex configurations, instead of directly define the model architecture with iterative
component definition and initialization, another recommended method for object initialization is via the config files.

For the RPN model, the model configuration file can be stored in the format of yaml or json, which are equivalent.
Below, we show the detailed configurations of an object in both yaml and json, where the "object" can denote
the "rpn model", "rpn layer", "rpn head", "expansion function", "reconciliation function", "remainder function",
and even the internal "pre-processing functions", "activation functions", etc.

object config stored in file "./configs/object_config.yaml"
```
object_config
    object_class: object_class_detailed_information
    object_parameters:
        parameter_1: value_1
        ...
        parameter_n: value_n
```

object config stored in file "./configs/object_config.json"
```
{
    "object_config":{
        "object_class": object_class_information
        "object_parameters":{
            "parameter_1": value_1,
            ...
            "parameter_n": value_n
        }
    }
}
```

## Object Instantiation from Configurations

Based on the object configurations represented above, the config class introduced in this module can
load them from file, and instantiate the object based on the configuration information.

For instance, for the "object_config" stored in the yaml/json files shown above,
the configuration information can be loaded and used to initiate the object with the following python code:

```python
# This is a Python code block
from tinybig.util import get_obj_from_str
from tinybig.config import config

# load the above config from file "./configs/object_config.yaml"
# json file can be loaded in a similar way with "config.load_json(...)"
configs = config.load_yaml(cache_dir='./configs', config_file='object_config.yaml')

class_name = configs["object_class"]
parameters = configs["object_parameters"]
obj = get_obj_from_str(class_name)(**parameters)
```

## Classes in this Module

This module contains the following categories of config classes:

* Base Config Template
* RPN Model Config
"""

from tinybig.config.base_config import config
from tinybig.config.rpn_config import rpn_config
