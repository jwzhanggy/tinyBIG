
"""
This tinyBIG toolkit package implements the modules that can be used to design and build the RPN model.

Modules implemented in this package:
- TBD
"""

__version__ = '0.1.0.post6'

from . import model, module, config
from . import remainder, expansion, compression, reconciliation
from . import learner, data, output, metric, koala
from . import visual, util

__all__ = [
    'model',
    'module',
    'config',
    'remainder',
    'expansion',
    'compression',
    'reconciliation',
    'learner',
    'data',
    'output',
    'metric',
    'koala',
    'visual',
    'util'
]
