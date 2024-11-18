# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Basic Compression Functions #
###############################

from tinybig.expansion import identity_expansion, reciprocal_expansion, linear_expansion


class identity_compression(identity_expansion):
    def __init__(self, name='identity_compression', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)


class reciprocal_compression(reciprocal_expansion):
    def __init__(self, name='reciprocal_compression', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)


class linear_compression(linear_expansion):
    def __init__(self, name='linear_compression', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
