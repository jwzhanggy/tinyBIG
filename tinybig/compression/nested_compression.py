# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Nested Compressions #
#######################

from tinybig.expansion import nested_expansion


class nested_compression(nested_expansion):
    def __init__(self, name='nested_compression', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
