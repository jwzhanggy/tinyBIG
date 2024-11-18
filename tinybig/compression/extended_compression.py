# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################
# Extended Compressions #
#########################

from tinybig.expansion import extended_expansion


class extended_compression(extended_expansion):
    def __init__(self, name='extended_compression', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
