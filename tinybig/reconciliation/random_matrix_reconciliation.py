# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Low-rank parameter reconciliation functions.

This module contains the low-rank parameter reconciliation functions,
including lorr_reconciliation, hm_reconciliation, lphm_reconciliation, and dual_lphm_reconciliation.
"""

import torch
import torch.nn.functional as F

from tinybig.reconciliation import fabrication


#######################################
# Random Matrix based reconciliations #
#######################################


class random_matrix_adaption_reconciliation(fabrication):
    pass

