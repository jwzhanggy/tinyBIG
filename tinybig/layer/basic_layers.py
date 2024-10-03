# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# RPN Multi-Head Layer Module #
###############################

from tinybig.module.base_layer import rpn_layer
from tinybig.head.basic_heads import perceptron_head, svm_head, kan_head, naive_bayes_head, pgm_head


class perceptron_layer(rpn_layer):

    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_layer',
        enable_bias: bool = True,
        # optional parameters
        with_taylor: bool = False,
        d: int = 2,
        with_dual_lphm: bool = False,
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        with_batch_norm: bool = False,
        with_relu: bool = False,
        with_softmax: bool = True,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            perceptron_head(
                m=m, n=n,
                enable_bias=enable_bias,
                device=device,
                with_taylor=with_taylor,
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr,
                with_residual=with_residual,
                d=d, r=r,
                channel_num=channel_num,
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_softmax=with_softmax,
                *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class svm_layer(rpn_layer):

    def __init__(
        self, m: int, n: int,
        name: str = 'svm_layer',
        kernel: str = 'linear',
        base_range: tuple = (-1, 1),
        num_interval: int = 10,
        epsilon: float = 1.0,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            svm_head(
                m=m, n=n,
                kernel=kernel,
                base_range=base_range,
                num_interval=num_interval,
                epsilon=epsilon,
                enable_bias=enable_bias,
                device=device,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                channel_num=channel_num,
                *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class kan_layer(rpn_layer):

    def __init__(
        self, m: int, n: int,
        grid_range=(-1, 1), t: int = 5, d: int = 3,
        name: str = 'perceptron_head',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False, r: int = 3,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            kan_head(
                m=m, n=n,
                grid_range=grid_range,
                t=t, d=d,
                enable_bias=enable_bias,
                with_lorr=with_lorr, r=r,
                channel_num=channel_num,
                device=device,
                *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class naive_bayes_layer(rpn_layer):

    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_layer',
        distribution: str = 'normal',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            naive_bayes_head(
                m=m, n=n,
                enable_bias=enable_bias,
                distribution=distribution,
                device=device,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                channel_num=channel_num,
                *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class pgm_layer(rpn_layer):
    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_layer',
        distribution: str = 'normal',
        d: int = 2, with_replacement: bool = False,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            pgm_head(
                m=m, n=n,
                enable_bias=enable_bias,
                distribution=distribution,
                d=d, with_replacement=with_replacement,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                channel_num=channel_num,
                device=device, *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)
