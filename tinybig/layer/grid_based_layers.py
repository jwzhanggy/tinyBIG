# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Grid based RPN Layer Module #
###############################

from tinybig.module.base_layer import layer
from tinybig.head.grid_based_heads import grid_interdependence_head, grid_compression_head
from tinybig.fusion.concatenation_fusion import concatenation_fusion
from tinybig.fusion.metric_fusion import mean_fusion


class grid_interdependence_layer(layer):

    def __init__(
        self,
        h: int, w: int, in_channel: int, out_channel: int,
        d: int = 1,
        width: int = 1,
        name: str = 'grid_interdependence_layer',
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        packing_strategy: str = 'densest_packing',
        with_batch_norm: bool = True,
        with_relu: bool = True,
        with_residual: bool = False,
        enable_bias: bool = False,
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        print('* grid_interdependence_layer, width:', width)
        heads = [
            grid_interdependence_head(
                h=h, w=w, d=d,
                in_channel=in_channel, out_channel=out_channel,
                patch_shape=patch_shape,
                p_h=p_h, p_h_prime=p_h_prime,
                p_w=p_w, p_w_prime=p_w_prime,
                p_d=p_d, p_d_prime=p_d_prime,
                p_r=p_r,
                cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
                packing_strategy=packing_strategy,
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_residual=with_residual,
                enable_bias=enable_bias,
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr, r=r,
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        assert len(heads) >= 1
        m, n = heads[0].get_m(), heads[0].get_n()
        if len(heads) > 1:
            head_fusion = mean_fusion(dims=[head.get_n() for head in heads])
        else:
            head_fusion = None
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, head_fusion=head_fusion, device=device, *args, **kwargs)

    def get_output_grid_shape(self):
        assert len(self.heads) >= 1
        return self.heads[0].get_output_grid_shape()


class grid_compression_layer(layer):

    def __init__(
        self,
        h: int, w: int, channel_num: int,
        d: int = 1,
        name: str = 'grid_compression_layer',
        pooling_metric: str = 'batch_max',
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        with_dropout: bool = False, p: float = 0.5,
        packing_strategy: str = 'densest_packing',
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        print('* grid_compression_layer')
        heads = [
            grid_compression_head(
                h=h, w=w, d=d,
                channel_num=channel_num,
                pooling_metric=pooling_metric,
                patch_shape=patch_shape,
                p_h=p_h, p_h_prime=p_h_prime,
                p_w=p_w, p_w_prime=p_w_prime,
                p_d=p_d, p_d_prime=p_d_prime,
                p_r=p_r,
                cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
                packing_strategy=packing_strategy,
                with_dropout=with_dropout, p=p,
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ]
        assert len(heads) >= 1
        m, n = heads[0].get_m(), heads[0].get_n()
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)

    def get_output_grid_shape(self):
        assert len(self.heads) >= 1
        return self.heads[0].get_output_grid_shape()