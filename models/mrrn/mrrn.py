import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from models.sdas.model_utils import (
        normalization,
        Downsample,
        zero_module,
        AttentionBlock
)

from models.cgattention import CascadedGroupAttention
from models.masa import MaSA
from models.tkesa import TKESA
from models.gcam import GCAM
from models.glfa import GLFA
from models.msdi import MSDI
from models.ffm import FFM
from models.masa import MaSA
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.ConvTranspose2d(in_channels=channels,
                                           out_channels=self.out_channels,
                                           kernel_size=4,
                                           stride=2, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv:
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x


class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, True)
            self.x_upd = Upsample(channels, True)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                 channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d( channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        return self.skip_connection(x) + h



class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        channel_mult,
        attention_mult,
        num_heads = 4,
        num_heads_upsample=-1,
        num_head_channels = 64,
    ):

        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.channel_mult = channel_mult

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, ch, 3, padding=1)]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        attn_ratio = 4
        kernels = [5] * num_heads
        key_dim = (ch // num_heads) // attn_ratio

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        out_channels=int(mult * model_channels),
                    ),
                    # GLFA(in_channels=int(mult * model_channels))
                ]
                ch = int(mult * model_channels)
                if ds in attention_mult:
                    layers.append(
                                # SS2D(d_model=ch),
                                # CascadedGroupAttention(
                                #     dim=ch,
                                #     key_dim=key_dim,
                                #     num_heads=num_heads,
                                #     resolution=out_ch,
                                #     kernels=kernels
                                # )
                                # MaSA(ch),
                                TKESA(ch),
                                # GLFA(in_channels=ch),
                                # GCAM(ch),
                                # AttentionBlock(
                                #     ch,
                                #     num_heads=num_heads,
                                #     num_head_channels=num_head_channels,
                                # )
                    )

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            down=True,
                        ),
                        # GLFA(in_channels=out_ch)
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch


        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
            ),
            # MaSA(ch),
            TKESA(ch),
            # GCAM(ch),
            # SS2D(d_model=ch),
            # CascadedGroupAttention(
            #                         dim=ch,
            #                         key_dim=key_dim,
            #                         num_heads=num_heads,
            #                         resolution=out_ch,
            #                         kernels=kernels
            #                     )
            # AttentionBlock(
            #     ch,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            # ),
            ResBlock(
                ch,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        self.att_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                
                layers = [
                    ResBlock(
                        ch + ich,
                        # ch,
                        out_channels=int(model_channels * mult),
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_mult:
                    layers.append(
                            # SS2D(d_model=ch),
                            # CascadedGroupAttention(
                            #         dim=ch,
                            #         key_dim=key_dim,
                            #         num_heads=num_heads,
                            #         resolution=out_ch,
                            #         kernels=kernels
                            #     )
                            # MaSA(ch),
                            TKESA(ch),
                            # GCAM(ch),
                            # AttentionBlock(
                            #     ch,
                            #     num_heads=num_heads_upsample,##
                            #     num_head_channels=num_head_channels,
                            # )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            up=True,
                        )
                    )
                    ds //= 2
                # att_layer = [
                #     # FFM(dim1=ch,channel=ich)
                #     TKESA(ch)
                # ]
                self.output_blocks.append(nn.Sequential(*layers))
                # self.att_blocks.append(nn.Sequential(*att_layer))
                self._feature_size += ch
                
        # self.masa = MaSA(ch)
        self.gcma = GCAM(ch)
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )


    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        h = x # [16, 256, 64, 64]
        for module in self.input_blocks:
            h = module(h) # [16, 128, 64, 64] [16, 128, 64, 64] [16, 128, 64, 64] [16, 128, 32, 32] [16, 256, 32, 32](att) [16, 256, 32, 32](att) [16, 256, 16, 16] [16, 512, 16, 16](att) [16, 512, 16, 16](att)
            hs.append(h)
        h = self.middle_block(h) # [16, 512, 16, 16]
        for module in self.output_blocks:
        # for module, att_module in zip(self.output_blocks, self.att_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            
            h = module(h)# [16, 512, 16, 16] [16, 512, 16, 16] [16, 512, 32, 32] [16, 256, 32, 32] [16, 256, 32, 32] [16, 256, 64, 64] [16, 128, 64, 64] [16, 128, 64, 64] [16, 128, 64, 64]
        h = self.gcma(h)
        return self.out(h)


class MRRNstructionLayer(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 num_res_blocks,
                 hide_channels_ratio,
                 channel_mult,
                 attention_mult
                 ):

        super(MRRNstructionLayer, self).__init__()
        """
        model_helper:
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()
        """

        self.inplanes=inplanes
        self.instrides=instrides
        self.num_res_blocks=num_res_blocks
        self.attention_mult=attention_mult
        i = 0
        for block_name in self.inplanes:
            module= UNetModel(
                in_channels=self.inplanes[block_name],
                out_channels=self.inplanes[block_name],
                model_channels=int(hide_channels_ratio*self.inplanes[block_name]),
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                attention_mult=attention_mult
            )
            # if i == 0:
            #     down = False
            #     module = HLFAE(in_channels=self.inplanes[block_name],out_channels=self.inplanes[block_name],down=down)
            # else:
            #     down = True
            #     module = HLFAE(in_channels=self.inplanes[block_name],out_channels=self.inplanes[block_name],down=down)
            # i = i + 1
            # module = HLFAE(in_channels=self.inplanes[block_name],out_channels=self.inplanes[block_name],down=False)
            self.add_module('{}_mrrn'.format(block_name),module)
        self.msdi_1 = MSDI([256,512])
        self.msdi_2 = MSDI([512,512])
        self.msdi_3 = MSDI([512,256])


    def forward(self, inputs,train=False):
        block_feats = inputs['block_feats'] # block1,block2,block3,block4
        mrrn_feats = { block_name:getattr(self,'{}_mrrn'.format(block_name))(block_feats[block_name]) for block_name in block_feats}# block1[16,256,64,64],block2[16,512,32,32],block3[16,512,16,16],block4[16,256,8,8]
        #TODO
        #多尺度融合block1和block4，block2和block3
        # inputs_1 = [mrrn_feats['block1'],mrrn_feats['block2']]
        # inputs_2 = [mrrn_feats['block2'],mrrn_feats['block3']]
        # inputs_3 = [mrrn_feats['block3'],mrrn_feats['block4']]
        # mrrn_feats['block2'] = self.msdi_1(inputs_1)
        # mrrn_feats['block3'] = self.msdi_2(inputs_2)
        # mrrn_feats['block4'] = self.msdi_3(inputs_3)
        # for block_name in block_feats:
        #     print("block_feat:",block_feats[block_name].shape,"mrrn_feats:",mrrn_feats[block_name].shape)
        residual={ block_name: (block_feats[block_name] - mrrn_feats[block_name] )**2 for block_name in block_feats}# block1[16,256,64,64],block2[16,512,32,32],block3[16,512,16,16],block4[16,256,8,8]
        return {'feats_mrrn':mrrn_feats,'residual':residual}


    def get_outplanes(self):
        return self.inplanes

    def get_outstrides(self):
        return self.instrides
