from torch import nn
from models.layer_norm import LayerNorm
from models.register_models import register_neck


@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = #levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        return fpn_feats, new_fpn_masks