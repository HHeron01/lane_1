import torch
import torchvision as tv
from torch import nn
from mmdet3d.models.builder import NECKS
from mmcv.runner import force_fp32, auto_fp16


def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod

class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),
            # nn.BatchNorm1d(sh * sw),
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),
            # nn.BatchNorm1d(sh * sw),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(),
        )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(sc),
            ))

    def forward(self, x):
        x = x.view([x.size()[0], x.size()[1], self.image_featmap_size[1] * self.image_featmap_size[2]])
        bev_view = self.fc_transform(x)  #
        # bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        # bev_view = bev_view.view(4, 512, 25, 5)
        bev_view = bev_view.view([bev_view.size()[0], bev_view.size()[1], self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        return bev_view



@NECKS.register_module()
class VRM(nn.Module):
    def __init__(self, output_size=(200, 48), input_channel=512, N=1):
        super().__init__()
        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 5))
        self.s64transformer = FCTransform_((1024, 9, 16), (256, 25, 5))
        self.N = N

        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)
                ),
                downsample=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(1024),
                )

            )
        )

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Sequential(
                    nn.Conv2d(input_channel, 128, 1),
                    nn.BatchNorm2d(128),
                )
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Sequential(
                    nn.Conv2d(128, 64, 1),
                    nn.BatchNorm2d(64),
                )
            ),
        )

    @auto_fp16()
    def forward(self, img_feats):
        S32_img_feat = img_feats[-1]
        S64_img_feat = self.down(S32_img_feat)
        # _, output_dim, ouput_H, output_W = x.shape
        # x = x.view(B, N, output_dim, ouput_H, output_W)
        bev_32 = self.s32transformer(S32_img_feat)
        bev_64 = self.s64transformer(S64_img_feat)
        bev_feat = torch.cat([bev_64, bev_32], dim=1)
        bev_feat = self.bev_up_new(bev_feat)

        return bev_feat, S32_img_feat

@NECKS.register_module()
class Swin_VRM(nn.Module):
    def __init__(self, output_size=(200, 48), input_channel=384, N=1):
        super().__init__()
        self.s32transformer = FCTransform_((384, 18, 32), (192, 25, 5))
        self.s64transformer = FCTransform_((768, 9, 16), (192, 25, 5))
        self.N = N

        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(input_channel * 2),
                    nn.ReLU(),
                    nn.Conv2d(input_channel * 2, input_channel * 2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(input_channel * 2)
                ),
                downsample=nn.Sequential(
                    nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(input_channel * 2),
                )

            )
        )

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Sequential(
                    nn.Conv2d(input_channel, 128, 1),
                    nn.BatchNorm2d(128),
                )
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Sequential(
                    nn.Conv2d(128, 64, 1),
                    nn.BatchNorm2d(64),
                )
            ),
        )

    @auto_fp16()
    def forward(self, img_feats):
        # print("img_feats:", len(img_feats))
        # print(img_feats[0].shape, img_feats[1].shape, img_feats[2].shape)
        S32_img_feat = img_feats[-1]
        S64_img_feat = self.down(S32_img_feat)
        # _, output_dim, ouput_H, output_W = x.shape
        # x = x.view(B, N, output_dim, ouput_H, output_W)
        bev_32 = self.s32transformer(S32_img_feat)
        bev_64 = self.s64transformer(S64_img_feat)
        bev_feat = torch.cat([bev_64, bev_32], dim=1)
        bev_feat = self.bev_up_new(bev_feat)

        # print("bev_feat:", bev_feat.shape, S32_img_feat.shape)

        return bev_feat, S32_img_feat


