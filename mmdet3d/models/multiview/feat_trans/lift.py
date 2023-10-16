import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import NECKS

def gen_dx_bx(grid_cfg):
    dx = torch.Tensor(grid_cfg.grid_res)
    bx = torch.Tensor([off + res / 2.0 for off, res in zip(grid_cfg.offset, grid_cfg.grid_res)])
    n1 = grid_cfg.grid_size[0] / grid_cfg.grid_res[0] - 1
    n2 = grid_cfg.grid_size[1] / grid_cfg.grid_res[1] - 1
    n3 = grid_cfg.grid_height / grid_cfg.grid_res[2] - 1
    nx = torch.LongTensor([n1, n2, n3])
    return dx, bx, nx

def gen_dx_bx_new(grid_cfg):
    dx = torch.Tensor(grid_cfg['grid_res'])
    bx = torch.Tensor([off + res / 2.0 for off, res in zip(grid_cfg['offset'], grid_cfg['grid_res'])])
    n1 = grid_cfg['grid_size'][0] / grid_cfg['grid_res'][0] - 1
    n2 = grid_cfg['grid_size'][1] / grid_cfg['grid_res'][1] - 1
    n3 = grid_cfg['grid_height'] / grid_cfg['grid_res'][2] - 1
    nx = torch.Tensor([n1, n2, n3])
    return dx, bx, nx


def create_frustum(final_dim, downsample, grid_cfg):
    """Create a frustum (D x H x W x 3), have D*H*W points, each point is
    (u,v,d).
    (u,v) is (x,y) pixel coords in input image, d is the depth.
    """
    # make grid in image plane
    ogfH, ogfW = final_dim
    fH, fW = int(ogfH // downsample), int(ogfW // downsample)
    if grid_cfg.depth_discretization == 'UD':
        depth_dist = torch.arange(*grid_cfg.dbound, dtype=torch.float)
    elif grid_cfg.depth_discretization == "LID":
        dmin, dmax, interval = grid_cfg.dbound
        num = int((dmax - dmin) / interval)
        depth_dist = torch.tensor([dmin + (dmax - dmin) / num / (num + 1) * i * (i + 1) for i in range(num)],
                                  dtype=torch.float)
    else:
        raise TypeError("unknown depth discretization")
    ds = depth_dist.view(-1, 1, 1).expand(-1, fH, fW)
    D, _, _ = ds.shape
    xs = torch.linspace(
        0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    ys = torch.linspace(
        0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
    frustum = torch.stack((xs, ys, ds), -1)
    # print("frustum ", frustum.shape)
    return nn.Parameter(frustum, requires_grad=False), depth_dist

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        # save kept for backward
        ctx.save_for_backward(kept)
        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None

@NECKS.register_module()
class Lift(nn.Module):

    def __init__(self, grid_cfg, img_size, downsample, channel_in, channel_out, **kwargs):
        super().__init__()

        dx, bx, nx = gen_dx_bx_new(grid_cfg)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.frustum, self.depth_dist = create_frustum(img_size, downsample, grid_cfg)
        self.D, _, _, _ = self.frustum.shape
        self.channel_out = channel_out
        self.downsample = downsample
        self.img_size = img_size
        self.depthnet = nn.Conv2d(channel_in, self.D + channel_out, kernel_size=1, padding=0)
        self.use_quickcumsum = True
        self.height_channel = nx[2]

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, undistorts):
        """Determine the (x,y,z) locations (in the ego frame) of the points in
        the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # with torch.cuda.amp.autocast(enabled=False):
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        # points = points.float()
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
       # points_t = points[0][1]
        #trans_t = trans.view(B, N, 1, 1, 1, 3)[0][1]
        #points_t = points_t + trans_t
        points += trans.view(B, N, 1, 1, 1, 3)
        return points
 #根据相机内外参将视锥中的点投影到ego坐标系,输出BxNxDxHxWx3
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, undistorts):
        """Determine the (x,y,z) locations (in the ego frame) of the points in
        the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # with torch.cuda.amp.autocast(enabled=False):
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        # points = points.float()
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
       # points_t = points[0][1]
       # trans_t = trans.view(B, N, 1, 1, 1, 3)[0][1]
       # points_t = points_t + trans_t
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)  # (B*N*D*H*W x 3),xyz
        batch_ix = torch.cat([torch.full(
            [Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix),
                               1)  # (B*N*D*H*W x 4), xyzi
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]  # (K x 4), xyzi
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        #  - 然后是一个神奇的操作，对每个体素中的点云特征进行sumpooling，
        # 代码中使用了cumsum_trick，巧妙地运用前缀和以及上述argsort的索引。输出是去重之后的Voxel特征，BxCxZxXxY
        # gem_feats  (M x 4), xyzi
        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2],
        geom_feats[:, 1], geom_feats[:, 0]] = x
        # geom_feats B x N x D x H/downsample x W/downsample x 3
        # collapse Z
        if self.neck == '2d':
            final = torch.cat(final.unbind(dim=2), 1)
        else:
            final = final.permute(0, 1, 3, 4, 2)
        return final

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)  # (B*N*D*H*W x 3),xyz
        batch_ix = torch.cat([torch.full(
            [Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix),
                               1)  # (B*N*D*H*W x 4), xyzi
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]  # (K x 4), xyzi
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        # gem_feats  (M x 4), xyzi
        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, int(self.nx[2].item()), int(self.nx[1].item()), int(self.nx[0].item())), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2],
        geom_feats[:, 1], geom_feats[:, 0]] = x
        # geom_feats B x N x D x H/downsample x W/downsample x 3
        # collapse Z
        if self.neck == '2d':
            final = final.unbind(dim=2)[0]
        else:
            final = final.permute(0, 1, 3, 4, 2)
        return final

    def forward(self, x, intrins, imu2c, post_rots, post_trans, undistorts, **kwargs):
        self.neck = kwargs['neck']
        x = np.squeeze(x)
        x = self.depthnet(x)
        # print("camfeat after depthnet",x.shape)
        depth = self.get_depth_dist(x[:, :self.D])
        # print("depth",depth.shape, self.D, self.C)  160, 240
        new_x = depth[:, :self.D].unsqueeze(
            1) * x[:, self.D:(self.D + self.channel_out)].unsqueeze(2)
        x = new_x.view(-1, intrins.shape[1], self.channel_out, self.D,
                       int(self.img_size[0] // self.downsample), int(self.img_size[1] // self.downsample))
        x = x.permute(0, 1, 3, 4, 5, 2)
        c2imu = torch.inverse(imu2c)
        rots = c2imu[..., :3, :3]
        trans = c2imu[..., :3, -1]
        intrins = intrins[..., :3, :3]
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, undistorts)
        x = self.voxel_pooling(geom, x)
        return x

















"""
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/workspace/lane')
import torch.nn.functional as F
from mmdet3d.models.builder import NECKS


def gen_dx_bx(grid_cfg):
    dx = torch.Tensor(grid_cfg['grid_res'])
    bx = torch.Tensor([off + res / 2.0 for off, res in zip(grid_cfg['offset'], grid_cfg['grid_res'])])
    n1 = grid_cfg['grid_size'][0] / grid_cfg['grid_res'][0] - 1
    n2 = grid_cfg['grid_size'][1] / grid_cfg['grid_res'][1] - 1
    n3 = grid_cfg['grid_height'] / grid_cfg['grid_res'][2] - 1
    nx = torch.Tensor([n1, n2, n3])
    return dx, bx, nx


# 生成视锥
def create_frustum(final_dim, downsample, grid_cfg):
    Create a frustum (D x H x W x 3), have D*H*W points, each point is
    (u,v,d).
    (u,v) is (x,y) pixel coords in input image, d is the depth.
    # make grid in image plane
    ogfH, ogfW = final_dim
    fH, fW = int(ogfH // downsample), int(ogfW // downsample)
    if grid_cfg.depth_discretization == 'UD':
        depth_dist = torch.arange(*grid_cfg.dbound, dtype=torch.float)
    elif grid_cfg.depth_discretization == "LID":
        dmin, dmax, interval = grid_cfg.dbound
        num = int((dmax - dmin) / interval)
        depth_dist = torch.tensor([dmin + (dmax - dmin) / num / (num + 1) * i * (i + 1) for i in range(num)],
                                  dtype=torch.float) # 生成深度张量
    else:
        raise TypeError("unknown depth discretization")
    ds = depth_dist.view(-1, 1, 1).expand(-1, fH, fW)
    D, _, _ = ds.shape
    xs = torch.linspace(
        0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    ys = torch.linspace(
        0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
    frustum = torch.stack((xs, ys, ds), -1)  # （D， fH, fW, 3） 3是（h,w,d）坐标
    # print("frustum ", frustum.shape)
    return nn.Parameter(frustum, requires_grad=False), depth_dist

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])  # 删选前后不相等的值

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1])) # 错位相减

    return x, geom_feats

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])  # 前后行不相同
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1])) # 错位相减，得到差值
        # save kept for backward
        ctx.save_for_backward(kept)
        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None

@NECKS.register_module()
class Lift(nn.Module):

    def __init__(self, grid_cfg, img_size, downsample, channel_in, channel_out, **kwargs):
        super().__init__()

        dx, bx, nx = gen_dx_bx(grid_cfg)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.frustum, self.depth_dist = create_frustum(img_size, downsample, grid_cfg)
        self.D, _, _, _ = self.frustum.shape
        self.channel_out = channel_out
        self.downsample = downsample
        self.img_size = img_size
        self.depthnet = nn.Conv2d(channel_in, self.D + channel_out, kernel_size=1, padding=0)
        self.use_quickcumsum = True
        self.height_channel = nx[2]

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, undistorts):
        Determine the (x,y,z) locations (in the ego frame) of the points in
        the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        # with torch.cuda.amp.autocast(enabled=False):
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        # points = points.float()
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        #points_t = points[0][1]
        #trans_t = trans.view(B, N, 1, 1, 1, 3)[0][1]
        #points_t = points_t + trans_t
        points += trans.view(B, N, 1, 1, 1, 3)  # 2, 1, 110, 60, 40
        return points
# 视锥从像素投影到ego坐标系

     根据变换后的点和点云构建bev特征
- VoxelPooling()：根据视锥点云在ego周围空间的位置索引，把点云特征分配到BEV pillar中，然后对同一个pillar中的点云特征进行sum-pooling处理，输出B,C,X,Y的BEV特征
  - 首先将点云特征reshape成MxC，其中M=BxNxDxHxW
  - 然后将GetGeometry()输出的空间点云转换到体素坐标下，得到对应的体素坐标。并通过范围参数过滤掉无用的点
  - 将体素坐标展平，reshape成一维的向量，然后对体素坐标中B、X、Y、Z的位置索引编码，然后对位置进行argsort，这样就把属于相同BEV pillar的体素放在相邻位置，得到点云在体素中的索引
  - 然后是一个神奇的操作，对每个体素中的点云特征进行sumpooling，代码中使用了cumsum_trick，巧妙地运用前缀和以及上述argsort的索引。输出是去重之后的Voxel特征，BxCxZxXxY
  - 最后使用unbind将Z维度切片，然后cat到C的维度上。Z维度为1，实际效果就是去掉了Z维度，输出为BxCxXxY的BEV 特征图
    
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape  # 
        Nprime = B * N * D* H * W
        # 8, 1, 110, 60, 40, 64, Nprime4224000
        # flatten x
        x = x.reshape(Nprime, C) # 点云特征

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()   # 转换到体素坐标
        geom_feats = geom_feats.view(Nprime, 3)  # (B*N*D*H*W x 3),xyz  3就是（x, y, z）  4224000, 3
        batch_ix = torch.cat([torch.full(
            [Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix),
                               1)  # (B*N*D*H*W x 4), xyzi
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]  # (K x 4), xyzi  展平，相加的目的是什么---在每一个体素去最大？
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()  # 对他们的几何特征排序, 764749--K,对每一个相同坐标？
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 直接·对这些特征重排序，后面会reshape,然后做一些操作
        # (764749, 4),
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)  # sumpooling，这里传入的是x, geom_feats
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        # gem_feats  (M x 4), xyzi,x :(M, 64)
        # griddify (B x C x Z x X x Y)
        #final = torch.zeros(
        #    (B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        final = torch.zeros(
            (B, C, int(self.nx[2].item()), int(self.nx[1].item()), int(self.nx[0].item())), device=x.device)
        
        final[geom_feats[:, 3], :, geom_feats[:, 2],    # (2, 64, 9, 239, 95)
        geom_feats[:, 1], geom_feats[:, 0]] = x  
        # geom_feats B x N x D x H/downsample x W/downsample x 3
        # collapse Z
        if self.neck == '2d':
           final = torch.cat(final.unbind(dim=2), 1)# 去掉
        #if self.neck == '2d':
        #   final = final.unbind(dim=2)[0]# 去掉z  (18, 64, 239, 95)
        else:
            final = final.permute(0, 1, 3, 4, 2)  
        return final

    def forward(self, x, intrins, imu2c, post_rots, post_trans, undistorts, **kwargs):
        self.neck = kwargs['neck']
        x = np.squeeze(x)  #(8,1, 4, 40, 60)
        x = self.depthnet(x)  # (8, 174, 160, 240)
        # x = self.depthnet(x.squeeze(0))
        # print("camfeat after depthnet",x.shape)
        depth = self.get_depth_dist(x[:, :self.D])  # DEPTH(8, 110, 160, 240), self.D:110
        # print("depth",depth.shape, self.D, self.C)
        new_x = depth[:, :self.D].unsqueeze(
            1) * x[:, self.D:(self.D + self.channel_out)].unsqueeze(2)   # (N， 64， 110， 160， 240)
        x = new_x.view(-1, intrins.shape[1], self.channel_out, self.D,
                       int(self.img_size[0] // self.downsample), int(self.img_size[1] // self.downsample))  # 前D列操作，改变形状
        x = x.permute(0, 1, 3, 4, 5, 2)    # (8, 1, 64, 110, 60, 40)
        c2imu = torch.inverse(imu2c)
        rots = c2imu[..., :3, :3]
        trans = c2imu[..., :3, -1]
        intrins = intrins[..., :3, :3]
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, undistorts)   # 转换到自车空间
        x = self.voxel_pooling(geom, x)
        return x

"""







