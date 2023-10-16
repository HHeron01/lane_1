import torch

def smart_collate_fn(batch):

    imgs = []
    intrins = []
    extrinsics = []
    post_rots = []
    post_trans = []
    undists = []
    bda_rots = []
    rots = []
    trans = []
    grids = []
    drop_idxs = []

    gt_masks = []
    mask_hafs = []
    mask_vafs = []
    mask_offsets = []
    mask_zs = []

    gt_lanes = []
    file_paths = []
    for data in batch:
        img, intrin, extrinsic, post_rot, post_tran, undist, \
        bda_rot, rot, tran, grid, drop_idx = data['img_inputs']

        imgs.append(img)
        intrins.append(intrin)
        extrinsics.append(extrinsic)
        post_rots.append(post_rot)
        post_trans.append(post_tran)
        undists.append(undist)
        bda_rots.append(bda_rot)
        rots.append(rot)
        trans.append(tran)
        grids.append(grid)
        drop_idxs.append(drop_idx)

        gt_mask, mask_haf, mask_vaf, mask_offset, mask_z = data['maps']

        gt_masks.append(gt_mask)
        mask_hafs.append(mask_haf)
        mask_vafs.append(mask_vaf)
        mask_offsets.append(mask_offset)
        mask_zs.append(mask_z)

        gt_lane = data['gt_lanes']
        file_path = data['file_path']

        gt_lanes.append(gt_lane)
        file_paths.append(file_path)

    data_batch = {
        'img_metas': torch.tensor([]),
        'img_inputs': (torch.stack(imgs), torch.stack(intrins), torch.stack(extrinsics), torch.stack(post_rots),
                       torch.stack(post_trans), torch.stack(undists), torch.stack(bda_rots), torch.stack(rots),
                 torch.stack(trans), torch.stack(grids), torch.stack(drop_idxs)),
        "maps": (torch.stack(gt_masks), torch.stack(mask_hafs), torch.stack(mask_vafs), torch.stack(mask_offsets),
                 torch.stack(mask_zs)),
        'gt_lanes': gt_lanes,
        'file_path': file_paths,
    }


    #
    # data_batch = {
    #     # 'imgs': torch.stack([data['imgs'] for data in batch]),
    #     # 'trans': torch.stack([data['trans'] for data in batch]),
    #     # 'rots': torch.stack([data['rots'] for data in batch]),
    #     # 'extrinsics': torch.stack([data['extrinsics'] for data in batch]),
    #     # 'intrins': torch.stack([data['intrins'] for data in batch]),
    #     # 'undists': torch.stack([data['undists'] for data in batch]),
    #     # 'post_trans': torch.stack([data['post_trans'] for data in batch]),
    #     # 'post_rots': torch.stack([data['post_rots'] for data in batch]),
    #     # 'mask_offset': torch.stack([data['mask_offset'] for data in batch]),
    #     # 'mask_haf': torch.stack([data['mask_haf'] for data in batch]),
    #     # 'mask_vaf': torch.stack([data['mask_vaf'] for data in batch]),
    #     # 'mask_z': torch.stack([data['mask_z'] for data in batch]),
    #     # 'grid': torch.stack([data['grid'] for data in batch]),
    #     # 'drop_idx': torch.stack([data['drop_idx'] for data in batch]),
    #     # 'img_metas': torch.stack([data['img_metas'] for data in batch]),
    #     # 'img_inputs': torch.stack([data['img_inputs'] for data in batch]),
    #     # 'maps': torch.stack([data['maps'] for data in batch]),
    #     'gt_lanes': [data['gt_lanes'] for data in batch],
    #     'file_path': [data['file_path'] for data in batch],
    # }

    return data_batch