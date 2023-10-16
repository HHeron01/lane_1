import torch
import numpy as np
import math
import cv2
from torch import nn

INFINITY = 987654.


def match_proposals_with_targets(model, proposals, targets, t_pos=15., t_neg=18.):
    num_category = model.num_category
    n_strips = model.n_strips
    n_offsets = model.n_offsets
    # repeat proposals and targets to generate all combinations
    num_proposals = proposals.shape[0]
    num_targets = targets.shape[0]
    # pad proposals and target for the valid_offset_mask's trick
    proposals_pad = proposals.new_zeros(proposals.shape[0], proposals.shape[1] + 1)
    proposals_pad[:, :-1] = proposals
    proposals = proposals_pad
    targets_pad = targets.new_zeros(targets.shape[0], targets.shape[1] + 1)
    targets_pad[:, :-1] = targets
    targets = targets_pad

    proposals = torch.repeat_interleave(proposals, num_targets,
                                        dim=0)  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b]

    targets = torch.cat(num_proposals * [targets])  # applying this 2 times on [c, d] gives [c, d, c, d]

    # get start and the intersection of offsets
    targets_starts = targets[:, num_category] #* n_strips
    proposals_starts = proposals[:, num_category] * n_strips

    starts = torch.max(targets_starts, proposals_starts).round().long()
    last_vis_idx = targets_starts.new_zeros(targets_starts.shape, dtype=torch.long)
    target_vis = targets[:, model.num_category + 3 + 2 * n_offsets:]
    x_s, y_s = torch.nonzero(target_vis == 1, as_tuple=True)
    # x_s, y_s = torch.nonzero(target_vis, as_tuple=True)
    new_last_vis_idx = np.zeros(last_vis_idx.shape, dtype=np.int)
    new_last_vis_idx[x_s.cpu().numpy()] = y_s.cpu().numpy().astype(np.int)
    ends = torch.from_numpy(new_last_vis_idx).to(targets.device)

    lengths = ends - starts + 1
    ends[lengths < 0] = starts[lengths < 0] - 1
    lengths[lengths < 2] = 0

    valid_offsets_mask = targets.new_zeros(targets.shape)
    all_indices = torch.arange(valid_offsets_mask.shape[0], dtype=torch.long, device=targets.device)
    #   put a one on index `start`, giving [0, 1, 0, 0, 0]
    valid_offsets_mask[all_indices, num_category + 3 + starts] = 1.
    valid_offsets_mask[all_indices, num_category + 3 + ends + 1] -= 1.

    # the cumsum gives [0, 1, 1, 0, 0], the correct mask for the offsets
    valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) != 0.
    invalid_offsets_mask = ~valid_offsets_mask
    # cv2.imwrite('./valid_offsets_mask.png', valid_offsets_mask.detach().cpu().numpy() * 100)

    # compute distances
    # this compares [ac, ad, bc, bd], i.e., all combinations
    # debug_targets = targets[1, :] * valid_offsets_mask.float()
    # debug_proposals = proposals[1, :] * valid_offsets_mask.float()

    distances = torch.abs((targets - proposals) * valid_offsets_mask.float()).sum(dim=1) / (lengths.float() + 1e-9
                                                                                            )  # avoid division by zero

    distances[lengths == 0] = INFINITY
    invalid_offsets_mask = invalid_offsets_mask.view(num_proposals, num_targets, invalid_offsets_mask.shape[1])
    distances = distances.view(num_proposals, num_targets)  # d[i,j] = distance from proposal i to target j

    positives = distances.min(dim=1)[0] < t_pos
    negatives = distances.min(dim=1)[0] > t_neg

    if positives.sum() == 0:
        target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
    else:
        target_positives_indices = distances[positives].argmin(dim=1)
    invalid_offsets_mask = invalid_offsets_mask[positives, target_positives_indices]

    return positives, invalid_offsets_mask[:, :-1], negatives, target_positives_indices



