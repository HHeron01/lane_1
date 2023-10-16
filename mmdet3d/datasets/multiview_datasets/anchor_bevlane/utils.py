import matplotlib
import numpy as np
import math

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams['figure.figsize'] = (35, 30)



def resample_laneline_in_y(input_lane, y_steps, x_min, poly_order=3, sample_step=1, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])
    y_max = np.max(input_lane[:, 1])

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    # f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    # f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    # #
    # x_values = f_x(y_steps)
    # z_values = f_z(y_steps)

    # x_values = input_lane[:, 0]
    # z_values = input_lane[:, 2]

    xs_gt = input_lane[:, 0]
    ys_gt = input_lane[:, 1]
    zs_gt = input_lane[:, 2]

    poly_params_yx = np.polyfit(ys_gt, xs_gt, deg=poly_order)
    poly_params_yz = np.polyfit(ys_gt, zs_gt, deg=poly_order)

    y_min, y_max = np.min(ys_gt), np.max(ys_gt)
    y_min = math.floor(y_min)
    y_max = math.ceil(y_max)

    # y_sample = np.array(range(y_min, y_max, sample_step))
    # y_values = np.array(y_sample, dtype=np.float32)
    y_values = y_steps

    x_values = np.polyval(poly_params_yx, y_values)
    z_values = np.polyval(poly_params_yz, y_values)

    #adjust coordination
    # x_values = x_values - x_min

    if out_vis:
        output_visibility_y = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        output_visibility_x = np.logical_and(x_values >= x_min, x_values <= -x_min)
        output_visibility = output_visibility_x * output_visibility_y
        # x_values = x_values[output_visibility]
        # z_values = z_values[output_visibility]
        return x_values, z_values, output_visibility.astype(np.float32)# + 1e-9
    return x_values, z_values


def resample_laneline_in_y_with_vis(input_lane, y_steps, vis_vec):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    f_vis = interp1d(input_lane[:, 1], vis_vec, fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)
    vis_values = f_vis(y_steps)

    x_values = x_values[vis_values > 0.5]
    y_values = y_steps[vis_values > 0.5]
    z_values = z_values[vis_values > 0.5]
    return np.array([x_values, y_values, z_values]).T