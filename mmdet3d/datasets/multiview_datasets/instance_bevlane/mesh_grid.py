from typing import List
import cv2
import numpy as np
import torch



class MeshGrid(object):
    def __init__(self, width_range, depth_range, width_res, depth_res) -> None:
        """
        Init: Need 6 params for initing mesh grid

        Input:
            Plan A: unit m
                width: start, end, res
                depth: start, end, res
            Plan B: unit grid pixel
                width: size, offset, res
                depth: size, offset, res

        MeshGrid vs RFU-baselink
                        (w_end, d_end)
            -----------------
            |       ^ y     |
            |       |       |
            |       |---> x |
            |     origin    |
            |               |
            -----------------
        (w_start, d_start)
        """
        self.update_grid_param(width_range, depth_range, width_res, depth_res)
        self.generate_grid_map()

    def __call__(self):
        pass

    def update_grid_param(self, width_range: List, depth_range: List,
                          width_res: float, depth_res: float):
        self.width_start = width_range[0]
        self.width_end = width_range[1]
        self.depth_start = depth_range[0]
        self.depth_end = depth_range[1]
        self.width_res = width_res
        self.depth_res = depth_res
        self.width_res_2 = self.width_res * 0.5  # 0.1
        self.depth_res_2 = self.depth_res * 0.5  # 0.15
        self.width = self.width_end - self.width_start
        self.depth = self.depth_end - self.depth_start
        self.accuracy = 2

    def get_start(self):
        return self.width_start, self.depth_start

    def get_end(self):
        return self.width_end, self.depth_end

    def get_res(self):
        return self.width_res, self.depth_res

    def generate_grid_map(self):
        grid_width = round(self.width / self.width_res)
        grid_depth = round(self.depth / self.depth_res)
        self.grid_size = (grid_width, grid_depth)  # (192， 320)
        u = round(-self.width_start / self.width_res)
        v = round(self.depth_end / self.depth_res)
        self.grid_offset = (u, v)  # (96, 320)
        self.grid_res = (self.width_res, self.depth_res)

        self.img = np.random.rand(grid_depth, grid_width, 3)

    def get_grid_size(self):
        return self.grid_size

    def get_grid_offset(self):
        return self.grid_offset

    def get_grid_res(self):
        return self.grid_res

    def is_index_outside(self, u, v):
        return (u >= self.grid_size[0]) or (u < 0) or \
               (v >= self.grid_size[1]) or (v < 0)

    def is_pos_outside(self, x, y):
        re1 = (x >= self.width_end)
        re2 = (x <= self.width_start)
        re3 = (y >= self.depth_end)
        re4 = (y <= self.depth_start)
        return re1 or re2 or re3 or re4

    def show_grid(self, win_name="mesh_grid", wait_time=3000, amplify=1):
        disp_shape = (self.img.shape[1] * amplify, self.img.shape[0] * amplify)
        disp_img = cv2.resize(self.img, disp_shape)
        cv2.imshow(win_name, disp_img)
        cv2.waitKey(wait_time)

    def save_grid(self, file_name="./mesh_grid.jpeg", amplify=1):
        disp_shape = (self.img.shape[1] * amplify, self.img.shape[0] * amplify)
        disp_img = cv2.resize(self.img * 255, disp_shape)
        cv2.imwrite(file_name, disp_img)

    def get_center_index(self):
        return self.grid_offset

    def get_center_pos(self):
        return (-self.width_start, self.depth_end)


    def get_index_by_pos(self, x, y):
        """
        RFU coordinate system for IMU/EGO-CAR
            y
            ^
            |
            |   .(x0, y0)
            |
            |---------> x
            O
        output: [u, v] in grid
            the line at up and left is belong the cell
        """
        # u = int((x - self.width_start) / self.width_res)
        # v = int(-(y - self.depth_end) / self.depth_res)
        u = round((x - self.width_start) / self.width_res)
        v = round(-(y - self.depth_end) / self.depth_res)
        return u, v

    def get_anchor_offset_x(self):
        pass


    def get_pos_by_index(self, u, v):
        """
        mesh-grid
            O----------->u(x)
            |
            |
            |
            |      .(u0, v0)
            |
            v(y)
        output: [x, y] in RFU-baselink
            pos is left right corner, not cell center
        """
        x = self.width_start + u * self.width_res
        y = self.depth_end - v * self.depth_res
        return x, y

    def draw_center_point(self):
        poi = self.get_center_index()
        cv2.circle(self.img, poi, radius=3, color=(0, 0, 0), thickness=2)

    def draw_point_by_index(self, index):
        poi = (index[0], index[1])
        cv2.circle(self.img, poi, radius=3, color=(0, 255, 0), thickness=2)

    def draw_point_by_pos(self, pos):
        index = self.get_index_by_pos(pos[0], pos[1])
        self.draw_point_by_index(index)

    def get_cell_center_pos(self, u, v):
        offset_x = int(u) * self.width_res + self.width_res_2
        offset_y = int(v) * self.depth_res + self.depth_res_2
        x = self.width_start + offset_x
        y = self.depth_end - offset_y
        return round(x, self.accuracy), round(y, self.accuracy)

    def get_cell_corners(self, u, v):
        center_x, center_y = self.get_cell_center_pos(u, v)
        corner_left_up = (center_x - self.width_res_2, center_y + self.depth_res_2)
        corner_left_down = (center_x - self.width_res_2, center_y - self.depth_res_2)
        corner_right_up = (center_x + self.width_res_2, center_y + self.depth_res_2)
        corner_right_down = (center_x + self.width_res_2, center_y - self.depth_res_2)
        return corner_left_up, corner_left_down, corner_right_up, corner_right_down

    def get_offset_with_cell_center(self, x, y):
        u, v = self.get_index_by_pos(x, y)
        center_x, center_y = self.get_cell_center_pos(u, v)
        offset_x = x - center_x
        offset_y = y - center_y

        ratio_offset_x = offset_x / self.width_res
        ratio_offset_y = offset_y / self.depth_res
        # print("ratio_offset_x:", ratio_offset_x, ratio_offset_y)
        # return round(offset_x, self.accuracy), round(offset_y, self.accuracy)
        # 像素到bev---自车（最后的米坐标系）应该是格子在做
        return round(ratio_offset_x, self.accuracy), round(ratio_offset_y, self.accuracy)

    def get_offset_with_cell_ld(self, x, y):
        bev_x = x/self.width_res
        bev_y = y/self.depth_res
        
        offset_x = bev_x - round(bev_x) + 0.5
        offset_y = bev_y - round(bev_y) + 0.5
        return round(offset_x, self.accuracy), round(offset_y, self.accuracy)

    # @torchsnooper.snoop()
    def make_grid(self):
        """
        Constructs an array representing the corners of
        an orthographic grid in camera coordinate.
        """
        width, depth = self.grid_size
        x_offset, z_offset = self.grid_offset
        x_res, z_res = self.grid_res
        x_coords = torch.arange(0., width, x_res) + x_offset
        z_coords = torch.arange(0., depth, z_res) + z_offset
        print('x_coords', x_coords.shape)
        print('z_coords', z_coords.shape)
        zz, xx = torch.meshgrid(z_coords, x_coords)
        print('xx', xx.shape)
        print('zz', zz.shape)
        out = torch.stack([xx, torch.full_like(xx, 1), zz], dim=-1)
        print('out', out.shape)
        return out

    def make_grid_2(self):
        width, depth = self.grid_size
        xoff, yoff = self.grid_offset
        xcoords = torch.linspace(xoff, width + xoff, int(width / self.grid_res[0]))
        ycoords = torch.linspace(yoff, depth + yoff, int(depth / self.grid_res[1]))
        print('xcoords', xcoords.shape)
        print('ycoords', ycoords.shape)
        yy, xx = torch.meshgrid(ycoords, xcoords)
        print('xx', xx.shape)
        print('yy', yy.shape)
        return torch.stack([xx, yy, torch.full_like(xx, 1.08)], dim=-1)



"""
TEST FUNCTION
"""

def test_api(mesh):
    out = mesh.get_start()
    out = mesh.get_end()
    out = mesh.get_res()
    out = mesh.get_grid_size()
    out = mesh.get_grid_offset()
    out = mesh.get_grid_res()
    out = mesh.get_center_index()
    out = mesh.is_index_outside(out[0], out[1])
    out = mesh.get_center_pos()
    out = mesh.is_pos_outside(out[0], out[1])

    out = mesh.is_pos_outside(8.0, 3.0)
    out = mesh.is_pos_outside(88.0, 3.0)
    out = mesh.is_pos_outside(8.0, 333.0)

    out = mesh.get_cell_center_pos(0, 0)
    out = mesh.get_cell_center_pos(1, 1)
    out = mesh.get_cell_corners(0, 0)
    out = mesh.get_cell_corners(1, 1)

    out = mesh.get_offset_with_cell_center(mesh.width_start, mesh.depth_end)
    out = mesh.get_offset_with_cell_center(-19.1, 65.7)

    out = mesh.get_offset_with_cell_ld(-19.2, 66.)
    out = mesh.get_offset_with_cell_ld(19.2, -30.)
    pass

def test_show(mesh):
    center = mesh.get_center_index()
    mesh.draw_center_point()

    index = mesh.get_index_by_pos(0., 50.)
    mesh.draw_point_by_index(index)
    pos = (19.2, -30.0)
    index = mesh.get_index_by_pos(pos[0], pos[1])
    mesh.draw_point_by_index(index)

    pos = mesh.get_pos_by_index(index[0], index[1])
    mesh.draw_point_by_pos(pos)

    mesh.show_grid()
    mesh.save_grid()

def test_make_grid(mesh):
    mesh.make_grid_2()


if __name__ == "__main__":
    WIDTH_RANGE = (-19.2, 19.2)  # unit:m
    DEPTH_RANGE = (-30., 66.)  # unit:m
    WIDTH_RES = 0.2  # unit:m
    DEPTH_RES = 0.6  # unit:m
    mesh = MeshGrid(WIDTH_RANGE, DEPTH_RANGE, WIDTH_RES, DEPTH_RES)

    test_api(mesh)

