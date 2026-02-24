import os
CUDA_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"  # 改成你实际安装的版本

os.environ["CUDA_PATH"] = CUDA_PATH
os.environ["PATH"]   = rf"{CUDA_PATH}\bin;{CUDA_PATH}\libnvvp;" + os.environ.get("PATH", "")
os.environ["INCLUDE"]= rf"{CUDA_PATH}\include;" + os.environ.get("INCLUDE", "")
os.environ["LIB"]    = rf"{CUDA_PATH}\lib\x64;" + os.environ.get("LIB", "")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;8.9"  # 3090 + 4090

import numpy as np
import collections
import torch
import math
from pytorch3d.structures import Meshes
from torch import nn
import json
from pytorch3d.utils import ico_sphere

from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
import struct

from gsplat import rasterization
import trimesh
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pytorch3d.renderer import (
    TexturesVertex,
)
import cv2
import time
from tqdm import tqdm
import gc
import re

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
BaseCamera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Camera(BaseCamera):
    @property
    def K(self):
        K = np.eye(3)
        if self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL" or self.model == "RADIAL" or self.model == "SIMPLE_RADIAL_FISHEYE" or self.model == "RADIAL_FISHEYE":
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[0]
            K[0, 2] = self.params[1]
            K[1, 2] = self.params[2]
        elif self.model == "PINHOLE" or self.model == "OPENCV" or self.model == "OPENCV_FISHEYE" or self.model == "FULL_OPENCV" or self.model == "FOV" or self.model == "THIN_PRISM_FISHEYE":
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[1]
            K[0, 2] = self.params[2]
            K[1, 2] = self.params[3]
        else:
            raise NotImplementedError
        return K


def camera_to_intrinsic(camera):
    '''
    camera object to intrinsic matrix
    fx 0  cx
    0  fy cy
    0  0  1
    '''
    return np.array([
        [camera.params[0], 0, camera.params[2]],
        [0, camera.params[1], camera.params[3]],
        [0, 0, 1]
    ])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

    def to_transform_mat(self):
        '''
        R, t matrix
        '''
        R = self.qvec2rotmat()
        t = self.tvec
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @property
    def world_to_camera(self) -> np.ndarray:
        R = qvec2rotmat(self.qvec)
        t = self.tvec
        world2cam = np.eye(4)
        world2cam[:3, :3] = R
        world2cam[:3, 3] = t
        return world2cam


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


##################################
def create_arrow_mesh(
        start_pt: torch.Tensor,
        end_pt: torch.Tensor,
        radius: float = 0.01,
        n_sectors: int = 16,
        color_start=(1.0, 0.0, 0.0),
        color_end=(0.0, 1.0, 0.0)
) -> Meshes:
    """
    在世界坐标系里创建一个从 start_pt 指向 end_pt 的简易“箭头”。
    - 用圆柱模拟，底部/顶部不同颜色。

    Args:
        start_pt: (3,) 张量, 箭头尾部在世界坐标的坐标
        end_pt:   (3,) 张量, 箭头头部在世界坐标的坐标
        radius:   圆柱体半径
        n_sectors:  圆周方向离散的段数（越大越圆）
        color_start: 底色 (R, G, B)
        color_end:   顶色 (R, G, B)
    Returns:
        一个 PyTorch3D 的 Meshes 对象（batch=1），其顶点已经变换到世界坐标系。
    """
    device = start_pt.device

    # 1) 算出箭头向量
    vec = end_pt - start_pt  # shape (3,)
    length = torch.norm(vec)
    if length < 1e-7:
        # 如果 start==end，就返回一个非常小的点
        verts = torch.tensor([[0, 0, 0]], device=device, dtype=torch.float32)
        faces = torch.tensor([[0, 0, 0]], device=device, dtype=torch.int64)
        colors = torch.tensor([[color_start]], device=device, dtype=torch.float32)
        return Meshes(verts=[verts], faces=[faces],
                      textures=TexturesVertex(verts_features=colors))

    # 2) 在局部坐标系下生成从 z=0 到 z=1 的圆柱 (不包含顶部圆锥，简单示例)
    #    - 底部圆是 z=0, 半径=1
    #    - 顶部圆是 z=1, 半径=1
    #    - 然后再统一缩放到所需半径和长度
    # 先生成圆周的点
    theta = torch.linspace(0, 2 * math.pi, n_sectors, device=device)[:-1]  # 去掉最后一个与第一个重复
    circle_x = torch.cos(theta)
    circle_y = torch.sin(theta)
    # 顶点: 两个圈 + center? 这里我们只做外壁即可
    # bottom circle (z=0)
    bottom_verts = torch.stack([circle_x, circle_y, torch.zeros_like(circle_x)], dim=-1)  # (n_sectors-1, 3)
    # top circle (z=1)
    top_verts = bottom_verts.clone()
    top_verts[:, 2] = 1.0

    verts_local = torch.cat([bottom_verts, top_verts], dim=0)  # shape (2*(n_sectors-1), 3)

    # 3) 拼接三角形索引 (环绕建墙)
    #    底圈索引 [0..n-1], 顶圈索引 [n..2n-1]
    n = bottom_verts.shape[0]
    faces = []
    for i in range(n):
        i_next = (i + 1) % n
        # 两个三角形 (i, i_next, i+n), (i_next, i+n, i_next+n)
        faces.append([i, i_next, i + n])
        faces.append([i_next, i + n, i_next + n])
    faces = torch.tensor(faces, device=device, dtype=torch.int64)

    # 4) 给每个顶点设置颜色 (底-> color_start, 顶-> color_end)
    #    先给底部一个颜色, 顶部另一个颜色
    color_start_t = torch.tensor(color_start, device=device, dtype=torch.float32)
    color_end_t = torch.tensor(color_end, device=device, dtype=torch.float32)
    colors_local_bottom = color_start_t.unsqueeze(0).expand(n, -1)  # shape (n,3)
    colors_local_top = color_end_t.unsqueeze(0).expand(n, -1)  # shape (n,3)
    colors_local = torch.cat([colors_local_bottom, colors_local_top], dim=0)  # (2n,3)

    # 5) 把局部圆柱从半径=1, 高度=1 -> 缩放到 (radius, length)
    #    且让 z 轴对齐 vec 方向, bottom->start_pt
    #    做法：
    #      a) scale X,Y by radius, scale Z by length
    #      b) 旋转: z 轴(0,0,1) -> vec/|vec|
    #      c) 平移到 start_pt
    #
    #  a) scale
    #    local: (x, y, z)
    #    scaled: (x*radius, y*radius, z*length)
    scaled_verts = verts_local.clone()
    scaled_verts[:, 0] *= radius
    scaled_verts[:, 1] *= radius
    scaled_verts[:, 2] *= length

    #  b) 旋转
    #    需要一个旋转矩阵R: 把(0,0,1)映射到 direction = vec/|vec|
    #    可使用简单的"轴-角"方法，或 PyTorch3D 的辅助函数。
    direction = vec / length
    z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    # 用一个小函数来计算旋转(此处演示简易做法)
    R = _rotate_vec_a_to_b(z_axis, direction)

    # 应用 R
    scaled_verts = scaled_verts @ R.T  # (N,3)

    #  c) 平移
    scaled_verts += start_pt.unsqueeze(0)

    # 6) 组装为 Meshes
    mesh = Meshes(
        verts=[scaled_verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=colors_local.unsqueeze(0))
    )
    return mesh


def _rotate_vec_a_to_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    给定两个单位向量 a, b, 计算一个 3x3 旋转矩阵 R, 使得 R*a = b.
    简易版本，用于演示。
    """
    EPS = 1e-8
    a = a / (a.norm() + EPS)
    b = b / (b.norm() + EPS)

    cross = torch.cross(a, b)
    dot = (a * b).sum()
    if dot < -1.0 + EPS:
        # a, b 近似相反，把 a 旋转 180 度
        # 需找任意轴与 a 垂直
        # 例如: 选择 a 与(1,0,0)不平行，就可以 cross
        axis = torch.cross(a, torch.tensor([1.0, 0.0, 0.0], device=a.device))
        if axis.norm() < EPS:
            axis = torch.cross(a, torch.tensor([0.0, 1.0, 0.0], device=a.device))
        axis = axis / axis.norm()
        return _rotation_about_axis(axis, math.pi)
    else:
        # 常规情况
        # 公式: R = I + [v_x] + [v_x]^2 * 1/(1+dot)
        #      其中 v_x 是 cross(a,b) 的叉乘矩阵
        vx = _cross_matrix(cross)
        R = torch.eye(3, device=a.device) + vx + vx @ vx * (1.0 / (1.0 + dot + EPS))
        return R


def _cross_matrix(v: torch.Tensor) -> torch.Tensor:
    """
    给定 v=(x,y,z), 返回其对应的叉乘矩阵 [v_x]:
      [  0, -z,  y ]
      [  z,  0, -x ]
      [ -y,  x,  0 ]
    """
    x, y, z = v
    return torch.tensor([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], device=v.device, dtype=v.dtype)


def _rotation_about_axis(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rodrigues公式：绕单位轴 axis 旋转 angle 角度的 3x3 矩阵。
    """
    axis = axis / axis.norm()
    K = _cross_matrix(axis)
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    R = I + K * math.sin(angle) + K @ K * (1 - math.cos(angle))
    return R


# 计算相机中心在世界坐标中的位置
def get_camera_center(world2cam: torch.Tensor) -> torch.Tensor:
    """
    world2cam: shape (4,4)，表示 世界->相机
    返回 camera_center_world: shape (3,)
    """
    R = world2cam[:3, :3]
    t = world2cam[:3, 3]
    # camera_center = -R^T * t
    c = -R.T @ t
    return c




def create_sphere_mesh(
        center: torch.Tensor,
        radius: float = 0.05,
        color=(1.0, 0.0, 0.0),
        ico_level: int = 1,
        device: torch.device = torch.device("cuda:0")
) -> Meshes:
    """
    使用 PyTorch3D 的 ico_sphere 生成一个球体，并平移到指定世界坐标 center。
    给所有顶点赋予同一种颜色。

    Args:
        center: (3,) 张量, 球心在「世界坐标系」的位置.
        radius: 球体半径
        color:  (R, G, B), [0~1] 范围
        ico_level: ico_sphere 的细分等级(0~5), 越大网格越精细
        device:  放在哪个设备 (CPU 或 GPU)
    Returns:
        一个 PyTorch3D 的 `Meshes` 对象(batch=1)，其顶点/面都在世界坐标下。
    """
    # 1) 先生成一个以(0,0,0)为中心, 半径=1 的基础球体
    base_sphere = ico_sphere(level=ico_level, device=device)  # (batch=1) 的 Meshes

    # 2) 获取顶点、面
    base_verts = base_sphere.verts_packed()  # shape (V,3)
    base_faces = base_sphere.faces_packed()  # shape (F,3)

    # 3) 缩放 + 平移
    #    缩放: 让原本半径=1 的球体变成半径=radius
    #    平移: 顶点坐标 += center
    sphere_verts = base_verts * radius
    sphere_verts = sphere_verts + center.unsqueeze(0)  # 广播加, shape(1,3)

    # 4) 创建一个新的 Meshes 对象
    sphere_mesh = Meshes(verts=[sphere_verts], faces=[base_faces])

    # 5) 给每个顶点都赋予同一种颜色
    color_t = torch.tensor(color, dtype=torch.float32, device=device).view(1, 3)  # shape (1,3)
    color_per_vert = color_t.expand(sphere_verts.shape[0], -1).unsqueeze(0)  # shape (1,V,3)

    sphere_mesh.textures = TexturesVertex(verts_features=color_per_vert)

    return sphere_mesh

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

class VectorAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, axis=-1):
        defaults = dict(lr=lr, betas=betas, eps=eps, axis=axis)
        super(VectorAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VectorAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            axis = group['axis']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1-b1)
                if axis is not None:
                    dim = grad.shape[axis]
                    grad_norm = torch.norm(grad, dim=axis).unsqueeze(axis).repeat_interleave(dim, dim=axis)
                    grad_sq = grad_norm * grad_norm
                    g2.mul_(b2).add_(grad_sq, alpha=1-b2)
                else:
                    g2.mul_(b2).add_(grad.square(), alpha=1-b2)

                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                gr = m1 / (eps + m2.sqrt())
                p.data.sub_(gr, alpha=lr)

def read_blender(root_path, resize_ratio=1.0):
    train_path = os.path.join(root_path, 'train')
    image4_path = os.path.join(train_path, 'image4')
    data_np_path = os.path.join(train_path, 'data_np')

    img_list = os.listdir(image4_path)
    print(f"all data images: {len(img_list)}")

    img_name0 = img_list[0]
    img_path0 = os.path.join(image4_path, img_name0)
    # img size
    img0 = cv2.imread(img_path0)
    cam_height, cam_width = img0.shape[:2]
    # 缩放后的分辨率
    cam_height = int(cam_height * resize_ratio)
    cam_width = int(cam_width * resize_ratio)

    np_path0 = os.path.join(data_np_path, img_name0.split('.')[0] + '.npz')
    np_data0 = np.load(np_path0, allow_pickle=True)
    contents0 = np_data0['Cameras'].item()
    cam_model = 'PINHOLE'
    cam_id = 1
    cam_cx = cam_width / 2
    cam_cy = cam_height / 2

    fovx = contents0["camera_angle_x"]
    fovy = focal2fov(fov2focal(fovx, cam_height), cam_width)
    focal_length_x = fov2focal(fovx, cam_width)
    focal_length_y = fov2focal(fovy, cam_height)

    cam_params = np.array([focal_length_x, focal_length_y, cam_cx, cam_cy])

    colmap_camera = Camera(cam_id, cam_model, cam_width, cam_height, cam_params)

    # self.poses
    poses = []
    for image_name in img_list:
        np_path = os.path.join(data_np_path, image_name.split('.')[0] + '.npz')
        np_data = np.load(np_path, allow_pickle=True)
        Cameras = np_data['Cameras'].item()
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(Cameras['transform_matrix'])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        poses.append(pose)

    return poses, colmap_camera, img_list, image4_path

def read_nerf(root_path, resize_ratio=1.0):
    json_path = os.path.join(root_path, 'transforms_train.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']

    img_path0 = os.path.join(root_path, frames[0]['file_path'])
    # img size
    img0 = cv2.imread(img_path0)
    cam_height, cam_width = img0.shape[:2]
    # 缩放后的分辨率
    cam_height = int(cam_height * resize_ratio)
    cam_width = int(cam_width * resize_ratio)

    # camera parameters
    fovx = data['camera_angle_x']
    fovy = focal2fov(fov2focal(fovx, cam_height), cam_width)
    focal_length_x = fov2focal(fovx, cam_width)
    focal_length_y = fov2focal(fovy, cam_height)

    cam_model = 'PINHOLE'
    cam_id = 1
    cam_cx = cam_width / 2
    cam_cy = cam_height / 2
    cam_params = np.array([focal_length_x, focal_length_y, cam_cx, cam_cy])
    colmap_camera = Camera(cam_id, cam_model, cam_width, cam_height, cam_params)

    # self.poses and self.image_list
    poses = []
    img_list = []
    for frame in frames:
        file_path = frame['file_path']
        img_name = file_path.split('/')[-1]
        img_list.append(img_name)

        # get the world-to-camera transform and set R, T
        c2w = np.array(frame['transform_matrix'])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        poses.append(pose)
    image4_path = os.path.join(root_path, 'train')

    return poses, colmap_camera, img_list, image4_path

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_metashape(root_path, resize_ratio=1.0):
    try:
        cameras_extrinsic_file = os.path.join(root_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(root_path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(root_path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(root_path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 读取相机内参
    camera0 = cam_intrinsics[0]
    original_width = camera0.width
    original_height = camera0.height
    cam_height = int(original_height * resize_ratio)
    cam_width = int(original_width * resize_ratio)

    cam_model = camera0.model
    if cam_model == "PINHOLE":
        orig_flx = camera0.params[0]
        orig_fly = camera0.params[1]
        orig_FovX = focal2fov(orig_flx, original_width)
        orig_FovY = focal2fov(orig_fly, original_height)

        focal_length_x = fov2focal(orig_FovX, cam_width)
        focal_length_y = fov2focal(orig_FovY, cam_height)
        cam_id = 1
        cam_cx = cam_width / 2
        cam_cy = cam_height / 2
    else:
        orig_flx = camera0.params[0]
        orig_FovX = focal2fov(orig_flx, original_width)
        orig_FovY = focal2fov(orig_flx, original_height)

        focal_length_x = fov2focal(orig_FovX, cam_width)
        focal_length_y = fov2focal(orig_FovY, cam_height)
        cam_id = 1
        cam_cx = cam_width / 2
        cam_cy = cam_height / 2

    cam_params = np.array([focal_length_x, focal_length_y, cam_cx, cam_cy])
    colmap_camera = Camera(cam_id, cam_model, cam_width, cam_height, cam_params)

    # 读取相机外参
    poses = []
    img_list = []
    for idx, key in enumerate(cam_extrinsics):
        cam_extrinsic = cam_extrinsics[key]
        img_name = os.path.basename(cam_extrinsic.name)
        img_list.append(img_name)

        R = qvec2rotmat(cam_extrinsic.qvec)
        T = cam_extrinsic.tvec
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        poses.append(pose)
    image4_path = os.path.join(root_path, "images")

    return poses, colmap_camera, img_list, image4_path


def get_normal_img_mask(image_lst, image4_path, img_height, img_width, device):
    """
    获取正常的图像列表和掩码
    """
    # 读取 RGB 图像（BGR → RGB）
    gt_image_lst = []
    gt_tensor_lst = []
    gt_mask_lst = []
    for i, frame_ in enumerate(tqdm(image_lst)):
        gt_image_path = os.path.join(image4_path, frame_)
        gt_image_all = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
        # ========== 提取 RGB 和 mask ==========
        # if gt_image_all.shape[2] == 4:
        #     # 有 alpha 通道
        #     gt_image = gt_image_all[:, :, :3]
        #     gt_mask = gt_image_all[:, :, 3]
        #     gt_mask = (gt_mask / 255.0).astype(np.uint8)
        #     gt_image = gt_image * gt_mask[:, :, None]
        # else:
        gt_image = gt_image_all[:, :, :3]
        # mask = 非黑即白
        gray_img = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        threshold = 5  # 0~255 之间调节
        gt_mask = np.where(gray_img < threshold, 0, 1).astype(np.uint8)
        gt_image = gt_image * gt_mask[:, :, None]
            # gt_mask = np.ones_like(gt_image[:, :, 0])
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # ========== 缩放 RGB 图像和 mask ==========
        if gt_image.shape[:2] != (img_height, img_width):
            gt_image = cv2.resize(gt_image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        gt_image_lst.append(gt_image)
        # 归一化到 [0, 1] 并转换为 tensor
        gt_tensor = torch.tensor(gt_image, dtype=torch.float32) / 255.0
        gt_tensor = gt_tensor.unsqueeze(0).to(device)  # (1, H, W, 3)
        gt_tensor_lst.append(gt_tensor)
        gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        gt_mask_lst.append(gt_mask_tensor)

    return gt_image_lst, gt_tensor_lst, gt_mask_lst

def get_metashape_img_mask(image_lst, image4_path, img_height, img_width, device):
    """
        获取tree图像列表和掩码
    """
    # 读取 RGB 图像（BGR → RGB）
    gt_image_lst = []
    gt_tensor_lst = []
    gt_mask_lst = []

    mask_path = image4_path.replace('images', 'masks')

    for i, frame_ in enumerate(tqdm(image_lst)):
        gt_image_path = os.path.join(image4_path, frame_)
        gt_image_all = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
        # ========== 提取 RGB 和 mask ==========
        gt_image = gt_image_all
        # mask = 非黑即白
        mask_frame_path = os.path.join(mask_path, frame_.replace(".JPG", ".jpg"))
        gt_mask_all = cv2.imread(mask_frame_path, cv2.IMREAD_UNCHANGED)
        if gt_mask_all.max() > 1:
            gt_mask_all = gt_mask_all / 255.0
        gt_mask_all = cv2.rotate(gt_mask_all, cv2.ROTATE_90_CLOCKWISE)
        gt_mask = gt_mask_all.astype(np.uint8)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # ========== 缩放 RGB 图像和 mask ==========
        if gt_image.shape[:2] != (img_height, img_width):
            gt_image = cv2.resize(gt_image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        gt_image_lst.append(gt_image)
        # 归一化到 [0, 1] 并转换为 tensor
        gt_tensor = torch.tensor(gt_image, dtype=torch.float32) / 255.0
        gt_tensor = gt_tensor.unsqueeze(0).to(device)  # (1, H, W, 3)
        gt_tensor_lst.append(gt_tensor)
        gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        gt_mask_lst.append(gt_mask_tensor)

    return gt_image_lst, gt_tensor_lst, gt_mask_lst


def batched_dot(x, y):
    return (x * y).sum(dim=-1)

class blender_scene3D_mesh3dGS_subdivide_full_rand:
    def __init__(self, root_path, color_lr, offset_lr, sphere_radius, ico_level, device, use_vectorAdam, color_with_offset,
                 resize_ratio=1.0, use_nerf=False, use_metashape=False):
        self.device = device
        self.root_path = root_path

        self.color_lr = color_lr
        self.offset_lr = offset_lr

        self.use_vectorAdam = use_vectorAdam
        self.color_with_offset = color_with_offset
        self.resize_ratio = resize_ratio

        if use_nerf:
            poses, colmap_camera, image_list, image4_path = read_nerf(root_path, resize_ratio=resize_ratio)
        elif use_metashape:
            poses, colmap_camera, image_list, image4_path = read_metashape(root_path, resize_ratio=resize_ratio)
        else:
            poses, colmap_camera, image_list, image4_path = read_blender(root_path, resize_ratio=resize_ratio)
        self.poses = poses
        self.colmap_camera = colmap_camera # 已经改过size了
        self.image_list = image_list

        self.poses = np.array(self.poses)
        self.img_height, self.img_width = self.colmap_camera.height, self.colmap_camera.width
        self.intrinsic_mat = camera_to_intrinsic(self.colmap_camera)
        print('Intrinsic Matrix:', self.intrinsic_mat)
        print('Poses:', self.poses.shape)


        if use_metashape:
            gt_image_lst, gt_tensor_lst, gt_mask_lst = get_metashape_img_mask(self.image_list, image4_path, self.img_height, self.img_width, self.device)
        else:
            gt_image_lst, gt_tensor_lst, gt_mask_lst = get_normal_img_mask(self.image_list, image4_path, self.img_height, self.img_width, self.device)

        self.gt_image = gt_image_lst
        self.gt_tensor = gt_tensor_lst
        self.gt_mask = gt_mask_lst

        self.gt_image = np.array(self.gt_image)
        self.gt_tensor = torch.cat(self.gt_tensor, dim=0)
        self.gt_mask_tensor = torch.cat(self.gt_mask, dim=0).to(self.device).unsqueeze(-1)

        print('GT Image:', len(self.gt_image), self.gt_image.shape)
        print('GT Tensor:', len(self.gt_tensor), self.gt_tensor.shape)

        self.num_imgs = len(self.image_list)

        masked_tensor = self.gt_tensor * self.gt_mask_tensor  # (N, H, W, 3)
        valid_pixel_count = self.gt_mask_tensor.sum(dim=(0, 1, 2))  # scalar

        # 为防止除以0，添加一个 epsilon
        eps = 1e-6
        average_color = masked_tensor.sum(dim=(0, 1, 2)) / (valid_pixel_count + eps)
        average_color = tuple(average_color.tolist())

        print('Average Color (masked):', average_color)

        # 创建一个初始球体 Mesh
        point_world = np.array([0.0, 0.0, 0.0])
        # point_world = np.array([0.0, 0.0, sphere_radius // 3])
        self.init_mesh = create_sphere_mesh(device=self.device, ico_level=ico_level, radius=sphere_radius, color=average_color,
                                            center=torch.tensor(point_world, device=self.device, dtype=torch.float32))
        num_verts = self.init_mesh.verts_packed().shape[0]
        print('num_verts:', num_verts)

        self.init_mesh_verts_offsets = nn.Parameter(torch.zeros(num_verts, 3, device=self.device))
        self.colors_offsets_activation = torch.sigmoid
        # 获取原始颜色
        base_colors = self.init_mesh.textures.verts_features_packed()  # (num_verts, 3)
        # 反 Sigmoid 得到可训练参数
        init_color_params = torch.logit(base_colors, eps=1e-4)  # 避免极端值导致梯度爆炸
        # 颜色偏移量 直接变成可训练参数
        self.init_mesh_verts_colors = nn.Parameter(init_color_params)

        lr_param_groups = self.get_lr_param_groups(rate=1.0)

        self.subdivide = SubdivideMeshes()

        print('frame_scene3D init done')

    def get_lr_param_groups(self, rate=1.0):
        """
        为 init_mesh_verts_offsets（位置偏移）和 init_mesh_verts_colors_offsets（颜色偏移）指定不同的学习率。
        位置偏移用于变形，颜色偏移用于调整外观。
        """

        # 位置偏移的学习率较高，因为形状变化通常需要更大的调整
        # 网格顶点 offsets: shape = [N, 3], 把最后一维(3)看作向量 => axis=-1
        param_groups_verts = []
        param_groups_verts.append({
            "params": [self.init_mesh_verts_offsets],
            "lr": self.offset_lr * rate
        })


        if self.use_vectorAdam:
            self.optimizer_verts = VectorAdam(param_groups_verts)
        else:
            self.optimizer_verts = torch.optim.Adam(param_groups_verts, lr=0.0, eps=1e-15)

        # 颜色偏移的学习率较低，避免训练过程中颜色变化过快或过于剧烈
        # 颜色也可以加入： 如果 shape = [N, 3], 同理 axis=-1
        param_groups_color = []
        if self.color_with_offset:
            param_groups_color.append({
                "params": [self.init_mesh_verts_offsets],
                "lr": self.offset_lr * rate
            })

        param_groups_color.append({
            "params": [self.init_mesh_verts_colors],
            "lr": self.color_lr * rate
        })
        self.optimizer_color = torch.optim.Adam(param_groups_color, lr=0.0, eps=1e-15)

        return param_groups_color

    def K(self, p, q):
        num = math.factorial(p) * math.factorial(q)
        den = math.factorial(p + q + 2)
        return num / den

    def compute_triangles_statistics_batched(self, verts: torch.Tensor, eps=1e-14):
        """
        给定一批三角形(2D)，每个三角形由3个点 (x,y) 表示。
        verts.shape = [N, 3, 2], 其中 N 可以是任意数量

        返回:
          area: [N] 张量，三角形面积 (>=0)
          centroid: [N,2] 张量，三角形质心 (mx, my)
          cov: [N,2,2] 张量，对应的协方差矩阵 (可能在面积极小处退化)

        对于 area < eps 的三角形，会在输出里做 mask 处理，使其结果为 0。
        """
        A = verts[:, 0, :]  # [N,2]
        B = verts[:, 1, :]
        C = verts[:, 2, :]

        AB = B - A
        AC = C - A
        cross_val = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
        area = 0.5 * torch.abs(cross_val)  # [N]

        centroid = (A + B + C) / 3.0  # [N,2]

        xA = A[:, 0]
        yA = A[:, 1]
        L = AB[:, 0]
        N_ = AB[:, 1]
        M = AC[:, 0]
        O = AC[:, 1]

        # precompute
        i00 = 2.0 * area * self.K(0, 0)
        i10 = 2.0 * area * self.K(1, 0)
        i01 = 2.0 * area * self.K(0, 1)
        i20 = 2.0 * area * self.K(2, 0)
        i11 = 2.0 * area * self.K(1, 1)
        i02 = 2.0 * area * self.K(0, 2)

        # x^2
        int_x2 = (xA ** 2) * i00 + (L ** 2) * i20 + (M ** 2) * i02 \
                 + 2.0 * xA * L * i10 + 2.0 * xA * M * i01 + 2.0 * L * M * i11
        Ex2 = int_x2 / torch.clamp_min(area, eps)

        # y^2
        int_y2 = (yA ** 2) * i00 + (N_ ** 2) * i20 + (O ** 2) * i02 \
                 + 2.0 * yA * N_ * i10 + 2.0 * yA * O * i01 + 2.0 * N_ * O * i11
        Ey2 = int_y2 / torch.clamp_min(area, eps)

        # x*y
        int_xy = (xA * yA) * i00 \
                 + (xA * N_ + yA * L) * i10 \
                 + (xA * O + yA * M) * i01 \
                 + (L * N_) * i20 \
                 + (M * O) * i02 \
                 + (L * O + M * N_) * i11
        Exy = int_xy / torch.clamp_min(area, eps)

        mx = centroid[:, 0]
        my = centroid[:, 1]

        cov_xx = Ex2 - mx * mx
        cov_yy = Ey2 - my * my
        cov_xy = Exy - mx * my

        cov = torch.zeros(verts.shape[0], 2, 2, dtype=verts.dtype, device=verts.device)
        cov[:, 0, 0] = cov_xx
        cov[:, 1, 1] = cov_yy
        cov[:, 0, 1] = cov_xy
        cov[:, 1, 0] = cov_xy

        valid_mask = (area > eps)
        area_out = area.clone()
        centroid_out = centroid.clone()
        cov_out = cov.clone()
        area_out[~valid_mask] = 0.0
        centroid_out[~valid_mask] = 0.0
        cov_out[~valid_mask] = 0.0
        return area_out, centroid_out, cov_out

    def mesh_facets_to_3d_covars(self,
                                 mesh: Meshes,
                                 short_z: float = 1e-3,
                                 eps=1e-14,
                                 sigma=1.732):  # 1.732 1.414
        """
        从 PyTorch3D 的网格 `mesh` 中, 获取每个三角面, 并计算:
          - means_3d: [F, 3] 每个面的质心
          - covars_3d: [F, 3, 3], 近似的 3D 协方差(椭球),
                       其面内主轴= 2D 协方差特征值, z方向= short_z^2 (极薄)
          - valid_mask: [F], 表示非退化面
        可直接传给 rasterization(..., means=means_3d, covars=covars_3d, colors=colors, ...)
        """

        device = mesh.device
        dtype = mesh.verts_packed().dtype

        faces = mesh.faces_packed()  # [F,3]
        verts = mesh.verts_packed()  # [V,3]
        face_verts = verts[faces]  # [F,3,3]

        # (1) 面法线, 在 PyTorch3D 里可用:
        face_normals = mesh.faces_normals_packed().to(dtype)  # [F,3]
        # 若需要单位化, 确认 face_normals 是否已归一化, 可强制:
        face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + eps)

        # (2) 三角形顶点 A,B,C
        A = face_verts[:, 0, :]  # [F,3]
        B = face_verts[:, 1, :]
        C = face_verts[:, 2, :]

        # 构造 AB, AC
        AB = B - A
        AC = C - A
        cross_val = torch.cross(AB, AC, dim=1)  # [F,3]
        area_3d = 0.5 * torch.norm(cross_val, dim=1)  # [F]
        valid_mask = (area_3d > eps)

        # (3) 建立平面基 (u_3d, v_3d)
        AB_len = torch.norm(AB, dim=1, keepdim=True)
        u_3d = AB / (AB_len + eps)  # [F,3]
        cross_temp = torch.cross(face_normals, u_3d, dim=1)  # [F,3]
        cross_temp_len = torch.norm(cross_temp, dim=1, keepdim=True)
        v_3d = cross_temp / (cross_temp_len + eps)  # [F,3]

        # (4) 投影到 2D
        B2x = batched_dot(AB, u_3d)  # [F]
        B2y = batched_dot(AB, v_3d)
        C2x = batched_dot(AC, u_3d)
        C2y = batched_dot(AC, v_3d)

        A2 = torch.zeros(face_verts.shape[0], 2, dtype=dtype, device=device)
        B2 = torch.stack([B2x, B2y], dim=-1)
        C2 = torch.stack([C2x, C2y], dim=-1)
        tri_2d = torch.stack([A2, B2, C2], dim=1)  # [F,3,2]

        # (5) 调用 2D 协方差
        area_2d, centroid_2d, cov_2d = self.compute_triangles_statistics_batched(tri_2d, eps=eps)
        # 质心 => 3D
        cx = centroid_2d[:, 0]
        cy = centroid_2d[:, 1]
        means_3d = A + cx.unsqueeze(1) * u_3d + cy.unsqueeze(1) * v_3d  # [F,3]

        # (6) 2D 特征分解 => lam1, lam2 => 构造旋转 R => cov3D
        evals, evecs = torch.linalg.eigh(cov_2d)  # [F,2], [F,2,2], ascending
        # flip 大特征值排在前
        evals = evals.flip(dims=(1,))  # [F,2]
        evecs = evecs.flip(dims=(2,))  # [F,2,2]
        lam1 = evals[:, 0].clamp_min(eps)
        lam2 = evals[:, 1].clamp_min(eps)

        # 原始 1-sigma 椭圆面积
        ellipse_area = math.pi * torch.sqrt(lam1 * lam2).clamp_min(eps)  # [F]

        # 三角形面积
        area_2d_clamped = area_2d.clamp_min(1e-14)  # [F]

        # 计算 alpha，使得 alpha * 椭圆面积 = 三角形面积
        alpha = area_2d_clamped / ellipse_area  # [F]

        # 将协方差特征值乘以 alpha 即可
        lam1 = lam1 * alpha
        lam2 = lam2 * alpha

        # lam1 = lam1 * (sigma ** 2)  # sigma的宽度
        # lam2 = lam2 * (sigma ** 2)

        # 接下来一样构造对应的 3D covariance……
        v1_2d = evecs[:, :, 0]  # [F,2]
        v2_2d = evecs[:, :, 1]

        # 构造 3D 主轴
        x_temp3d = v1_2d[:, :, None][:, 0, :] * u_3d + v1_2d[:, :, None][:, 1, :] * v_3d  # [F,3]
        y_temp3d = v2_2d[:, :, None][:, 0, :] * u_3d + v2_2d[:, :, None][:, 1, :] * v_3d
        x_norm = x_temp3d.norm(dim=1, keepdim=True) + eps
        y_norm = y_temp3d.norm(dim=1, keepdim=True) + eps
        x_axis3d = x_temp3d / x_norm
        y_axis3d = y_temp3d / y_norm
        z_axis3d = face_normals  # [F,3], (已归一化)

        # 有需要可再强制正交
        # y_axis3d = torch.cross(z_axis3d, x_axis3d, dim=1)
        # y_axis3d = y_axis3d / (y_axis3d.norm(dim=1, keepdim=True)+1e-14)

        # R_i = [x_axis3d_i, y_axis3d_i, z_axis3d_i], shape=[F,3,3]
        F_ = face_verts.shape[0]
        R = torch.zeros(F_, 3, 3, dtype=dtype, device=device)
        R[:, :, 0] = x_axis3d
        R[:, :, 1] = y_axis3d
        R[:, :, 2] = z_axis3d

        # 对角尺度 diag(lam1, lam2, short_z^2)
        # => Cov3D = R * diag(...) * R^T
        # lam1, lam2 => [F], => broadcast
        lam_mat = torch.zeros(F_, 3, 3, dtype=dtype, device=device)
        lam_mat[:, 0, 0] = lam1
        lam_mat[:, 1, 1] = lam2
        lam_mat[:, 2, 2] = short_z ** 2  # z 方向非常短

        # Cov3D = R * lam_mat * R^T
        # 注意: 需要批量矩阵乘法
        # shape [F,3,3], [F,3,3]
        covars_3d = R.bmm(lam_mat.bmm(R.transpose(1, 2)))  # [F,3,3]

        # (7) 颜色
        verts_colors = mesh.textures.verts_features_packed()  # (num_verts, 3) 每个顶点的颜色

        # 获取每个面的三个顶点颜色
        faces_colors = verts_colors[faces]  # (num_faces, 3, 3)
        face_avg_color = faces_colors.mean(dim=1)  # (num_faces, 3)

        return means_3d, covars_3d, face_avg_color, valid_mask

    def save_mesh(self, save_path: str):
        """
        保存带颜色的 mesh 到指定路径（.ply 格式）

        Args:
            save_path (str): 保存文件的路径，如 'output.ply'
        """
        # 获取 mesh 顶点、面、颜色信息
        base_verts = self.init_mesh.verts_packed().detach().cpu().numpy()  # (num_verts, 3)
        base_faces = self.init_mesh.faces_packed().detach().cpu().numpy()  # (num_faces, 3)

        # 应用偏移
        updated_verts = base_verts + self.init_mesh_verts_offsets.detach().cpu().numpy()  # (num_verts, 3)

        # 颜色 Sigmoid 还原 (0-1) -> (0-255)
        updated_colors = self.colors_offsets_activation(self.init_mesh_verts_colors).detach().cpu().numpy()
        updated_colors = (updated_colors * 255).astype(np.uint8)  # 转换为 0-255 颜色

        # 创建带颜色的 mesh
        mesh = trimesh.Trimesh(vertices=updated_verts, faces=base_faces, vertex_colors=updated_colors)

        # 保存到 .ply 文件
        mesh.export(save_path)

    def render_mesh_loss_batch_rand(self, indices, laplacian_mode='uniform'):
        """
        在指定batch中随机选取n个样本进行渲染
        :param indices: 随机选取的样本索引
        """
        # 获取原始顶点坐标
        base_verts = self.init_mesh.verts_packed()  # (num_verts, 3)
        base_faces = self.init_mesh.faces_packed()  # (num_faces, 3)

        # 应用位置偏移
        updated_verts = base_verts + self.init_mesh_verts_offsets  # (num_verts, 3)

        # 颜色使用 Sigmoid 还原
        updated_colors = self.colors_offsets_activation(self.init_mesh_verts_colors)  # (num_verts, 3)

        # 重新构造 mesh
        mesh_new = Meshes(
            verts=[updated_verts],
            faces=[base_faces],
            textures=TexturesVertex(verts_features=[updated_colors])
        )

        faces_xyz, faces_covars_3d, faces_rgb, faces_valid_mask = self.mesh_facets_to_3d_covars(mesh=mesh_new)

        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(mesh_new)

        # mesh normal consistency
        loss_normal = mesh_normal_consistency(mesh_new)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(mesh_new, method=laplacian_mode)

        num_gs = faces_xyz.shape[0]

        # 应用筛选到关键参数
        selected_poses = self.poses[indices.cpu().numpy()]  # 形状变为 (selected_num, 4, 4)
        selected_num_imgs = indices.shape[0]

        colors, alphas, meta = rasterization(
            means=faces_xyz, opacities=torch.ones(num_gs, device=self.device),
            colors=faces_rgb, covars=faces_covars_3d, quats=None, scales=None,
            viewmats=torch.tensor(selected_poses, dtype=torch.float32).to(self.device),
            Ks=torch.tensor(self.intrinsic_mat, dtype=torch.float32).repeat(selected_num_imgs, 1, 1).to(self.device),
            width=self.img_width, height=self.img_height,
            render_mode='RGB+D',
            packed=True, sparse_grad=False, channel_chunk=32, tile_size=32,
            near_plane=0.01, far_plane=1e10)

        rendered_images = colors[..., :3]  # 提取前 3 个通道 torch.Size([1, 1440, 1920, 3])
        depth_map = colors[..., 3]  # 提取第 4 个通道 torch.Size([1, 1440, 1920])
        ################################################################################

        # return rendered_images, alphas, depth_map, self.image_list_all[idx], loss_edge, loss_normal, loss_laplacian
        return rendered_images, alphas, depth_map, [self.image_list[i] for i in indices.cpu().numpy()], loss_edge, loss_normal, loss_laplacian

    def subdivide_mesh(self, method='uniform', rate=0.9):
        """
        细分 mesh，支持：
          - uniform: 所有面细分（Loop Subdivision）
          - adaptive: 选择性细分（基于梯度选定的面）

        细分后，重新初始化可训练参数，并更新 optimizer
        """
        with torch.no_grad():
            # 获取原始顶点坐标
            old_verts = self.init_mesh.verts_packed()  # (num_verts, 3)
            old_faces = self.init_mesh.faces_packed()  # (num_faces, 3)
            # 应用位置偏移
            old_verts = old_verts + self.init_mesh_verts_offsets  # (num_verts, 3)
            # 颜色使用 Sigmoid 还原
            old_colors = self.colors_offsets_activation(self.init_mesh_verts_colors)  # (num_verts, 3)
            # 重新构造 mesh
            old_mesh = Meshes(
                verts=[old_verts],
                faces=[old_faces],
                textures=TexturesVertex(verts_features=[old_colors])
            )

        if method == 'uniform':
            # **全局细分**，使用 `SubdivideMeshes`，并传递颜色 `feats`
            new_mesh, new_colors = self.subdivide(old_mesh, feats=old_colors)
        else:
            raise ValueError(f"Unknown subdivision method: {method}")

        # **更新 mesh**
        num_verts = new_mesh.verts_packed().shape[0]
        num_faces = new_mesh.faces_packed().shape[0]

        # **处理颜色信息**
        if new_colors is None:
            print("[subdivide_mesh] WARNING: No new colors found. Initializing default gray color.")
            new_colors = torch.ones((num_verts, 3), device=self.device) * 0.5  # 默认灰色
        base_colors = new_colors  # (num_verts, 3)

        # **转换颜色到 logit 形式**
        init_color_params = torch.logit(base_colors, 1e-4)  # 避免极端值
        self.init_mesh_verts_colors = nn.Parameter(init_color_params)

        # **重建 `TexturesVertex`**
        self.init_mesh = Meshes(
            verts=[new_mesh.verts_packed()],
            faces=[new_mesh.faces_packed()],
            textures=TexturesVertex(verts_features=[base_colors])
        )

        # **重置 offset**
        self.init_mesh_verts_offsets = nn.Parameter(torch.zeros(num_verts, 3, device=self.device))

        # **重置 optimizer**
        lr_param_groups = self.get_lr_param_groups(rate=rate)
        # self.optimizer = torch.optim.Adam(lr_param_groups, lr=0.0, eps=1e-15)

        print(f"[subdivide_mesh] Done subdivision ({method}).")
        print(f"  Old V={old_verts.shape[0]}, New V={num_verts}, Old F={old_faces.shape[0]}, New F={num_faces}")

def update_full_sub_batch(scene, num_batch, epoch, color_epoch,
                      silhouette_weight, silhouette_edge, silhouette_laplacian, silhouette_normal,
                      alpha_rgb, alpha_edge, alpha_normal, alpha_laplacian,
                      gt_rgb_all, rand_batch_num):
    total_loss = 0
    total_loss_RGB = 0

    # 获取总数据量
    total_imgs = scene.num_imgs
    # 生成全局随机索引
    global_indices = torch.randperm(total_imgs, device=scene.device)

    # 计算实际子批次数量
    sub_batch_num = (total_imgs + rand_batch_num - 1) // rand_batch_num

    for sub_idx in range(sub_batch_num):
        # 清空优化器梯度（每个batch单独更新）
        scene.optimizer_verts.zero_grad()
        scene.optimizer_color.zero_grad()

        # 计算当前子批次的索引范围
        start = sub_idx * rand_batch_num
        end = min((sub_idx + 1) * rand_batch_num, total_imgs)
        current_indices = global_indices[start:end]

        # 当前batch的数据计算
        # num_imgs = scene.num_imgs_lst[batch_idx]
        (rendered_images, rendered_alpha, rendered_depth, pc_obj_names,
         loss_edge, loss_normal, loss_laplacian) = scene.render_mesh_loss_batch_rand(
            laplacian_mode=laplacian_mode, indices=current_indices,
        )
        gt_rgb = gt_rgb_all[current_indices]
        gt_mask = scene.gt_mask_tensor[current_indices]

        # 计算各损失项
        loss_RGB = F.mse_loss(rendered_images, gt_rgb)
        loss_silhouette = F.mse_loss(rendered_alpha, gt_mask)

        # 根据epoch阶段选择损失组合
        if epoch < int(color_epoch):
            loss = (
                silhouette_weight * loss_silhouette +
                silhouette_edge * loss_edge +
                silhouette_laplacian * loss_laplacian +
                silhouette_normal * loss_normal
            )
        else:
            loss = (
                alpha_rgb * loss_RGB +
                0.5 * silhouette_weight * loss_silhouette +
                alpha_edge * loss_edge +
                alpha_normal * loss_normal +
                alpha_laplacian * loss_laplacian
            )

        # 反向传播与参数更新（每个batch执行一次）
        loss.backward()
        scene.optimizer_verts.step()
        scene.optimizer_color.step()

        # 累加Loss用于统计
        total_loss += loss.item()
        total_loss_RGB += loss_RGB.item()

    return total_loss, total_loss_RGB


def calculate_geometric_subdivide_epochs(total_epochs, num_subdivide, ratio=2 / 3, x=2.0):
    # 计算前段 epoch 数量（offset 更新阶段）
    front_epochs = int(total_epochs * ratio)

    # 等比数列求和公式：S = a * (r^n - 1) / (r - 1)，其中 a 是首项，r 是公比，n 是项数
    num_phases = num_subdivide + 1
    r = x
    if r == 1:
        # 等差数列特例处理
        sum_ratio = num_phases
    else:
        sum_ratio = (r**num_phases - 1) / (r - 1)

    # 计算首项 a，使得整个等比数列之和为 front_epochs
    a = front_epochs / sum_ratio

    # 计算每个阶段的结束时间点
    subdivide_epochs = []
    current_epoch = 0
    for i in range(num_subdivide):
        current_epoch += int(round(a * r**i))
        subdivide_epochs.append(min(current_epoch, front_epochs))

    return subdivide_epochs, front_epochs

def get_next_test_name(father_path, prefix):
    """
    自动检测父路径下是否有符合前缀的目录，自动编号生成下一个可用名称。
    :param father_path: 保存测试结果的路径，例如 root_path/save_test/synthetic_tree
    :param prefix: 前缀，例如 "gsplat_adj_20250402"
    :return: (new_test_name, full_path)
    """
    os.makedirs(father_path, exist_ok=True)

    # 匹配形如 gsplat_adj_20250402_00005 的文件夹
    pattern = re.compile(f"^{re.escape(prefix)}_(\\d{{4}})$")
    existing_nums = []

    for name in os.listdir(father_path):
        match = pattern.match(name)
        if match:
            existing_nums.append(int(match.group(1)))

    next_num = 1 if not existing_nums else max(existing_nums) + 1
    new_test_name = f"{prefix}_{next_num:04d}"
    full_path = os.path.join(father_path, new_test_name)

    return new_test_name, full_path
#######################################
if __name__ == "__main__":



    original_path = rf".\github_cache"
    sub_name = 'other_e572_Black'

    root_path = rf"{original_path}\Objaverse\{sub_name}"
    obj_name = root_path.split('\\')[-1].split('_Black')[0]
    save_path_root = rf"result\Objaverse\{obj_name}"


    use_nerf = False
    use_metashape = False
    resize_ratio = 1.0  # size // 2 tree 0.1
    device = torch.device("cuda:0")
    print("device: ", device)


    more_training = 1
    sub_batch_num = 25
    epochs = int(400 * (25 // sub_batch_num) * more_training)
    ico_level = 4
    sphere_radius = 0.8

    lr_rate_de = 0.7
    color_lr = 5e-4
    offset_lr = 5e-4
    subdivide_epoch = 'stable'
    laplacian_mode = 'uniform'  # 'uniform', 'cot', 'cotcurv'

    alpha_rgb = 1.0
    alpha_edge = 1.0
    alpha_normal = 0.0
    alpha_laplacian = 5.0

    silhouette_weight = 2.0
    silhouette_edge = 1.0
    silhouette_normal = 0.0
    silhouette_laplacian = 5.0

    use_vectorAdam = True
    use_subdivide = False

    color_front_ratio = 3/4
    x_ratio = 1.5
    num_subdivide = 2
    if not use_subdivide:
        ico_level = ico_level + num_subdivide
    color_with_offset = True
    blender_full_scene_meshGS_rand = blender_scene3D_mesh3dGS_subdivide_full_rand(root_path, color_lr, offset_lr, sphere_radius,
                                                                                  ico_level, device, use_vectorAdam,
                                                                                  color_with_offset, resize_ratio=resize_ratio,
                                                                                  use_nerf=use_nerf, use_metashape=use_metashape)
    num_all_data = blender_full_scene_meshGS_rand.num_imgs
    rand_batch_num = num_all_data // sub_batch_num  # = 253 // 25 = 10

    prefix = (fr'full {"vectorAdam" if use_vectorAdam else "Adam"} subdivide_{num_subdivide if use_subdivide else "N"} x_ratio{x_ratio}'
                 fr'{"color_with_offset" if color_with_offset else "color_on_offset"} sub_batch_num{sub_batch_num} rand batch_{rand_batch_num} icp_{ico_level}')

    father_path = os.path.join(original_path, save_path_root, 'mesh3dGS_ours_batch')

    test_name, save_img_path = get_next_test_name(father_path, prefix)
    print("save_img_path: ", save_img_path)
    os.makedirs(save_img_path, exist_ok=True)

    txt_path = os.path.join(save_img_path, 'log.txt')
    with open(txt_path, 'w') as f:
        f.write(f"epochs: {epochs}\n")
        f.write(f'scale ellipse area S1 mesh3dGS mesh_loss \n')
        f.write(f"ico_level: {ico_level}, sphere_radius: {sphere_radius}, epochs: {epochs}\n")
        f.write(
            f"alpha_rgb: {alpha_rgb}, alpha_edge: {alpha_edge}, alpha_normal: {alpha_normal}, alpha_laplacian: {alpha_laplacian}\n")
        f.write(f"color_lr: {color_lr}, offset_lr: {offset_lr}\n")
        f.write(f'{subdivide_epoch} epoch / 1 subdivide, uniform, lr_rate_de {lr_rate_de}\n')
        f.write(
            f"silhouette_weight: {silhouette_weight}, silhouette_edge: {silhouette_edge}, silhouette_normal: {silhouette_normal}, silhouette_laplacian: {silhouette_laplacian}\n")
        f.write(f"laplacian_mode: {laplacian_mode}\n")
        f.write(f'optimizer:\n')
        f.write(f'color_front_ratio: {color_front_ratio}, x_ratio: {x_ratio}\n')
        f.write(f'color stage using ({"color and offset" if color_with_offset else "color only"}) Adam\n')
        f.write(f'offset stage using (only offset) {"vectorAdam" if use_vectorAdam else "Adam"}\n')
        f.write(f'subdivide: {"subdivide" if use_subdivide else "no_subdivide"} subdivide\n')
        f.write(f'num_subdivide: {num_subdivide}\n')
        f.write(f'sub_batch_num: {sub_batch_num}\n')
        f.write(f'rand_batch: rand batch_{rand_batch_num}\n')
        f.write(f'resize_ratio: {resize_ratio}\n')
        f.write(f'use_nerf: {use_nerf}\n')


    gt_rgb = blender_full_scene_meshGS_rand.gt_tensor
    print('GT RGB:', len(gt_rgb), gt_rgb.shape)
    start_time = time.time()
    # tqdm 进度条
    progress_bar = tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True)

    num_batch = sub_batch_num
    total_loss_lst = []
    total_loss_RGB_lst = []
    time_lst = []
    memory_lst = []
    train_time = time.time()
    for epoch in progress_bar:
        sub_divide_epochs, offset_state = calculate_geometric_subdivide_epochs(epochs, num_subdivide=num_subdivide,
                                                                     ratio=color_front_ratio, x=x_ratio)

        total_loss, total_loss_RGB = update_full_sub_batch(scene=blender_full_scene_meshGS_rand, num_batch=num_batch, epoch=epoch,
                                                       color_epoch=offset_state, silhouette_weight=silhouette_weight,
                                                       silhouette_edge=silhouette_edge, silhouette_laplacian=silhouette_laplacian,
                                                       silhouette_normal=silhouette_normal, alpha_rgb=alpha_rgb,
                                                       alpha_edge=alpha_edge, alpha_normal=alpha_normal,
                                                       alpha_laplacian=alpha_laplacian, gt_rgb_all=gt_rgb,
                                                       rand_batch_num=rand_batch_num)

        if use_subdivide:
            for sub_epoch in sub_divide_epochs:
                if epoch == sub_epoch:
                    # save all image
                    with torch.no_grad():
                        for i in range(blender_full_scene_meshGS_rand.num_imgs):
                            indices = torch.tensor([i], device=device)
                            (rendered_images, rendered_alpha, rendered_depth, pc_obj_names,
                             loss_edge, loss_normal, loss_laplacian) = blender_full_scene_meshGS_rand.render_mesh_loss_batch_rand(
                                laplacian_mode=laplacian_mode, indices=indices
                            )
                            rendered_images_np = rendered_images[0].squeeze().detach().cpu().numpy()
                            folder = os.path.join(save_img_path, f"subdivide_{sub_epoch}")
                            os.makedirs(folder, exist_ok=True)
                            img_name = blender_full_scene_meshGS_rand.image_list[i]
                            save_path = os.path.join(folder, img_name)
                            rendered_images_np = (rendered_images_np * 255).astype(np.uint8)
                            cv2.imwrite(save_path, rendered_images_np)

                        gc.collect()
                        torch.cuda.empty_cache()

                    # 细分 mesh
                    blender_full_scene_meshGS_rand.subdivide_mesh(method='uniform', rate=lr_rate_de)

        # **在 tqdm 进度条中实时显示 RGB Loss**
        progress_bar.set_postfix(RGB_Loss=total_loss_RGB / num_batch)
        total_loss_lst.append(total_loss / num_batch)
        total_loss_RGB_lst.append(total_loss_RGB / num_batch)
        time_lst.append(time.time() - train_time)
        memory_lst.append(torch.cuda.memory_allocated(device) / 1024 / 1024)  # 以 MB 为单位

        if (epoch + 1) % int(epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch}, Loss: {total_loss_RGB / num_batch}, time: {time.time() - start_time}")
            with open(txt_path, 'a') as f:
                f.write(f"Epoch {epoch}, Loss: {total_loss_RGB / num_batch}, time: {time.time() - start_time}\n")
            start_time = time.time()
            with torch.no_grad():
                indices50 = torch.tensor([50], device=device)
                (rendered_images, rendered_alpha, rendered_depth, pc_obj_names,
                 loss_edge, loss_normal, loss_laplacian) = blender_full_scene_meshGS_rand.render_mesh_loss_batch_rand(
                    laplacian_mode=laplacian_mode, indices=indices50
                )
                rendered_images_np = rendered_images[0].squeeze().detach().cpu().numpy()
                GT_rgb_np = gt_rgb[50].squeeze().detach().cpu().numpy()
                big_img = np.concatenate([rendered_images_np, GT_rgb_np], axis=1)
                plt.imshow(big_img)
                plt.axis('off')
                plt.title(f"Epoch {epoch}, RGB Loss: {total_loss_RGB / num_batch:.4f}")
                save_path = os.path.join(save_img_path, f"render {epoch}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
                plt.close()

                # 保存 mesh
                save_mesh_path = os.path.join(save_img_path, f"mesh {epoch}.ply")
                blender_full_scene_meshGS_rand.save_mesh(save_mesh_path)

            gc.collect()
            torch.cuda.empty_cache()

    # save npz loss
    loss_npz_path = os.path.join(save_img_path, f"loss.npz")
    np.savez(loss_npz_path, total_loss=total_loss_lst, total_loss_RGB=total_loss_RGB_lst, time=time_lst, memory=memory_lst)

    # save all image
    with torch.no_grad():
        for i in range(blender_full_scene_meshGS_rand.num_imgs):
            indices = torch.tensor([i], device=device)
            (rendered_images, rendered_alpha, rendered_depth, pc_obj_names,
             loss_edge, loss_normal, loss_laplacian) = blender_full_scene_meshGS_rand.render_mesh_loss_batch_rand(
                laplacian_mode=laplacian_mode, indices=indices
            )
            rendered_images_np = rendered_images[0].squeeze().detach().cpu().numpy()
            folder_render = os.path.join(save_img_path, f"render")
            os.makedirs(folder_render, exist_ok=True)
            img_name = blender_full_scene_meshGS_rand.image_list[i]
            save_path = os.path.join(folder_render, img_name)
            rendered_images_np = (rendered_images_np * 255).astype(np.uint8)[:, :, ::-1]  # 转换为 RGB 格式
            cv2.imwrite(save_path, rendered_images_np)

            folder_gt = os.path.join(save_img_path, f"gt")
            os.makedirs(folder_gt, exist_ok=True)
            GT_rgb_np = gt_rgb[i].squeeze().detach().cpu().numpy()
            save_path = os.path.join(folder_gt, img_name)
            GT_rgb_np = (GT_rgb_np * 255).astype(np.uint8)[:, :, ::-1]  # 转换为 RGB 格式
            cv2.imwrite(save_path, GT_rgb_np)
