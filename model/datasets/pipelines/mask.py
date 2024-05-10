# import cv2
# import math
# import numpy
# from ..builder import PIPELINES
#
#
# @PIPELINES.register_module()
# class SampleMaskVertices(object):
#     def __init__(self,
#                  center_sampling=False,
#                  num_ray=18):
#         super().__init__()
#         self.center_sampling = center_sampling
#         assert num_ray > 0
#         self.num_ray = num_ray
#
#     # after padding
#     def __call__(self, results):
#         assert results['with_mask']
#         gt_mask = results['gt_mask'].masks[0]
#         center, contour, KEEP = self.get_mass_center(gt_mask)
#         vertices = self.sample_mask_vertices(
#             center, contour, KEEP, results['pad_shape'][:2])
#         results['gt_mask_vertices'] = vertices
#         results['mass_center'] = center
#         return results
#
#     def get_mass_center(self, mask):
#         contour, _ = cv2.findContours(
#             mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         contour = sorted(
#             contour, key=lambda x: cv2.contourArea(x), reverse=True)
#         contour = contour[0][:, 0, :]  # x, y coordinate of contour
#         contour_info = cv2.moments(contour)
#         KEEP = False
#         if contour_info['m00'] > 0.:
#             KEEP = True
#         if KEEP:
#             mass_x = contour_info['m10'] / contour_info['m00']
#             mass_y = contour_info['m01'] / contour_info['m00']
#             center = numpy.array([mass_x, mass_y])
#         else:
#             center = numpy.array([-1., -1.])
#         return center, contour, KEEP
#
#     def sample_mask_vertices(self, center, contour, KEEP=True, max_shape=None):
#         vertices = numpy.empty(
#             (2, self.num_ray), dtype=numpy.float32)
#         vertices.fill(-1)
#         if not KEEP:
#             return vertices
#         num_pts = contour.shape[0]
#         if num_pts <= self.num_ray:
#             vertices[:, :num_pts] = contour.transpose()
#             return vertices
#         inside_contour = cv2.pointPolygonTest(contour, center, False) > 0
#         if self.center_sampling and inside_contour:
#             c_x, c_y = center
#             x = contour[:, 0] - center[0]
#             y = contour[:, 1] - center[1]
#             angle = numpy.arctan2(y, x) * 180 / numpy.pi
#             angle[angle < 0] += 360
#             angle = angle.astype(numpy.uint32)
#             distance = numpy.sqrt(x ** 2 + y ** 2)
#             angles, distances = [], []
#             for ang in range(0, 360, 360 // self.num_ray):
#                 if ang in angle:
#                     dist = distance[ang == angle].max()
#                     angles.append(ang)
#                     distances.append(dist)
#                 else:
#                     for increment in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
#                         aux_ang = ang + increment
#                         if aux_ang in angle:
#                             dist = distance[aux_ang == angle].max()
#                             angles.append(aux_ang)
#                             distances.append(dist)
#                             break
#             angles = numpy.array(angles)
#             distances = numpy.array(distances)
#             angles = angles / 180 * numpy.pi
#             sin = numpy.sin(angles)
#             cos = numpy.cos(angles)
#             vertex_x = c_x + distances * cos
#             vertex_y = c_y + distances * sin
#         else:
#             interval = math.ceil(num_pts / self.num_ray)
#             vertex_x = contour[::interval, 0]
#             vertex_y = contour[::interval, 1]
#         if max_shape is not None:
#             vertex_x = numpy.clip(vertex_x, 0, max_shape[1] - 1)
#             vertex_y = numpy.clip(vertex_y, 0, max_shape[0] - 1)
#         partial_vertices = numpy.vstack(
#             (vertex_x, vertex_y))  # 2 x num_vertices
#         vertices[:, :partial_vertices.shape[1]] = partial_vertices
#         return vertices
#
#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'num_ray={self.num_ray})')
#         return repr_str


import random

import cv2
import math
import numpy
import numpy as np
from numpy.random import sample

from ..builder import PIPELINES


@PIPELINES.register_module()
class SampleMaskVertices(object):
    def __init__(self,
                 center_sampling=False,
                 num_ray=12):
        super().__init__()
        self.center_sampling = center_sampling
        assert num_ray > 0
        self.num_ray = num_ray

    # after padding
    def __call__(self, results):
        assert results['with_mask']
        gt_mask = results['gt_mask'].masks[0]
        center, contour, KEEP = self.get_mass_center(gt_mask)

        vertices = self.sample_mask_vertices(
            center, contour, KEEP, results['pad_shape'][:2])
        results['gt_mask_vertices'] = vertices
        results['mass_center'] = center
        return results

    # 给两个定点之间插值
    def interpolate_points(self, ps, pe):
        xs, ys = ps
        xe, ye = pe
        points = []
        dx = xe - xs
        dy = ye - ys
        if dx != 0:
            scale = dy / dx
            if xe > xs:
                x_interpolated = list(range(math.ceil(xs), math.floor(xe) + 1))
            else:
                x_interpolated = list(range(math.floor(xs), math.ceil(xe) - 1, -1))
            for x in x_interpolated:
                y = ys + (x - xs) * scale
                points.append([x, y])
        if dy != 0:
            scale = dx / dy
            if ye > ys:
                y_interpolated = list(range(math.ceil(ys), math.floor(ye) + 1))
            else:
                y_interpolated = list(range(math.floor(ys), math.ceil(ye) - 1, -1))
            for y in y_interpolated:
                x = xs + (y - ys) * scale
                points.append([x, y])
        if xe > xs:
            points = sorted(points, key=lambda x: x[0])
        else:
            points = sorted(points, key=lambda x: -x[0])
        return points

    # 修剪点集
    def prune_points(self, points, th=0.1):
        points_pruned = [points[0]]
        for i in range(1, len(points)):
            x1, y1 = points_pruned[-1]
            x2, y2 = points[i]
            dist = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if dist > th:
                points_pruned.append(points[i])
        return points_pruned

    # 选取插值后一定的点
    def sample_points(self, points, sample_rate=0.5):
        points = np.array(points)
        k = int(len(points) * sample_rate)
        index = sorted(random.sample(list(range(len(points))), k))
        points = points[index]
        return points

    def get_mass_center(self, mask):
        contour, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if contour:
            contour = sorted(
                contour, key=lambda x: cv2.contourArea(x), reverse=True)
            contour = contour[0][:, 0, :]  # x, y coordinate of contour
            # print(contour.shape)
        else:
            print("没有找到轮廓")

        # 点的顺序（顺时针为 True）
        contour_x = np.array(contour.transpose()[0])
        contour_y = np.array(contour.transpose()[-1])
        xc = np.append(contour_x, contour_x[0])
        yc = np.append(contour_y, contour_y[0])
        area = 0
        for i in range(self.num_ray):
            x1, y1 = xc[i], yc[i]
            x2, y2 = xc[i + 1], yc[i + 1]
            area += (x2 - x1) * (y2 - y1)
        if not area < 0:
            contour = contour[::-1, :]

        # 对多边形进行插值
        prob = np.random.uniform()
        points_interpolated = []
        if prob < 0.5:  # 50% 的几率进行多边形增强
            points = contour
            points_interpolated.append(points[0])
            for i in range(0, len(points) - 1):
                points_i = self.interpolate_points(points[i], points[i + 1])
                points_interpolated += points_i
                points_interpolated.append(points[i + 1])
            points_interpolated = self.prune_points(points_interpolated)
            points_interpolated = self.sample_points(points_interpolated)
            contour = np.array(points_interpolated, dtype=numpy.float32)  # [num_pts , 2]
        else:
            contour = contour

        contour_info = cv2.moments(contour)
        KEEP = False
        if contour_info['m00'] > 0.:
            KEEP = True
        if KEEP:
            mass_x = contour_info['m10'] / contour_info['m00']
            mass_y = contour_info['m01'] / contour_info['m00']
            center = numpy.array([mass_x, mass_y])
        else:
            center = numpy.array([-1., -1.])
        return center, contour, KEEP

    def sample_mask_vertices(self, center, contour, KEEP=True, max_shape=None):
        vertices = numpy.empty(
            (2, self.num_ray), dtype=numpy.float32)
        vertices.fill(-1)
        if not KEEP:
            return vertices

        num_pts = contour.shape[0]  # [num_pts, 2]
        if num_pts <= self.num_ray:
            vertices[:, :num_pts] = contour.transpose()
            return vertices

        inside_contour = cv2.pointPolygonTest(contour, center, False) > 0

        if self.center_sampling and inside_contour:
            c_x, c_y = center
            x = contour[:, 0] - center[0]
            y = contour[:, 1] - center[1]
            angle = numpy.arctan2(y, x) * 180 / numpy.pi
            angle[angle < 0] += 360
            angle = angle.astype(numpy.uint32)
            distance = numpy.sqrt(x ** 2 + y ** 2)
            angles, distances = [], []
            for ang in range(0, 360, 360 // self.num_ray):
                if ang in angle:
                    dist = distance[ang == angle].max()
                    angles.append(ang)
                    distances.append(dist)
                else:
                    for increment in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                        aux_ang = ang + increment
                        if aux_ang in angle:
                            dist = distance[aux_ang == angle].max()
                            angles.append(aux_ang)
                            distances.append(dist)
                            break
            angles = numpy.array(angles)
            distances = numpy.array(distances)
            angles = angles / 180 * numpy.pi
            sin = numpy.sin(angles)
            cos = numpy.cos(angles)
            vertex_x = c_x + distances * cos
            vertex_y = c_y + distances * sin
        else:
            interval = math.ceil(num_pts / self.num_ray)
            vertex_x = contour[::interval, 0]
            vertex_y = contour[::interval, 1]
        if max_shape is not None:
            vertex_x = numpy.clip(vertex_x, 0, max_shape[1] - 1)
            vertex_y = numpy.clip(vertex_y, 0, max_shape[0] - 1)
        partial_vertices = numpy.vstack(
            (vertex_x, vertex_y))  # 2 x num_vertices
        vertices[:, :partial_vertices.shape[1]] = partial_vertices  # [2, 12]

        xs = np.array(vertices[0])
        ys = np.array(vertices[-1])

        # 设置多边形起点
        start = np.argmin(xs ** 2 + ys ** 2)
        vertices = vertices.transpose()
        vertices = np.concatenate([vertices[start:], vertices[:start]], 0)
        vertices = vertices.transpose()

        return vertices

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_ray={self.num_ray})')
        return repr_str
