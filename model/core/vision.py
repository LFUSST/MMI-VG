import cv2
import mmcv
import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import pycocotools.mask as maskUtils
import matplotlib as mpl
from matplotlib import colors as mplc
import matplotlib.figure as mplfigure
from detectron2.utils.visualizer import GenericMask, VisImage

EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_expr_bbox(filename,
                     pred_bbox,
                     outfile=None,
                     gt_bbox=None,
                     pred_bbox_color='red',
                     gt_bbox_color='blue',
                     thickness=3):
    plt.clf()
    _, axe = plt.subplots()

    pred_bbox_color = color_val_matplotlib(pred_bbox_color)
    gt_bbox_color = color_val_matplotlib(gt_bbox_color)

    img = mmcv.imread(filename).astype(np.uint8)
    img = np.ascontiguousarray(img)

    pred_bbox_int = pred_bbox.long().cpu()
    pred_bbox_poly = [[pred_bbox_int[0], pred_bbox_int[1]], [pred_bbox_int[2], pred_bbox_int[1]],
                      [pred_bbox_int[2], pred_bbox_int[3]], [pred_bbox_int[0], pred_bbox_int[3]]]
    pred_bbox_poly = numpy.array(pred_bbox_poly).reshape((4, 2))
    pred_polygon = Polygon(pred_bbox_poly)
    pred_patch = PatchCollection([pred_polygon], facecolor='none', edgecolors=[
        pred_bbox_color], linewidths=thickness)

    axe.add_collection(pred_patch)
    plt.imshow(cv2.cvtColor(axe.get_image(), cv2.COLOR_BGR2RGB))

    plt.axis('off')  # 关闭坐标轴显示

    plt.show()

    if gt_bbox is not None:
        gt_bbox_int = gt_bbox.long().cpu()
        gt_bbox_poly = [[gt_bbox_int[0], gt_bbox_int[1]], [gt_bbox_int[0], gt_bbox_int[3]],
                        [gt_bbox_int[2], gt_bbox_int[3]], [gt_bbox_int[2], gt_bbox_int[1]]]
        gt_bbox_poly = numpy.array(gt_bbox_poly).reshape((4, 2))
        gt_polygon = Polygon(gt_bbox_poly)
        gt_patch = PatchCollection(
            [gt_polygon], facecolor='none', edgecolors=[gt_bbox_color], linewidths=thickness)
        axe.add_collection(gt_patch)

    axe.axis('off')
    axe.imshow(img)
    plt.show()  # 显示图像
    # plt.savefig(outfile)

    # plt.close()


def imshow_expr_mask(filename,
                     pred_mask,
                     outfile=None,
                     gt_mask=None,
                     overlay=True):
    if not overlay:
        plt.clf()
        plt.axis('off')
        pred_mask = maskUtils.decode(pred_mask).astype(bool)
        plt.imshow(pred_mask, "gray")
        # plt.savefig(outfile.replace(".jpg", "_pred.jpg"))
        if gt_mask is not None:
            plt.clf()
            plt.axis('off')
            gt_mask = maskUtils.decode(gt_mask).astype(bool)
            assert gt_mask.shape == pred_mask.shape
            plt.imshow(gt_mask, "gray")
            # plt.savefig(outfile.replace(".jpg", "_gt.jpg"))
            plt.show()
        # plt.close()
    else:
        img = cv2.imread(filename)[:, :, ::-1]
        height, width = img.shape[:2]
        img = numpy.ascontiguousarray(img).clip(0, 255).astype(numpy.uint8)
        output_pred = VisImage(img, scale=1.)
        pred_mask = maskUtils.decode(pred_mask)
        pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        assert pred_mask.shape[0] == height and pred_mask.shape[1] == width
        pred_mask = GenericMask(pred_mask, height, width)
        for segment in pred_mask.polygons:
            polygon = mpl.patches.Polygon(
                segment.reshape(-1, 2),
                fill=True,
                facecolor=mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,),
                edgecolor=mplc.to_rgb([0., 0., 0.]) + (1,),
                linewidth=2
            )
            output_pred.ax.add_patch(polygon)

        plt.imshow(output_pred.get_image())

        plt.axis('off')  # 关闭坐标轴显示

        plt.show()  # 显示图像

        # cv2.imwrite(outfile.replace(".jpg", "_pred.jpg"),
        #             output_pred.get_image()[:, :, ::-1])
        if gt_mask is not None:
            output_gt = VisImage(img, scale=1.)
            gt_mask = maskUtils.decode(gt_mask)
            gt_mask = cv2.resize(gt_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            assert gt_mask.shape[0] == height and gt_mask.shape[1] == width
            gt_mask = GenericMask(gt_mask, height, width)
            for segment in gt_mask.polygons:
                polygon = mpl.patches.Polygon(
                    segment.reshape(-1, 2),
                    fill=True,
                    facecolor=mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,),
                    edgecolor=mplc.to_rgb([0., 0., 0.]) + (1,),
                    linewidth=2
                )
                output_gt.ax.add_patch(polygon)
            # 显示图像（确保 ax 是 matplotlib 的 Axes 对象）

            plt.imshow(output_gt.get_image())

            plt.axis('off')  # 关闭坐标轴显示

            plt.show()  # 显示图像
            # cv2.imwrite(outfile.replace(".jpg", "_gt.jpg"),
            #             output_gt.get_image()[:, :, ::-1])

