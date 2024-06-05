import SimpleITK as sitk
import torch
import glob
import numpy as np
import os
import scipy.spatial
import pandas as pd
from tqdm import tqdm
from dataset import resize_image

ckpt_path = (
    "/data1/songwei/Segment/checkpoints/baselinev8/epoch=115-valid_loss=0.5323.ckpt"
)
prediction_path = "/data1/songwei/Segment/predictions/test/"
roi_size = (128, 128, 128)
overlap = (int(128 * 0.5), int(128 * 0.5), int(128 * 0.5))
device = torch.device("cuda:3")
gaussian_infer = True
flip_tta = False
paths = [prediction_path]
print(ckpt_path)

ckpt = torch.load(ckpt_path, map_location="cpu")
for k in list(ckpt["state_dict"].keys()):
    if k.startswith("model."):
        ckpt["state_dict"][k[6:]] = ckpt["state_dict"].pop(k)

from models.unet import MyNet

# from models.global_local_unet import MyNet

model = MyNet(1, 1)
model.load_state_dict(ckpt["state_dict"])
model.cuda(device).eval()


def slide_window_inference(image, model, roi_size, overlap, device=device):
    """
    使用滑动窗口方法进行推理。

    Args:
        image (torch.Tensor): 输入图像，形状为 (1, 1, D, H, W)。
        model (torch.nn.Module): 用于推理的深度学习模型。
        roi_size (tuple): 每个滑动窗口的大小 (D_p, H_p, W_p)。
        overlap (tuple): 相邻滑动窗口之间的重叠 (D_o, H_o, W_o)。
        device (str): 用于推理的设备，默认为 "cuda"。

    Returns:
        torch.Tensor: 模型预测的输出图像，形状与输入图像相同 (1, 1, D, H, W)。
    """

    # 获取图像维度
    _, _, D, H, W = image.shape
    global_image = image.cpu().numpy()[0, 0]
    global_image = resize_image(global_image, new_shape=(128, 128, 128))
    global_image = (
        torch.from_numpy(global_image).float().unsqueeze(0).unsqueeze(0).to(device)
    )

    # 计算滑动步长
    stride_d = roi_size[0] - overlap[0]
    stride_h = roi_size[1] - overlap[1]
    stride_w = roi_size[2] - overlap[2]

    # 初始化输出图像和计数数组
    output_image = torch.zeros_like(image, device=device)

    # Create a Gaussian distribution with a large center and smaller surroundings
    x = np.linspace(-5, 5, roi_size[0])
    y = np.linspace(-5, 5, roi_size[1])
    z = np.linspace(-5, 5, roi_size[2])
    x, y, z = np.meshgrid(x, y, z)
    d = np.sqrt(x**2 + y**2 + z**2)

    # Create the Gaussian distribution
    sigma, mu = 4.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
    g = torch.from_numpy(g).float().to(device)
    if not gaussian_infer:
        g = torch.ones_like(g)

    # 滑动窗口循环
    for d in range(0, D - roi_size[0] + 1, stride_d):
        for h in range(0, H - roi_size[1] + 1, stride_h):
            for w in range(0, W - roi_size[2] + 1, stride_w):
                # 提取滑动窗口
                patch = image[
                    :, :, d : d + roi_size[0], h : h + roi_size[1], w : w + roi_size[2]
                ]

                # 将滑动窗口送入模型进行推理
                with torch.no_grad():

                    patch = patch.to(device)
                    output_patch = model(patch)[0]
                    if flip_tta:
                        patch_flipx = model(torch.flip(patch, [4]))[0]
                        patch_flipy = model(torch.flip(patch, [3]))[0]
                        patch_flipyx = model(torch.flip(patch, [3, 4]))[0]
                        output_patch = (
                            output_patch
                            + torch.flip(patch_flipx, [4])
                            + torch.flip(patch_flipy, [3])
                            + torch.flip(patch_flipyx, [3, 4])
                        )

                # 将模型输出累加到输出图像
                output_image[
                    :, :, d : d + roi_size[0], h : h + roi_size[1], w : w + roi_size[2]
                ] += (output_patch * g)

    # 使用计数数组对输出图像进行归一化
    output_image = output_image

    return output_image


def get_hausdorff(test_image, result_image):
    """Compute the Hausdorff distance."""

    result_statistics = sitk.StatisticsImageFilter()
    result_statistics.Execute(result_image)

    if result_statistics.GetSum() == 0:
        hd = np.nan
        return hd

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 3D
    e_test_image = sitk.BinaryErode(test_image, (1, 1, 1))
    e_result_image = sitk.BinaryErode(result_image, (1, 1, 1))

    h_test_image = sitk.Subtract(test_image, e_test_image)
    h_result_image = sitk.Subtract(result_image, e_result_image)

    h_test_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_test_image))).tolist()
    h_result_indices = np.flip(
        np.argwhere(sitk.GetArrayFromImage(h_result_image))
    ).tolist()

    test_coordinates = [
        test_image.TransformIndexToPhysicalPoint(x) for x in h_test_indices
    ]
    result_coordinates = [
        test_image.TransformIndexToPhysicalPoint(x) for x in h_result_indices
    ]

    def get_distances_from_a_to_b(a, b):
        kd_tree = scipy.spatial.KDTree(a, leafsize=100)
        return kd_tree.query(b, k=1, eps=0, p=2)[0]

    d_test_to_result = get_distances_from_a_to_b(test_coordinates, result_coordinates)
    d_result_to_test = get_distances_from_a_to_b(result_coordinates, test_coordinates)

    hd = max(np.percentile(d_test_to_result, 95), np.percentile(d_result_to_test, 95))

    return hd


def get_metrics(test_image, label_image):
    """Compute the precision and recall."""
    test_array = sitk.GetArrayFromImage(test_image).flatten()
    result_array = sitk.GetArrayFromImage(label_image).flatten()
    if np.all(test_array == 0) and np.all(result_array == 0):
        return 0, 0, 0, 0, 0, 0, 0

    true_positive = np.sum(test_array * result_array)
    false_positive = np.sum((1 - test_array) * result_array)
    false_negative = np.sum(test_array * (1 - result_array))

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0
    )
    dice = (
        (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
        if (2 * true_positive + false_positive + false_negative)
        else 0
    )

    hd95 = get_hausdorff(test_image, label_image)
    return (sensitivity, precision, dice, hd95)


def calculate(prediction_path, ground_truth_path):
    metrics_list = []

    test_list = [
        x
        for x in os.listdir(prediction_path)
        if x.endswith(".nii.gz") or x.endswith(".mha")
    ]
    for name in tqdm(sorted(test_list)):
        try:
            test_image_path = os.path.join(ground_truth_path, name)
            result_image_path = os.path.join(prediction_path, name)
            test_image = sitk.ReadImage(test_image_path)
            result_image = sitk.ReadImage(result_image_path)
            assert test_image.GetSize() == result_image.GetSize()

            # Copy meta information
            result_image.CopyInformation(test_image)

            # Apply mask for treated aneurysms
            test_image = test_image > 0.5
            result_image = result_image > 0.5

            # Compute metrics
            try:
                h95 = get_hausdorff(test_image, result_image)
            except Exception as e:
                # print(f"Error calculating Hausdorff for {name}: {e}")
                h95 = 0

            (sensitivity, precision, dice, h95) = get_metrics(test_image, result_image)

            # Append metrics for current image to list
            metrics_list.append(
                {
                    "name": name,
                    "sensitivity": sensitivity,
                    "precision": precision,
                    "dice": dice,
                    "hausdorff95": h95,
                }
            )
        except Exception as e:
            print(f"Failed to process {name}: {e}")
            continue

    # Create DataFrame from list of metrics
    info_df = pd.DataFrame(metrics_list)
    return info_df


def predict_from_path(
    model,
    roi_size,
    overlap,
    images_path="/data1/songwei/WorkStation/Data/raw/Dataset011_AortaSeg/fold1/*",
):
    image_list = glob.glob(images_path)
    for image_path in tqdm(image_list):
        image_meta = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_meta).astype(np.float32)

        # image = np.clip(image, 0, np.percentile(image, 99.9))
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = 2 * (image - 0.5)
        image = np.clip(image, -1000, 500)
        image = (image - np.mean(image)) / np.std(image)
        image = torch.from_numpy(image.copy()).float().unsqueeze(0).unsqueeze(0)
        # 进行滑动窗口推理
        output_image = slide_window_inference(image, model, roi_size, overlap)
        output_image = np.round(torch.sigmoid(output_image)[0][0].cpu().numpy()).astype(
            np.uint8
        )
        # 将 output_image 保存为 nii.gz
        output_image = sitk.GetImageFromArray(output_image)
        output_image.CopyInformation(image_meta)
        sitk.WriteImage(
            output_image,
            os.path.join(
                prediction_path, os.path.basename(image_path).replace("_0000", "")
            ),
        )


predict_from_path(model, roi_size, overlap)
label_path = "/data1/songwei/WorkStation/Data/raw/Dataset011_AortaSeg/labelsTr"


results = []
for p in paths:
    new_fold = (
        calculate(
            prediction_path=p,
            ground_truth_path=label_path,
        ),
    )[0]

    # print(p)
    print(new_fold.describe())
    results.append(new_fold)
