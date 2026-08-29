import math
from math import sin, cos, acos, degrees

import numpy as np
import cv2
from skimage import transform as trans
import torch
import torch.nn.functional as F
import threading

import torchvision
from torchvision.transforms import v2

import kornia.geometry.transform as kgm
import kornia.color as kc

torchvision.disable_beta_transforms_warning()

# <--left profile
src1 = np.array(
    [
        [51.642, 50.115],
        [57.617, 49.990],
        [35.740, 69.007],
        [51.157, 89.050],
        [57.025, 89.702],
    ],
    dtype=np.float32,
)

# <--left
src2 = np.array(
    [
        [45.031, 50.118],
        [65.568, 50.872],
        [39.677, 68.111],
        [45.177, 86.190],
        [64.246, 86.758],
    ],
    dtype=np.float32,
)

# ---frontal
src3 = np.array(
    [
        [39.730, 51.138],
        [72.270, 51.138],
        [56.000, 68.493],
        [42.463, 87.010],
        [69.537, 87.010],
    ],
    dtype=np.float32,
)

# -->right
src4 = np.array(
    [
        [46.845, 50.872],
        [67.382, 50.118],
        [72.737, 68.111],
        [48.167, 86.758],
        [67.236, 86.190],
    ],
    dtype=np.float32,
)

# -->right profile
src5 = np.array(
    [
        [54.796, 49.990],
        [60.771, 50.115],
        [76.673, 69.007],
        [55.388, 89.702],
        [61.257, 89.050],
    ],
    dtype=np.float32,
)

# Pitch Up
src6 = np.array(
    [
        [39.730, 55.000],  # LE (Shifted Down)
        [72.270, 55.000],  # RE (Shifted Down)
        [56.000, 64.000],  # Nose (Shifted Up/Closer to eyes)
        [42.463, 78.000],  # LM (Shifted Up)
        [69.537, 78.000],  # RM (Shifted Up)
    ],
    dtype=np.float32,
)

# Pitch Down
src7 = np.array(
    [
        [39.730, 45.000],  # LE (Shifted Up)
        [72.270, 45.000],  # RE (Shifted Up)
        [56.000, 75.000],  # Nose (Shifted Down/Further from eyes)
        [42.463, 95.000],  # LM (Shifted Down)
        [69.537, 95.000],  # RM (Shifted Down)
    ],
    dtype=np.float32,
)

src = np.array([src1, src2, src3, src4, src5, src6, src7])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

arcface_src = np.expand_dims(arcface_src, axis=0)

# Definisci i punti di riferimento come tensore PyTorch
arcface_src_cuda = torch.tensor(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=torch.float32,
)  # Shape: (5, 2)

# F-09: module-level constant matrices for color-space conversions (avoid re-allocating per call)
_RGB_TO_YUV = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001],
    ],
    dtype=torch.float32,
)

_YUV_TO_RGB = torch.tensor(
    [[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]],
    dtype=torch.float32,
)

_RGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float32,
)

_XYZ_WHITE_D65 = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)

_XYZ_TO_RGB = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=torch.float32,
)

# --- Thread-safe caches for deterministic geometric masks & kernels ---
_faded_mask_cache: dict[tuple, torch.Tensor] = {}
_faded_mask_lock = threading.Lock()

_gaussian_kernel_cache: dict[tuple, torch.Tensor] = {}
_gaussian_kernel_lock = threading.Lock()

_laplace_kernel_cache: dict[str, torch.Tensor] = {}
_laplace_kernel_lock = threading.Lock()


def gamma_decode_srgb_to_linear_rgb(srgb: torch.Tensor) -> torch.Tensor:
    """
    Converts sRGB [0, 1] to Linear RGB [0, 1].
    Uses Kornia's optimized CUDA kernels for exact piecewise sRGB decoding.
    Note: Kornia refers to non-linear sRGB simply as 'rgb'.
    """
    return kc.rgb_to_linear_rgb(srgb.clamp(0.0, 1.0))


def gamma_encode_linear_rgb_to_srgb(linear_rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts Linear RGB [0, 1] to sRGB [0, 1].
    Uses Kornia's optimized CUDA kernels for exact piecewise sRGB encoding.
    Note: Kornia refers to non-linear sRGB simply as 'rgb'.
    """
    return kc.linear_rgb_to_rgb(linear_rgb.clamp(0.0, 1.0))


def pad_image_by_size(img, image_size):
    # Se image_size non è una tupla, crea una tupla con altezza e larghezza uguali
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)

    # Larghezza e altezza dell'immagine
    w, h = img.size(dim=2), img.size(dim=1)

    # Dimensioni target
    target_h, target_w = image_size

    # Verifica se la larghezza o l'altezza è inferiore alle dimensioni target
    if w < target_w or h < target_h:
        # Calcolo del padding necessario a destra e in basso
        pad_right = max(target_w - w, 0)  # Assicura che il padding sia non negativo
        pad_bottom = max(target_h - h, 0)  # Assicura che il padding sia non negativo

        # Aggiungi padding all'immagine (pad_left, pad_right, pad_top, pad_bottom)
        img = torch.nn.functional.pad(
            img, (0, pad_right, 0, pad_bottom), mode="constant", value=0
        )

    return img


def transform(img, center, output_size, scale, rotation):
    """
    OPTIMIZED: GPU Warping using Kornia.
    """
    # pad image by image size
    img = pad_image_by_size(img, output_size)

    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]

    # Kornia GPU affine
    M_tensor = torch.from_numpy(M).float().unsqueeze(0).to(img.device)
    img_b = img.unsqueeze(0) if img.dim() == 3 else img

    cropped = kgm.warp_affine(
        img_b.float(),
        M_tensor,
        dsize=(output_size, output_size),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)

    return cropped.to(img.dtype), M


def trans_points2d(pts, M):
    # Add a column of ones to the pts array to create homogeneous coordinates
    ones_column = np.ones((pts.shape[0], 1), dtype=np.float32)
    homogeneous_pts = np.hstack([pts, ones_column])

    # Perform the matrix multiplication for all points at once
    transformed_pts = np.dot(homogeneous_pts, M.T)

    # Return only the first two columns (x and y coordinates)
    return transformed_pts[:, :2]


def trans_points3d(pts, M):
    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)

    # Add a column of ones to the pts array to create homogeneous coordinates for 2D transformation
    ones_column = np.ones((pts.shape[0], 1), dtype=np.float32)
    homogeneous_pts = np.hstack([pts[:, :2], ones_column])

    # Perform the matrix multiplication for all points at once
    transformed_2d = np.dot(homogeneous_pts, M.T)

    # Scale the z-coordinate
    scaled_z = pts[:, 2] * scale

    # Combine the transformed 2D points with the scaled z-coordinate
    transformed_pts = np.hstack([transformed_2d[:, :2], scaled_z.reshape(-1, 1)])

    return transformed_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


def estimate_affine_matrix_3d23d(X, Y):
    """Using least-squares solution
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    """
    X_homo = np.hstack((X, np.ones([X.shape[0], 1])))  # n x 4
    P = np.linalg.lstsq(X_homo, Y, rcond=None)[0].T  # Affine matrix. 3 x 4
    return P


def P2sRt(P):
    """decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    """
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    # F-13: guard against zero-norm division
    if np.linalg.norm(R1) < 1e-9:
        return 1.0, np.eye(3), np.zeros(3)
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def matrix2angle(R):
    """get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return rx, ry, rz


def warp_affine_torchvision(
    img,
    matrix,
    image_size,
    rotation_ratio=0.0,  # Gardé pour compatibilité des anciens appels, mais ignoré
    border_value=0.0,
    border_mode="replicate",
    interpolation_value=v2.functional.InterpolationMode.NEAREST,
    device="cpu",
):
    """
    OPTIMIZED: Bypasses slow matrix decomposition and CPU trigonometry.
    Feeds the 2x3 affine matrix directly to Kornia for instant GPU rendering.
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # Convertir en tenseur si c'est un numpy array
    if isinstance(img, torch.Tensor):
        img_tensor = img.to(device).float()
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
    else:
        img_tensor = (
            torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
        )

    # Charger la matrice Numpy sur le GPU
    M_tensor = torch.from_numpy(matrix).float().unsqueeze(0).to(device)

    # Mapping des paramètres vers Kornia
    mode_kgm = (
        "nearest"
        if interpolation_value == v2.functional.InterpolationMode.NEAREST
        else "bilinear"
    )
    padding_mode_kgm = "border" if border_mode == "replicate" else "zeros"

    # Warping direct sur VRAM
    warped_img_tensor = kgm.warp_affine(
        img_tensor,
        M_tensor,
        dsize=image_size,
        mode=mode_kgm,
        padding_mode=padding_mode_kgm,
        align_corners=True,
    ).squeeze(0)

    # Restauration du type d'origine
    if isinstance(img, torch.Tensor):
        warped_img_tensor = warped_img_tensor.to(img.dtype)

    return warped_img_tensor


def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    # F-14: guard against zero variance (degenerate point cloud)
    var_sum = src_demean.var(axis=0).sum()
    if var_sum < 1e-10:
        return np.nan * T
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T


def get_matrix(lmk, templates):
    if templates.shape[0] == 1:
        return umeyama(lmk, templates[0], True)[0:2, :]
    test_lmk = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_error, best_matrix = float("inf"), []
    for i in np.arange(templates.shape[0]):
        matrix = umeyama(lmk, templates[i], True)[0:2, :]
        error = np.sum(
            np.sqrt(np.sum((np.dot(matrix, test_lmk.T).T - templates[i]) ** 2, axis=1))
        )
        if error < min_error:
            min_error, best_matrix = error, matrix
    return best_matrix


def align_crop(
    img, lmk, image_size, mode="arcfacemap", interpolation=v2.InterpolationMode.NEAREST
):
    if mode != "arcfacemap":
        if mode == "arcface112":
            templates = float(image_size) / 112.0 * arcface_src
        else:
            factor = float(image_size) / 128.0
            # F-05: copy before modifying to avoid mutating the module-level template
            templates = arcface_src.copy() * factor
            templates[:, 0] += factor * 8.0
    else:
        templates = float(image_size) / 112.0 * src_map[112]

    matrix = get_matrix(lmk, templates)
    #'''
    # warped = cv2.warpAffine(
    #    img,
    #    matrix,
    #    (image_size, image_size),
    #    borderValue=0.0,
    #    borderMode=cv2.BORDER_REPLICATE,
    # )
    #'''
    warped = warp_affine_torchvision(
        img,
        matrix,
        (image_size, image_size),
        rotation_ratio=57.2958,
        border_value=0.0,
        border_mode="replicate",
        interpolation_value=v2.functional.InterpolationMode.NEAREST,
        device=img.device,
    )

    return warped, matrix


def get_arcface_template(image_size=112, mode="arcface112"):
    if mode == "arcface112":
        template = float(image_size) / 112.0 * arcface_src
    elif mode == "arcface128":
        factor = float(image_size) / 128.0
        template = arcface_src * factor
        template[:, 0] += factor * 8.0
    else:
        template = float(image_size) / 112.0 * src_map[112]

    return template


# lmk is prediction; src is template
def estimate_norm_arcface_template(lmk, src=arcface_src):
    assert lmk.shape == (5, 2)
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")

    for i in np.arange(src.shape[0]):
        # Use instance-based estimate for compatibility
        if hasattr(trans.SimilarityTransform, "from_estimate"):
            tform = trans.SimilarityTransform.from_estimate(lmk, src[i])
        else:
            tform = trans.SimilarityTransform()
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        # results is (2, 5), we need it to be (5, 2) to match src[i]
        results = np.dot(M, lmk_tran.T)
        results = results.T

        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    # print(src[min_index])
    return min_M, min_index


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode="arcface112"):
    assert lmk.shape == (5, 2)
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")

    if mode != "arcfacemap":
        if mode == "arcface112":
            src = float(image_size) / 112.0 * arcface_src
        else:
            factor = float(image_size) / 128.0
            # F-05: copy before modifying to avoid mutating the module-level template
            src = arcface_src.copy() * factor
            src[:, 0] += factor * 8.0
    else:
        src = float(image_size) / 112.0 * src_map[112]

    for i in np.arange(src.shape[0]):
        # Use instance-based estimate for compatibility
        if hasattr(trans.SimilarityTransform, "from_estimate"):
            tform = trans.SimilarityTransform.from_estimate(lmk, src[i])
        else:
            tform = trans.SimilarityTransform()
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        # results is (2, 5), we need it to be (5, 2) to match src[i]
        results = np.dot(M, lmk_tran.T)
        results = results.T

        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    # print(src[min_index])
    return min_M, min_index


def warp_face_by_bounding_box(img, bboxes, image_size=112):
    """
    OPTIMIZED: GPU Warping using Kornia.
    """
    # pad image by image size
    img = pad_image_by_size(img, image_size)

    source_points = np.array(
        [
            [bboxes[0], bboxes[1]],
            [bboxes[2], bboxes[1]],
            [bboxes[0], bboxes[3]],
            [bboxes[2], bboxes[3]],
        ]
    ).astype(np.float32)

    target_points = np.array(
        [[0, 0], [image_size, 0], [0, image_size], [image_size, image_size]]
    ).astype(np.float32)

    tform = trans.SimilarityTransform.from_estimate(source_points, target_points)
    M = tform.params[0:2]

    M_tensor = torch.from_numpy(M).float().unsqueeze(0).to(img.device)
    img_b = img.unsqueeze(0) if img.dim() == 3 else img

    img_warped = kgm.warp_affine(
        img_b.float(),
        M_tensor,
        dsize=(image_size, image_size),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)

    return img_warped.to(img.dtype), M


def warp_face_by_face_landmark_5(
    img,
    kpss,
    image_size=112,
    mode="arcface112",
    interpolation=v2.InterpolationMode.NEAREST,
):
    """
    OPTIMIZED: GPU Warping using Kornia.
    """
    # pad image by image size
    img = pad_image_by_size(img, image_size)

    M, pose_index = estimate_norm(kpss, image_size, mode=mode)

    M_tensor = torch.from_numpy(M).float().unsqueeze(0).to(img.device)
    img_b = img.unsqueeze(0) if img.dim() == 3 else img

    mode_kgm = (
        "nearest" if interpolation == v2.InterpolationMode.NEAREST else "bilinear"
    )

    img_warped = kgm.warp_affine(
        img_b.float(),
        M_tensor,
        dsize=(image_size, image_size),
        mode=mode_kgm,
        align_corners=True,
    ).squeeze(0)

    return img_warped.to(img.dtype), M


def getRotationMatrix2D(center, output_size, scale, rotation, is_clockwise=True):
    scale_ratio = scale
    if not is_clockwise:
        rotation = -rotation
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]

    return M


def invertAffineTransform(M):
    """
    t = trans.SimilarityTransform()
    t.params[0:2] = M
    IM = t.inverse.params[0:2, :]

    Returns a 2x3 affine matrix (the inverse), not the full 3x3 homogeneous form.
    All callers should receive and use a 2x3 matrix.
    """
    M_H = np.vstack([M, np.array([0, 0, 1])])
    IM = np.linalg.inv(M_H)

    # F-02: return only the 2x3 affine rows, not the full 3x3 homogeneous matrix
    return IM[0:2, :]


def warp_face_by_bounding_box_for_landmark_68(img, bbox, input_size):
    """
    OPTIMIZED: GPU Warping using Kornia.
    """
    # pad image by image size
    img = pad_image_by_size(img, input_size[0])

    scale = 195 / np.subtract(bbox[2:], bbox[:2]).max()
    translation = (256 - np.add(bbox[2:], bbox[:2]) * scale) * 0.5
    affine_matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]])

    M_tensor = torch.from_numpy(affine_matrix).float().unsqueeze(0).to(img.device)
    img_b = img.unsqueeze(0) if img.dim() == 3 else img

    crop_image = (
        kgm.warp_affine(
            img_b.float(),
            M_tensor,
            dsize=(input_size[1], input_size[0]),
            mode="bilinear",
            align_corners=True,
        )
        .squeeze(0)
        .to(img.dtype)
    )

    # Post-processing (CLAHE on CPU if too dark, kept as original as it's a rare fallback)
    if torch.mean(crop_image.to(dtype=torch.float32)[0, :, :]) < 30:
        lab = cv2.cvtColor(
            crop_image.permute(1, 2, 0).to("cpu").numpy(), cv2.COLOR_RGB2Lab
        )
        # F-06: CLAHE requires uint8; convert L channel from float Lab range to uint8 and back
        clahe = cv2.createCLAHE(clipLimit=2)
        L_u8 = (lab[:, :, 0] * 2.55).clip(0, 255).astype(np.uint8)
        L_eq = clahe.apply(L_u8)
        lab[:, :, 0] = L_eq.astype(np.float32) / 2.55
        crop_image = (
            torch.from_numpy(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB))
            .to(img.device)
            .permute(2, 0, 1)
        )

    return crop_image, affine_matrix


def warp_face_by_bounding_box_for_landmark_98(img, bbox_org, input_size):
    """
    :param img: raw image
    :param bbox: the bbox for the face
    :param input_size: tuple input image size
    :return:
    """
    # pad image by image size
    img = pad_image_by_size(img, input_size[0])

    ##preprocess
    bbox = bbox_org.copy()
    min_face = 20
    base_extend_range = [0.2, 0.3]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    if bbox_width <= min_face or bbox_height <= min_face:
        return None, None
    add = int(max(bbox_width, bbox_height))

    bimg = torch.nn.functional.pad(img, (add, add, add, add), "constant", 0)

    bbox += add

    face_width = (1 + 2 * base_extend_range[0]) * bbox_width
    center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    ### make the box as square
    bbox[0] = center[0] - face_width // 2
    bbox[1] = center[1] - face_width // 2
    bbox[2] = center[0] + face_width // 2
    bbox[3] = center[1] + face_width // 2

    # crop
    bbox = bbox.astype(np.int32)
    crop_image = bimg[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]

    h, w = (crop_image.size(dim=1), crop_image.size(dim=2))

    t_resize = v2.Resize((input_size[1], input_size[0]), antialias=False)
    crop_image = t_resize(crop_image)

    return crop_image, [h, w, bbox[1], bbox[0], add]


def create_bounding_box_from_face_landmark_106_98_68(face_landmark_106_98_68):
    min_x, min_y = np.min(face_landmark_106_98_68, axis=0)
    max_x, max_y = np.max(face_landmark_106_98_68, axis=0)
    # F-16: use int32 to avoid overflow for coordinates > 32767
    bounding_box = np.array([min_x, min_y, max_x, max_y]).astype(np.int32)
    return bounding_box


def convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    face_landmark_5 = np.stack(
        [
            np.mean(face_landmark_68[lm_idx[[1, 2]], :], 0),  # left eye
            np.mean(face_landmark_68[lm_idx[[3, 4]], :], 0),  # right eye
            face_landmark_68[lm_idx[0], :],  # nose
            face_landmark_68[lm_idx[5], :],  # lip
            face_landmark_68[lm_idx[6], :],  # lip
        ],
        axis=0,
    )

    if np.any(face_landmark_68_score):
        face_landmark_5_score = np.stack(
            [
                np.mean(face_landmark_68_score[lm_idx[[1, 2]], :], 0),  # left eye
                np.mean(face_landmark_68_score[lm_idx[[3, 4]], :], 0),  # right eye
                face_landmark_68_score[lm_idx[0], :],  # nose
                face_landmark_68_score[lm_idx[5], :],  # lip
                face_landmark_68_score[lm_idx[6], :],  # lip
            ],
            axis=0,
        )
    else:
        face_landmark_5_score = np.array([])

    return face_landmark_5, face_landmark_5_score


def convert_face_landmark_98_to_5(face_landmark_98, face_landmark_98_score):
    face_landmark_5 = np.array(
        [
            face_landmark_98[96],  # eye left
            face_landmark_98[97],  # eye-right
            face_landmark_98[54],  # nose,
            face_landmark_98[76],  # lip left
            face_landmark_98[82],  # lip right
        ]
    )

    face_landmark_5_score = np.array(
        [
            face_landmark_98_score[96],  # eye left
            face_landmark_98_score[97],  # eye-right
            face_landmark_98_score[54],  # nose,
            face_landmark_98_score[76],  # lip left
            face_landmark_98_score[82],  # lip right
        ]
    )

    return face_landmark_5, face_landmark_5_score


# TUFA's dense 314-point set. It is not a dataset topology, so these indices were
# recovered from the export fork's structure prompts: TUFA addresses a landmark by its
# anchor position in a canonical mean face, so an anchor that Prompt/shape_314.npz
# shares with Prompt/shape_98.npz is the same semantic point. The five below are exact
# (0.0000 px) anchor matches for WFLW 96/97/54/76/82, and running both graphs on the
# same crop puts the predicted pairs within 0.15 px of each other on the CPU EP and
# 0.05 px through the app's TensorRT path (a wrong pairing is ~87 px off, for scale).
LANDMARK_314_EYE_LEFT = 312
LANDMARK_314_EYE_RIGHT = 313
LANDMARK_314_NOSE_TIP = 154
LANDMARK_314_LIP_LEFT = 51
LANDMARK_314_LIP_RIGHT = 258
# Eye corners, used only for the use_mean_eyes variant. WFLW's lid points (62/66,
# 70/74) have no exact counterpart in the 314 set, so this averages the two corners
# per eye instead of four points; on the canonical face that lands within a pixel of
# the four-point mean.
LANDMARK_314_EYE_LEFT_CORNERS = [28, 126]  # WFLW 60 (outer), 64 (inner)
LANDMARK_314_EYE_RIGHT_CORNERS = [186, 283]  # WFLW 68 (inner), 72 (outer)


def convert_face_landmark_314_to_5(face_landmark_314, face_landmark_314_score):
    face_landmark_5 = np.array(
        [
            face_landmark_314[LANDMARK_314_EYE_LEFT],  # eye left
            face_landmark_314[LANDMARK_314_EYE_RIGHT],  # eye-right
            face_landmark_314[LANDMARK_314_NOSE_TIP],  # nose,
            face_landmark_314[LANDMARK_314_LIP_LEFT],  # lip left
            face_landmark_314[LANDMARK_314_LIP_RIGHT],  # lip right
        ]
    )

    face_landmark_5_score = np.array(
        [
            face_landmark_314_score[LANDMARK_314_EYE_LEFT],  # eye left
            face_landmark_314_score[LANDMARK_314_EYE_RIGHT],  # eye-right
            face_landmark_314_score[LANDMARK_314_NOSE_TIP],  # nose,
            face_landmark_314_score[LANDMARK_314_LIP_LEFT],  # lip left
            face_landmark_314_score[LANDMARK_314_LIP_RIGHT],  # lip right
        ]
    )

    return face_landmark_5, face_landmark_5_score


def convert_face_landmark_106_to_5(face_landmark_106):
    face_landmark_5 = np.array(
        [
            face_landmark_106[38],  # eye left
            face_landmark_106[88],  # eye-right
            face_landmark_106[86],  # nose,
            face_landmark_106[52],  # lip left
            face_landmark_106[61],  # lip right
        ]
    )

    return face_landmark_5


def convert_face_landmark_203_to_5(face_landmark_203, use_mean_eyes=False):
    if use_mean_eyes:
        eye_left = np.mean(
            face_landmark_203[[0, 6, 12, 18]], axis=0
        )  # Average of left eye points
        eye_right = np.mean(
            face_landmark_203[[24, 30, 36, 42]], axis=0
        )  # Average of right eye points
    else:
        eye_left = face_landmark_203[197]  # Specific left eye point
        eye_right = face_landmark_203[198]  # Specific right eye point

    nose = face_landmark_203[201]  # Nose
    lip_left = face_landmark_203[48]  # Left lip corner
    lip_right = face_landmark_203[66]  # Right lip corner

    face_landmark_5 = np.array([eye_left, eye_right, nose, lip_left, lip_right])

    return face_landmark_5


def convert_face_landmark_478_to_5(face_landmark_478, use_mean_eyes=False):
    if use_mean_eyes:
        eye_left = np.mean(
            face_landmark_478[[472, 471, 470, 469]], axis=0
        )  # Average of left eye points
        eye_right = np.mean(
            face_landmark_478[[477, 476, 475, 474]], axis=0
        )  # Average of right eye points
    else:
        eye_left = face_landmark_478[468]  # Specific left eye point
        eye_right = face_landmark_478[473]  # Specific right eye point

    nose = face_landmark_478[4]  # Nose
    lip_left = face_landmark_478[61]  # Left lip corner
    lip_right = face_landmark_478[291]  # Right lip corner

    face_landmark_5 = np.array([eye_left, eye_right, nose, lip_left, lip_right])

    return face_landmark_5


def convert_face_landmark_x_to_5(pts, **kwargs):
    pts_score = kwargs.get("pts_score", [])
    use_mean_eyes = kwargs.get("use_mean_eyes", False)

    if pts.shape[0] == 5:
        return pts
    elif pts.shape[0] == 68:
        pt5 = convert_face_landmark_68_to_5(
            face_landmark_68=pts, face_landmark_68_score=pts_score
        )
    elif pts.shape[0] == 98:
        pt5 = convert_face_landmark_98_to_5(
            face_landmark_98=pts, face_landmark_98_score=pts_score
        )
    elif pts.shape[0] == 314:
        pt5 = convert_face_landmark_314_to_5(
            face_landmark_314=pts, face_landmark_314_score=pts_score
        )
    elif pts.shape[0] == 106:
        pt5 = convert_face_landmark_106_to_5(face_landmark_106=pts)
    elif pts.shape[0] == 203:
        pt5 = convert_face_landmark_203_to_5(
            face_landmark_203=pts, use_mean_eyes=use_mean_eyes
        )
    elif pts.shape[0] == 478:
        pt5 = convert_face_landmark_478_to_5(
            face_landmark_478=pts, use_mean_eyes=use_mean_eyes
        )
    else:
        raise ValueError(f"Unknow shape: {pts.shape}")

    return pt5


def test_bbox_landmarks(img, bbox, kpss, caption="image", show_kpss_label=False):
    image = img.permute(1, 2, 0).to("cpu").numpy().copy()
    if len(bbox) > 0:
        box = bbox.astype(int)
        color = (255, 0, 0)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

    if len(kpss) > 0:
        for i in range(kpss.shape[0]):
            kps = kpss[i].astype(int)
            color = (0, 0, 255)
            cv2.circle(image, (kps[0], kps[1]), 1, color, 2)
            text = None
            if show_kpss_label:
                if kpss.shape[0] == 5:
                    match i:
                        case 0:
                            text = "LE"
                        case 1:
                            text = "RE"
                        case 2:
                            text = "NO"
                        case 3:
                            text = "LM"
                        case 4:
                            text = "RM"
                else:
                    text = str(i)

                image = cv2.putText(
                    image,
                    text,
                    (kps[0], kps[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    False,
                )

    cv2.imshow(caption, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_multi_bbox_landmarks(
    img, bboxes, kpss, caption="image", show_kpss_label=False
):
    if len(bboxes) > 0 and len(kpss) > 0:
        for i in range(np.array(kpss).shape[0]):
            test_bbox_landmarks(
                img,
                bboxes[i],
                kpss[i],
                caption=caption,
                show_kpss_label=show_kpss_label,
            )
    elif len(bboxes) > 0:
        for i in range(np.array(bboxes).shape[0]):
            test_bbox_landmarks(
                img, bboxes[i], [], caption=caption, show_kpss_label=show_kpss_label
            )
    elif len(kpss) > 0:
        for i in range(np.array(kpss).shape[0]):
            test_bbox_landmarks(
                img, [], kpss[i], caption=caption, show_kpss_label=show_kpss_label
            )


def detect_img_color(img):
    frame = img.permute(1, 2, 0)

    b = frame[:, :, :1]
    g = frame[:, :, 1:2]
    r = frame[:, :, 2:]

    # computing the mean
    b_mean = torch.mean(b.to(float))
    g_mean = torch.mean(g.to(float))
    r_mean = torch.mean(r.to(float))

    # displaying the most prominent color
    if b_mean > g_mean and b_mean > r_mean:
        return "BGR"
    elif g_mean > r_mean and g_mean > b_mean:
        return "GBR"

    return "RGB"


def get_face_orientation(face_size, lmk):
    assert lmk.shape == (5, 2)
    src = np.squeeze(arcface_src, axis=0)
    src = float(face_size) / 112.0 * src

    # Use instance-based estimate for compatibility
    if hasattr(trans.SimilarityTransform, "from_estimate"):
        tform = trans.SimilarityTransform.from_estimate(lmk, src)
    else:
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, src)

    angle_deg_to_front = np.rad2deg(tform.rotation)

    return angle_deg_to_front


def rgb_to_yuv(image, normalize=False):
    """
    Convert an RGB image to YUV.
    Args:
        image (torch.Tensor): The input image tensor in RGB format (C, H, W) with values in the range [0, 255].
    Returns:
        torch.Tensor: The image tensor in YUV format (C, H, W).
    """
    if normalize:
        # Ensure the image is in the range [0, 1]
        image = torch.div(image, 255.0)

    # F-09: use module-level constant matrix, moved to device on demand
    conversion_matrix = _RGB_TO_YUV.to(device=image.device, dtype=image.dtype)

    # Apply the conversion matrix
    yuv_image = torch.tensordot(
        image.permute(1, 2, 0), conversion_matrix, dims=1
    ).permute(2, 0, 1)

    return yuv_image


def yuv_to_rgb(image, normalize=False):
    """
    Convert a YUV image to RGB.
    Args:
        image (torch.Tensor): The input image tensor in YUV format (C, H, W) with values in the range [0, 1].
    Returns:
        torch.Tensor: The image tensor in RGB format (C, H, W).
    """
    # F-09: use module-level constant matrix, moved to device on demand
    conversion_matrix = _YUV_TO_RGB.to(device=image.device, dtype=image.dtype)

    # Apply the conversion matrix
    rgb_image = torch.tensordot(
        image.permute(1, 2, 0), conversion_matrix, dims=1
    ).permute(2, 0, 1)

    # Ensure the image is in the range [0, 1]
    rgb_image = torch.clamp(rgb_image, 0, 1)

    if normalize:
        rgb_image = torch.mul(rgb_image, 255.0)

    return rgb_image


def rgb_to_lab(rgb: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Converts an RGB image to LAB color space.
    OPTIMIZED: Replaced manual PyTorch math with Kornia's fused CUDA kernels.
    Expects (C, H, W) or (B, C, H, W).
    """
    if normalize:
        rgb = rgb / 255.0

    # Ensure batch dimension exists for Kornia
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
        lab = kc.rgb_to_lab(rgb).squeeze(0)
    else:
        lab = kc.rgb_to_lab(rgb)

    return lab


def lab_to_rgb(lab: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Converts a LAB image to RGB color space.
    OPTIMIZED: Replaced manual PyTorch math with Kornia's fused CUDA kernels.
    Expects (C, H, W) or (B, C, H, W).
    """
    # Ensure batch dimension exists for Kornia
    if lab.dim() == 3:
        lab = lab.unsqueeze(0)
        rgb = kc.lab_to_rgb(lab).squeeze(0)
    else:
        rgb = kc.lab_to_rgb(lab)

    if normalize:
        rgb = rgb * 255.0

    return rgb


def rgb_to_hsv(image):
    device = (
        image.device
    )  # Ensure operations happen on the same device as the input image

    # Convert image to float if needed
    image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()

    r, g, b = image[0], image[1], image[2]  # Split the RGB channels

    max_val, _ = torch.max(
        image, dim=0
    )  # Max value per pixel across RGB channels, shape [512, 512]
    min_val, _ = torch.min(
        image, dim=0
    )  # Min value per pixel across RGB channels, shape [512, 512]
    delta = max_val - min_val  # Difference between max and min, shape [512, 512]

    # Initialize Hue, Saturation, and Value tensors as float32
    h = torch.zeros_like(max_val, dtype=torch.float32).to(device)
    s = torch.zeros_like(max_val, dtype=torch.float32).to(device)
    v = max_val  # Value is max_val (no need to change dtype)

    # Avoid division by zero: only compute where delta != 0
    mask = delta != 0

    # Hue calculation based on which color channel is the maximum
    r_mask = max_val == r
    g_mask = max_val == g
    b_mask = max_val == b

    h[mask & r_mask] = ((g - b) / delta % 6)[mask & r_mask]
    h[mask & g_mask] = (((b - r) / delta) + 2)[mask & g_mask]
    h[mask & b_mask] = (((r - g) / delta) + 4)[mask & b_mask]

    h = h * 60.0  # Scale hue to [0, 360] range
    h = h / 360.0  # Normalize hue to [0, 1]

    # Saturation calculation: only compute where max_val != 0
    s[max_val != 0] = (delta / max_val)[max_val != 0]

    # Stack the HSV channels together
    hsv_image = torch.stack([h, s, v], dim=0)

    return hsv_image


def hsv_to_rgb(hsv_image):
    device = (
        hsv_image.device
    )  # Ensure operations happen on the same device as the input image

    h, s, v = hsv_image[0], hsv_image[1], hsv_image[2]  # Split the HSV channels

    h = h * 360.0  # Convert hue back to [0, 360] range

    c = v * s  # Chroma
    x = c * (1 - torch.abs((h / 60.0) % 2 - 1))  # Second largest component of the color
    m = v - c  # Match value

    # Initialize r, g, b with zeros
    r = torch.zeros_like(h, device=device)
    g = torch.zeros_like(h, device=device)
    b = torch.zeros_like(h, device=device)

    # Conditions for different hue ranges
    h1 = (0 <= h) & (h < 60)
    h2 = (60 <= h) & (h < 120)
    h3 = (120 <= h) & (h < 180)
    h4 = (180 <= h) & (h < 240)
    h5 = (240 <= h) & (h < 300)
    h6 = (300 <= h) & (h < 360)

    # Apply the color transformation logic based on hue ranges
    r[h1] = c[h1]
    g[h1] = x[h1]
    b[h1] = 0

    r[h2] = x[h2]
    g[h2] = c[h2]
    b[h2] = 0

    r[h3] = 0
    g[h3] = c[h3]
    b[h3] = x[h3]

    r[h4] = 0
    g[h4] = x[h4]
    b[h4] = c[h4]

    r[h5] = x[h5]
    g[h5] = 0
    b[h5] = c[h5]

    r[h6] = c[h6]
    g[h6] = 0
    b[h6] = x[h6]

    # Add m to match the value and scale the RGB channels back to [0, 1]
    r = r + m
    g = g + m
    b = b + m

    # Stack the RGB channels together
    rgb_image = torch.stack([r, g, b], dim=0)

    return rgb_image


def sharpen(img):
    device = img.device  # Ensure we use the same device

    # Convert img to float and normalize it
    img = img.float() / 255.0

    # Gaussian smoothing using PyTorch's functional API (approximation of Gaussian blur)
    # F-07: pass sigma as keyword arg; the first positional param is sigma, second is kernel_size
    gauss_kernel = get_gaussian_kernel(sigma=1.5, kernel_size=5).to(
        device
    )  # Create a Gaussian kernel for blurring
    img = img.unsqueeze(0)  # Add batch dimension for convolution
    gauss_out = torch.nn.functional.conv2d(
        img, gauss_kernel, padding=2, groups=img.size(1)
    )
    gauss_out = gauss_out.squeeze(0)  # Remove batch dimension

    alpha = 1.5
    img_out = (img.squeeze(0) - gauss_out) * alpha + img.squeeze(0)

    # Clamp values between 0 and 1, then scale back to [0, 255]
    img_out = torch.clamp(img_out, 0.0, 1.0) * 255.0

    return img_out.to(torch.uint8)


def get_gaussian_kernel(sigma: float, kernel_size: int = 5) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for convolution.
    OPTIMIZED: Thread-safe caching added to prevent re-allocating
    the mathematical kernel tensor on every sharpen pass.
    """
    cache_key = (float(sigma), int(kernel_size))

    with _gaussian_kernel_lock:
        if cache_key in _gaussian_kernel_cache:
            return _gaussian_kernel_cache[cache_key]

    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    g_kernel = torch.outer(g, g)
    g_kernel = g_kernel.unsqueeze(0).unsqueeze(0)  # Make it 4D for convolution
    final_kernel = g_kernel.expand(
        3, 1, kernel_size, kernel_size
    )  # Apply to each channels

    with _gaussian_kernel_lock:
        _gaussian_kernel_cache[cache_key] = final_kernel

    return final_kernel


# Live Portrait
# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt101(pt101, use_lip=True):
    """
    parsing the 2 points according to the 101 points, which cancels the roll
    """
    # the former version use the eye center, but it is not robust, now use interpolation
    pt_left_eye = np.mean(pt101[[39, 42, 45, 48]], axis=0)  # left eye center
    pt_right_eye = np.mean(pt101[[51, 54, 57, 60]], axis=0)  # right eye center

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt101[75] + pt101[81]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2


def parse_pt2_from_pt98(pt98, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 98 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(
            pt98[[66, 60, 62, 64]], axis=0
        )  # Average of left eye points
        pt_right_eye = np.mean(
            pt98[[74, 68, 70, 72]], axis=0
        )  # Average of right eye points
    else:
        pt_left_eye = pt98[96]  # Specific left eye point
        pt_right_eye = pt98[97]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt98[76] + pt98[82]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt106(pt106, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 106 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(
            pt106[[33, 35, 40, 39]], axis=0
        )  # Average of left eye points
        pt_right_eye = np.mean(
            pt106[[87, 89, 94, 93]], axis=0
        )  # Average of right eye points
    else:
        pt_left_eye = pt106[38]  # Specific left eye point
        pt_right_eye = pt106[88]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt203(pt203, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 203 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(
            pt203[[0, 6, 12, 18]], axis=0
        )  # Average of left eye points
        pt_right_eye = np.mean(
            pt203[[24, 30, 36, 42]], axis=0
        )  # Average of right eye points
    else:
        pt_left_eye = pt203[197]  # Specific left eye point
        pt_right_eye = pt203[198]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2


def parse_pt2_from_pt478(pt478, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to the 203 points, which cancels the roll
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(
            pt478[[472, 471, 470, 469]], axis=0
        )  # Average of left eye points
        pt_right_eye = np.mean(
            pt478[[477, 476, 475, 474]], axis=0
        )  # Average of right eye points
    else:
        pt_left_eye = pt478[468]  # Specific left eye point
        pt_right_eye = pt478[473]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt478[61] + pt478[291]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2


def parse_pt2_from_pt314(pt314, use_lip=True, use_mean_eyes=False):
    """
    parsing the 2 points according to TUFA's dense 314 points, which cancels the roll

    Indices come from the anchor correspondence documented above
    convert_face_landmark_314_to_5. Without this branch parse_pt2_from_pt_x would fall
    into its ``shape[0] > 101`` case and read the first 101 points as if they were the
    101-point topology, which for this x-sorted set is the left third of the face.
    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt314[LANDMARK_314_EYE_LEFT_CORNERS], axis=0)
        pt_right_eye = np.mean(pt314[LANDMARK_314_EYE_RIGHT_CORNERS], axis=0)
    else:
        pt_left_eye = pt314[LANDMARK_314_EYE_LEFT]  # Specific left eye point
        pt_right_eye = pt314[LANDMARK_314_EYE_RIGHT]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (
            pt314[LANDMARK_314_LIP_LEFT] + pt314[LANDMARK_314_LIP_RIGHT]
        ) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2


def parse_pt2_from_pt68(pt68, use_lip=True):
    """
    parsing the 2 points according to the 68 points, which cancels the roll
    """
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    if use_lip:
        pt5 = np.stack(
            [
                np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
                np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
                pt68[lm_idx[0], :],  # nose
                pt68[lm_idx[5], :],  # lip
                pt68[lm_idx[6], :],  # lip
            ],
            axis=0,
        )

        pt2 = np.stack([(pt5[0] + pt5[1]) / 2, (pt5[3] + pt5[4]) / 2], axis=0)
    else:
        pt2 = np.stack(
            [
                np.mean(pt68[lm_idx[[1, 2]], :], 0),  # left eye
                np.mean(pt68[lm_idx[[3, 4]], :], 0),  # right eye
            ],
            axis=0,
        )

    return pt2


def parse_pt2_from_pt5(pt5, use_lip=True):
    """
    parsing the 2 points according to the 5 points, which cancels the roll
    """
    pt_left_eye = pt5[0]  # Specific left eye point
    pt_right_eye = pt5[1]  # Specific right eye point

    if use_lip:
        # use lip
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt5[3] + pt5[4]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)

    return pt2


def parse_pt2_from_pt9(pt9, use_lip=True):
    """
    parsing the 2 points according to the 9 points, which cancels the roll
    ['right eye right', 'right eye left', 'left eye right', 'left eye left', 'nose tip', 'lip right', 'lip left', 'upper lip', 'lower lip']
    """
    if use_lip:
        pt2 = np.stack(
            [
                (pt9[2] + pt9[3]) / 2,  # eye center (left+right eye avg)
                (pt9[5] + pt9[6]) / 2,  # lip center
            ],
            axis=0,
        )
    else:
        pt2 = np.stack(
            [
                (pt9[2] + pt9[3]) / 2,
                (pt9[0] + pt9[1]) / 2,
            ],
            axis=0,
        )

    return pt2


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_pt2_from_pt_x(pts, use_lip=True, use_mean_eyes=False):
    if pts.shape[0] == 101:
        pt2 = parse_pt2_from_pt101(pts, use_lip=use_lip)
    elif pts.shape[0] == 106:
        pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 68:
        pt2 = parse_pt2_from_pt68(pts, use_lip=use_lip)
    elif pts.shape[0] == 5:
        pt2 = parse_pt2_from_pt5(pts, use_lip=use_lip)
    elif pts.shape[0] == 203:
        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 98:
        pt2 = parse_pt2_from_pt98(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 478:
        pt2 = parse_pt2_from_pt478(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] == 314:
        pt2 = parse_pt2_from_pt314(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
    elif pts.shape[0] > 101:
        # take the first 101 points
        pt2 = parse_pt2_from_pt101(pts[:101], use_lip=use_lip)
    elif pts.shape[0] == 9:
        pt2 = parse_pt2_from_pt9(pts, use_lip=use_lip)
    else:
        raise Exception(f"Unknow shape: {pts.shape}")

    if not use_lip:
        # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]

    return pt2


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_rect_from_landmark(
    pts,
    scale=1.5,
    need_square=True,
    vx_ratio=0,
    vy_ratio=0,
    use_deg_flag=False,
    **kwargs,
):
    """parsing center, size, angle from 101/68/5/x landmarks
    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size
    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area

    judge with pts.shape
    """
    pt2 = parse_pt2_from_pt_x(
        pts,
        use_lip=kwargs.get("use_lip", True),
        use_mean_eyes=kwargs.get("use_mean_eyes", False),
    )

    uy = pt2[1] - pt2[0]
    norm_uy = np.linalg.norm(uy)
    if norm_uy <= 1e-3:
        uy = np.array([0, 1], dtype=np.float32)
    else:
        uy /= norm_uy
    ux = np.array((uy[1], -uy[0]), dtype=np.float32)

    # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
    # print(uy)
    # print(ux)
    # F-15: clamp acos argument to [-1, 1] to prevent domain errors from floating-point drift
    angle = acos(max(-1.0, min(1.0, ux[0])))
    if ux[1] < 0:
        angle = -angle

    # rotation matrix
    M = np.array([ux, uy])

    # calculate the size which contains the angle degree of the bbox, and the center
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m

    size *= scale  # scale size
    center = (
        center0 + ux * center1[0] + uy * center1[1]
    )  # counterclockwise rotation, equivalent to M.T @ center1.T
    center = (
        center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    )  # considering the offset in vx and vy direction

    if use_deg_flag:
        angle = degrees(angle)

    return center, size, angle


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def parse_bbox_from_landmark(pts, **kwargs):
    center, size, angle = parse_rect_from_landmark(pts, **kwargs)
    cx, cy = center
    w, h = size

    # calculate the vertex positions before rotation
    bbox = np.array(
        [
            [cx - w / 2, cy - h / 2],  # left, top
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],  # right, bottom
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float32,
    )

    # construct rotation matrix
    bbox_rot = bbox.copy()
    R = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )

    # calculate the relative position of each vertex from the rotation center, then rotate these positions, and finally add the coordinates of the rotation center
    bbox_rot = (bbox_rot - center) @ R.T + center

    return {
        "center": center,  # 2x1
        "size": size,  # scalar
        "angle": angle,  # rad, counterclockwise
        "bbox": bbox,  # 4x2
        "bbox_rot": bbox_rot,  # 4x2
    }


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def _estimate_similar_transform_from_pts(
    pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=-0.1, flag_do_rot=True, **kwargs
):
    """calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image
    pts: landmark, 101 or 68 points or other points, Nx2
    scale: the larger scale factor, the smaller face ratio
    vx_ratio: x shift
    vy_ratio: y shift, the smaller the y shift, the lower the face region
    rot_flag: if it is true, conduct correction
    """
    center, size, angle = parse_rect_from_landmark(
        pts,
        scale=scale,
        vx_ratio=vx_ratio,
        vy_ratio=vy_ratio,
        use_lip=kwargs.get("use_lip", True),
        use_mean_eyes=kwargs.get("use_mean_eyes", False),
    )

    s = dsize / size[0]  # scale
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=np.float32)  # center of dsize

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]  # ori center
        tcx, tcy = tgt_center[0], tgt_center[1]  # target center
        # need to infer
        M_INV = np.array(
            [
                [s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                [
                    -s * sintheta,
                    s * costheta,
                    tcy - s * (-sintheta * cx + costheta * cy),
                ],
            ],
            dtype=np.float32,
        )
    else:
        M_INV = np.array(
            [
                [s, 0, tgt_center[0] - s * center[0]],
                [0, s, tgt_center[1] - s * center[1]],
            ],
            dtype=np.float32,
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)

    # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
    return M_INV, M[:2, ...]


def warp_face_by_face_landmark_x(img, pts, **kwargs):
    """
    OPTIMIZED: Uses Kornia (GPU) instead of torchvision+scikit-image for affine transformation.
    Eliminates CPU-bound trigonometry and matrix decomposition bottlenecks.
    """
    dsize = kwargs.get("dsize", 224)  # Default LivePortrait size
    scale = kwargs.get("scale", 1.5)
    vy_ratio = kwargs.get("vy_ratio", -0.1)
    # Out-of-bounds sampling fill. Defaults to "zeros" (unchanged behaviour).
    # Callers that crop with a wide scale and/or a face near the image edge
    # (e.g. the VR180 perspective crops fed to PerformRecast) can pass
    # "border" to replicate edge pixels instead of producing black borders,
    # which otherwise yield all/mostly-black face crops.
    padding_mode = kwargs.get("padding_mode", "zeros")

    # Pad image if necessary
    img = pad_image_by_size(img, dsize)

    # Calculate matrix from landmarks (Fast CPU operation, no bottleneck here)
    M_o2c, M_c2o = _estimate_similar_transform_from_pts(
        pts,
        dsize=dsize,
        scale=scale,
        vy_ratio=vy_ratio,
        flag_do_rot=kwargs.get("flag_do_rot", True),
    )

    # 100% GPU Warping using Kornia
    # Bypasses the slow t = trans.SimilarityTransform() completely
    warped_img = transform_img_kgm(
        img.float(), M_o2c, dsize=dsize, padding_mode=padding_mode
    )

    # Ensure we return the original dtype (usually uint8 or float32 depending on pipeline)
    warped_img = warped_img.to(img.dtype)

    return warped_img, M_o2c, M_c2o


def create_faded_inner_mask(
    size: tuple[int, int],
    border_thickness: int,
    fade_thickness: int,
    blur_radius: int = 3,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a mask with a thick black border and a faded white center towards the border.
    The white edges are smoothed using Gaussian blur.

    OPTIMIZED: Added thread-safe caching. Generating 512x512 meshgrids and running
    Gaussian blurs dynamically 60 times a second severely fragments VRAM.
    This caches the final tensor based on its deterministic parameters.
    Parameters:
    - size: Tuple (height, width) for the mask size.
    - border_thickness: The thickness of the outer black border.
    - fade_thickness: The thickness over which the white center fades into the black border.
    - blur_radius: The radius for the Gaussian blur to smooth the white edges.
    - device: Device to perform the computation ('cuda' for GPU, 'cpu' for CPU).

    Returns:
    - mask: A PyTorch tensor containing the mask.
    """
    cache_key = (
        size[0],
        size[1],
        border_thickness,
        fade_thickness,
        blur_radius,
        str(device),
    )

    with _faded_mask_lock:
        if cache_key in _faded_mask_cache:
            return _faded_mask_cache[cache_key]

    height, width = size
    mask = torch.zeros(
        (height, width), dtype=torch.float32, device=device
    )  # Start with a black mask

    # Define the inner region
    inner_start = border_thickness
    inner_end_x = width - border_thickness
    inner_end_y = height - border_thickness

    # Create grid for distance calculations on the specified device
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    # Calculate distances to the nearest border for each point
    dist_to_left = x_indices - inner_start
    dist_to_right = inner_end_x - x_indices - 1
    dist_to_top = y_indices - inner_start
    dist_to_bottom = inner_end_y - y_indices - 1

    # Calculate minimum distance to any border
    dist_to_border = torch.minimum(
        torch.minimum(dist_to_left, dist_to_right),
        torch.minimum(dist_to_top, dist_to_bottom),
    )

    # Mask inside the fading region
    fade_region = (dist_to_border >= 0) & (dist_to_border < fade_thickness)
    mask[fade_region] = dist_to_border[fade_region] / fade_thickness

    # Mask in the full white region
    white_region = dist_to_border >= fade_thickness
    mask[white_region] = 1.0

    # Apply Gaussian blur to smooth the white edges
    mask = mask[None, None, :, :]  # Aggiungi batch e channel
    mask = torchvision.transforms.functional.gaussian_blur(
        mask, kernel_size=(blur_radius, blur_radius), sigma=(blur_radius / 2)
    )
    mask = mask[0, 0, :, :]  # Rimuovi batch e channel

    with _faded_mask_lock:
        _faded_mask_cache[cache_key] = mask

    return mask


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def prepare_paste_back(
    mask_crop, crop_M_c2o, dsize, interpolation=v2.InterpolationMode.BILINEAR
):
    """
    OPTIMIZED: prepare mask for later image paste back using direct GPU warping (Kornia).
    """
    # pad image by image size
    mask_crop = pad_image_by_size(mask_crop, (dsize[0], dsize[1]))

    # Use Kornia to warp the mask back to original space in one GPU pass
    mask_ori = transform_img_kgm(mask_crop.float(), crop_M_c2o, dsize=dsize)

    return mask_ori


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/crop.py
def paste_back(
    img_crop, M_c2o, img_ori, mask_ori, interpolation=v2.InterpolationMode.BILINEAR
):
    """
    OPTIMIZED: paste back the image using Kornia for Affine Transform
    and PyTorch in-place operations for blending to save VRAM and latency.
    """
    dsize = (img_ori.shape[1], img_ori.shape[2])

    # pad image by image size
    img_crop = pad_image_by_size(img_crop, dsize)

    # Transform the crop back to the original image space using Kornia
    output = transform_img_kgm(img_crop.float(), M_c2o, dsize=dsize)

    # F-03: clone before in-place ops to avoid mutating the caller's tensor
    img_ori_float = img_ori.clone().float()

    # Highly optimized in-place blending to avoid memory fragmentation
    # output = mask_ori * output + (1 - mask_ori) * img_ori
    output.mul_(mask_ori)
    output.add_(img_ori_float.mul_(1.0 - mask_ori))

    # Clamp and convert back to uint8
    output.clamp_(0, 255)

    return output.to(torch.uint8)


@torch.no_grad()
def paste_back_adv(
    img_crop: torch.Tensor,
    M_c2o: np.ndarray,
    img: torch.Tensor,
    mask_crop: torch.Tensor,
    interpolation=None,
) -> torch.Tensor:
    """
    Paste back the transformed cropped image onto the original image with a mask.
    OPTIMIZED: Eliminates full-frame padding. Calculates the bounding box of the
    transformed face, warps locally via Kornia, and blends in-place.
    Saves massive VRAM and compute time on high-resolution (HD/4K) footage.
    """
    device = img.device

    # 1. Calculate the bounding box of the transformed crop in the original image space
    crop_h, crop_w = img_crop.shape[1], img_crop.shape[2]
    corners = np.array(
        [[0, 0], [0, crop_h - 1], [crop_w - 1, 0], [crop_w - 1, crop_h - 1]],
        dtype=np.float32,
    )

    x = M_c2o[0, 0] * corners[:, 0] + M_c2o[0, 1] * corners[:, 1] + M_c2o[0, 2]
    y = M_c2o[1, 0] * corners[:, 0] + M_c2o[1, 1] * corners[:, 1] + M_c2o[1, 2]

    # Constrain the bounding box to the actual dimensions of the original image
    left = max(math.floor(np.min(x)), 0)
    top = max(math.floor(np.min(y)), 0)
    right = min(math.ceil(np.max(x)), img.shape[2])
    bottom = min(math.ceil(np.max(y)), img.shape[1])

    bw = right - left
    bh = bottom - top

    # Failsafe: if the bounding box is completely off-screen, return the original image
    if bw <= 0 or bh <= 0:
        return img

    # 2. Adjust transformation matrix for the localized bounding box
    M_local = M_c2o.copy()
    M_local[0, 2] -= left
    M_local[1, 2] -= top

    M_tensor = torch.from_numpy(M_local).float().to(device)

    # 3. Ensure input crops are float32 [0, 1] for precise Kornia warping
    img_crop_f = img_crop.float() / 255.0 if img_crop.dtype == torch.uint8 else img_crop
    mask_crop_f = (
        mask_crop.float() / 255.0 if mask_crop.dtype == torch.uint8 else mask_crop
    )

    # 4. Warp img_crop and mask_crop LOCALLY using Kornia (100% GPU, zero padding overhead)
    img_warped = kgm.warp_affine(
        img_crop_f.unsqueeze(0),
        M_tensor.unsqueeze(0),
        dsize=(bh, bw),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)

    mask_warped = kgm.warp_affine(
        mask_crop_f.unsqueeze(0),
        M_tensor.unsqueeze(0),
        dsize=(bh, bw),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)

    mask_warped.clamp_(0.0, 1.0)

    # 5. Extract original region, blend in-place, and insert back
    img_is_uint8 = img.dtype == torch.uint8
    img_float = img.float() / 255.0 if img_is_uint8 else img

    target_roi = img_float[:, top:bottom, left:right]

    # In-place blending logic
    blended_roi = (img_warped * mask_warped) + (target_roi * (1.0 - mask_warped))

    # Cast back to the original image tensor
    if img_is_uint8:
        img[:, top:bottom, left:right] = torch.clamp(blended_roi * 255.0, 0, 255).to(
            torch.uint8
        )
    else:
        # If the original image was already float, assign directly
        img[:, top:bottom, left:right] = blended_roi

    return img


def paste_back_kgm(img_crop, M_c2o, img_ori, mask_ori):
    """paste back the image"""
    dsize = (img_ori.shape[1], img_ori.shape[2])

    # pad image by image size
    img_crop = pad_image_by_size(img_crop, (img_ori.shape[1], img_ori.shape[2]))

    img_crop = img_crop.float()
    img_back = transform_img_kgm(img_crop, M_c2o, dsize=dsize)
    img_back = torch.clip(mask_ori * img_back + (1 - mask_ori) * img_ori, 0, 255)

    return img_back.to(torch.uint8)


def transform_img_kgm(
    img,
    M,
    dsize,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
    fill_value=(0, 0, 0),
):
    """Conduct similarity or affine transformation to the image using PyTorch CUDA.

    Args:
    img (torch.Tensor): Input image tensor (C x H x W)
    M (torch.Tensor): 2x3 or 3x3 transformation matrix
    dsize (tuple[int, int]): size of the output image (height, width).
    mode(str, optional): interpolation mode to calculate output values 'bilinear' | 'nearest'. Default: "bilinear"
    align_corners (bool, optional): mode for grid_generation. Default: True
    fill_value (Tensor, optional): tensor of shape that fills the padding area. Only supported for RGB. Default: zeros(3)

    Returns:
    torch.Tensor: Transformed image
    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)

    # Convert M to a torch tensor if it is a numpy.ndarray
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).to(img.device, dtype=img.dtype)

        # Prepare the transformation matrix
    M = M[:2, :]  # Ensure it's a 2x3 matrix

    img_transformed = kgm.warp_affine(
        src=img.unsqueeze(0),
        M=M[None],
        dsize=(_dsize[0], _dsize[1]),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        fill_value=fill_value,
    )
    img_transformed = img_transformed.squeeze(0)

    return img_transformed


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def calculate_distance_ratio(
    lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6
) -> np.ndarray:
    return np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) / (
        np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps
    )


def calc_eye_close_ratio(
    lmk: np.ndarray, target_eye_ratio: np.ndarray = None
) -> np.ndarray:
    """
    Calculates the Eye Aspect Ratio (EAR) with strict projection safeguards.
    Includes Profile Occlusion Detection and Symmetric Blink Harmonization
    to completely eliminate "fisheyes" and "lazy eyes".

    Args:
        lmk: Array of shape (N, 203, 2) or (1, 203, 2) containing landmarks.
        target_eye_ratio: Optional target ratio to concatenate.

    Returns:
        np.ndarray: The safely clamped and harmonized eye ratios.
    """
    # 1. Calculate raw horizontal width of the eyes
    raw_left_width = np.linalg.norm(lmk[:, 0] - lmk[:, 12], axis=1, keepdims=True)
    raw_right_width = np.linalg.norm(lmk[:, 24] - lmk[:, 36], axis=1, keepdims=True)

    # SAFEGUARD A: Profile Occlusion Detection (The Fisheye Fix)
    # If one eye is significantly narrower horizontally than the other (< 55%),
    # the face is turned. The hidden eye's 2D landmarks are unreliable.
    left_occluded = raw_left_width < (raw_right_width * 0.55)
    right_occluded = raw_right_width < (raw_left_width * 0.55)

    # SAFEGUARD B: Clamp minimum width to prevent ZeroDivision on extreme squishing
    min_eye_width = 4.0
    left_eye_width = np.maximum(raw_left_width, min_eye_width)
    right_eye_width = np.maximum(raw_right_width, min_eye_width)

    # 2. Calculate vertical height of the eyes
    left_eye_height = np.linalg.norm(lmk[:, 6] - lmk[:, 18], axis=1, keepdims=True)
    right_eye_height = np.linalg.norm(lmk[:, 30] - lmk[:, 42], axis=1, keepdims=True)

    # 3. Calculate Base Ratios
    lefteye_close_ratio = left_eye_height / left_eye_width
    righteye_close_ratio = right_eye_height / right_eye_width

    # SAFEGUARD C: Apply Occlusion Lock
    # Force the hidden eye to perfectly mirror the visible eye's EAR.
    # This prevents the network from rendering a bulging wide-open eye.
    lefteye_close_ratio = np.where(
        left_occluded, righteye_close_ratio, lefteye_close_ratio
    )
    righteye_close_ratio = np.where(
        right_occluded, lefteye_close_ratio, righteye_close_ratio
    )

    # SAFEGUARD D: Symmetric Blink Harmonization (Anti "Lazy-Eye")
    blink_threshold = 0.28
    is_blinking = (lefteye_close_ratio < blink_threshold) & (
        righteye_close_ratio < blink_threshold
    )

    avg_ratio = (lefteye_close_ratio + righteye_close_ratio) / 2.0

    lefteye_close_ratio = np.where(is_blinking, avg_ratio, lefteye_close_ratio)
    righteye_close_ratio = np.where(is_blinking, avg_ratio, righteye_close_ratio)

    # SAFEGUARD E: Hard clamp the final ratio to biologically plausible limits.
    max_safe_ear = 0.45
    lefteye_close_ratio = np.clip(lefteye_close_ratio, 0.0, max_safe_ear)
    righteye_close_ratio = np.clip(righteye_close_ratio, 0.0, max_safe_ear)

    if target_eye_ratio is not None:
        return np.concatenate(
            [lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1
        )
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
# def calc_lip_close_ratio(lmk: np.ndarray) -> np.ndarray:
#    return calculate_distance_ratio(lmk, 90, 102, 48, 66)
def calc_lip_close_ratio(lmk: np.ndarray) -> np.ndarray:
    """
    Calculates the Mouth Aspect Ratio (MAR) with strict projection safeguards.
    Prevents division by zero on profile faces or extreme pouting,
    which causes the lower face to collapse or the mouth to stretch unnaturally.

    Args:
        lmk: Array of shape (N, 203, 2) or (1, 203, 2) containing landmarks.

    Returns:
        np.ndarray: The clamped lip ratios to safely feed the retargeting network.
    """
    # 1. Calculate horizontal width of the mouth (Denominator)
    # Indices based on 203-point format: Left mouth corner (48), Right mouth corner (66)
    mouth_width = np.linalg.norm(lmk[:, 48] - lmk[:, 66], axis=1, keepdims=True)

    # SAFEGUARD A: Clamp minimum width to prevent MAR explosion.
    # A mouth width below 8.0 pixels implies extreme profile, heavy occlusion, or severe pout.
    min_mouth_width = 8.0
    mouth_width = np.maximum(mouth_width, min_mouth_width)

    # 2. Calculate vertical height of the lips (Numerator)
    # Indices: Upper lip center (90), Lower lip center (102)
    lip_height = np.linalg.norm(lmk[:, 90] - lmk[:, 102], axis=1, keepdims=True)

    # 3. Calculate Base Ratio
    mar = lip_height / mouth_width

    # SAFEGUARD B: Hard clamp the final ratio to biologically plausible limits (0.0 to 0.85).
    # Normal human mouth aspect ratio rarely exceeds 0.75 even when shouting or yawning.
    max_safe_mar = 0.85
    mar = np.clip(mar, 0.0, max_safe_mar)

    return mar


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/camera.py
def headpose_pred_to_degree(pred):
    """
    Converts a headpose prediction to degrees.

    Args:
        pred: (bs, 66) or (bs, 1) or other shapes.
              (bs, 66) indicates a classification task with 66 classes.

    Returns:
        degree: Converted headpose prediction in degrees if input shape is (bs, 66).
                Otherwise, returns the input as is.
    """
    # Check if pred is (bs, 66)
    if pred.ndim > 1 and pred.shape[1] == 66:
        # Get the device of the input tensor
        device = pred.device

        # Create an index tensor [0, 1, 2, ..., 65]
        idx_tensor = [idx for idx in range(0, 66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)

        # Apply softmax to get probabilities over the 66 classes
        pred = torch.nn.functional.softmax(pred, dim=1)

        # Calculate the weighted sum (degree estimation)
        # This step computes the sum of probabilities * indices and scales the result
        degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 97.5

        return degree

    # If input is not (bs, 66), return it unchanged
    return pred


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/camera.py
def get_rotation_matrix(pitch_, yaw_, roll_) -> torch.Tensor:
    """
    Calculates the 3D rotation matrix from Euler angles (in degrees).
    OPTIMIZED: Direct device allocation eliminates CPU-to-GPU PCIe transfer overhead.
    """

    # Helper to securely cast to tensor on the right device if needed
    def _to_tensor(val):
        if isinstance(val, torch.Tensor):
            return val
        return torch.tensor(val, dtype=torch.float32)

    pitch = _to_tensor(pitch_)
    yaw = _to_tensor(yaw_)
    roll = _to_tensor(roll_)

    # Convert degrees to radians
    pitch = pitch / 180.0 * torch.pi
    yaw = yaw / 180.0 * torch.pi
    roll = roll / 180.0 * torch.pi

    device = pitch.device
    dtype = pitch.dtype

    # Ensure 2D shape (bs, 1)
    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    bs = pitch.shape[0]

    # FIXED: Allocate directly on VRAM. No .to(device) chaining.
    ones = torch.ones([bs, 1], device=device, dtype=dtype)
    zeros = torch.zeros([bs, 1], device=device, dtype=dtype)

    # Rotation matrix around x-axis (pitch)
    rot_x = torch.cat(
        [
            ones,
            zeros,
            zeros,
            zeros,
            torch.cos(pitch),
            -torch.sin(pitch),
            zeros,
            torch.sin(pitch),
            torch.cos(pitch),
        ],
        dim=1,
    ).reshape([bs, 3, 3])

    # Rotation matrix around y-axis (yaw)
    rot_y = torch.cat(
        [
            torch.cos(yaw),
            zeros,
            torch.sin(yaw),
            zeros,
            ones,
            zeros,
            -torch.sin(yaw),
            zeros,
            torch.cos(yaw),
        ],
        dim=1,
    ).reshape([bs, 3, 3])

    # Rotation matrix around z-axis (roll)
    rot_z = torch.cat(
        [
            torch.cos(roll),
            -torch.sin(roll),
            zeros,
            torch.sin(roll),
            torch.cos(roll),
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=1,
    ).reshape([bs, 3, 3])

    # Combine the rotations (z, y, x)
    rot = rot_z @ rot_y @ rot_x

    # Return the transposed rotation matrix
    return rot.permute(0, 2, 1)


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def transform_keypoint(kp_info: dict):
    """
    Transforms the keypoints using the pose (pitch, yaw, roll), shift (translation), and expression deformation.

    Args:
        kp_info: A dictionary containing the following keys:
            - 'kp': Tensor of shape (bs, k, 3), the keypoints.
            - 'pitch', 'yaw', 'roll': Tensors representing head pose angles.
            - 't': Translation vector (bs, 3).
            - 'exp': Expression deformation vector (bs, k, 3).
            - 'scale': Scaling factor.

    Returns:
        kp_transformed: Transformed keypoints of shape (bs, k, 3).
    """
    kp = kp_info["kp"]  # (bs, k, 3) keypoints
    pitch, yaw, roll = kp_info["pitch"], kp_info["yaw"], kp_info["roll"]
    t, exp = kp_info["t"], kp_info["exp"]
    scale = kp_info["scale"]

    # Convert pose angles to degrees
    pitch = headpose_pred_to_degree(pitch)
    yaw = headpose_pred_to_degree(yaw)
    roll = headpose_pred_to_degree(roll)

    # Determine the batch size
    bs = kp.shape[0]

    # Determine the number of keypoints
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # For shape (bs, num_kpx3)
    else:
        num_kp = kp.shape[1]  # For shape (bs, num_kp, 3)

    # Get the rotation matrix based on pitch, yaw, and roll
    rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

    # Apply the transformation: s * (R * x_c,s + exp) + t
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # Apply scaling

    # Apply translation, only to x and y (ignore z)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]

    return kp_transformed


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_eyeball_direction(
    eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs
):
    if eyeball_direction_x > 0:
        delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
        delta_new[0, 15, 0] += eyeball_direction_x * 0.001
    else:
        delta_new[0, 11, 0] += eyeball_direction_x * 0.001
        delta_new[0, 15, 0] += eyeball_direction_x * 0.0007

    delta_new[0, 11, 1] += eyeball_direction_y * -0.0005
    delta_new[0, 15, 1] += eyeball_direction_y * -0.0006
    blink = -eyeball_direction_y / 2.0

    delta_new[0, 11, 1] += blink * -0.0005
    delta_new[0, 13, 1] += blink * 0.0004
    delta_new[0, 15, 1] += blink * -0.0006
    delta_new[0, 16, 1] += blink * 0.00025

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_smile(smile, delta_new, **kwargs):
    delta_new[0, 20, 1] += smile * -0.01
    delta_new[0, 14, 1] += smile * -0.02
    delta_new[0, 17, 1] += smile * 0.0065
    delta_new[0, 17, 2] += smile * 0.003
    delta_new[0, 13, 1] += smile * -0.00275
    delta_new[0, 16, 1] += smile * -0.00275
    delta_new[0, 3, 1] += smile * -0.0035
    delta_new[0, 7, 1] += smile * -0.0035

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_wink(wink, delta_new, **kwargs):
    delta_new[0, 11, 1] += wink * 0.001
    delta_new[0, 13, 1] += wink * -0.0003
    delta_new[0, 17, 0] += wink * 0.0003
    delta_new[0, 17, 1] += wink * 0.0003
    delta_new[0, 3, 1] += wink * -0.0003

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_eyebrow(eyebrow, delta_new, **kwargs):
    if eyebrow > 0:
        delta_new[0, 1, 1] += eyebrow * 0.001
        delta_new[0, 2, 1] += eyebrow * -0.001
    else:
        delta_new[0, 1, 0] += eyebrow * -0.001
        delta_new[0, 2, 0] += eyebrow * 0.001
        delta_new[0, 1, 1] += eyebrow * 0.0003
        delta_new[0, 2, 1] += eyebrow * -0.0003

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_zero(lip_variation_zero, delta_new, **kwargs):
    delta_new[0, 19, 0] += lip_variation_zero

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_one(lip_variation_one, delta_new, **kwargs):
    delta_new[0, 14, 1] += lip_variation_one * 0.001
    delta_new[0, 3, 1] += lip_variation_one * -0.0005
    delta_new[0, 7, 1] += lip_variation_one * -0.0005
    delta_new[0, 17, 2] += lip_variation_one * -0.0005

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_two(lip_variation_two, delta_new, **kwargs):
    delta_new[0, 20, 2] += lip_variation_two * -0.001
    delta_new[0, 20, 1] += lip_variation_two * -0.001
    delta_new[0, 14, 1] += lip_variation_two * -0.001

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_lip_variation_three(lip_variation_three, delta_new, **kwargs):
    delta_new[0, 19, 1] += lip_variation_three * 0.001
    delta_new[0, 19, 2] += lip_variation_three * 0.0001
    delta_new[0, 17, 1] += lip_variation_three * -0.0001

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_mov_x(mov_x, delta_new, **kwargs):
    delta_new[0, 5, 0] += mov_x

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/gradio_pipeline.py
@torch.no_grad()
def update_delta_new_mov_y(mov_y, delta_new, **kwargs):
    delta_new[0, 5, 1] += mov_y

    return delta_new


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def calc_combined_eye_ratio(c_d_eyes_i, source_lmk, device="cuda"):
    """
    FIX: Averages the driving eye ratios to prevent left-eye dominance bias.
    Ensures symmetric baseline retargeting for the LivePortrait generator.
    """
    c_s_eyes = calc_eye_close_ratio(source_lmk[None])
    c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(device)

    # Safely extract left and right eye ratios
    left_eye_ratio = c_d_eyes_i[0][0]
    right_eye_ratio = c_d_eyes_i[0][1] if len(c_d_eyes_i[0]) > 1 else left_eye_ratio

    # Calculate the mean to harmonize the retargeting delta
    mean_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

    c_d_eyes_i_numpy_m = np.array([mean_eye_ratio], dtype=np.float32)

    # Minimum 0.08 clamp to prevent eyelid mesh overlapping (Z-fighting)
    c_d_eyes_i_numpy = np.array([max(c_d_eyes_i_numpy_m[0], 0.08)], dtype=np.float32)
    c_d_eyes_i_tensor = torch.from_numpy(c_d_eyes_i_numpy).reshape(1, 1).to(device)

    # Format: [c_s,eyes, c_d,eyes,i]
    combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)

    return combined_eye_ratio_tensor


def calc_independent_eye_ratios(
    c_d_eyes_i: np.ndarray, source_lmk: np.ndarray, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates separate retargeting tensors for Left and Right eyes.
    Enables the 'Split-Eye' asymmetric blink (winking) trick.
    """
    c_s_eyes = calc_eye_close_ratio(source_lmk[None])
    c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(device)

    # Safely extract left and right eye ratios
    left_eye_ratio = float(c_d_eyes_i[0][0])
    right_eye_ratio = (
        float(c_d_eyes_i[0][1]) if len(c_d_eyes_i[0]) > 1 else left_eye_ratio
    )

    # Clamp to 0.08 minimum to avoid 3D mesh overlap (Z-fighting on eyelids)
    left_val = np.array([max(left_eye_ratio, 0.08)], dtype=np.float32)
    right_val = np.array([max(right_eye_ratio, 0.08)], dtype=np.float32)

    left_tensor = torch.from_numpy(left_val).reshape(1, 1).to(device)
    right_tensor = torch.from_numpy(right_val).reshape(1, 1).to(device)

    # Format: [c_s_left, c_s_right, target_specific_eye]
    ratio_left_target = torch.cat([c_s_eyes_tensor, left_tensor], dim=1)
    ratio_right_target = torch.cat([c_s_eyes_tensor, right_tensor], dim=1)

    return ratio_left_target, ratio_right_target


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/live_portrait_wrapper.py
def calc_combined_lip_ratio(c_d_lip_i, source_lmk, device="cuda"):
    c_s_lip = calc_lip_close_ratio(source_lmk[None])
    c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(device)
    # c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).to(device).reshape(1, 1) # 1x1
    c_d_lip_i_numpy = np.array(
        [c_d_lip_i[0]], dtype=np.float32
    )  # Assicurati che sia un array NumPy
    c_d_lip_i_tensor = torch.from_numpy(c_d_lip_i_numpy).to(device).reshape(1, 1)  # 1x1
    # [c_s,lip, c_d,lip,i]
    combined_lip_ratio_tensor = torch.cat(
        [c_s_lip_tensor, c_d_lip_i_tensor], dim=1
    )  # 1x2

    return combined_lip_ratio_tensor


# imported from https://github.com/KwaiVGI/LivePortrait/blob/main/src/utils/helper.py
def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, "batch size must be equal"

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def apply_laplace_filter(img: torch.Tensor) -> torch.Tensor:
    """
    Applies a Laplacian edge-detection filter to a grayscale image tensor.
    OPTIMIZED: Caches the Laplace tensor strictly per-device to prevent allocating
    a new 3x3 matrix in VRAM on every single frame.
    """
    device_str = str(img.device)

    with _laplace_kernel_lock:
        laplace_kernel = _laplace_kernel_cache.get(device_str)
        if laplace_kernel is None:
            laplace_kernel = (
                torch.tensor(
                    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                    dtype=torch.float32,
                    device=img.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            _laplace_kernel_cache[device_str] = laplace_kernel

    # Erweitere den Graustufen-Bild-Tensor für Faltung (Batches und Kanäle hinzufügen)
    img_expanded = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) für die Faltung

    # Faltung mit dem Laplace-Kernel durchführen
    laplacian = torch.nn.functional.conv2d(img_expanded, laplace_kernel, padding=1)

    return laplacian.squeeze(0).squeeze(0)  # (H, W)


def _map_jpeg_quality(
    base_q: int,
    face_scale: float,
    *,
    gamma: float = 0.6,
    strength: float = 1.0,
    q_min: int = 5,
    q_max: int = 95,
) -> int:
    """
    base_q:  UI-Wert [1..100]
    face_scale: tform.scale = (Original-Gesichtsbreite)/512
                <1 -> Gesicht war kleiner (upsampled); >1 -> größer (downsampled)
    gamma:    Krümmung (0.4..0.8 fühlt sich gut an)
    strength: 0..1, wie stark die Skalierung vom base_q abweicht
    q_min/q_max: harte Klammern (vermeidet Extremwerte)
    """
    # numerisch stabil klammern
    f = max(0.25, min(4.0, float(face_scale)))  # weicher Arbeitsbereich
    # norm >1, wenn f<1; norm <1, wenn f>1
    norm = (1.0 / f) ** gamma

    # weiche Mischung: bei strength=0 → base_q, bei 1 → volle Skalierung
    q_scaled = base_q * norm
    q_eff = (1.0 - strength) * base_q + strength * q_scaled

    # sauber runden + clampen
    q_eff = int(round(max(q_min, min(q_max, q_eff))))
    return q_eff


def jpegBlur(img: torch.Tensor, q: int) -> torch.Tensor:
    """
    img: [3,H,W], float32 (0..255) ODER (0..1) ODER uint8 (0..255)
    q:   1..100
    """
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError("Image must have shape [3,H,W].")
    device = img.device

    # in uint8 konvertieren (sauber klammern + runden)
    if img.dtype == torch.float32 or img.dtype == torch.float64:
        # Erkennen, ob 0..1 -> auf 0..255 hochskalieren
        mx = float(img.max().detach().item())
        if mx <= 1.0 + 1e-6:
            img_uint8 = (img.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
        else:
            img_uint8 = img.clamp(0, 255).round().to(torch.uint8).cpu()
    elif img.dtype == torch.uint8:
        img_uint8 = img.cpu()
    else:
        img_uint8 = img.to(torch.float32).clamp(0, 255).round().to(torch.uint8).cpu()

    # Encoding erwartet [H,W,C] auf CPU
    img_hwc = img_uint8.contiguous()  # .permute(1, 2, 0).contiguous()
    buf = torchvision.io.encode_jpeg(img_hwc, quality=int(q))
    out_hwc = torchvision.io.decode_jpeg(buf)  # [H,W,C], uint8
    out_chw = out_hwc  # .permute(2, 0, 1).contiguous()       # [3,H,W]

    return out_chw.to(device=device, dtype=torch.float32)


def histogram_matching(
    source_image: torch.Tensor, target_image: torch.Tensor, diffslider: float
) -> torch.Tensor:
    """
    Exact Histogram Matching using a GPU-accelerated Look-Up Table (LUT).
    OPTIMIZED: Operates in Linear RGB space to prevent unnatural contrast crushing,
    and uses 1024 bins to preserve shadow details during linear quantization.
    """
    device = source_image.device
    dtype = source_image.dtype
    NUM_BINS = 1024

    # 1. Normalize [0, 1] and decode to Linear RGB for physically accurate matching
    s_img_linear = gamma_decode_srgb_to_linear_rgb(
        source_image.to(device, dtype=torch.float32) / 255.0
    )
    t_img_linear = gamma_decode_srgb_to_linear_rgb(
        target_image.to(device, dtype=torch.float32) / 255.0
    )

    # 2. Create validity masks to exclude padding (pure black) independently
    s_valid = s_img_linear.sum(dim=0) > 0.01
    t_valid = t_img_linear.sum(dim=0) > 0.01

    matched_target_linear = torch.empty_like(t_img_linear)

    for channel in range(3):
        source_channel = s_img_linear[channel]
        target_channel = t_img_linear[channel]

        # Extract ONLY valid pixels for accurate statistics
        src_valid_vals = source_channel[s_valid]
        tgt_valid_vals = target_channel[t_valid]

        # Failsafe if image is entirely black (rare detection failure)
        if src_valid_vals.numel() == 0:
            src_valid_vals = torch.tensor([0.0], device=device)
        if tgt_valid_vals.numel() == 0:
            tgt_valid_vals = torch.tensor([0.0], device=device)

        # Clamp and scale to integer bins
        src_bins = torch.clamp(src_valid_vals * (NUM_BINS - 1), 0, NUM_BINS - 1).to(
            torch.int64
        )
        tgt_bins = torch.clamp(tgt_valid_vals * (NUM_BINS - 1), 0, NUM_BINS - 1).to(
            torch.int64
        )

        # Compute histograms safely ignoring the background
        src_hist = torch.bincount(src_bins, minlength=NUM_BINS).float()
        tgt_hist = torch.bincount(tgt_bins, minlength=NUM_BINS).float()

        src_hist += 1e-6
        tgt_hist += 1e-6

        # Compute CDFs
        src_cdf = torch.cumsum(src_hist, dim=0)
        src_cdf = src_cdf / src_cdf[-1]

        tgt_cdf = torch.cumsum(tgt_hist, dim=0)
        tgt_cdf = tgt_cdf / tgt_cdf[-1]

        # Build the Look-Up Table (LUT)
        indices = torch.searchsorted(src_cdf, tgt_cdf, right=False)
        indices = torch.clamp(indices, 0, NUM_BINS - 1)
        lut = indices.float() / float(NUM_BINS - 1)

        # Apply the LUT mapping to the entire target channel
        tgt_all_bins = torch.clamp(target_channel * (NUM_BINS - 1), 0, NUM_BINS - 1).to(
            torch.int64
        )
        matched_target_linear[channel] = lut[tgt_all_bins]

        # Restore pure black padding safely to prevent grey halos at warp edges
        matched_target_linear[channel][~t_valid] = target_channel[~t_valid]

    # 3. Blend result in Linear space based on the diffslider [0, 100]
    alpha = diffslider / 100.0
    final_image_linear = (1.0 - alpha) * t_img_linear + alpha * matched_target_linear

    # 4. Re-encode to sRGB, scale back to [0, 255], clamp, and match original dtype
    final_image_srgb = gamma_encode_linear_rgb_to_srgb(final_image_linear)
    final_image_out = torch.clamp(final_image_srgb * 255.0, 0.0, 255.0)

    return final_image_out.to(dtype=dtype)


def histogram_matching_withmask(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: torch.Tensor,
    diffslider: float,
) -> torch.Tensor:
    """
    Exact Histogram Matching using a GPU-accelerated LUT, restricted to a mask.
    OPTIMIZED: Operates in Linear RGB space to prevent contrast crushing, with 1024 bins
    to maintain shadow precision. Validates against pure black padding.
    """
    device = source_image.device
    dtype = source_image.dtype
    NUM_BINS = 1024

    # 1. Format user mask safely for boolean indexing
    base_mask = mask.to(device=device, dtype=torch.bool)
    if base_mask.dim() == 3 and base_mask.size(0) == 1:
        base_mask = base_mask.squeeze(0)

    # 2. Normalize [0, 1] and decode to Linear RGB for physically accurate matching
    s_img_linear = gamma_decode_srgb_to_linear_rgb(
        source_image.to(device, dtype=torch.float32) / 255.0
    )
    t_img_linear = gamma_decode_srgb_to_linear_rgb(
        target_image.to(device, dtype=torch.float32) / 255.0
    )

    # 3. Exclude pure black padding pixels independently
    s_valid = s_img_linear.sum(dim=0) > 0.01
    t_valid = t_img_linear.sum(dim=0) > 0.01

    # Calculate statistics strictly where mask overlaps with actual image data
    s_mask_valid = base_mask & s_valid
    t_mask_valid = base_mask & t_valid

    # Early exit to prevent NaN division and save VRAM if no face is detected
    if not s_mask_valid.any() or not t_mask_valid.any():
        return target_image.to(dtype=dtype)

    # Clone target to preserve unmasked background areas perfectly
    matched_target_linear = t_img_linear.clone()

    for channel in range(3):
        source_channel = s_img_linear[channel]
        target_channel = t_img_linear[channel]

        # Extract ONLY strictly valid masked pixels for statistical calculation
        src_masked_vals = source_channel[s_mask_valid]
        tgt_masked_vals = target_channel[t_mask_valid]

        # Scale to integer bins
        src_bins = torch.clamp(src_masked_vals * (NUM_BINS - 1), 0, NUM_BINS - 1).to(
            torch.int64
        )
        tgt_bins = torch.clamp(tgt_masked_vals * (NUM_BINS - 1), 0, NUM_BINS - 1).to(
            torch.int64
        )

        # Compute histograms safely
        src_hist = torch.bincount(src_bins, minlength=NUM_BINS).float()
        tgt_hist = torch.bincount(tgt_bins, minlength=NUM_BINS).float()

        src_hist += 1e-6
        tgt_hist += 1e-6

        # Compute CDFs
        src_cdf = torch.cumsum(src_hist, dim=0)
        src_cdf = src_cdf / src_cdf[-1]

        tgt_cdf = torch.cumsum(tgt_hist, dim=0)
        tgt_cdf = tgt_cdf / tgt_cdf[-1]

        # Build the Look-Up Table (LUT)
        indices = torch.searchsorted(src_cdf, tgt_cdf, right=False)
        indices = torch.clamp(indices, 0, NUM_BINS - 1)
        lut = indices.float() / float(NUM_BINS - 1)

        # APPLY LUT ONLY TO VALID TARGET MASKED PIXELS
        tgt_apply_bins = torch.clamp(
            target_channel[t_mask_valid] * (NUM_BINS - 1), 0, NUM_BINS - 1
        ).to(torch.int64)
        matched_target_linear[channel][t_mask_valid] = lut[tgt_apply_bins]

    # 4. Blend result in Linear space based on the diffslider [0, 100]
    alpha = diffslider / 100.0
    final_image_linear = (1.0 - alpha) * t_img_linear + alpha * matched_target_linear

    # 5. Re-encode to sRGB, scale back to [0, 255], clamp, and cast to original dtype
    final_image_srgb = gamma_encode_linear_rgb_to_srgb(final_image_linear)
    final_image_out = torch.clamp(final_image_srgb * 255.0, 0.0, 255.0)

    return final_image_out.to(dtype=dtype)


def apply_reinhard_color_transfer(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    diffslider: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Unified Reinhard Statistical Color Transfer in LAB Space.
    OPTIMIZED: Safely ignores padding independently for source and target to prevent temporal flickering.
    """
    device = source_image.device
    dtype = source_image.dtype

    # 1. Prepare Data & Normalize
    s_img = source_image.to(device, dtype=torch.float32) / 255.0
    t_img = target_image.to(device, dtype=torch.float32) / 255.0

    # 2. Base validity: Exclude pure black padding pixels INDEPENDENTLY
    s_valid = s_img.sum(dim=0) > 0.01
    t_valid = t_img.sum(dim=0) > 0.01

    # 3. Handle Optional Mask dynamically for both source and target
    if mask is not None:
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # [H, W]
        s_mask = (mask > 0.05) & s_valid
        t_mask = (mask > 0.05) & t_valid
    else:
        # Global mode: just use all non-black pixels independently
        s_mask = s_valid
        t_mask = t_valid

    # Fallback: if target mask is empty, return original target
    if t_mask.sum() < 100:
        return target_image.to(dtype=dtype)

    # Failsafe: Prevent division by zero if source is entirely black
    if s_mask.sum() == 0:
        s_mask = torch.ones_like(s_mask)

    # 4. Convert to LAB Space
    s_lab = rgb_to_lab(s_img, normalize=False)
    t_lab = rgb_to_lab(t_img, normalize=False)

    # 5. Calculate Stats INDEPENDENTLY on Valid Pixels Only
    s_masked = s_lab[:, s_mask]
    t_masked = t_lab[:, t_mask]

    s_mean = s_masked.mean(dim=1).view(3, 1, 1)
    s_std = s_masked.std(dim=1).view(3, 1, 1)

    t_mean = t_masked.mean(dim=1).view(3, 1, 1)
    t_std = t_masked.std(dim=1).view(3, 1, 1)

    t_std += 1e-6  # Safety against division by zero

    # 6. Apply Transfer globally to the Target
    t_lab_trans = (t_lab - t_mean) * (s_std / t_std) + s_mean

    # 7. Clamp LAB values to biological/visual limits to prevent artifacts
    t_lab_trans[0].clamp_(0.0, 100.0)  # L (Lightness)
    t_lab_trans[1].clamp_(-127.0, 127.0)  # A (Green-Red)
    t_lab_trans[2].clamp_(-127.0, 127.0)  # B (Blue-Yellow)

    # 8. Convert back to RGB and clamp
    result = lab_to_rgb(t_lab_trans, normalize=False)
    result = torch.clamp(result, 0.0, 1.0)

    # 9. Blend based on diffslider
    alpha = diffslider / 100.0
    final = (1.0 - alpha) * t_img + alpha * result

    return torch.clamp(final * 255.0, 0.0, 255.0).to(dtype=dtype)


@torch.no_grad()
def apply_adain_color_transfer(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    blend_amount: float = 100.0,
    calc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Applies statistical color transfer (AdaIN) from the target to the source in LAB space.
    OPTIMIZED: Statistical masks decoupled to prevent VRAM graph leaks and temporal flickering.
    """
    device = source.device
    dtype = source.dtype
    eps = 1e-6

    # 1. Normalize inputs to [0, 1] as float32 for precision math
    src_norm = source.to(device=device, dtype=torch.float32) / 255.0
    tgt_norm = target.to(device=device, dtype=torch.float32) / 255.0

    # Format blending mask securely
    mask_f32 = mask.to(device=device, dtype=torch.float32)
    if mask_f32.dim() == 2:
        mask_f32 = mask_f32.unsqueeze(0)

    # 2. Prepare the Calculation Mask (Statistics Core)
    if calc_mask is None:
        core_mask = -F.max_pool2d(
            -mask_f32.unsqueeze(0), kernel_size=31, stride=1, padding=15
        ).squeeze(0)

        calc_mask_ready = (core_mask > 0.8).float()

        if torch.sum(calc_mask_ready) < 100:
            calc_mask_ready = mask_f32.clone()
    else:
        calc_mask_ready = calc_mask.to(device=device, dtype=torch.float32)
        if calc_mask_ready.dim() == 2:
            calc_mask_ready = calc_mask_ready.unsqueeze(0)

    # Filter out pure black padding pixels INDEPENDENTLY for Source and Target
    s_non_black = (src_norm.sum(dim=0, keepdim=True) > 0.01).float()
    t_non_black = (tgt_norm.sum(dim=0, keepdim=True) > 0.01).float()

    s_calc_mask = calc_mask_ready * s_non_black
    t_calc_mask = calc_mask_ready * t_non_black

    s_calc_mask_sum = torch.sum(s_calc_mask, dim=(1, 2), keepdim=True) + eps
    t_calc_mask_sum = torch.sum(t_calc_mask, dim=(1, 2), keepdim=True) + eps

    # 3. Convert to LAB Space
    src_lab = rgb_to_lab(src_norm, normalize=False)
    tgt_lab = rgb_to_lab(tgt_norm, normalize=False)

    # 4. Compute Means and Variances strictly inside the INDEPENDENT core masks
    src_mean = (
        torch.sum(src_lab * s_calc_mask, dim=(1, 2), keepdim=True) / s_calc_mask_sum
    )
    tgt_mean = (
        torch.sum(tgt_lab * t_calc_mask, dim=(1, 2), keepdim=True) / t_calc_mask_sum
    )

    src_var = (
        torch.sum(s_calc_mask * (src_lab - src_mean) ** 2, dim=(1, 2), keepdim=True)
        / s_calc_mask_sum
    )
    tgt_var = (
        torch.sum(t_calc_mask * (tgt_lab - tgt_mean) ** 2, dim=(1, 2), keepdim=True)
        / t_calc_mask_sum
    )

    src_std = torch.sqrt(src_var + eps)
    tgt_std = torch.sqrt(tgt_var + eps)

    # 5. Apply AdaIN transformation globally in LAB space
    src_matched_lab = ((src_lab - src_mean) / src_std) * tgt_std + tgt_mean

    # 6. Clamp LAB values to valid biological/visual ranges to prevent artifacts
    src_matched_lab[0].clamp_(0.0, 100.0)  # L (Lightness)
    src_matched_lab[1].clamp_(-127.0, 127.0)  # A (Red/Green)
    src_matched_lab[2].clamp_(-127.0, 127.0)  # B (Blue/Yellow)

    # 7. Convert back to RGB
    src_matched_rgb = lab_to_rgb(src_matched_lab, normalize=False) * 255.0
    src_matched_rgb = torch.clamp(src_matched_rgb, 0.0, 255.0)

    # 8. Blend based on user amount and the SOFT APPLICATION MASK
    alpha = blend_amount / 100.0
    source_f32 = source.to(device=device, dtype=torch.float32)

    result = (src_matched_rgb * mask_f32 * alpha) + (
        source_f32 * (1.0 - (mask_f32 * alpha))
    )

    return torch.clamp(result, 0.0, 255.0).to(dtype=dtype)


def transform_t(img, center, output_size, scale, rotation):
    device = img.device
    dtype = img.dtype
    img = pad_image_by_size(img, output_size)

    scale_ratio = scale
    rot_rad = torch.tensor(rotation * torch.pi / 180.0, device=device, dtype=dtype)
    cos_theta = torch.cos(rot_rad) * scale_ratio
    sin_theta = torch.sin(rot_rad) * scale_ratio

    a = cos_theta
    b = sin_theta
    c = -sin_theta
    d = cos_theta

    cx, cy = center
    cx = cx * scale_ratio
    cy = cy * scale_ratio
    tx = -cx
    ty = -cy
    tx_final = output_size / 2
    ty_final = output_size / 2
    tx_total = tx_final + a * tx + b * ty
    ty_total = ty_final + c * tx + d * ty

    M = torch.tensor([[a, b, tx_total], [c, d, ty_total]], dtype=dtype, device=device)
    img_batch = img.unsqueeze(0)
    grid = torch.nn.functional.affine_grid(
        M.unsqueeze(0), img_batch.size(), align_corners=False
    )
    cropped_batch = torch.nn.functional.grid_sample(
        img_batch, grid, align_corners=False, mode="bilinear"
    )
    cropped = cropped_batch.squeeze(0)

    return cropped, M


def trans_points2d_t(pts, M):
    if pts.dim() != 2 or pts.size(1) != 2:
        raise ValueError("pts deve essere un tensore 2D con dimensione (N, 2)")
    ones_column = torch.ones((pts.size(0), 1), dtype=pts.dtype, device=pts.device)
    homogeneous_pts = torch.cat([pts, ones_column], dim=1)
    transformed_pts = homogeneous_pts @ M.T

    return transformed_pts[:, :2]


def invertAffineTransform_t(M):
    if M.dim() == 2 and M.size() == (2, 3):
        M_H = torch.cat(
            [M, torch.tensor([[0, 0, 1]], device=M.device, dtype=M.dtype)], dim=0
        )
        IM_H = torch.inverse(M_H)
        IM = IM_H[:2, :]
    else:
        raise ValueError("M deve essere di dimensione (2, 3)")

    return IM


def get_face_orientation_t(face_size: float, lmk: torch.Tensor) -> torch.Tensor:
    global arcface_src_cuda
    assert lmk.shape == (5, 2), "lmk deve essere un tensore di forma (5, 2)"
    device = lmk.device

    # Ensure reference template is on the correct device
    if device != arcface_src_cuda.device:
        arcface_src_cuda = arcface_src_cuda.to(device)

    src_scaled = (face_size / 112.0) * arcface_src_cuda  # Shape: (5, 2)

    # Compute centroids
    centroid_lmk = lmk.mean(dim=0, keepdim=True)  # Shape: (1, 2)
    centroid_src = src_scaled.mean(dim=0, keepdim=True)  # Shape: (1, 2)

    # Center landmarks
    lmk_centered = lmk - centroid_lmk  # Shape: (5, 2)
    src_centered = src_scaled - centroid_src  # Shape: (5, 2)

    # Calculate scaling factor securely
    norm_lmk = torch.norm(lmk_centered, dim=1).pow(2).sum().unsqueeze(0)
    norm_src = torch.norm(src_centered, dim=1).pow(2).sum().unsqueeze(0)

    # Avoid zero division risk
    scale = torch.sqrt(norm_src / (norm_lmk + 1e-8))

    lmk_scaled = lmk_centered * scale

    # Calculate covariance matrix
    covariance = torch.mm(src_centered.t(), lmk_scaled)  # Shape: (2, 2)

    # FIXED: Use modern torch.linalg.svd. Note that Vh is returned already transposed.
    U, S, Vh = torch.linalg.svd(covariance)

    # Calculate rotation matrix
    R = torch.mm(U, Vh)  # Shape: (2, 2)

    # Check determinant to ensure a valid rotation (no reflections)
    det = torch.linalg.det(R)
    if det < 0:
        U[:, -1] *= -1.0
        R = torch.mm(U, Vh)

    # Calculate angle in radians and convert to degrees
    angle_rad = torch.atan2(R[1, 0], R[0, 0])  # Shape (1,)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg


def calculate_lmk_rotation_translation(source_landmarks, target_landmarks):
    """
    Calcola la matrice di rotazione e traslazione tra due insiemi di punti di landmark.

    :param source_landmarks: numpy array di dimensione (203, 2) o (203, 3) - Landmark sorgente.
    :param target_landmarks: numpy array di dimensione (203, 2) o (203, 3) - Landmark target.
    :return: (R, t) - Matrice di rotazione e vettore di traslazione.
    """

    # Step 1: Calcola i centri di massa di ciascun insieme di punti
    source_center = np.mean(source_landmarks, axis=0)
    target_center = np.mean(target_landmarks, axis=0)

    # Step 2: Centra i punti rispetto al centro di massa
    centered_source = source_landmarks - source_center
    centered_target = target_landmarks - target_center

    # Step 3: Calcola la matrice di covarianza
    covariance_matrix = np.dot(centered_source.T, centered_target)

    # Step 4: Applica la decomposizione SVD
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Step 5: Calcola la matrice di rotazione
    R = np.dot(Vt.T, U.T)

    # Step 6: Correggi eventuali riflessioni (per mantenere la det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Step 7: Calcola la traslazione
    t = target_center - np.dot(source_center, R)

    return R, t


def rotation_matrix_to_angle(R):
    """
    Converti la matrice di rotazione 2x2 in un angolo (in gradi).
    """
    # Calcola l'angolo di rotazione in radianti
    angle_rad = np.arctan2(R[1, 0], R[0, 0])  # Usando la matrice di rotazione
    # Converti l'angolo in gradi
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def get_matrix_lmk_rotation_translation(R, t):
    """
    Combina la matrice di rotazione e il vettore di traslazione in un'istanza SimilarityTransform.

    :param R: Matrice di rotazione 2x2.
    :param t: Vettore di traslazione 2x1.
    :return: Istanza di SimilarityTransform con rotazione e traslazione.
    """
    # Estrai l'angolo di rotazione dalla matrice di rotazione
    rotation_angle = rotation_matrix_to_angle(R)

    # Crea un'istanza di SimilarityTransform usando l'angolo di rotazione e la traslazione
    t = trans.SimilarityTransform(rotation=np.radians(rotation_angle), translation=t)

    M = t.params[0:2]

    return M


def calc_face_yaw_pitch(kps):
    """
    Estimates the Yaw (Left/Right) and Pitch (Up/Down) angles based on 5 landmarks.
    Returns values roughly in degrees.
    """
    # kps order: 0:Left Eye, 1:Right Eye, 2:Nose, 3:Left Mouth, 4:Right Mouth
    le = kps[0]
    re = kps[1]
    n = kps[2]
    lm = kps[3]
    rm = kps[4]

    # --- YAW CALCULATION (Left - Right) ---
    # Compare distance from Nose to Left Eye vs Nose to Right Eye
    # We use horizontal distances (x-coordinates)
    dist_n_le = n[0] - le[0]
    dist_re_n = re[0] - n[0]

    # Avoid division by zero
    diff = dist_n_le - dist_re_n
    sum_dist = dist_n_le + dist_re_n + 1e-6

    # Ratio roughly between -1 and 1. Scaled to approx degrees.
    # Positive = Looking Right (User's Right, Image Left)
    # Negative = Looking Left
    yaw = (diff / sum_dist) * 90.0

    # --- PITCH CALCULATION (Up - Down) ---
    # Compare center of eyes to nose vs nose to center of mouth
    eye_center_y = (le[1] + re[1]) / 2
    mouth_center_y = (lm[1] + rm[1]) / 2

    dist_eye_nose = n[1] - eye_center_y
    dist_nose_mouth = mouth_center_y - n[1]

    # Typical ratio is around 0.8 to 1.0 for frontal
    # If eye-nose distance decreases -> Looking Up
    # If eye-nose distance increases -> Looking Down

    ratio = dist_eye_nose / (dist_nose_mouth + 1e-6)

    # Heuristic adjustment to center around 0 degrees
    # This is a rough approximation suitable for masking triggers
    pitch = (1.0 - ratio) * 90.0

    return yaw, pitch
