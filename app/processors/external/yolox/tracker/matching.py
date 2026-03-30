import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_cost = cost_matrix[row_ind, col_ind]
    matched_mask = matched_cost <= thresh
    if matched_mask.any():
        matches = np.stack([row_ind[matched_mask], col_ind[matched_mask]], axis=1)
    else:
        matches = np.empty((0, 2), dtype=int)
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(row_ind[matched_mask].tolist()))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(col_ind[matched_mask].tolist()))
    return matches, unmatched_a, unmatched_b


def _bbox_ious(atlbrs, btlbrs):
    atlbrs = np.asarray(atlbrs, dtype=float)
    btlbrs = np.asarray(btlbrs, dtype=float)
    if atlbrs.ndim == 1:
        atlbrs = atlbrs[np.newaxis]
    if btlbrs.ndim == 1:
        btlbrs = btlbrs[np.newaxis]

    ax1, ay1, ax2, ay2 = atlbrs[:, 0], atlbrs[:, 1], atlbrs[:, 2], atlbrs[:, 3]
    bx1, by1, bx2, by2 = btlbrs[:, 0], btlbrs[:, 1], btlbrs[:, 2], btlbrs[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union_area = a_area[:, None] + b_area[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=float)

    cost_matrix = 1 - _bbox_ious(atlbrs, btlbrs)
    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
