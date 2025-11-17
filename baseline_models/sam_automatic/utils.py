import numpy as np
import torch


def dice_loss(
    predicted_mask: torch.Tensor, ground_truth_mask: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Calculate the Dice loss.

    Args:
    - predicted_mask: The predicted mask.
    - ground_truth_mask: The ground truth mask.
    - epsilon: The epsilon value.

    Returns:
    - The Dice loss.
    """
    predicted_mask = torch.sigmoid(predicted_mask)
    intersection = (predicted_mask * ground_truth_mask).sum()
    return 1 - (2.0 * intersection + epsilon) / (
        predicted_mask.sum() + ground_truth_mask.sum() + epsilon
    )


def mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> torch.Tensor:
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """

    N, H, W = mask1.shape
    M, H, W = mask2.shape

    mask1 = mask1.view(N, H * W)
    mask2 = mask2.view(M, H * W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0.0, device=mask1.device),
        intersection / union,
    )

    return ret


def get_ious(predicted_masks, gt_masks, positive_threshold=0.5):
    """
    Args:
        predicted_masks
        gt_masks
        positive_threshold = in the prediction, threshold to consider the prediction is positive
        match_iou = minimum iou to consider there is a match
    Returns:
        matrix of IoUs between predicted_masks and gt_masks

    """
    num_preds = predicted_masks.shape[0]
    num_gt = gt_masks.shape[0]

    ious = torch.zeros((num_preds, num_gt))

    for i in range(num_preds):
        for j in range(num_gt):
            ious[i, j] = mask_iou(
                (torch.sigmoid(predicted_masks[i]) > positive_threshold).type(
                    torch.float
                ),
                gt_masks[j].type(torch.float),
            )

    return ious


def get_matches(ious, match_threshold=0.2):
    """
    Args:
        ious : matrix of ious (num_preds, num_gt)
        positive_threshold = in the prediction, threshold to consider the prediction is positive
        match_iou = minimum iou to consider there is a match
    Returns:
        vector of size num_gt with the index of the matched predicted mask, or -1 if no match

    """

    ious_match = torch.max(ious, axis=0)
    matches = ious_match.indices
    values = ious_match.values
    matches[values < match_threshold] = -1
    return matches


def compute_metrics_torch(predicted_mask, ground_truth_mask):

    # Basic metrics
    TP = (predicted_mask & ground_truth_mask).sum()
    FP = (predicted_mask & ~ground_truth_mask).sum()
    TN = (~predicted_mask & ~ground_truth_mask).sum()
    FN = (~predicted_mask & ground_truth_mask).sum()

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Intersection
    intersection = torch.logical_and(predicted_mask, ground_truth_mask).sum()

    # Union
    union = torch.logical_or(predicted_mask, ground_truth_mask).sum()

    # IoU
    iou = intersection / union

    return accuracy, f1_score, iou, TP.item(), FP.item(), TN.item(), FN.item()


def compute_metrics(
    predicted_mask: np.array, ground_truth_mask: np.array
) -> tuple[float, float, float]:
    """Compute accuracy, F1 score for binary masks.

    Args:
    - predicted_mask: Predicted binary mask.
    - ground_truth_mask: Ground truth binary mask.

    Returns:
    - accuracy, f1_score, IoU
    """

    # Basic metrics
    TP = np.sum((predicted_mask == 1) & (ground_truth_mask == 1))
    FP = np.sum((predicted_mask == 1) & (ground_truth_mask == 0))
    TN = np.sum((predicted_mask == 0) & (ground_truth_mask == 0))
    FN = np.sum((predicted_mask == 0) & (ground_truth_mask == 1))

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Intersection
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()

    # Union
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()

    # IoU
    iou = intersection / union

    return accuracy, f1_score, iou
