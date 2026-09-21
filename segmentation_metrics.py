import torch


def confusion_matrix(prediction, target, num_classes):
    """Return a rows=target, columns=prediction confusion matrix."""
    prediction = prediction.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = ((target >= 0) & (target < num_classes) &
             (prediction >= 0) & (prediction < num_classes))
    encoded = target[valid] * num_classes + prediction[valid]
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion(confusion):
    """Calculate per-class and aggregate IoU/Dice metrics."""
    confusion = confusion.to(dtype=torch.float64)
    true_positive = confusion.diag()
    target_pixels = confusion.sum(dim=1)
    predicted_pixels = confusion.sum(dim=0)
    union = target_pixels + predicted_pixels - true_positive
    dice_denominator = target_pixels + predicted_pixels

    iou = torch.full_like(true_positive, float("nan"))
    dice = torch.full_like(true_positive, float("nan"))
    iou[union > 0] = true_positive[union > 0] / union[union > 0]
    dice[dice_denominator > 0] = (
        2.0 * true_positive[dice_denominator > 0] / dice_denominator[dice_denominator > 0]
    )
    total = confusion.sum()
    pixel_accuracy = true_positive.sum() / total if total > 0 else torch.tensor(float("nan"))
    valid_iou = ~torch.isnan(iou)
    valid_dice = ~torch.isnan(dice)
    foreground_iou = iou[1:]
    valid_foreground_iou = ~torch.isnan(foreground_iou)
    return {
        "iou": iou,
        "dice": dice,
        "miou": iou[valid_iou].mean() if valid_iou.any() else torch.tensor(float("nan")),
        "foreground_miou": (
            foreground_iou[valid_foreground_iou].mean()
            if valid_foreground_iou.any() else torch.tensor(float("nan"))
        ),
        "mean_dice": dice[valid_dice].mean() if valid_dice.any() else torch.tensor(float("nan")),
        "pixel_accuracy": pixel_accuracy,
    }
