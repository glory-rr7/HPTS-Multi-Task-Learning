import unittest

import torch

from segmentation_metrics import confusion_matrix, metrics_from_confusion


class SegmentationMetricsTest(unittest.TestCase):
    def test_perfect_four_class_prediction(self):
        target = torch.tensor([[[0, 1], [2, 3]]])
        confusion = confusion_matrix(target.clone(), target, num_classes=4)
        metrics = metrics_from_confusion(confusion)
        self.assertTrue(torch.equal(confusion, torch.eye(4, dtype=torch.long)))
        self.assertTrue(torch.allclose(metrics["iou"], torch.ones(4, dtype=torch.float64)))
        self.assertTrue(torch.allclose(metrics["dice"], torch.ones(4, dtype=torch.float64)))
        self.assertEqual(metrics["foreground_miou"].item(), 1.0)

    def test_known_binary_confusion_values(self):
        metrics = metrics_from_confusion(torch.tensor([[1, 1], [1, 1]]))
        self.assertTrue(torch.allclose(
            metrics["iou"], torch.tensor([1 / 3, 1 / 3], dtype=torch.float64)
        ))
        self.assertTrue(torch.allclose(
            metrics["dice"], torch.tensor([0.5, 0.5], dtype=torch.float64)
        ))
        self.assertEqual(metrics["pixel_accuracy"].item(), 0.5)

    def test_invalid_targets_are_excluded(self):
        confusion = confusion_matrix(
            torch.tensor([0, 1, 0]), torch.tensor([0, 1, 255]), num_classes=2
        )
        self.assertTrue(torch.equal(confusion, torch.eye(2, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
