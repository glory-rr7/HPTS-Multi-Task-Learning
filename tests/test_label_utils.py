import unittest

import torch

from dataloader.label_utils import decode_segmentation_label


class LabelUtilsTest(unittest.TestCase):
    def test_grayscale_class_map_is_preserved(self):
        label = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.uint8)
        decoded = decode_segmentation_label(label)
        self.assertEqual(decoded.dtype, torch.long)
        self.assertTrue(torch.equal(decoded, label.long()))

    def test_legacy_rgb_palette_is_decoded(self):
        label = torch.tensor([
            [[255, 0], [255, 0]], [[255, 255], [0, 0]], [[255, 0], [0, 255]],
        ], dtype=torch.uint8)
        self.assertTrue(torch.equal(
            decode_segmentation_label(label), torch.tensor([[[0, 1], [2, 3]]])
        ))

    def test_signatr_rgb_palette_is_decoded(self):
        label = torch.tensor([
            [[0, 0], [255, 255]], [[0, 255], [0, 255]], [[255, 0], [0, 0]],
        ], dtype=torch.uint8)
        self.assertTrue(torch.equal(
            decode_segmentation_label(label), torch.tensor([[[0, 1], [2, 3]]])
        ))

    def test_unknown_rgb_color_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cannot identify RGB label palette"):
            decode_segmentation_label(torch.zeros((3, 1, 1), dtype=torch.uint8), "bad.png")


if __name__ == "__main__":
    unittest.main()
