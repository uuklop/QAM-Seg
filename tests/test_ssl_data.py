"""Dual-batch scaffolding. Synthetic tensors only; isic_loader is not used."""

import os
import tempfile
import unittest

import numpy as np

from qamseg.data import (
    UNLABELED_TODO,
    build_dual_loaders,
    build_ssl_datasets,
    iter_dual_batches,
    load_unlabeled_array,
)


class DualLoaderTests(unittest.TestCase):
    def test_synthetic_batches_have_labeled_and_unlabeled_tensors(self):
        labeled, unlabeled = build_ssl_datasets(length=4, size=16, seed=0)
        labeled_loader, unlabeled_loader = build_dual_loaders(
            labeled, unlabeled, batch_size=2, shuffle=False
        )
        batches = list(iter_dual_batches(labeled_loader, unlabeled_loader))
        self.assertEqual(len(batches), 2)
        images_l, masks_l, images_u = batches[0]
        self.assertEqual(tuple(images_l.shape), (2, 3, 16, 16))
        self.assertEqual(tuple(masks_l.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(images_u.shape), (2, 3, 16, 16))
        self.assertGreaterEqual(float(masks_l.min()), 0.0)
        self.assertLessEqual(float(masks_l.max()), 1.0)
        self.assertTrue(set(masks_l.unique().tolist()).issubset({0.0, 1.0}))

    def test_missing_unlabeled_npy_names_the_todo(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_unlabeled_array("/tmp/qamseg-missing-unlabeled.npy")
        message = str(ctx.exception)
        self.assertIn("TODO", message)
        self.assertIn("does not download", message)
        self.assertIn("ISIC", UNLABELED_TODO)

    def test_local_npy_roundtrip_without_a_dataset_download(self):
        array = np.zeros((2, 8, 8, 3), dtype=np.float32)
        array[0, 2:5, 2:5, :] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unlabeled.npy")
            np.save(path, array)
            images = load_unlabeled_array(path)
        self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
        self.assertAlmostEqual(float(images[0, :, 3, 3].mean()), 1.0)
        self.assertAlmostEqual(float(images[0, :, 0, 0].mean()), 0.0)

    def test_module_does_not_expose_isic_loader(self):
        import qamseg.data as data

        self.assertFalse(hasattr(data, "isic_loader"))


if __name__ == "__main__":
    unittest.main()
