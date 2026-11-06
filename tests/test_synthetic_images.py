"""Tests for toolbox.DataZoo.image.synthetic.

Pure numpy — always runs. The on-disk tree tests need Pillow (a
``vision``-extra dependency, also pulled in by matplotlib) and use
``importorskip``; the ImageLoader end-to-end test additionally
needs torch/torchvision and runs in the extras-smoke CI job.
Determinism asserted here is the generator's core contract: one
seed, consumed in a fixed order, reproduces the dataset exactly —
arrays in memory and PNG bytes on disk.
"""

import numpy as np
import pytest

from toolbox.DataZoo.image import SHAPES, generate_dataset, write_dataset
from toolbox.vision.data.folder_loader import get_image_paths_and_labels

# --- in-memory generation -------------------------------------------------


def test_generate_dataset_shapes_labels_and_order():
    images, labels, class_names = generate_dataset(4, size=32, seed=1)
    assert class_names == tuple(sorted(SHAPES))
    assert images.shape == (4 * len(SHAPES), 32, 32, 3)
    assert images.dtype == np.uint8
    assert labels.dtype == np.int64
    # Ordered by class, then index: four of each label, in order.
    np.testing.assert_array_equal(labels, np.repeat(np.arange(len(SHAPES)), 4))


def test_generate_dataset_is_deterministic_per_seed():
    first = generate_dataset(3, seed=7)
    second = generate_dataset(3, seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    other = generate_dataset(3, seed=8)
    assert not np.array_equal(first[0], other[0])


def test_zero_noise_images_have_exactly_two_levels():
    images, _, _ = generate_dataset(3, seed=3, noise=0.0)
    for image in images:
        # Background + foreground only, identical across channels:
        # the shape is the only signal.
        assert len(np.unique(image)) == 2
        np.testing.assert_array_equal(image[:, :, 0], image[:, :, 2])


def test_shape_foreground_fractions_are_sane():
    images, labels, class_names = generate_dataset(6, seed=5, noise=0.0, size=64)
    for label, shape in enumerate(class_names):
        for image in images[labels == label]:
            foreground = np.mean(image[:, :, 0] == image.max())
            assert 0.03 < foreground < 0.85, (shape, foreground)


def test_noise_adds_levels_and_scales_with_the_level():
    clean, _, _ = generate_dataset(2, seed=11, noise=0.0)
    low, _, _ = generate_dataset(2, seed=11, noise=0.05)
    high, _, _ = generate_dataset(2, seed=11, noise=0.3)
    # Same seed, same geometry: the difference from clean is the
    # noise alone, and it grows with the level.
    assert len(np.unique(high[0])) > 50
    diff_low = np.abs(low[0].astype(int) - clean[0].astype(int)).mean()
    diff_high = np.abs(high[0].astype(int) - clean[0].astype(int)).mean()
    assert 0 < diff_low < diff_high


def test_classes_argument_controls_the_label_order():
    _, labels, class_names = generate_dataset(2, classes=("rectangle", "bars"), seed=0)
    assert class_names == ("bars", "rectangle")
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


def test_single_class_dataset():
    images, labels, class_names = generate_dataset(3, classes=("circle",), seed=2)
    assert class_names == ("circle",)
    assert images.shape == (3, 64, 64, 3)
    np.testing.assert_array_equal(labels, [0, 0, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"classes": ("triangle",)},
        {"classes": ()},
        {"classes": ("circle", "circle")},
        {"n_per_class": 0},
        {"size": 8},
        {"noise": -0.1},
        {"noise": 1.1},
    ],
)
def test_generate_dataset_rejects_bad_arguments(kwargs):
    n = kwargs.pop("n_per_class", 1)
    with pytest.raises(ValueError):
        generate_dataset(n, **kwargs)


# --- class-folder trees -----------------------------------------------------


def test_write_dataset_tree_matches_folder_loader(tmp_path):
    pytest.importorskip("PIL")
    written = write_dataset(tmp_path, 3, seed=4, size=32)
    assert len(written) == 3 * len(SHAPES)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(SHAPES)

    paths, labels, class_to_label = get_image_paths_and_labels(tmp_path)
    assert class_to_label == {name: i for i, name in enumerate(sorted(SHAPES))}
    assert len(paths) == len(labels) == len(written)
    # Every path carries its own folder's label and really is one of
    # the written PNGs.
    assert sorted(paths) == sorted(written)
    for path, label in zip(paths, labels, strict=True):
        folder = path.rsplit("/", 2)[-2]
        assert class_to_label[folder] == label
        assert path.endswith(".png")


def test_write_dataset_labels_match_generate_dataset(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    write_dataset(tmp_path, 2, seed=9, size=32, noise=0.0)
    images, _, class_names = generate_dataset(2, seed=9, size=32, noise=0.0)
    # The file for class c, index i holds generate_dataset's row for
    # that class and index — the disk and memory surfaces agree.
    for label, shape in enumerate(class_names):
        for index in range(2):
            path = tmp_path / shape / f"{shape}_{index:03d}.png"
            on_disk = np.asarray(Image.open(path))
            np.testing.assert_array_equal(on_disk, images[label * 2 + index])


def test_write_dataset_is_byte_identical_per_seed(tmp_path):
    pytest.importorskip("PIL")
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_dataset(first, 2, seed=13, size=32)
    write_dataset(second, 2, seed=13, size=32)
    first_files = sorted(first.rglob("*.png"))
    second_files = sorted(second.rglob("*.png"))
    assert [p.name for p in first_files] == [p.name for p in second_files]
    for a, b in zip(first_files, second_files, strict=True):
        assert a.read_bytes() == b.read_bytes()


def test_written_images_open_with_expected_size_and_mode(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    write_dataset(tmp_path, 1, seed=0, size=48)
    for path in tmp_path.rglob("*.png"):
        with Image.open(path) as image:
            assert image.size == (48, 48)
            assert image.mode == "RGB"


def test_tree_loads_end_to_end_through_image_loader(tmp_path):
    pytest.importorskip("PIL")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from toolbox.vision.data.image_loader import ImageLoader
    from toolbox.vision.transformations.train_test_transform import (
        train_test_image_transformations,
    )

    write_dataset(tmp_path, 2, seed=6, size=64)
    paths, labels, _ = get_image_paths_and_labels(tmp_path)
    _, test_transform = train_test_image_transformations(resize=32)
    dataset = ImageLoader(paths, labels, transform=test_transform)
    assert len(dataset) == 2 * len(SHAPES)
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert label == labels[0]
    batch_images, batch_labels = ImageLoader.collate_fn([dataset[0], dataset[1]])
    assert batch_images.shape == (2, 3, 32, 32)
    assert batch_labels.dtype == torch.int64
