"""Deterministic synthetic shape images for demos and tests.

``DataZoo/image`` stood empty from 2026-10-27 (when the broken
``make_test_image`` export was dropped) until this module landed,
and the vision and fourier code hand-rolled its test images inline.
This is the one canonical generator: simple labeled shapes —
circles, rectangles, vertical bars — drawn with numpy only, from a
single seeded generator consumed in a fixed order, so the same
seed reproduces the same dataset byte-for-byte.

Two surfaces:

- :func:`generate_dataset` returns the images in memory as a uint8
  array of shape ``(n, size, size, 3)`` with integer labels.
- :func:`write_dataset` writes the same dataset as a class-folder
  tree — ``root/<class>/<class>_000.png`` … — in exactly the layout
  :func:`toolbox.vision.data.folder_loader.get_image_paths_and_labels`
  consumes (one folder per class, ``.png`` files, class labels from
  the sorted folder names; the label order here is the same sorted
  order, so in-memory and on-disk labels agree). Pillow — a
  ``vision``-extra dependency — is imported lazily inside
  :func:`write_dataset`, so generating arrays needs only numpy, a
  core dependency.

The label signal is the shape alone: foreground/background levels
are re-drawn per image and Gaussian noise (``noise``, as a fraction
of the 0–255 scale) is added to every image from the same seeded
stream, so neither brightness nor a fixed noise pattern separates
the classes.
"""

from pathlib import Path

import numpy as np

#: The shape classes, in the sorted order used for labels.
SHAPES = ("bars", "circle", "rectangle")

_MIN_SIZE = 16


def _draw_shape(rng, shape, size):
    """One noiseless grayscale image (``size`` x ``size`` float array).

    Foreground/background levels and the shape's geometry are all
    drawn from ``rng`` in a fixed order per shape kind.
    """
    background = float(rng.integers(0, 61))
    foreground = float(rng.integers(175, 256))
    yy, xx = np.mgrid[0:size, 0:size]
    if shape == "circle":
        center = rng.uniform(0.32, 0.68, size=2) * size
        radius = rng.uniform(0.16, 0.32) * size
        mask = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= radius**2
    elif shape == "rectangle":
        width, height = rng.uniform(0.30, 0.55, size=2) * size
        x0 = rng.uniform(0, size - width)
        y0 = rng.uniform(0, size - height)
        mask = (xx >= x0) & (xx < x0 + width) & (yy >= y0) & (yy < y0 + height)
    elif shape == "bars":
        mask = np.zeros((size, size), dtype=bool)
        n_bars = int(rng.integers(2, 5))
        edges = np.sort(rng.uniform(0, size, size=2 * n_bars))
        for i in range(n_bars):
            mask[:, int(edges[2 * i]) : int(edges[2 * i + 1])] = True
    else:  # pragma: no cover - generate_dataset validates the names
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    return np.where(mask, foreground, background)


def generate_dataset(
    n_per_class=8,
    *,
    classes=SHAPES,
    size=64,
    seed=0,
    noise=0.05,
):
    """Generate a labeled synthetic dataset in memory.

    Args:
        n_per_class: Images per class (at least 1).
        classes: Shape names from :data:`SHAPES`; label ``i`` is
            ``classes`` sorted, so the labels match the class-folder
            mapping :func:`write_dataset` produces on disk.
        size: Image width/height in pixels (at least 16).
        seed: Seed for the single generator behind every draw —
            the same seed reproduces the identical dataset.
        noise: Gaussian noise level as a fraction of the 0–255
            scale (0 = clean two-level images, 1 = overwhelming).

    Returns:
        tuple: ``(images, labels, class_names)`` — a uint8 array of
        shape ``(n_per_class * len(classes), size, size, 3)``, an
        int64 label array, and the class names in label order.
        Images are ordered by class, then by index within the class.
    """
    classes = tuple(classes)
    if not classes:
        raise ValueError("classes must name at least one shape")
    unknown = [name for name in classes if name not in SHAPES]
    if unknown:
        raise ValueError(f"unknown shapes {unknown}; choose from {SHAPES}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"classes must not repeat: {classes}")
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be at least 1, got {n_per_class}")
    if size < _MIN_SIZE:
        raise ValueError(f"size must be at least {_MIN_SIZE}, got {size}")
    if not 0.0 <= noise <= 1.0:
        raise ValueError(f"noise must lie in [0, 1], got {noise}")

    class_names = tuple(sorted(classes))
    rng = np.random.default_rng(seed)
    images = np.empty((n_per_class * len(class_names), size, size, 3), dtype=np.uint8)
    labels = np.empty(n_per_class * len(class_names), dtype=np.int64)
    row = 0
    for label, shape in enumerate(class_names):
        for _ in range(n_per_class):
            gray = _draw_shape(rng, shape, size)
            # The noise field is drawn even when noise == 0 so the
            # stream (and therefore the geometry) does not depend
            # on the noise level.
            gray = gray + rng.normal(0.0, noise * 255.0, size=(size, size))
            images[row] = np.repeat(
                np.clip(gray, 0, 255).round().astype(np.uint8)[:, :, None],
                3,
                axis=2,
            )
            labels[row] = label
            row += 1
    return images, labels, class_names


def write_dataset(
    root,
    n_per_class=8,
    *,
    classes=SHAPES,
    size=64,
    seed=0,
    noise=0.05,
):
    """Write a generated dataset as a class-folder tree of PNGs.

    The layout is the one ``toolbox.vision.data.folder_loader``
    reads: ``root/<class>/<class>_<index:03d>.png``, one folder per
    class. Needs Pillow (the ``vision`` extra); generating arrays
    with :func:`generate_dataset` does not.

    Args:
        root: Destination directory (created, with parents).
        n_per_class, classes, size, seed, noise: As in
            :func:`generate_dataset`.

    Returns:
        list of str: The written file paths, ordered by class
        (sorted) then index — the same order as the in-memory
        dataset for the same arguments.
    """
    try:
        from PIL import Image
    except ImportError as err:  # pragma: no cover - env-dependent
        raise ImportError(
            "write_dataset needs Pillow (the ToolBox 'vision' extra)"
        ) from err
    images, _, class_names = generate_dataset(
        n_per_class, classes=classes, size=size, seed=seed, noise=noise
    )
    root = Path(root)
    paths = []
    row = 0
    for shape in class_names:
        class_dir = root / shape
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(n_per_class):
            path = class_dir / f"{shape}_{index:03d}.png"
            Image.fromarray(images[row]).save(path)
            paths.append(str(path))
            row += 1
    return paths
