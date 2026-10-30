"""Tests for toolbox.vision and toolbox.automation.

The pure-stdlib helpers run in the core suite. Everything that needs
the ``vision`` or ``automation`` optional extras is guarded with
``importorskip`` so the core CI job (core deps only) skips it, while
the dedicated extras-smoke CI job — which installs both extras —
exercises it for real. Asserted behaviors were probe-verified against
the real implementations first.
"""

import importlib

import pytest

from toolbox.vision.data.folder_loader import get_image_paths_and_labels


def _make_tree(base, layout):
    for class_name, filenames in layout.items():
        class_dir = base / class_name
        class_dir.mkdir(parents=True)
        for name in filenames:
            (class_dir / name).write_bytes(b"x")


# --- vision: folder_loader (pure stdlib, always runs) -------------------


def test_folder_loader_class_map_is_sorted_and_labels_match_paths(tmp_path):
    _make_tree(tmp_path, {"dogs": ["d.jpeg"], "cats": ["a.jpg", "b.PNG"]})
    paths, labels, class_to_label = get_image_paths_and_labels(tmp_path)
    assert class_to_label == {"cats": 0, "dogs": 1}
    assert len(paths) == len(labels) == 3
    # Iteration order of the class folders is not guaranteed; what must
    # hold is that every path carries its own folder's label.
    for path, label in zip(paths, labels, strict=True):
        folder = path.rsplit("/", 2)[-2]
        assert class_to_label[folder] == label


def test_folder_loader_filters_extensions_case_insensitively(tmp_path):
    _make_tree(
        tmp_path,
        {"cats": ["a.jpg", "b.JPEG", "c.PnG", "notes.txt", "noext"]},
    )
    paths, labels, _ = get_image_paths_and_labels(tmp_path)
    assert sorted(p.rsplit("/", 1)[-1] for p in paths) == [
        "a.jpg",
        "b.JPEG",
        "c.PnG",
    ]
    assert labels == [0, 0, 0]


def test_folder_loader_ignores_files_in_the_base_dir(tmp_path):
    _make_tree(tmp_path, {"cats": ["a.jpg"]})
    (tmp_path / "stray.jpg").write_bytes(b"x")
    paths, _, _ = get_image_paths_and_labels(tmp_path)
    assert len(paths) == 1
    assert paths[0].endswith("cats/a.jpg")


def test_folder_loader_empty_base_dir(tmp_path):
    assert get_image_paths_and_labels(tmp_path) == ([], [], {})


def test_folder_loader_empty_class_dir_still_gets_a_label(tmp_path):
    _make_tree(tmp_path, {"cats": [], "dogs": ["d.jpg"]})
    paths, labels, class_to_label = get_image_paths_and_labels(tmp_path)
    assert class_to_label == {"cats": 0, "dogs": 1}
    assert len(paths) == 1
    assert labels == [1]


# --- vision: heavy submodules (vision extra) ----------------------------


def test_image_loader_len_getitem_and_collate():
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image

    from toolbox.vision.data.image_loader import ImageLoader

    paths = []
    # tmp_path-free: build the images in a TemporaryDirectory so this
    # test needs no pytest fixture beyond the imports above.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        for i, color in enumerate(["red", "blue"]):
            path = Path(tmp) / f"img{i}.png"
            Image.new("RGB", (8, 8), color=color).save(path)
            paths.append(str(path))

        loader = ImageLoader(paths, [3, 5])
        assert len(loader) == 2
        image, label = loader[1]
        assert image.size == (8, 8)
        assert label == 5

        tensor_loader = ImageLoader(
            paths, [3, 5], transform=lambda img: torch.zeros(3, 4, 4)
        )
        batch = [tensor_loader[0], tensor_loader[1]]
        images, labels = ImageLoader.collate_fn(batch)
        assert images.shape == (2, 3, 4, 4)
        assert labels.tolist() == [3, 5]


def test_train_test_transformations_shapes():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from toolbox.vision.transformations.train_test_transform import (
        train_test_image_transformations,
    )

    _, test_transform = train_test_image_transformations(resize=16)
    tensor = test_transform(Image.new("RGB", (24, 10)))
    assert tensor.shape == (3, 16, 16)


def test_efficient_net_rejects_unknown_model_size():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from toolbox.vision.models.efficient_net_pretrained import EfficientNetModel

    with pytest.raises(ValueError, match="model_size"):
        EfficientNetModel(model_size="b9", pretrained=False)


def test_efficient_net_b0_forward_with_explicit_num_classes():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch

    from toolbox.vision.models.efficient_net_pretrained import EfficientNetModel

    model = EfficientNetModel(
        model_size="b0", num_classes=3, pretrained=False, device=torch.device("cpu")
    )
    # predict() does not switch modes; in train mode BatchNorm rejects a
    # single 1x1 spatial value per channel, so evaluate explicitly.
    model.model.eval()
    probs = model.predict(torch.zeros(2, 3, 32, 32))
    assert probs.shape == (2, 3)
    assert probs.sum(dim=1).tolist() == pytest.approx([1.0, 1.0], abs=1e-5)


# --- automation: random_folder_opener (stdlib in-file, extra-gated) -----


@pytest.fixture()
def subfolder_module():
    # The module itself is stdlib-only, but importing it runs
    # toolbox.automation's __init__, which needs loguru (automation extra).
    pytest.importorskip("loguru")
    return importlib.import_module("toolbox.automation.random_folder_opener")


def test_get_all_subfolders_is_recursive(subfolder_module, tmp_path):
    (tmp_path / "a" / "a1").mkdir(parents=True)
    (tmp_path / "b").mkdir()
    found = subfolder_module.get_all_subfolders(str(tmp_path))
    assert sorted(found) == sorted(
        [
            str(tmp_path / "a"),
            str(tmp_path / "a" / "a1"),
            str(tmp_path / "b"),
        ]
    )


def test_get_all_subfolders_skips_hidden_underscore_and_venv(
    subfolder_module, tmp_path
):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "_private").mkdir()
    (tmp_path / "venv" / "inner").mkdir(parents=True)
    (tmp_path / "keep").mkdir()
    found = subfolder_module.get_all_subfolders(str(tmp_path))
    assert found == [str(tmp_path / "keep")]


def test_get_all_subfolders_empty_dir(subfolder_module, tmp_path):
    assert subfolder_module.get_all_subfolders(str(tmp_path)) == []


# --- automation: heavy submodules (automation extra) --------------------


def test_arxiv_titlecleaner():
    pytest.importorskip("loguru")
    pytest.importorskip("bs4")
    pytest.importorskip("tkinter")
    module = importlib.import_module("toolbox.automation.ArXivSearcher")
    assert module.titlecleaner("  A paper\nabout  things! ") == "A paperabout things "


@pytest.mark.parametrize(
    ("module_name", "deps"),
    [
        ("toolbox.automation", ["loguru"]),
        ("toolbox.automation.SeleniumInitializer", ["selenium", "webdriver_manager"]),
        ("toolbox.automation.LinkedInHelper.scraper", ["selenium"]),
        ("toolbox.automation.cli", ["click", "loguru"]),
    ],
)
def test_automation_modules_import(module_name, deps):
    for dep in deps:
        pytest.importorskip(dep)
    importlib.import_module(module_name)
