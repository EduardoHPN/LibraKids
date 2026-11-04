"""Testes do loader de dados processados."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tqdm")

from libras.data.loader import list_classes, load_arrays, split_data  # noqa: E402


def _make_dataset(root, counts, shape=(30, 438)):
    for cls, n in counts.items():
        (root / cls).mkdir(parents=True)
        for i in range(n):
            np.save(root / cls / f"{i}.npy", np.full(shape, i, dtype=np.float32))


class TestListClasses:
    def test_filters_small_classes(self, tmp_path):
        _make_dataset(tmp_path, {"a": 3, "b": 2, "c": 5})
        assert list_classes(tmp_path) == ["a", "c"]

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_classes(tmp_path / "nope")


class TestLoadArrays:
    def test_shapes_and_one_hot(self, tmp_path):
        _make_dataset(tmp_path, {"a": 3, "b": 4})
        X, y = load_arrays(["a", "b"], tmp_path, expected_shape=(30, 438))
        assert X.shape == (7, 30, 438)
        assert y.shape == (7, 2)
        assert (y.sum(axis=1) == 1).all()

    def test_skips_wrong_shape(self, tmp_path):
        _make_dataset(tmp_path, {"a": 3})
        np.save(tmp_path / "a" / "bad.npy", np.zeros((10, 438), dtype=np.float32))
        X, _ = load_arrays(["a"], tmp_path, expected_shape=(30, 438))
        assert len(X) == 3


class TestSplitData:
    @pytest.fixture
    def split(self, tmp_path):
        _make_dataset(tmp_path, {"a": 3, "b": 10, "c": 20}, shape=(4, 8))
        classes = ["a", "b", "c"]
        X, y = load_arrays(classes, tmp_path)
        return split_data(X, y, classes), len(X)

    def test_every_class_in_every_split(self, split):
        data, _ = split
        for y in (data.y_train, data.y_val, data.y_test):
            assert set(np.argmax(y, axis=1)) == {0, 1, 2}

    def test_no_samples_lost(self, split):
        data, total = split
        assert len(data.X_train) + len(data.X_val) + len(data.X_test) == total
        assert data.num_classes == 3
