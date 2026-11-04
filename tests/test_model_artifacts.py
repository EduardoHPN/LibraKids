"""Testes dos artefatos de modelo que entram na imagem Docker."""
from __future__ import annotations

import warnings

import pytest

from libras.config import PATHS

joblib = pytest.importorskip("joblib")
pytest.importorskip("sklearn")

_LETTER_CLASSES = list("ABCDEFGILMNOPQRSTUVWY")


def test_label_encoder_loads_without_warning_and_keeps_class_order():
    """A ordem das classes define a saída do modelo; mudá-la troca as letras preditas."""
    path = PATHS["models_letters"] / "label_encoder.pkl"
    # Warning vira erro: pega o InconsistentVersionWarning de um pkl salvo
    # com outra versão do scikit-learn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encoder = joblib.load(path)

    assert list(encoder.classes_) == _LETTER_CLASSES
    assert list(encoder.inverse_transform(range(len(_LETTER_CLASSES)))) == _LETTER_CLASSES
