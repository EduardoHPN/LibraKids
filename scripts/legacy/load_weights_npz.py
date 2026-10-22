from __future__ import annotations

import sys

import numpy as np
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Dense, Dropout, LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from libras.config import PATHS
from libras.utils.io import load_json


def build_compatible_model(num_classes: int, num_frames: int, keypoint_dim: int) -> Sequential:
    """Reproduz EXATAMENTE a arquitetura do Colab."""
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True, activation='tanh',
                 recurrent_dropout=0.2),
            input_shape=(num_frames, keypoint_dim),
        ),
        BatchNormalization(),
        Dropout(0.5),

        Bidirectional(LSTM(128, return_sequences=False, activation='tanh',
                           recurrent_dropout=0.2)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main() -> None:
    print("=" * 55)
    print("  Carregando pesos via .npz")
    print("=" * 55)

    models_dir = PATHS["models"]

    # Carrega metadata
    meta = load_json(models_dir / "metadata.json")
    num_classes = meta["num_classes"]
    # Formato de entrada vem do metadata.json do Colab (1662, 438 ou 405).
    num_frames = meta.get("num_frames", 30)
    keypoint_dim = meta.get("keypoint_dim", 1662)
    print(f"Classes: {num_classes} | entrada: ({num_frames}, {keypoint_dim})")

    # Reconstrói arquitetura
    print("\n🔧 Reconstruindo arquitetura local...")
    model = build_compatible_model(num_classes, num_frames, keypoint_dim)

    # Build para inicializar os pesos
    dummy = np.zeros((1, num_frames, keypoint_dim), dtype=np.float32)
    _ = model(dummy)

    # Lista layers locais que têm pesos
    print("\nLayers locais (com pesos):")
    local_layers_with_weights = []
    for i, layer in enumerate(model.layers):
        if layer.get_weights():
            shapes = [w.shape for w in layer.get_weights()]
            print(f"  [{i}] {layer.name}: {shapes}")
            local_layers_with_weights.append((i, layer))

    # Carrega .npz
    npz_path = models_dir / "weights_raw.npz"
    if not npz_path.exists():
        print(f"\n❌ Arquivo não encontrado: {npz_path}")
        sys.exit(1)

    print(f"\n📥 Carregando {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    # Reconstrói lista de layers do Colab a partir da metadata
    layer_order_arr = data['__layer_order__']
    colab_layers = []
    for item in layer_order_arr:
        layer_id, n_weights = str(item).split('|')
        n_weights = int(n_weights)
        colab_layers.append((layer_id, n_weights))

    print(f"\nLayers do Colab (com pesos):")
    for layer_id, n in colab_layers:
        print(f"  {layer_id}: {n} arrays")

    # Verifica que o número bate
    if len(colab_layers) != len(local_layers_with_weights):
        print(f"\n❌ Mismatch! Colab={len(colab_layers)} vs Local={len(local_layers_with_weights)}")
        sys.exit(1)

    # Aplica pesos
    print(f"\n🔌 Aplicando pesos...")
    for (local_idx, local_layer), (colab_id, n_weights) in zip(
        local_layers_with_weights, colab_layers
    ):
        # Reconstrói lista de pesos desse layer
        colab_weights = [data[f"{colab_id}_w{j}"] for j in range(n_weights)]

        local_shapes = [w.shape for w in local_layer.get_weights()]
        colab_shapes = [w.shape for w in colab_weights]

        if local_shapes != colab_shapes:
            print(f"  ❌ {local_layer.name}: shape mismatch!")
            print(f"      Local: {local_shapes}")
            print(f"      Colab: {colab_shapes}")
            sys.exit(1)

        local_layer.set_weights(colab_weights)
        print(f"  ✓ {local_layer.name}")

    # Salva
    output = models_dir / "best_model.keras"
    model.save(str(output))
    print(f"\n✅ Modelo salvo: {output}")
    print("   Agora rode: python scripts/04_evaluate.py")


if __name__ == "__main__":
    main()