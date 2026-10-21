"""
Data augmentation para sequências de keypoints.

Aplica transformações realistas sobre os keypoints do MediaPipe
para multiplicar o conjunto de treino artificialmente.

Importante:
  • Augmentation é aplicada APENAS no treino, nunca em val/teste
  • As transformações preservam a semântica do sinal em Libras
  • NÃO usamos flip horizontal porque lateralidade importa em Libras
"""

from __future__ import annotations

import numpy as np

from libras.features.keypoints import FACE_DIM, HAND_DIM, POSE_DIM


def add_gaussian_noise(seq: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Adiciona ruído gaussiano aos keypoints.
    Simula pequenas imprecisões na detecção do MediaPipe.
    """
    noise = np.random.normal(0, std, seq.shape).astype(np.float32)
    return seq + noise


def time_warp(seq: np.ndarray, max_factor: float = 0.2) -> np.ndarray:
    """
    Distorce temporalmente a sequência (acelera/desacelera regiões).
    Simula execução mais rápida ou lenta do mesmo sinal.
    """
    num_frames = len(seq)
    factor = np.random.uniform(1 - max_factor, 1 + max_factor)
    new_length = max(2, int(num_frames * factor))

    # Reamostra para o novo comprimento, depois volta ao original
    indices_resample = np.linspace(0, num_frames - 1, new_length).astype(int)
    resampled = seq[indices_resample]

    indices_back = np.linspace(0, new_length - 1, num_frames).astype(int)
    return resampled[indices_back]


def spatial_jitter(seq: np.ndarray, max_shift: float = 0.02) -> np.ndarray:
    """
    Aplica deslocamento espacial pequeno em todos os keypoints.
    Simula o usuário ligeiramente fora do centro do quadro.
    """
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)

    # Estrutura: pose (x,y,z,vis) + face (x,y,z) + mão esq/dir (x,y,z cada).
    # Apenas x e y são deslocados; z e visibility ficam intactos.
    pose_shift = np.tile([shift_x, shift_y, 0.0, 0.0], POSE_DIM // 4)
    face_shift = np.tile([shift_x, shift_y, 0.0], FACE_DIM // 3)
    hand_shift = np.tile([shift_x, shift_y, 0.0], HAND_DIM // 3)
    shift_vec = np.concatenate([pose_shift, face_shift, hand_shift, hand_shift])

    return (seq + shift_vec).astype(np.float32)


def scale_keypoints(seq: np.ndarray, max_scale: float = 0.1) -> np.ndarray:
    """
    Escala suavemente os keypoints (zoom in/out leve).
    Simula usuário mais próximo ou mais distante da câmera.
    """
    scale = np.random.uniform(1 - max_scale, 1 + max_scale)
    return (seq * scale).astype(np.float32)


def augment_sequence(seq: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """
    Aplica augmentations aleatoriamente.
    Cada transformação tem probabilidade `prob` de ser aplicada.
    """
    out = seq.copy()
    if np.random.random() < prob:
        out = add_gaussian_noise(out, std=0.008)
    if np.random.random() < prob:
        out = time_warp(out, max_factor=0.15)
    if np.random.random() < prob:
        out = spatial_jitter(out, max_shift=0.015)
    if np.random.random() < prob:
        out = scale_keypoints(out, max_scale=0.08)
    return out


def augment_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    multiplier: int = 5,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica augmentation para multiplicar o conjunto de treino.

    Para cada amostra original, gera `multiplier - 1` versões aumentadas.
    O multiplicador inclui a amostra original (multiplier=5 → 1 original + 4 aug).

    Parameters
    ----------
    X_train     : (N, num_frames, keypoint_dim)
    y_train     : (N, num_classes) one-hot
    multiplier  : quantas amostras por original (mínimo 2)
    random_seed : reprodutibilidade

    Returns
    -------
    X_aug, y_aug : conjuntos aumentados
    """
    np.random.seed(random_seed)

    n_original = len(X_train)
    n_total = n_original * multiplier

    X_aug = np.empty((n_total, *X_train.shape[1:]), dtype=np.float32)
    y_aug = np.empty((n_total, *y_train.shape[1:]), dtype=y_train.dtype)

    # Coloca originais primeiro
    X_aug[:n_original] = X_train
    y_aug[:n_original] = y_train

    # Gera amostras aumentadas
    for i in range(n_original):
        for j in range(1, multiplier):
            idx = n_original * j + i
            X_aug[idx] = augment_sequence(X_train[i], prob=0.7)
            y_aug[idx] = y_train[i]

    # Embaralha
    perm = np.random.permutation(n_total)
    return X_aug[perm], y_aug[perm]