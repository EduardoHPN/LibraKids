"""
Passo 13 — Monta o conjunto de treino (.npy) a partir do cache de landmarks.

Lê os landmarks brutos extraídos por scripts/11_extract_landmarks.py e grava
uma sequência por vídeo no formato de data/processed_custom/ — o mesmo que
scripts/03_train.py consome:

    <out>/<Classe>/<video>.npy   (num_frames, keypoint_dim) float32

Os vetores saem de `libras.features.keypoints`, as mesmas funções que a API
aplica aos landmarks do app/navegador. Treinando com landmarks do MediaPipe
Tasks, o modelo vê em produção exatamente o que viu no treino (ver
docs/VALIDACAO_MEDIAPIPE_TASKS.md).

Formatos:
  padrão       438 dimensões (pose x/y/z/vis + face reduzida + mãos)
  --no-pose-z  405 dimensões (sem o z da pose — mais robusto entre runtimes)

Uso:
    python scripts/13_build_training_set.py --cache data/landmarks \
        --out data/processed_custom_tasks
    python scripts/13_build_training_set.py --cache data/landmarks \
        --out data/processed_custom_tasks_sem_z --no-pose-z

Depois, no Colab: python scripts/03_train.py --config configs/tasks.yaml
(ou configs/tasks_sem_z.yaml).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from libras.features.keypoints import (
    NO_POSE_Z_TOTAL_DIM, TOTAL_DIM, keypoints_from_arrays, keypoints_from_arrays_no_pose_z,
)
from libras.features.landmark_cache import load_landmark_cache, sequence_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta .npy de treino a partir do cache de landmarks")
    parser.add_argument("--cache", type=Path, required=True,
                        help="Pasta de saída do passo 11 (contém tasks/ e/ou legacy/)")
    parser.add_argument("--backend", choices=["tasks", "legacy"], default="tasks",
                        help="De qual extração montar (padrão: tasks — o que os clientes usam)")
    parser.add_argument("--out", type=Path, required=True, help="Pasta de saída dos .npy")
    parser.add_argument("--no-pose-z", action="store_true",
                        help=f"Remove o z da pose ({TOTAL_DIM} → {NO_POSE_Z_TOTAL_DIM} dimensões)")
    parser.add_argument("--num-frames", type=int, default=30)
    args = parser.parse_args()

    src = args.cache / args.backend
    files = sorted(src.glob("*/*.npz"))
    if not files:
        raise SystemExit(f"Nenhum .npz em {src} — rode antes scripts/11_extract_landmarks.py")
    if args.backend == "legacy":
        print("Atenção: montando a partir do Holistic legado — os clientes usam o MediaPipe Tasks.")

    builder = keypoints_from_arrays_no_pose_z if args.no_pose_z else keypoints_from_arrays
    dim = NO_POSE_Z_TOTAL_DIM if args.no_pose_z else TOTAL_DIM

    per_class: Counter[str] = Counter()
    versions: set[str] = set()
    for f in tqdm(files, desc=f"Montando {dim} dim"):
        d = load_landmark_cache(f)
        versions.add(str(d.get("mediapipe_version", "nao registrada no cache")))
        seq = sequence_features(d, builder, args.num_frames).astype(np.float32)
        assert seq.shape == (args.num_frames, dim), f"{f}: shape {seq.shape}"

        target = args.out / f.parent.name / f"{f.stem}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, seq)
        per_class[f.parent.name] += 1

    info = {
        "source_backend": args.backend,
        "mediapipe_versions": sorted(versions),
        "keypoint_dim": dim,
        "pose_z": not args.no_pose_z,
        "num_frames": args.num_frames,
        "videos": sum(per_class.values()),
        "per_class": dict(sorted(per_class.items())),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n{info['videos']} sequências ({dim} dim) em {args.out}")
    print(f"   MediaPipe: {', '.join(info['mediapipe_versions'])}")


if __name__ == "__main__":
    main()
