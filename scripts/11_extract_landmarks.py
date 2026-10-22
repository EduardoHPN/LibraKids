"""
Passo 11 — Extrai landmarks brutos dos vídeos, pelo Holistic legado OU pelo MediaPipe Tasks.

Primeira metade da validação numérica do docs/API_CONTRACT.md (passo 2.2.2):
o modelo foi treinado com o Holistic legado (mediapipe==0.10.9), mas o app e
o navegador usam o MediaPipe Tasks (HolisticLandmarker). Este script extrai os
landmarks dos MESMOS vídeos pelos dois caminhos; scripts/12_compare_landmarks.py
compara os resultados.

O HolisticLandmarker não existe em Python no mediapipe==0.10.9, então cada
backend roda no seu venv. Por isso este script só depende de cv2, mediapipe e
numpy (não importa o pacote `libras`):

    # Holistic legado — venv do projeto
    .venv/Scripts/python scripts/11_extract_landmarks.py --backend legacy \
        --videos <pasta raw_custom> --out <pasta cache>

    # MediaPipe Tasks — venv separado. 0.10.35: no Windows, o HolisticLandmarker
    # em Python das versões 0.10.14 a 0.10.21 aborta com "Check failed:
    # holder_ != nullptr The packet is empty" já no primeiro vídeo.
    py -3.11 -m venv .venv-tasks
    .venv-tasks/Scripts/python -m pip install mediapipe==0.10.35
    .venv-tasks/Scripts/python scripts/11_extract_landmarks.py --backend tasks \
        --model artifacts/mediapipe_tasks/holistic_landmarker.task \
        --videos <pasta raw_custom> --out <pasta cache>

O .task deve ser o MESMO do app/navegador:
https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task

Saída: <out>/<backend>/<Classe>/<video>.npz com, por frame, `pose` (T,33,4),
`face` (T,F,3), `left_hand`/`right_hand` (T,21,3) — NaN onde a parte não foi
detectada — e as máscaras `*_present`. Os frames NÃO são espelhados (igual ao
treino e ao contrato da API). É retomável: pula vídeos já extraídos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

POSE_POINTS, HAND_POINTS = 33, 21


def _array(landmarks, n_points: int, coords: int) -> np.ndarray:
    """Lista de landmarks → (n_points, coords); visibility ausente vira NaN."""
    out = np.full((n_points, coords), np.nan)
    for i, lm in enumerate(landmarks[:n_points]):
        values = [lm.x, lm.y, lm.z]
        if coords == 4:
            values.append(np.nan if lm.visibility is None else lm.visibility)
        out[i] = values
    return out


class LegacyExtractor:
    """mp.solutions.holistic — mesmos parâmetros de configs/default.yaml."""
    face_points = 468

    def __init__(self, args: argparse.Namespace) -> None:
        self._kwargs = dict(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.detection_confidence,
            min_tracking_confidence=args.tracking_confidence,
        )

    def start_video(self, fps: float) -> None:
        # Instância nova por vídeo: o tracking não vaza de um vídeo para outro
        # (igual a libras.data.preprocess.process_video).
        self._holistic = mp.solutions.holistic.Holistic(**self._kwargs)

    def end_video(self) -> None:
        self._holistic.close()

    def process(self, rgb: np.ndarray, frame_idx: int):
        rgb.flags.writeable = False
        r = self._holistic.process(rgb)
        lms = lambda part: part.landmark if part else None  # noqa: E731
        return (lms(r.pose_landmarks), lms(r.face_landmarks),
                lms(r.left_hand_landmarks), lms(r.right_hand_landmarks))


class TasksExtractor:
    """mediapipe.tasks HolisticLandmarker em modo VIDEO, opções padrão (= app/navegador)."""
    face_points = 478

    def __init__(self, args: argparse.Namespace) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self._vision = vision
        self._options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.model)),
            running_mode=vision.RunningMode.VIDEO,
        )

    def start_video(self, fps: float) -> None:
        self._landmarker = self._vision.HolisticLandmarker.create_from_options(self._options)
        self._ms_per_frame = 1000.0 / (fps or 30.0)

    def end_video(self) -> None:
        self._landmarker.close()

    def process(self, rgb: np.ndarray, frame_idx: int):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        r = self._landmarker.detect_for_video(image, int(round(frame_idx * self._ms_per_frame)))
        return (r.pose_landmarks or None, r.face_landmarks or None,
                r.left_hand_landmarks or None, r.right_hand_landmarks or None)


def extract_video(path: Path, extractor) -> dict[str, np.ndarray] | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    parts: dict[str, list] = {"pose": [], "face": [], "left_hand": [], "right_hand": []}
    shapes = {"pose": (POSE_POINTS, 4), "face": (extractor.face_points, 3),
              "left_hand": (HAND_POINTS, 3), "right_hand": (HAND_POINTS, 3)}

    extractor.start_video(fps)
    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # sem flip: frame original
            for name, lms in zip(parts, extractor.process(rgb, idx)):
                n, c = shapes[name]
                parts[name].append(_array(lms, n, c) if lms else np.full((n, c), np.nan))
            idx += 1
    finally:
        extractor.end_video()
        cap.release()

    if idx == 0:
        return None
    out = {name: np.stack(frames).astype(np.float32) for name, frames in parts.items()}
    for name in parts:
        out[f"{name}_present"] = ~np.isnan(out[name][:, :, 0]).all(axis=1)
    out["fps"] = np.float32(fps)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai landmarks brutos (legado ou Tasks)")
    parser.add_argument("--backend", choices=["legacy", "tasks"], required=True)
    parser.add_argument("--videos", type=Path, required=True, help="Pasta <Classe>/<video>.mp4")
    parser.add_argument("--out", type=Path, required=True, help="Pasta do cache .npz")
    parser.add_argument("--model", type=Path, help="holistic_landmarker.task (backend tasks)")
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-complexity", type=int, default=1)
    args = parser.parse_args()

    if args.backend == "tasks" and (args.model is None or not args.model.is_file()):
        parser.error("--model com o holistic_landmarker.task é obrigatório no backend tasks")

    extractor = LegacyExtractor(args) if args.backend == "legacy" else TasksExtractor(args)
    print(f"mediapipe {mp.__version__} — backend {args.backend}")

    videos = []
    for cls_dir in sorted(p for p in args.videos.iterdir() if p.is_dir()):
        vids = sorted(cls_dir.glob("*.mp4"))
        videos += vids[: args.limit_per_class] if args.limit_per_class else vids

    out_root = args.out / args.backend
    done = skipped = failed = 0
    start = time.time()
    for i, video in enumerate(videos, 1):
        target = out_root / video.parent.name / f"{video.stem}.npz"
        if target.exists():
            skipped += 1
            continue
        data = extract_video(video, extractor)
        if data is None:
            failed += 1
            print(f"  falhou: {video}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **data)
        done += 1
        if done % 10 == 0 or i == len(videos):
            rate = (time.time() - start) / done
            print(f"  {i}/{len(videos)} vídeos — {rate:.1f}s/vídeo")

    print(f"\nExtraídos: {done} | já existiam: {skipped} | falhas: {failed}")


if __name__ == "__main__":
    main()
