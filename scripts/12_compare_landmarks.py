"""
Passo 12 — Compara os landmarks do Holistic legado com os do MediaPipe Tasks.

Segunda metade da validação numérica do docs/API_CONTRACT.md (passo 2.2.2).
Lê o cache gerado por scripts/11_extract_landmarks.py (os mesmos vídeos pelos
dois backends) e mede:

  1. Detecção     — em quantos frames cada parte (pose, face, mãos) aparece
  2. Coordenadas  — diferença x/y nos frames em que os dois detectaram,
                    em larguras de ombro (independe da distância da câmera)
  3. Lateralidade — mão esquerda de um backend virando a direita do outro
  4. Vetores      — diferença relativa dos vetores finais (1662 e 438), montados
                    com libras.features.keypoints, a mesma função da API
  5. LSTM         — acurácia do modelo versionado com os vetores de cada backend
  6. Ablação      — troca partes do Tasks pelas do legado para achar a causa
                    de uma queda de acurácia

Uso:
    python scripts/12_compare_landmarks.py --cache <pasta do passo 11>
    python scripts/12_compare_landmarks.py --cache <pasta> --report docs/VALIDACAO_MEDIAPIPE_TASKS.md
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from libras.config import PATHS
from libras.features.keypoints import (
    FACE_SUBSET_INDICES, get_feature_builder, keypoints_from_arrays, keypoints_from_arrays_legacy,
)
from libras.features.landmark_cache import PARTS, load_landmark_cache as _load
from libras.features.landmark_cache import sequence_features as _features
from libras.utils.io import load_json

_L_SHOULDER, _R_SHOULDER = 11, 12


def _hybrid(leg: dict, tsk: dict, *, pose_z=False, hands=False, z_fit=None) -> dict:
    """Landmarks do Tasks com partes trocadas pelas do legado (para a ablação)."""
    d = {k: v.copy() for k, v in tsk.items()}
    n = min(len(leg["pose"]), len(tsk["pose"]))
    if pose_z:
        d["pose"][:n, :, 2] = leg["pose"][:n, :, 2]
    if z_fit is not None:
        d["pose"][..., 2] = z_fit[0] * d["pose"][..., 2] + z_fit[1]
    if hands:
        for h in ("left_hand", "right_hand"):
            d[h][:n] = leg[h][:n]
            d[f"{h}_present"][:n] = leg[f"{h}_present"][:n]
    return d


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _component_shares(share: np.ndarray) -> dict[str, float]:
    """Soma a fração por componente do vetor: pose (stride 4) + face/mãos (stride 3)."""
    pose_dim, hand_dim = 33 * 4, 21 * 3
    face_dim = len(share) - pose_dim - 2 * hand_dim
    pose = share[:pose_dim].reshape(-1, 4)
    face = share[pose_dim:pose_dim + face_dim].reshape(-1, 3)
    hands = share[pose_dim + face_dim:].reshape(-1, 3)
    return {
        "pose_xy": pose[:, :2].sum(), "pose_z": pose[:, 2].sum(), "pose_vis": pose[:, 3].sum(),
        "face_xy": face[:, :2].sum(), "face_z": face[:, 2].sum(),
        "hands_xy": hands[:, :2].sum(), "hands_z": hands[:, 2].sum(),
    }


def compare(cache: Path) -> tuple[str, dict]:
    pairs = []
    for f in sorted((cache / "legacy").glob("*/*.npz")):
        other = cache / "tasks" / f.parent.name / f.name
        if other.exists():
            pairs.append((f.parent.name, f.stem, _load(f), _load(other)))
    if not pairs:
        raise SystemExit(f"Nenhum vídeo extraído pelos dois backends em {cache}")

    det = {p: {"legacy": 0, "tasks": 0, "agree": 0} for p in PARTS}
    diffs: dict[str, list[np.ndarray]] = defaultdict(list)
    vis_nan = vis_total = 0
    vis_diffs: list[np.ndarray] = []
    swaps = hand_frames = 0
    total_frames = 0
    frame_mismatch = []

    for cls, name, leg, tsk in pairs:
        n = min(len(leg["pose"]), len(tsk["pose"]))
        if len(leg["pose"]) != len(tsk["pose"]):
            frame_mismatch.append(name)
        total_frames += n

        for p in PARTS:
            lp, tp = leg[f"{p}_present"][:n], tsk[f"{p}_present"][:n]
            det[p]["legacy"] += int(lp.sum())
            det[p]["tasks"] += int(tp.sum())
            det[p]["agree"] += int((lp == tp).sum())

            both = lp & tp
            if not both.any():
                continue
            # Escala: largura dos ombros do legado em cada frame (pose presente).
            sw = np.hypot(*(leg["pose"][:n, _L_SHOULDER, :2] - leg["pose"][:n, _R_SHOULDER, :2]).T)
            ok = both & leg["pose_present"][:n] & (sw > 1e-3)
            if not ok.any():
                continue
            d_xy = np.linalg.norm(leg[p][:n][ok][..., :2] - tsk[p][:n][ok][..., :2], axis=-1)
            d_xy = d_xy / sw[ok][:, None]
            if p == "face":
                diffs["face (lábios+sobrancelhas)"].append(d_xy[:, FACE_SUBSET_INDICES].ravel())
            if p == "pose":
                # 0–22: rosto, ombros, braços; 23–32: quadril e pernas (fora do quadro)
                diffs["pose (pontos 0–22)"].append(d_xy[:, :23].ravel())
                diffs["pose (pontos 23–32)"].append(d_xy[:, 23:].ravel())
            diffs[p].append(d_xy.ravel())

        both_pose = leg["pose_present"][:n] & tsk["pose_present"][:n]
        v_t = tsk["pose"][:n][both_pose][..., 3]
        vis_total += v_t.size
        vis_nan += int(np.isnan(v_t).sum())
        v_l = leg["pose"][:n][both_pose][..., 3]
        ok_v = ~np.isnan(v_t)
        vis_diffs.append(np.abs(v_l[ok_v] - v_t[ok_v]))

        # Lateralidade: a mão de um lado no legado está mais perto da mão do
        # OUTRO lado no Tasks? (compara o punho, ponto 0)
        for t in range(n):
            for side, other in (("left_hand", "right_hand"), ("right_hand", "left_hand")):
                if not leg[f"{side}_present"][t]:
                    continue
                wrist = leg[side][t, 0, :2]
                cands = {s: np.linalg.norm(tsk[s][t, 0, :2] - wrist)
                         for s in (side, other) if tsk[f"{s}_present"][t]}
                if not cands:
                    continue
                hand_frames += 1
                if min(cands, key=cands.get) == other and cands[other] < 0.1:
                    swaps += 1

    # ── Vetores finais e LSTM ────────────────────────────────
    from tensorflow.keras.models import load_model

    model = load_model(PATHS["models"] / "best_model.keras")
    classes = load_json(PATHS["models"] / "metadata.json")["classes"]
    _, num_frames, model_dim = model.input_shape
    builder = get_feature_builder(model_dim)

    vec_diff, vec_share = {}, {}
    for label, b in (("1662 (legado)", keypoints_from_arrays_legacy), ("438 (atual)", keypoints_from_arrays)):
        rel, sq = [], 0.0
        for _, _, leg, tsk in pairs:
            fl, ft = _features(leg, b, num_frames), _features(tsk, b, num_frames)
            rel.append(np.linalg.norm(ft - fl) / max(np.linalg.norm(fl), 1e-9))
            sq = sq + ((ft - fl) ** 2).sum(axis=0)
        vec_diff[label] = np.array(rel)
        vec_share[label] = sq / max(sq.sum(), 1e-12)

    X_leg = np.stack([_features(leg, builder, num_frames) for _, _, leg, _ in pairs]).astype(np.float32)
    X_tsk = np.stack([_features(tsk, builder, num_frames) for _, _, _, tsk in pairs]).astype(np.float32)
    y = np.array([classes.index(cls) for cls, *_ in pairs])
    p_leg = model.predict(X_leg, verbose=0)
    p_tsk = model.predict(X_tsk, verbose=0)
    top_leg, top_tsk = p_leg.argmax(1), p_tsk.argmax(1)
    top3 = lambda p: np.mean([y[i] in np.argsort(p[i])[-3:] for i in range(len(y))])  # noqa: E731
    true_prob_diff = np.abs(p_leg[np.arange(len(y)), y] - p_tsk[np.arange(len(y)), y])

    # ── Relatório ────────────────────────────────────────────
    L = []
    L.append(f"Vídeos comparados: **{len(pairs)}** ({total_frames} frames) — modelo LSTM com entrada "
             f"({num_frames}, {model_dim}).")
    if frame_mismatch:
        L.append(f"\nAtenção: {len(frame_mismatch)} vídeos com nº de frames diferente entre backends.")

    L.append("\n### 1. Detecção (frames em que a parte foi encontrada)\n")
    L.append("| Parte | Legado | Tasks | Mesma decisão |")
    L.append("|---|---|---|---|")
    for p in PARTS:
        L.append(f"| {p} | {_pct(det[p]['legacy'] / total_frames)} | {_pct(det[p]['tasks'] / total_frames)} "
                 f"| {_pct(det[p]['agree'] / total_frames)} |")

    L.append("\n### 2. Diferença de posição x/y (frames em que os dois detectaram)\n")
    L.append("Em larguras de ombro: 0,05 = 5% da distância entre os ombros.\n")
    L.append("| Parte | Mediana | p90 | p99 |")
    L.append("|---|---|---|---|")
    for p, arrs in diffs.items():
        a = np.concatenate(arrs)
        L.append(f"| {p} | {np.median(a):.3f} | {np.percentile(a, 90):.3f} | {np.percentile(a, 99):.3f} |")

    vd = np.concatenate(vis_diffs) if vis_diffs else np.array([np.nan])
    L.append(f"\n`visibility` da pose: ausente (NaN) em {_pct(vis_nan / max(vis_total, 1))} dos pontos do Tasks; "
             f"diferença mediana para o legado {np.nanmedian(vd):.3f} (p90 {np.nanpercentile(vd, 90):.3f}).")
    L.append(f"\nLateralidade: mão trocada (esquerda↔direita) em **{_pct(swaps / max(hand_frames, 1))}** "
             f"de {hand_frames} comparações de mão.")

    L.append("\n### 3. Vetores finais (após `normalize_sequence` para 30 frames)\n")
    L.append("Diferença relativa ‖tasks − legado‖ / ‖legado‖ por vídeo:\n")
    L.append("| Formato | Mediana | p90 | Máx |")
    L.append("|---|---|---|---|")
    for label, r in vec_diff.items():
        L.append(f"| {label} | {_pct(np.median(r))} | {_pct(np.percentile(r, 90))} | {_pct(r.max())} |")

    L.append("\nDe onde vem a diferença (fração de ‖tasks − legado‖² por componente):\n")
    L.append("| Formato | pose x/y | pose z | pose visibility | face x/y | face z | mãos x/y | mãos z |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, share in vec_share.items():
        c = _component_shares(share)
        L.append(f"| {label} | " + " | ".join(_pct(c[k]) for k in (
            "pose_xy", "pose_z", "pose_vis", "face_xy", "face_z", "hands_xy", "hands_z")) + " |")

    L.append("\n### 4. LSTM versionado\n")
    L.append("| Landmarks de | Top-1 | Top-3 |")
    L.append("|---|---|---|")
    L.append(f"| Holistic legado | {_pct(np.mean(top_leg == y))} | {_pct(top3(p_leg))} |")
    L.append(f"| MediaPipe Tasks | {_pct(np.mean(top_tsk == y))} | {_pct(top3(p_tsk))} |")
    L.append(f"\nMesma palavra no top-1 nos dois: **{_pct(np.mean(top_leg == top_tsk))}** dos vídeos. "
             f"Diferença na probabilidade da classe correta: mediana {np.median(true_prob_diff):.3f}, "
             f"p90 {np.percentile(true_prob_diff, 90):.3f}.")

    L.append("\nPor classe (top-1):\n")
    L.append("| Classe | Vídeos | Legado | Tasks |")
    L.append("|---|---|---|---|")
    for c, cname in enumerate(classes):
        m = y == c
        if m.any():
            L.append(f"| {cname} | {int(m.sum())} | {_pct(np.mean(top_leg[m] == c))} | {_pct(np.mean(top_tsk[m] == c))} |")

    # ── Ablação: quanto da diferença de acurácia cada parte explica ──
    both_pose = [(leg["pose_present"] & tsk["pose_present"]) for _, _, leg, tsk in pairs]
    z_tsk = np.concatenate([tsk["pose"][m][..., 2].ravel() for (_, _, _, tsk), m in zip(pairs, both_pose)])
    z_leg = np.concatenate([leg["pose"][m][..., 2].ravel() for (_, _, leg, _), m in zip(pairs, both_pose)])
    z_fit = np.polyfit(z_tsk, z_leg, 1)  # z do Tasks → escala do legado

    def _acc(datasets) -> tuple[float, float]:
        X = np.stack([_features(d, builder, num_frames) for d in datasets]).astype(np.float32)
        p = model.predict(X, verbose=0)
        return float(np.mean(p.argmax(1) == y)), float(top3(p))

    variants = [
        ("Holistic legado (referência)", [leg for _, _, leg, _ in pairs]),
        ("MediaPipe Tasks", [tsk for _, _, _, tsk in pairs]),
        ("Tasks, z da pose do legado", [_hybrid(l, t, pose_z=True) for _, _, l, t in pairs]),
        (f"Tasks, z da pose calibrado ({z_fit[0]:.3f}·z {z_fit[1]:+.3f})",
         [_hybrid(l, t, z_fit=z_fit) for _, _, l, t in pairs]),
        ("Tasks, mãos do legado", [_hybrid(l, t, hands=True) for _, _, l, t in pairs]),
        ("Tasks, z da pose e mãos do legado", [_hybrid(l, t, pose_z=True, hands=True) for _, _, l, t in pairs]),
        ("Legado com z da pose = 0", [_hybrid(l, l, z_fit=(0.0, 0.0)) for _, _, l, _ in pairs]),
    ]
    L.append("\n### 5. Ablação — o que explica a diferença de acurácia\n")
    L.append("Troca partes dos landmarks do Tasks pelas do legado e mede o LSTM de novo.\n")
    L.append("| Landmarks | Top-1 | Top-3 |")
    L.append("|---|---|---|")
    for label, datasets in variants:
        a1, a3 = _acc(datasets)
        L.append(f"| {label} | {_pct(a1)} | {_pct(a3)} |")

    summary = {
        "acc_legacy": float(np.mean(top_leg == y)),
        "acc_tasks": float(np.mean(top_tsk == y)),
        "agreement": float(np.mean(top_leg == top_tsk)),
    }
    return "\n".join(L), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara landmarks legado × MediaPipe Tasks")
    parser.add_argument("--cache", type=Path, required=True, help="Pasta de saída do passo 11")
    parser.add_argument("--report", type=Path, default=None, help="Salva o relatório em Markdown")
    args = parser.parse_args()

    report, summary = compare(args.cache)
    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows (cp1252) não tem "↔"
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n", encoding="utf-8")
        print(f"\nRelatório salvo em {args.report}")


if __name__ == "__main__":
    main()
