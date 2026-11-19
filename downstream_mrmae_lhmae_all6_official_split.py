#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
downstream_mrmae_lhmae_hypersigma_protocol.py

기능 (HyperSIGMA + LH-MAE downstream protocol PRESERVED; MR-MAE + LH-MAE Stage2 transfer)
- IN / UP / HC / HH:
    PU-style per-class split (train_num/class, seed)
- HU:
    Houston_official.mat의 official TR / TE split 사용
- Full-scene -> padding -> tiling -> tile-batchwise train/eval
- STREAM TR-only loss (타일 배치마다 TR 픽셀만 loss 계산)
- Eval은 ALL-CPU로 통일 (device mismatch 방지)

반영사항
1) results/train log:
   - 항상 같은 파일명 사용
   - 실행 시마다 overwrite 후 append
   - results.txt
   - train_log.txt

2) weight 저장 제거 (torch.save 안 함)

3) 모델 파라미터 수(total/trainable/frozen) 출력 + txt 기록

4) downstream에서 spectral adapter를 쓸지 여부를 옵션으로 제어
   - 기본값은 spatial-only
   - --use_spectral 붙이면 spectral branch 사용

5) spatial-only pretrained ckpt 사용 가능

6) HU dataset 지원
   - HU는 official TR/TE split 사용

7) MSDNet-style visualization

8) parameter counting extension

9) spectral adapter attention visualization

10) batch=1 inference latency measurement

11) HSBA complete disable
   - --disable_hsba 사용 시 HSBA 모듈 자체를 생성하지 않음
   - no-HSBA baseline / latency 측정 가능
   - 필요하면 checkpoint에서 HSBA 관련 key를 제거한 stripped ckpt 저장 가능

12) batch=1 MACs measurement
   - thop 사용
   - 단일 tile(batch=1) 기준 forward MACs 계산

13) HSBA ckpt loading skip
   - --skip_hsba_ckpt 사용 시 HSBA 모듈은 생성하지만 checkpoint에서는 HSBA 관련 key만 제외하고 load
   - spatial encoder는 pretrained ckpt 사용, HSBA는 random init 상태로 downstream 학습 가능
"""

import os
import re
import time
import argparse
import random
from functools import partial
from typing import Tuple, Dict, Any, List, Set

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from timm.models.vision_transformer import PatchEmbed
from timm.models import register_model

from model import split_data, utils, create_graph
from mae_lite.models.models_vit import Block
from util.pos_embed import get_2d_sincos_pos_embed

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None


# =================================================
# reproducibility (deterministic)
# =================================================
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# =================================================
# parameter counting utils
# =================================================
def count_parameters(model: nn.Module, print_detail: bool = False) -> Dict[str, int]:
    total = 0
    trainable = 0
    frozen = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n

    total_buffers = 0
    for b in model.buffers():
        total_buffers += b.numel()

    if print_detail:
        for name, p in model.named_parameters():
            tag = "[trainable]" if p.requires_grad else "[frozen]"
            print(f"{tag} {name:70s} {p.numel():>12d}")

    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
        "total_buffers": int(total_buffers),
    }


def count_named_parameter_group(
    model: nn.Module,
    prefixes: List[str],
    exact_names: Set[str] = None,
    trainable_only: bool = False,
    print_detail: bool = False,
) -> Dict[str, Any]:
    if exact_names is None:
        exact_names = set()

    total = 0
    matched = []

    for name, p in model.named_parameters():
        hit = any(name.startswith(pref) for pref in prefixes) or (name in exact_names)
        if not hit:
            continue

        if trainable_only and (not p.requires_grad):
            continue

        n = p.numel()
        total += n
        matched.append((name, int(n), bool(p.requires_grad)))

    if print_detail:
        for name, n, req in matched:
            tag = "[trainable]" if req else "[frozen]"
            print(f"{tag} {name:70s} {n:>12d}")

    return {
        "total_params": int(total),
        "matched": matched,
    }


def count_spectral_parameters(model: nn.Module, print_detail: bool = False) -> Dict[str, Any]:
    spectral_prefixes = [
        "backbone.spectral_adapter.",
        "backbone.spec_patch_embed.",
        "backbone.spec_ln.",
    ]
    spectral_exact = {"backbone.spectral_alpha"}

    total_pack = count_named_parameter_group(
        model,
        prefixes=spectral_prefixes,
        exact_names=spectral_exact,
        trainable_only=False,
        print_detail=print_detail,
    )
    trainable_pack = count_named_parameter_group(
        model,
        prefixes=spectral_prefixes,
        exact_names=spectral_exact,
        trainable_only=True,
        print_detail=False,
    )

    total = int(total_pack["total_params"])
    trainable = int(trainable_pack["total_params"])
    frozen = int(total - trainable)

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "matched": total_pack["matched"],
    }


def format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def format_macs(n: float) -> str:
    n = float(n)
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}G"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return f"{n:.0f}"


def estimate_model_size_mb(model: nn.Module, dtype_bytes: int = 4, include_buffers: bool = True) -> float:
    stats = count_parameters(model, print_detail=False)
    total_elems = stats["total_params"]
    if include_buffers:
        total_elems += stats["total_buffers"]
    return (total_elems * dtype_bytes) / (1024.0 ** 2)


# =================================================
# Robust MAT helpers
# =================================================
def ensure_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape={x.shape}")

    if x.shape[2] <= 16 and x.shape[0] > 16:
        return np.transpose(x, (1, 2, 0)).astype(np.float32)

    if x.shape[0] >= 32 and x.shape[1] == x.shape[2] and x.shape[0] != x.shape[1]:
        return np.transpose(x, (1, 2, 0)).astype(np.float32)

    return x.astype(np.float32)


def _get_first_ndarray_key(mat_dict: Dict[str, Any]) -> str:
    for k, v in mat_dict.items():
        if not k.startswith("__") and isinstance(v, np.ndarray):
            return k
    raise KeyError("No valid ndarray key found in .mat")


def _load_mat_any_key(mat_path: str, preferred_keys: Tuple[str, ...]) -> np.ndarray:
    pack = sio.loadmat(mat_path)
    for k in preferred_keys:
        if k in pack and isinstance(pack[k], np.ndarray):
            return pack[k]
    k0 = _get_first_ndarray_key(pack)
    return pack[k0]


def load_dataset(data_root: str, name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      cube: [H,W,C] float32
      gt  : [H,W]   int64, labels in 0..K
      meta: dict
        - for HU:
            meta["TR"], meta["TE"]
    """
    name = name.upper()
    meta: Dict[str, Any] = {}

    if name == "IN":
        cube = _load_mat_any_key(
            os.path.join(data_root, "IN", "Indian_pines_corrected.mat"),
            preferred_keys=("data", "indian_pines_corrected", "Indian_pines_corrected"),
        )
        gt = _load_mat_any_key(
            os.path.join(data_root, "IN", "Indian_pines_gt.mat"),
            preferred_keys=("groundT", "gt", "GT", "indian_pines_gt", "Indian_pines_gt"),
        )

    elif name == "UP":
        cube = _load_mat_any_key(
            os.path.join(data_root, "UP", "PaviaU.mat"),
            preferred_keys=("paviaU", "PaviaU", "data", "input"),
        )
        gt = _load_mat_any_key(
            os.path.join(data_root, "UP", "PaviaU_gt.mat"),
            preferred_keys=("paviaU_gt", "PaviaU_gt", "gt", "GT", "groundT"),
        )

    elif name == "HC":
        cube = _load_mat_any_key(
            os.path.join(data_root, "HC", "WHU_Hi_HanChuan.mat"),
            preferred_keys=("WHU_Hi_HanChuan", "data", "input"),
        )
        gt = _load_mat_any_key(
            os.path.join(data_root, "HC", "WHU_Hi_HanChuan_gt.mat"),
            preferred_keys=("WHU_Hi_HanChuan_gt", "gt", "GT", "groundT"),
        )

    elif name == "HH":
        cube = _load_mat_any_key(
            os.path.join(data_root, "HH", "WHU_Hi_HongHu.mat"),
            preferred_keys=("WHU_Hi_HongHu", "data", "input"),
        )
        gt = _load_mat_any_key(
            os.path.join(data_root, "HH", "WHU_Hi_HongHu_gt.mat"),
            preferred_keys=("WHU_Hi_HongHu_gt", "gt", "GT", "groundT"),
        )

    elif name == "HU":
        pack = sio.loadmat(os.path.join(data_root, "HU", "Houston_official.mat"))

        if "input" in pack:
            cube = pack["input"]
        elif "HSI" in pack:
            cube = pack["HSI"]
        elif "houston2013" in pack:
            cube = pack["houston2013"]
        else:
            cube = _load_mat_any_key(
                os.path.join(data_root, "HU", "Houston_official.mat"),
                preferred_keys=("input", "HSI", "houston2013", "data", "cube"),
            )

        if "GT" in pack:
            gt = pack["GT"]
        else:
            gt = _load_mat_any_key(
                os.path.join(data_root, "HU", "Houston_official.mat"),
                preferred_keys=("GT", "gt", "groundT"),
            )

        if "TR" in pack:
            meta["TR"] = np.asarray(pack["TR"])
        if "TE" in pack:
            meta["TE"] = np.asarray(pack["TE"])
        elif "VA" in pack:
            meta["TE"] = np.asarray(pack["VA"])

    else:
        raise ValueError(f"Unknown dataset: {name} (use IN/UP/HC/HH/HU)")

    cube = ensure_hwc(np.asarray(cube))
    gt = np.asarray(gt).astype(np.int64)

    if gt.ndim == 3 and gt.shape[-1] == 1:
        gt = gt[..., 0]

    if "TR" in meta and meta["TR"].ndim == 3 and meta["TR"].shape[-1] == 1:
        meta["TR"] = meta["TR"][..., 0]
    if "TE" in meta and meta["TE"].ndim == 3 and meta["TE"].shape[-1] == 1:
        meta["TE"] = meta["TE"][..., 0]

    return cube.astype(np.float32), gt, meta


# =================================================
# EL scene loader and preprocessing (HyperSIGMA original implementation)
# =================================================
_ENVI_DTYPE = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def _parse_envi_value(v: str):
    v = v.strip()
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        if inner == "":
            return []
        parts = [p.strip() for p in inner.split(",")]
        out = []
        for p in parts:
            if re.fullmatch(r"[-+]?\d+", p):
                out.append(int(p))
            else:
                try:
                    out.append(float(p))
                except Exception:
                    out.append(p.strip('"').strip("'"))
        return out
    if re.fullmatch(r"[-+]?\d+", v):
        return int(v)
    try:
        return float(v)
    except Exception:
        return v.strip('"').strip("'")


def load_envi_hdr(hdr_path: str) -> dict:
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].upper().startswith("ENVI"):
        lines = lines[1:]

    merged = []
    buf = ""
    in_brace = False
    for ln in lines:
        if "=" in ln and not in_brace:
            if buf:
                merged.append(buf)
                buf = ""
            buf = ln
            if "{" in ln and "}" not in ln:
                in_brace = True
        else:
            buf += " " + ln
            if in_brace and "}" in ln:
                in_brace = False
                merged.append(buf)
                buf = ""
    if buf:
        merged.append(buf)

    hdr = {}
    for ln in merged:
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        hdr[k.strip().lower()] = _parse_envi_value(v.strip())

    hdr.setdefault("interleave", "bsq")
    hdr.setdefault("byte order", 0)
    hdr.setdefault("header offset", 0)
    return hdr


def load_envi_img(img_path: str, hdr: dict) -> np.ndarray:
    W = int(hdr["samples"])
    H = int(hdr["lines"])
    C = int(hdr["bands"])
    dt_code = int(hdr["data type"])
    interleave = str(hdr.get("interleave", "bsq")).lower()
    byte_order = int(hdr.get("byte order", 0))
    offset = int(hdr.get("header offset", 0))

    if dt_code not in _ENVI_DTYPE:
        raise ValueError(f"[EL][ENVI] Unsupported data type code: {dt_code}")
    dtype = _ENVI_DTYPE[dt_code]

    dt = np.dtype(dtype)
    dt = dt.newbyteorder(">" if byte_order == 1 else "<")

    n = H * W * C
    with open(img_path, "rb") as f:
        if offset > 0:
            f.seek(offset, 0)
        arr = np.fromfile(f, dtype=dt, count=n)

    if arr.size != n:
        raise RuntimeError(f"[EL][ENVI] file too small? read={arr.size}, expected={n} ({img_path})")

    if interleave == "bsq":
        cube = arr.reshape((C, H, W)).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = arr.reshape((H, C, W)).transpose(0, 2, 1)
    elif interleave == "bip":
        cube = arr.reshape((H, W, C))
    else:
        raise ValueError(f"[EL][ENVI] Unsupported interleave: {interleave}")

    return cube.astype(np.float32)


def load_gt_png(gt_path: str) -> np.ndarray:
    img = Image.open(gt_path)
    arr = np.array(img)
    if arr.ndim == 2:
        return arr.astype(np.int64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(np.uint8)
        flat = rgb.reshape(-1, 3)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        bg = np.array([0, 0, 0], dtype=np.uint8)
        bg_idx = None
        for i, c in enumerate(uniq):
            if np.all(c == bg):
                bg_idx = i
                break
        mapped = inv.reshape(rgb.shape[0], rgb.shape[1]).astype(np.int64)
        if bg_idx is not None and bg_idx != 0:
            a = mapped.copy()
            a[mapped == 0] = -999
            a[mapped == bg_idx] = 0
            a[a == -999] = bg_idx
            mapped = a
        return mapped
    raise ValueError(f"[EL] Unsupported GT.png shape: {arr.shape}")


def list_el_sample_ids(data_root: str) -> List[int]:
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[EL] data_root not found: {data_root}")
    ids = []
    for name in os.listdir(data_root):
        p = os.path.join(data_root, name)
        if os.path.isdir(p) and name.isdigit():
            ids.append(int(name))
    return sorted(ids)


def load_electrolyzers_all_scenes(data_root: str, crop_bands: bool = True, band_start: int = 50, band_end_drop: int = 40):
    sample_ids = list_el_sample_ids(data_root)
    scenes = []

    for sid in sample_ids:
        sample_dir = os.path.join(os.path.abspath(data_root), str(sid))
        hdr_path = os.path.join(sample_dir, "HSI.hdr")
        img_path = os.path.join(sample_dir, "HSI.img")
        gt_path = os.path.join(sample_dir, "GT.png")

        if not (os.path.exists(hdr_path) and os.path.exists(img_path) and os.path.exists(gt_path)):
            raise FileNotFoundError(
                f"[EL] Missing files in {sample_dir}\n"
                f"  need: HSI.hdr, HSI.img, GT.png"
            )

        hdr = load_envi_hdr(hdr_path)
        cube = load_envi_img(img_path, hdr)
        if crop_bands:
            cube = cube[:, :, band_start:cube.shape[2] - band_end_drop]
        gt = load_gt_png(gt_path).astype(np.int64)

        if gt.shape[:2] != cube.shape[:2]:
            raise RuntimeError(f"[EL] shape mismatch sid={sid}: HSI={cube.shape[:2]} GT={gt.shape}")

        scenes.append({
            "sample_id": int(sid),
            "cube": cube.astype(np.float32),
            "gt": gt.astype(np.int64),
            "hdr": hdr,
            "sample_dir": sample_dir,
        })

    return scenes


def spectral_binning_np(hsi: np.ndarray) -> np.ndarray:
    h, w, bands = hsi.shape
    if bands % 2 != 0:
        hsi = hsi[:, :, :-1]
    return (hsi[:, :, 0::2] + hsi[:, :, 1::2]) / 2.0


def normalize_hsi_vector_wise_with_mask(hsi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsi = np.asarray(hsi, dtype=np.float32)
    mask = np.asarray(mask)
    normalized_hsi = np.copy(hsi)
    h, w, b = hsi.shape
    hsi_reshaped = hsi.reshape(-1, b)
    masked_indices = np.where(mask.reshape(-1) > 0)[0]
    if masked_indices.size == 0:
        return normalized_hsi
    relevant_pixels = hsi_reshaped[masked_indices]
    for i in range(relevant_pixels.shape[0]):
        pixel_vector = relevant_pixels[i]
        min_val = np.min(pixel_vector)
        max_val = np.max(pixel_vector)
        if max_val - min_val > 0:
            hsi_reshaped[masked_indices[i]] = (relevant_pixels[i] - min_val) / (max_val - min_val)
        else:
            hsi_reshaped[masked_indices[i]] = 0
    normalized_hsi = hsi_reshaped.reshape(h, w, b)
    normalized_hsi[mask == 0] = 0
    return normalized_hsi.astype(np.float32)


def get_el_scene_split(sample_ids: List[int], test_indices_sorted_positions: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    n = len(sample_ids)
    if any((idx < 0 or idx >= n) for idx in test_indices_sorted_positions):
        raise ValueError(f"[EL] invalid test index in {test_indices_sorted_positions}; valid range is 0..{n-1}")

    test_pos = sorted(set(int(x) for x in test_indices_sorted_positions))
    train_pos = [i for i in range(n) if i not in test_pos]
    train_ids = [sample_ids[i] for i in train_pos]
    test_ids = [sample_ids[i] for i in test_pos]
    return train_pos, test_pos, train_ids, test_ids


def prepare_el_hypersigma_legacy(args):
    """Legacy HyperSIGMA+LH-MAE EL protocol, copied from user-provided vis2 code.

    Train/test split is scene-wise 44/11 (default), PCA is fitted on *labeled
    pixels of train scenes only* (unlike IN/UP/HC/HH/HU full-image PCA),
    scenes are concatenated vertically BEFORE global padding & tiling.
    """
    scenes = load_electrolyzers_all_scenes(
        data_root=args.data_root,
        crop_bands=(not args.el_disable_repo_band_crop),
        band_start=args.el_band_start,
        band_end_drop=args.el_band_end_drop,
    )
    sample_ids = [x["sample_id"] for x in scenes]
    n_scenes = len(sample_ids)
    print(f"[EL] loaded scenes={n_scenes} sample_ids(first10)={sample_ids[:10]}")

    test_positions = [int(x) for x in args.el_test_indices.split(",") if str(x).strip() != ""]
    train_pos, test_pos, train_ids, test_ids = get_el_scene_split(sample_ids, test_positions)
    print(f"[EL split] test_positions={test_pos}")
    print(f"[EL split] test_sample_ids={test_ids}")
    print(f"[EL split] train_count={len(train_ids)} test_count={len(test_ids)}")

    proc_scenes = []
    for i, sc in enumerate(scenes):
        cube = sc["cube"]
        gt = sc["gt"]
        if args.el_use_spectral_binning:
            cube = spectral_binning_np(cube)
        if args.el_use_vector_norm:
            cube = normalize_hsi_vector_wise_with_mask(cube, gt)
        proc_scenes.append({
            "sample_id": sc["sample_id"],
            "cube": cube.astype(np.float32),
            "gt": gt.astype(np.int64),
            "is_test_scene": i in set(test_pos),
        })

    cubes = [x["cube"] for x in proc_scenes]
    gts = [x["gt"] for x in proc_scenes]

    if args.pca_components > 0:
        # fit PCA using labeled pixels from train scenes only
        X_train = []
        for i in train_pos:
            cube = cubes[i]
            gt = gts[i]
            X_train.append(cube[gt > 0])
        X_train = np.concatenate(X_train, axis=0).astype(np.float32)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.pca_components, whiten=False, svd_solver='auto', random_state=args.seed)
        pca.fit(X_train)
        cubes = [pca.transform(c.reshape(-1, c.shape[-1])).reshape(c.shape[0], c.shape[1], args.pca_components).astype(np.float32) for c in cubes]
        in_chans = args.pca_components
        print(f"[EL][PCA] fit on labeled TRAIN pixels only -> in_chans={in_chans}")
    else:
        in_chans = cubes[0].shape[-1]
        print(f"[EL][PCA] disabled -> in_chans={in_chans}")

    # HyperSIGMA original: do NOT change scene widths before concatenation.
    # With variable-width scenes, np.concatenate in the original code would fail.
    # Keep that behavior as the default rather than silently changing the data.
    widths = [int(c.shape[1]) for c in cubes]
    if len(set(widths)) != 1:
        scene_widths = [(int(sc["sample_id"]), width) for sc, width in zip(proc_scenes, widths)]
        if not args.el_pad_variable_widths:
            raise ValueError(
                "[EL][HyperSIGMA original] Scene widths are different: "
                f"{scene_widths}. The uploaded HyperSIGMA code uses "
                "np.concatenate(cubes, axis=0) WITHOUT any width correction. "
                "To reproduce it exactly, use equal-width input scenes. "
                "To opt in to the previous MR-MAE right-zero-padding behavior, "
                "add --el_pad_variable_widths to BOTH model experiments."
            )
        max_w = max(widths)
        print(
            f"[EL][OPT-IN WIDTH PAD] variable widths {sorted(set(widths))}; "
            f"right-zero-pad to {max_w}. NOT identical to original HyperSIGMA."
        )
        cubes = [
            np.pad(c, ((0, 0), (0, max_w-c.shape[1]), (0, 0)), mode="constant")
            for c in cubes
        ]
        gts = [
            np.pad(g, ((0, 0), (0, max_w-g.shape[1])), mode="constant")
            for g in gts
        ]

    # Exact HyperSIGMA original scene concatenation.
    data = np.concatenate(cubes, axis=0).astype(np.float32)
    data_gt = np.concatenate(gts, axis=0).astype(np.int64)
    H0, W0, _ = data.shape
    class_num = int(np.max(data_gt))

    scene_ranges = []
    row_start = 0
    for i, gt in enumerate(gts):
        h = gt.shape[0]
        row_end = row_start + h
        scene_ranges.append((row_start, row_end))
        row_start = row_end

    train_mask_hw = np.zeros((H0, W0), dtype=np.int32)
    test_mask_hw = np.zeros((H0, W0), dtype=np.int32)
    for i, (rs, re) in enumerate(scene_ranges):
        if i in train_pos:
            train_mask_hw[rs:re, :] = data_gt[rs:re, :]
        if i in test_pos:
            test_mask_hw[rs:re, :] = data_gt[rs:re, :]

    eval_ids = np.where(test_mask_hw.reshape(-1) > 0)[0].astype(np.int64)
    test_gt = test_mask_hw.copy()
    test_samples_gt = test_mask_hw.reshape(-1)

    train_index = np.flatnonzero(train_mask_hw.reshape(-1) > 0).astype(np.int64)
    test_index = eval_ids
    if len(np.intersect1d(train_index, test_index)):
        raise RuntimeError("[EL] Train/Test labeled pixel overlap")
    print(f"[EL][HyperSIGMA protocol] concat-scenes BEFORE tiling; train pixels={len(train_index)} test pixels={len(test_index)}")
    print(f"[EL][HyperSIGMA protocol] data={data.shape}, classes={class_num}")
    return {
        "data": data,
        "data_gt": data_gt,
        "train_map_hw": train_mask_hw,
        "test_gt": test_gt,
        "test_samples_gt": test_samples_gt,
        "train_index": train_index,
        "test_index": test_index,
        "split_desc": f"scene_split_train={len(train_ids)}_test={len(test_ids)}",
        "dataset_tag": "EL_sceneSplit",
        "pca_mode": "EL_train_labeled_pixels_only_HyperSIGMA_legacy",
        "extra_meta": {
            "train_positions": train_pos,
            "test_positions": test_pos,
            "train_sample_ids": train_ids,
            "test_sample_ids": test_ids,
        },
    }

# =================================================
# spectral helper
# =================================================
def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * (-np.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(length, embed_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class SpectralBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=False,
        )
        self.ln2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_attn: bool = False,
        average_attn_weights: bool = False,
    ):
        h = self.ln1(tokens)
        attn_out, attn_weights = self.attn(
            h, h, h,
            need_weights=return_attn,
            average_attn_weights=average_attn_weights,
        )
        tokens = tokens + attn_out

        h2 = self.ln2(tokens)
        ffn_out = self.ffn(h2)
        tokens = tokens + ffn_out

        if return_attn:
            return tokens, attn_weights
        return tokens


class SpectralAdapter3D(nn.Module):
    def __init__(
        self,
        in_chans: int = 100,
        k_depth: int = 15,
        d_model: int = 64,
        n_heads: int = 4,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        use_preconv: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.k = k_depth
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.use_preconv = use_preconv

        if self.use_preconv:
            self.preconv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=self.k,
                padding=self.k // 2,
                bias=False,
            )
        else:
            self.preconv = None

        self.band_emb = nn.Parameter(torch.randn(in_chans, d_model) * 0.02)

        spec_pos = get_1d_sincos_pos_embed(d_model, in_chans)
        self.register_buffer("spec_pos_embed", spec_pos, persistent=False)

        self.blocks = nn.ModuleList(
            [SpectralBlock(d_model, n_heads, mlp_ratio) for _ in range(depth)]
        )

        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.in_chans, f"Expected C={self.in_chans}, got {C}"

        if self.preconv is not None:
            x_perm = x.permute(0, 2, 3, 1)
            x_seq = x_perm.reshape(-1, 1, C)
            x_seq = self.preconv(x_seq)
            x_smooth = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x_smooth = x

        x_val = x_smooth.permute(0, 2, 3, 1)
        P = B * H * W
        x_val_flat = x_val.reshape(P, C)

        pixel_chunk = 4096

        if P <= pixel_chunk:
            tokens = x_val_flat.unsqueeze(-1) * self.band_emb.unsqueeze(0)
            tokens = tokens + self.spec_pos_embed.unsqueeze(0)
            tokens = tokens.permute(1, 0, 2)

            for blk in self.blocks:
                tokens = blk(tokens)

            tokens = tokens.permute(1, 0, 2)
            out = self.out_proj(tokens).squeeze(-1)
            out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return out

        outputs = []
        for start in range(0, P, pixel_chunk):
            end = min(P, start + pixel_chunk)
            x_chunk = x_val_flat[start:end]

            tokens = x_chunk.unsqueeze(-1) * self.band_emb.unsqueeze(0)
            tokens = tokens + self.spec_pos_embed.unsqueeze(0)
            tokens = tokens.permute(1, 0, 2)

            for blk in self.blocks:
                tokens = blk(tokens)

            tokens = tokens.permute(1, 0, 2)
            out_chunk = self.out_proj(tokens).squeeze(-1)
            outputs.append(out_chunk)

        out_flat = torch.cat(outputs, dim=0)
        out = out_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

    @torch.no_grad()
    def extract_attention_map_for_pixel(
        self,
        x: torch.Tensor,
        pixel_y: int,
        pixel_x: int,
        block_idx: int = 0,
        average_attn_weights: bool = False,
    ):
        assert x.ndim == 4 and x.shape[0] == 1, f"Expected x=[1,C,H,W], got {tuple(x.shape)}"
        B, C, H, W = x.shape
        assert C == self.in_chans, f"Expected C={self.in_chans}, got {C}"
        assert 0 <= pixel_y < H and 0 <= pixel_x < W, f"pixel ({pixel_y},{pixel_x}) out of range for H={H}, W={W}"
        assert 0 <= block_idx < len(self.blocks), f"block_idx={block_idx} out of range"

        vec = x[:, :, pixel_y, pixel_x]

        if self.preconv is not None:
            vec = self.preconv(vec.unsqueeze(1)).squeeze(1)

        tokens = vec.unsqueeze(-1) * self.band_emb.unsqueeze(0)
        tokens = tokens + self.spec_pos_embed.unsqueeze(0)
        tokens = tokens.permute(1, 0, 2).contiguous()

        captured_attn = None

        for i, blk in enumerate(self.blocks):
            if i == block_idx:
                tokens, attn = blk(
                    tokens,
                    return_attn=True,
                    average_attn_weights=average_attn_weights,
                )
                captured_attn = attn
            else:
                tokens = blk(tokens)

        if captured_attn is None:
            raise RuntimeError("Attention was not captured.")

        if average_attn_weights:
            captured_attn = captured_attn[0]
        else:
            captured_attn = captured_attn[0]

        return captured_attn.detach().float().cpu()


# =================================================
# checkpoint / hsba helpers
# =================================================
def is_hsba_key(k: str) -> bool:
    """
    HSBA-related checkpoint key detector.

    Supports keys such as:
      spectral_adapter.*
      spec_patch_embed.*
      spec_ln.*
      spectral_alpha

    Also supports common wrappers:
      module.*
      student.*
      backbone.*
      model.*
    """
    hsba_prefixes = (
        "spectral_adapter.",
        "spec_patch_embed.",
        "spec_ln.",
    )
    hsba_exact = {
        "spectral_alpha",
    }

    k2 = str(k)

    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "student.", "backbone.", "model."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
                changed = True

    if any(k2.startswith(pref) for pref in hsba_prefixes):
        return True
    if k2 in hsba_exact:
        return True
    return False


def strip_hsba_from_checkpoint_file(in_path: str, out_path: str):
    ckpt = torch.load(in_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        new_state = {k: v for k, v in state.items() if not is_hsba_key(k)}
        ckpt["model"] = new_state

    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        new_state = {k: v for k, v in state.items() if not is_hsba_key(k)}
        ckpt["model_state"] = new_state

    elif isinstance(ckpt, dict):
        new_state = {k: v for k, v in ckpt.items() if not is_hsba_key(k)}
        ckpt = new_state

    else:
        raise ValueError("Unsupported checkpoint format for stripping HSBA.")

    out_dir = os.path.dirname(out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"[STRIP HSBA CKPT] saved: {out_path}")


# =================================================
# MAE backbone
# =================================================
class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        k_depth=15,
        spectral_d_model: int = 64,
        spectral_depth: int = 1,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_pred_dim=None,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        use_hsba: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.use_hsba = bool(use_hsba)

        if self.use_hsba:
            self.spectral_adapter = SpectralAdapter3D(
                in_chans=in_chans,
                k_depth=k_depth,
                d_model=spectral_d_model,
                n_heads=4,
                depth=spectral_depth,
                mlp_ratio=4.0,
                use_preconv=True,
            )
            self.spectral_alpha = nn.Parameter(torch.tensor(0.1))
            self.spec_ln = norm_layer(embed_dim)
            self.spec_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        else:
            self.spectral_adapter = None
            self.spectral_alpha = None
            self.spec_ln = None
            self.spec_patch_embed = None

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_norm = norm_layer(decoder_embed_dim)
        decoder_pred_dim = patch_size ** 2 * in_chans if decoder_pred_dim is None else decoder_pred_dim
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_pred_dim, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.proj.bias is not None:
            torch.nn.init.constant_(self.patch_embed.proj.bias, 0.0)

        if self.use_hsba and self.spec_patch_embed is not None:
            w_spec = self.spec_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w_spec.view([w_spec.shape[0], -1]))
            if self.spec_patch_embed.proj.bias is not None:
                torch.nn.init.constant_(self.spec_patch_embed.proj.bias, 0.0)

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, ids_shuffle=None):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        if ids_shuffle is None:
            noise = torch.rand(N, L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

    def forward_spectral_for_decoder(self, x, mask_ratio, ids_shuffle):
        if (not self.use_hsba) or (self.spectral_adapter is None) or (self.spec_patch_embed is None):
            raise RuntimeError("HSBA is disabled, so forward_spectral_for_decoder cannot be used.")

        x_spec = self.spectral_adapter(x)
        spec_tokens = self.spec_patch_embed(x_spec)

        N, L, D = spec_tokens.shape
        len_keep = int(L * (1 - mask_ratio))

        ids_keep = ids_shuffle[:, :len_keep]
        spec_tokens_kept = torch.gather(
            spec_tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        return spec_tokens_kept

    def forward_encoder(self, x, mask_ratio, ids_shuffle=None, decode_idx=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore, ids_shuffle = self.random_masking(x, mask_ratio, ids_shuffle)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        latent = None
        for idx, blk in enumerate(self.blocks):
            if decode_idx is not None and decode_idx == idx:
                latent = self.norm(x)
            if decode_idx is not None and idx == self.depth - 1:
                x = blk(x, return_latent=True)
            else:
                x = blk(x)

        if decode_idx is not None:
            return latent, mask, ids_restore, ids_shuffle
        else:
            x = self.norm(x)
            return x, mask, ids_restore, ids_shuffle

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1,
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        if len(imgs.shape) == 4:
            target = self.patchify(imgs)
        else:
            target = imgs

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_spatial_patches_no_mask(self, imgs):
        B, C, H, W = imgs.shape

        spat_tokens = self.patch_embed(imgs)
        spat_tokens = spat_tokens + self.pos_embed[:, 1:, :]

        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(B, -1, -1)

        x = torch.cat([cls, spat_tokens], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        spat_patch = x[:, 1:, :]
        cls_out = x[:, :1, :]
        return spat_patch, cls_out

    def forward_fused_patches_no_mask(self, imgs):
        B, C, H, W = imgs.shape

        spat_tokens = self.patch_embed(imgs)
        spat_tokens = spat_tokens + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(B, -1, -1)
        spat_tokens_with_cls = torch.cat([cls, spat_tokens], dim=1)

        for blk in self.blocks:
            spat_tokens_with_cls = blk(spat_tokens_with_cls)

        spat_tokens_with_cls = self.norm(spat_tokens_with_cls)
        spat_patch = spat_tokens_with_cls[:, 1:, :]
        cls_out = spat_tokens_with_cls[:, :1, :]

        if (not self.use_hsba) or (self.spectral_adapter is None) or (self.spec_patch_embed is None):
            return spat_patch, cls_out, None

        x_spec = self.spectral_adapter(imgs)
        spec_tokens = self.spec_patch_embed(x_spec)
        spec_tokens = self.spec_ln(spec_tokens)

        fused_patch = spat_patch + self.spectral_alpha * spec_tokens
        return fused_patch, cls_out, spec_tokens

    def forward(self, imgs, mask_ratio=0.75, ids_shuffle=None, decode_idx=None):
        latent, mask, ids_restore, ids_shuffle = self.forward_encoder(
            imgs, mask_ratio, ids_shuffle, decode_idx=decode_idx
        )

        if decode_idx is not None:
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask, ids_shuffle

        if (not self.use_hsba) or (self.spectral_adapter is None) or (self.spec_patch_embed is None):
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask, ids_shuffle

        spec_tokens_kept = self.forward_spectral_for_decoder(
            imgs, mask_ratio, ids_shuffle
        )
        spec_tokens_kept = self.spec_ln(spec_tokens_kept)

        latent_fused = latent.clone()
        latent_fused[:, 1:, :] = latent_fused[:, 1:, :] + self.spectral_alpha * spec_tokens_kept

        pred = self.forward_decoder(latent_fused, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, ids_shuffle


# =================================================
# timm-style factories
# =================================================
@register_model
def mae_vit_tiny_patch16(pretrained=False, **kwargs):
    kwargs.setdefault("patch_size", 16)
    kwargs.setdefault("embed_dim", 192)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 12)
    kwargs.setdefault("decoder_embed_dim", 96)
    kwargs.setdefault("decoder_depth", 1)
    kwargs.setdefault("decoder_num_heads", 3)
    kwargs.setdefault("mlp_ratio", 4.0)
    kwargs.setdefault("norm_layer", partial(nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoderViT(**kwargs)
    return model


@register_model
def mae_vit_tiny_d13_patch16(pretrained=False, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=192,
        depth=13,
        num_heads=12,
        decoder_embed_dim=96,
        decoder_depth=1,
        decoder_num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_model
def mae_vit_small_patch16(pretrained=False, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        decoder_embed_dim=192,
        decoder_depth=1,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_model
def mae_vit_base_patch16(pretrained=False, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_model
def mae_vit_large_patch16(pretrained=False, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


@register_model
def mae_vit_huge_patch14(pretrained=False, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# =================================================
# Segmentation wrapper
# =================================================
class MAESegWrapper(nn.Module):
    def __init__(self, backbone, num_classes: int, img_size: int, use_spectral: bool = False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = backbone.embed_dim
        self.use_spectral = use_spectral

        p = backbone.patch_embed.patch_size
        if isinstance(p, (tuple, list)):
            p = p[0]
        self.patch_size = int(p)

        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        if self.use_spectral:
            patch_tokens, _, _ = self.backbone.forward_fused_patches_no_mask(x)
        else:
            patch_tokens, _ = self.backbone.forward_spatial_patches_no_mask(x)

        B, L, D = patch_tokens.shape

        Hp = self.img_size // self.patch_size
        Wp = self.img_size // self.patch_size
        assert Hp * Wp == L, f"L={L} != Hp*Wp={Hp*Wp}"

        feat_map = patch_tokens.reshape(B, Hp, Wp, D).permute(0, 3, 1, 2).contiguous()
        feat_up = F.interpolate(
            feat_map,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.head(feat_up)
        out = logits.permute(0, 2, 3, 1).reshape(B, self.img_size * self.img_size, self.num_classes)
        return out


# =================================================
# split helpers
# =================================================
def pu_split_per_class(gt_flat: np.ndarray, class_num: int, train_num: int, seed: int):
    rng = np.random.RandomState(seed)
    train_idx = []
    test_idx = []

    for c in range(1, class_num + 1):
        cls_idx = np.where(gt_flat == c)[0]
        rng.shuffle(cls_idx)

        n_train = min(train_num, len(cls_idx))
        train_idx.append(cls_idx[:n_train])
        test_idx.append(cls_idx[n_train:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx) if len(test_idx) else np.array([], dtype=np.int64)
    val_idx = np.array([], dtype=np.int64)

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def get_train_val_test_indices(dataset_name, data_gt, meta, class_num, train_num, seed):
    gt_flat = data_gt.reshape(-1)

    if dataset_name.upper() == "HU":
        if "TR" not in meta or "TE" not in meta:
            raise ValueError("HU dataset requires official TR/TE maps in Houston_official.mat")

        tr_map = np.asarray(meta["TR"]).astype(np.int64)
        te_map = np.asarray(meta["TE"]).astype(np.int64)

        if tr_map.ndim == 3 and tr_map.shape[-1] == 1:
            tr_map = tr_map[..., 0]
        if te_map.ndim == 3 and te_map.shape[-1] == 1:
            te_map = te_map[..., 0]

        if tr_map.shape != data_gt.shape or te_map.shape != data_gt.shape:
            raise ValueError(
                f"HU TR/TE shape mismatch: gt={data_gt.shape}, TR={tr_map.shape}, TE={te_map.shape}"
            )

        train_index = np.where(tr_map.reshape(-1) > 0)[0].astype(np.int64)
        test_index = np.where(te_map.reshape(-1) > 0)[0].astype(np.int64)
        val_index = np.array([], dtype=np.int64)

        split_msg = f"[SPLIT-HU official] train={len(train_index)} test={len(test_index)}"

    else:
        train_index, val_index, test_index = pu_split_per_class(
            gt_flat, class_num=class_num, train_num=train_num, seed=seed
        )
        split_msg = (
            f"[SPLIT-PU] seed={seed} train={len(train_index)} "
            f"test={len(test_index)} (train_num/class={train_num})"
        )

    return train_index, val_index, test_index, split_msg


# =================================================
# ckpt loading helper
# =================================================

def load_student_mae_ckpt(mae_model: nn.Module, ckpt_path: str, skip_hsba_ckpt: bool = False, use_spectral: bool = False):
    """Exact HyperSIGMA shape-only transfer; optional HSBA-skip for all6 compatibility.
    use_spectral is retained for call-site compatibility, not strict loading.
    WARNING: required pretrained tensors are not verified (legacy behavior).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    cleaned = {}
    target_keys = mae_model.state_dict().keys()
    skipped = []
    skipped_hsba = []
    loaded_hsba = []
    for k, v in state.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        if k2.startswith("student."):
            k2 = k2[len("student."):]
        if k2.startswith("decoder_") or k2 in {"mask_token", "decoder_pos_embed"}:
            skipped.append(k2)
            continue
        if k2 in {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed"}:
            skipped.append(k2)
            continue
        if skip_hsba_ckpt and is_hsba_key(k2):
            skipped.append(k2)
            skipped_hsba.append(k2)
            continue
        if k2 in target_keys:
            if mae_model.state_dict()[k2].shape == v.shape:
                cleaned[k2] = v
                if is_hsba_key(k2):
                    loaded_hsba.append(k2)
            else:
                skipped.append(f"{k2} (shape mismatch: ckpt={tuple(v.shape)} model={tuple(mae_model.state_dict()[k2].shape)})")
        else:
            skipped.append(k2)
    missing, unexpected = mae_model.load_state_dict(cleaned, strict=False)
    print(f"[MAE ckpt load] used={len(cleaned)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
    if skipped:
        print("  - skipped keys (ex):", skipped[:30])
    # Original loader printed stats only; all6 expects a small result dict for logging.
    # This return does NOT change the filtering/loading behavior.
    return {
        "used": len(cleaned),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_hsba": len(skipped_hsba),
        "loaded_hsba": len(loaded_hsba),
    }


def remove_decoder_parts(mae_model: nn.Module):
    removed = []
    for attr in [
        "decoder_blocks",
        "decoder_embed",
        "decoder_norm",
        "decoder_pred",
        "decoder_pos_embed",
        "mask_token",
    ]:
        if hasattr(mae_model, attr):
            delattr(mae_model, attr)
            removed.append(attr)
    print(f"[Remove decoder] {removed}")


# =================================================
# helper: TR pad + labels
# =================================================
def build_tr_pad_and_label0(data_gt_pad: np.ndarray, train_map_hw: np.ndarray):
    H0, W0 = train_map_hw.shape
    Hpad, Wpad = data_gt_pad.shape

    TR_pad = np.zeros((Hpad, Wpad), dtype=np.int32)
    TR_pad[:H0, :W0] = train_map_hw.astype(np.int32)

    label_pad0 = data_gt_pad.astype(np.int64) - 1
    return TR_pad, label_pad0


def make_tile_base_flat_indices(img_size: int, Wpad: int) -> np.ndarray:
    ys = np.arange(img_size, dtype=np.int64)[:, None]
    xs = np.arange(img_size, dtype=np.int64)[None, :]
    base = (ys * Wpad + xs).reshape(-1)
    return base


# =================================================
# logging helpers
# =================================================
def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


# =================================================
# Visualization helpers
# =================================================
def make_discrete_colormap(n_classes: int):
    import matplotlib.colors as mcolors

    if n_classes <= 20:
        base = plt.get_cmap("tab20", n_classes)
        colors = [base(i) for i in range(n_classes)]
    else:
        base = plt.get_cmap("hsv", n_classes)
        colors = [base(i) for i in range(n_classes)]

    colors[0] = (0, 0, 0, 1.0)
    cmap = mcolors.ListedColormap(colors)
    return cmap


def save_label_map(label_map, out_path, title=None, vmin=None, vmax=None):
    label_map = np.asarray(label_map)
    max_label = int(label_map.max()) if label_map.size > 0 else 0
    cmap = make_discrete_colormap(max_label + 1 if max_label + 1 > 1 else 2)

    plt.figure(figsize=(7, 6))
    plt.imshow(
        label_map,
        cmap=cmap,
        interpolation="nearest",
        vmin=0 if vmin is None else vmin,
        vmax=max_label if vmax is None else vmax,
    )
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[SAVE] label map: {out_path}")


def save_error_map(gt_map, pred_map, eval_ids, H, W, out_path, title="Error Map"):
    gt_flat = gt_map.reshape(-1)
    pred_flat = pred_map.reshape(-1)

    err = np.zeros((H * W,), dtype=np.int64)

    ids = np.asarray(eval_ids, dtype=np.int64)
    correct = gt_flat[ids] == pred_flat[ids]

    err[ids[correct]] = 1
    err[ids[~correct]] = 2
    err = err.reshape(H, W)

    import matplotlib.colors as mcolors

    cmap = mcolors.ListedColormap(
        [
            (0, 0, 0, 1.0),
            (0.2, 0.8, 0.2, 1),
            (0.9, 0.2, 0.2, 1),
        ]
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(err, cmap=cmap, interpolation="nearest", vmin=0, vmax=2)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[SAVE] error map: {out_path}")


def save_confusion_matrix_fig(cm, out_path, class_names=None, normalize=False, title=None):
    cm = np.asarray(cm, dtype=np.float64)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_show = cm / row_sum
    else:
        cm_show = cm

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_show, interpolation="nearest", cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)

    if title is None:
        title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    plt.title(title)

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i + 1) for i in range(num_classes)]

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_show.max() / 2.0 if cm_show.size > 0 else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_show[i, j]
            text = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] confusion matrix: {out_path}")


def save_overview_figure(gt_map, pred_map, err_map, out_path, suptitle=None):
    max_label = max(int(gt_map.max()), int(pred_map.max()))
    cmap_cls = make_discrete_colormap(max_label + 1 if max_label + 1 > 1 else 2)

    import matplotlib.colors as mcolors

    cmap_err = mcolors.ListedColormap(
        [
            (0, 0, 0, 1.0),
            (0.2, 0.8, 0.2, 1),
            (0.9, 0.2, 0.2, 1),
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(gt_map, cmap=cmap_cls, interpolation="nearest", vmin=0, vmax=max_label)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred_map, cmap=cmap_cls, interpolation="nearest", vmin=0, vmax=max_label)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(err_map, cmap=cmap_err, interpolation="nearest", vmin=0, vmax=2)
    axes[2].set_title("Error Map")
    axes[2].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] overview figure: {out_path}")


def build_error_map_array(gt_map, pred_map, eval_ids):
    H, W = gt_map.shape
    gt_flat = gt_map.reshape(-1)
    pred_flat = pred_map.reshape(-1)
    err = np.zeros((H * W,), dtype=np.int64)

    ids = np.asarray(eval_ids, dtype=np.int64)
    correct = gt_flat[ids] == pred_flat[ids]

    err[ids[correct]] = 1
    err[ids[~correct]] = 2
    return err.reshape(H, W)


def compute_each_acc_from_cm(cm: np.ndarray):
    cm = np.asarray(cm, dtype=np.float64)
    row_sum = cm.sum(axis=1)
    each_acc = np.divide(
        np.diag(cm),
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float64),
        where=row_sum > 0,
    )
    average_acc = float(each_acc.mean()) if each_acc.size > 0 else 0.0
    return each_acc, average_acc


def save_training_curve(train_epochs, train_losses, eval_epochs, eval_oas, out_path):
    plt.figure(figsize=(8, 5))

    ax1 = plt.gca()
    ax1.plot(train_epochs, train_losses, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if len(eval_epochs) > 0:
        ax2.plot(eval_epochs, eval_oas, marker="o", label="test OA")
    ax2.set_ylabel("test OA")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Training Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] training curve: {out_path}")


# =================================================
# Spectral attention visualization helpers
# =================================================
def save_spectral_attention_heatmap(
    attn_map: np.ndarray,
    out_path: str,
    title: str = "Spectral Self-Attention",
    xlabel: str = "Key band index",
    ylabel: str = "Query band index",
):
    attn_map = np.asarray(attn_map, dtype=np.float32)

    plt.figure(figsize=(7, 6))
    plt.imshow(attn_map, interpolation="nearest", cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] spectral attn heatmap: {out_path}")


def save_spectral_attention_profile(
    attn_map: np.ndarray,
    query_band_idx: int,
    out_path: str,
    title: str = None,
):
    attn_map = np.asarray(attn_map, dtype=np.float32)
    L = attn_map.shape[0]
    query_band_idx = int(np.clip(query_band_idx, 0, L - 1))

    profile = attn_map[query_band_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(L), profile)
    plt.xlabel("Key band index")
    plt.ylabel("Attention weight")
    if title is None:
        title = f"Attention Profile of Query Band {query_band_idx}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] spectral attn profile: {out_path}")


def save_multihead_spectral_attention(
    attn_heads: np.ndarray,
    out_dir: str,
    prefix: str,
):
    attn_heads = np.asarray(attn_heads, dtype=np.float32)
    num_heads = attn_heads.shape[0]

    for h in range(num_heads):
        save_spectral_attention_heatmap(
            attn_heads[h],
            os.path.join(out_dir, f"{prefix}_head{h}.png"),
            title=f"{prefix} - head {h}",
        )


# =================================================
# Latency / MACs helpers
# =================================================
@torch.no_grad()
def measure_batch1_latency(
    model: nn.Module,
    sample_b1: torch.Tensor,
    device: torch.device,
    warmup: int = 20,
    iters: int = 100,
) -> Dict[str, float]:
    assert sample_b1.ndim == 4 and sample_b1.shape[0] == 1, (
        f"Expected sample_b1=[1,C,H,W], got {tuple(sample_b1.shape)}"
    )

    model_was_training = model.training
    model.eval()

    x = sample_b1.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for _ in range(max(0, int(warmup))):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times_ms = []

    for _ in range(max(1, int(iters))):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)

    result = {
        "num_iters": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }

    if model_was_training:
        model.train()

    return result


@torch.no_grad()
def measure_batch1_macs(
    model: nn.Module,
    sample_b1: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    assert sample_b1.ndim == 4 and sample_b1.shape[0] == 1, (
        f"Expected sample_b1=[1,C,H,W], got {tuple(sample_b1.shape)}"
    )

    if thop_profile is None:
        return {
            "available": False,
            "reason": "thop is not installed",
            "macs": None,
            "params": None,
        }

    model_was_training = model.training
    model.eval()

    x = sample_b1.to(device, non_blocking=True)

    try:
        macs, params = thop_profile(
            model,
            inputs=(x,),
            verbose=False,
        )
        result = {
            "available": True,
            "reason": "",
            "macs": float(macs),
            "params": float(params),
        }
    except Exception as e:
        result = {
            "available": False,
            "reason": f"{type(e).__name__}: {str(e)}",
            "macs": None,
            "params": None,
        }

    if model_was_training:
        model.train()

    return result


# =================================================
# main
# =================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="datasets",
                        help="IN/UP/HC/HH/HU subfolders, or Electrolyzers-HSI root for EL")
    parser.add_argument("--dataset", type=str, default="UP", choices=["IN", "UP", "HC", "HH", "HU", "EL"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/mrmae_lhmae_stage2_6gpu/student_stage2.pth",
        help=(
            "Path to pretrained MAE checkpoint. "
            "Use 'none', 'null', 'no', 'random', or '' to skip checkpoint loading "
            "and train/evaluate from random initialization."
        ),
    )
    parser.add_argument("--remove_decoder", action="store_true")

    parser.add_argument("--use_spectral", action="store_true",
                        help="Use spectral branch in downstream. Default: spatial-only")

    parser.add_argument("--disable_hsba", action="store_true",
                        help="Completely remove HSBA modules from backbone for no-HSBA baseline / latency test")

    parser.add_argument(
        "--skip_hsba_ckpt",
        action="store_true",
        help=(
            "Create HSBA modules but exclude HSBA-related weights when loading checkpoint. "
            "Useful when loading spatial pretrained ckpt while training HSBA from random init."
        ),
    )

    parser.add_argument("--save_stripped_ckpt", action="store_true",
                        help="Save a copy of checkpoint with HSBA-related keys removed")
    parser.add_argument("--stripped_ckpt_path", type=str, default="",
                        help="Output path for stripped checkpoint")

    parser.add_argument("--spectral_depth", type=int, default=1,
                        help="MR-MAE Stage2 HSBA depth; pretrained checkpoint uses 1 (not legacy HyperSIGMA depth 2)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Optional results folder; default is HyperSIGMA-compatible name prefixed by MRMAE_ to avoid overwrites")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load dataset/ckpt and validate the complete downstream model, but skip training/evaluation")

    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--train_num", type=int, default=None,
                        help="Labeled training pixels per class. Default: IN=10, UP=20, HC=50, HH=50 (HyperSIGMA TPAMI 2025 Appendix D.1); HU uses official TR/TE, EL uses scene split.")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--accum_steps", type=int, default=1)

    parser.add_argument("--eval_start", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=30)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_train_each_epoch", type=int, default=1)

    parser.add_argument("--save_vis", action="store_true",
                        help="Save MSDNet-style visualizations at final evaluation")
    parser.add_argument("--vis_use_last_eval", action="store_true",
                        help="Use the last scheduled eval result for visualization if available")

    parser.add_argument("--print_param_detail", action="store_true",
                        help="Print all model parameters trainable/frozen")
    parser.add_argument("--print_spectral_param_detail", action="store_true",
                        help="Print spectral-related parameter names and counts")

    parser.add_argument("--save_spec_attn_vis", action="store_true",
                        help="Save spectral adapter self-attention heatmap")
    parser.add_argument("--spec_attn_vis_epochs", type=int, nargs="*", default=[],
                        help="Epochs to save spectral attention heatmap. Example: --spec_attn_vis_epochs 0 25 50 150")
    parser.add_argument("--spec_attn_block_idx", type=int, default=0,
                        help="Which spectral block attention to visualize")
    parser.add_argument("--spec_attn_tile_index", type=int, default=0,
                        help="Which tile index to visualize")
    parser.add_argument("--spec_attn_pixel_y", type=int, default=-1,
                        help="Pixel y inside selected tile. -1 means center")
    parser.add_argument("--spec_attn_pixel_x", type=int, default=-1,
                        help="Pixel x inside selected tile. -1 means center")
    parser.add_argument("--spec_attn_query_band_idx", type=int, default=-1,
                        help="Which query band profile to save. -1 means center band")

    parser.add_argument("--measure_latency", action="store_true",
                        help="Measure batch=1 inference latency on a single tile after training")
    parser.add_argument("--latency_warmup", type=int, default=20,
                        help="Number of warmup iterations for latency measurement")
    parser.add_argument("--latency_iters", type=int, default=100,
                        help="Number of timed iterations for latency measurement")
    parser.add_argument("--latency_tile_index", type=int, default=0,
                        help="Which tile index to use for batch=1 latency measurement")

    parser.add_argument("--measure_macs", action="store_true",
                        help="Measure batch=1 MACs on a single tile using thop")
    parser.add_argument("--macs_tile_index", type=int, default=0,
                        help="Which tile index to use for batch=1 MACs measurement")

    # EL-specific settings reproduce the original HyperSIGMA+LH-MAE EL pipeline.
    parser.add_argument("--el_test_indices", type=str,
                        default="0,1,4,5,6,7,8,12,41,46,50",
                        help="0-based positions among numerically sorted EL scenes")
    parser.add_argument("--el_disable_repo_band_crop", action="store_true",
                        help="Disable legacy EL spectral crop [50:-40]")
    parser.add_argument("--el_band_start", type=int, default=50)
    parser.add_argument("--el_band_end_drop", type=int, default=40)
    parser.add_argument("--el_use_spectral_binning", action="store_true")
    parser.add_argument("--el_use_vector_norm", action="store_true")
    parser.add_argument(
        "--el_pad_variable_widths", action="store_true",
        help="OPT-IN: right-zero-pad EL scenes to widest width; default preserves exact HyperSIGMA concatenation, which requires equal widths",
    )

    args = parser.parse_args()
    # HyperSIGMA TPAMI 2025, Appendix D.1: per-class counts used for
    # the main classification benchmark (HU uses the official TR/TE).
    official_train_per_class = {"IN": 10, "UP": 20, "HC": 50, "HH": 50}
    if args.train_num is None:
        args.train_num = official_train_per_class.get(args.dataset.upper(), 20)
    if args.train_num < 1:
        raise ValueError("--train_num must be >=1")
    if args.dataset.upper() in official_train_per_class:
        reference_count = official_train_per_class[args.dataset.upper()]
        print(f"[PROTOCOL] dataset={args.dataset}: train_num/class={args.train_num} "
              f"(HyperSIGMA paper reference={reference_count})")
    elif args.dataset.upper() == "HU":
        print("[PROTOCOL] HU: official TR/TE maps (train_num ignored)")
    elif args.dataset.upper() == "EL":
        print("[PROTOCOL] EL: scene-held-out split (train_num ignored)")
    if args.spectral_depth < 1:
        parser.error("--spectral_depth must be >= 1")
    if args.img_size <= 0 or args.patch_size <= 0 or args.img_size % args.patch_size != 0:
        parser.error("--img_size must be positive and divisible by --patch_size")
    if args.use_spectral and args.disable_hsba:
        print("[WARN] --disable_hsba overrides --use_spectral")

    ckpt_path = str(args.ckpt).strip()
    ckpt_is_disabled = ckpt_path == "" or ckpt_path.lower() in ["none", "null", "no", "random"]

    set_seed(args.seed)

    if args.disable_hsba and args.use_spectral:
        print("[WARN] --disable_hsba is on, so --use_spectral will be ignored.")
        args.use_spectral = False

    if args.disable_hsba and args.skip_hsba_ckpt:
        print("[WARN] --disable_hsba is on, so --skip_hsba_ckpt has no practical effect.")

    if args.disable_hsba and args.save_spec_attn_vis:
        print("[WARN] --disable_hsba is on, so spectral attention visualization will be skipped.")

    if args.save_spec_attn_vis and len(args.spec_attn_vis_epochs) == 0:
        args.spec_attn_vis_epochs = [0, args.max_epoch]
    spec_attn_epochs = set(int(e) for e in args.spec_attn_vis_epochs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print("[Device train]", device)
    print("[Mode]", "spectral" if args.use_spectral else "spatial-only")
    print("[HSBA]", "disabled" if args.disable_hsba else "enabled")
    print("[CKPT]", "disabled/random initialization" if ckpt_is_disabled else ckpt_path)
    print("[SKIP HSBA CKPT]", bool(args.skip_hsba_ckpt))

    img_size = args.img_size
    patch_size = args.patch_size
    pca_components = args.pca_components
    accum_steps = max(1, int(args.accum_steps))

    # --------------------------
    # Load data + preprocessing (keep each original HyperSIGMA protocol)
    # --------------------------
    if args.dataset.upper() == "EL":
        el_data = prepare_el_hypersigma_legacy(args)
        data = el_data["data"]
        data_gt = el_data["data_gt"]
        train_index = el_data["train_index"]
        test_index = el_data["test_index"]
        test_samples_gt = el_data["test_samples_gt"]
        test_gt = el_data["test_gt"]
        train_map_hw_el = el_data["train_map_hw"]
        split_desc = el_data["split_desc"]
        dataset_tag = el_data["dataset_tag"]
        pca_mode = el_data["pca_mode"]
        extra_meta = el_data["extra_meta"]
        H0, W0, _ = data.shape
        class_num = int(np.max(data_gt))
    else:
        data, data_gt, meta = load_dataset(args.data_root, args.dataset)
        H0, W0, _ = data.shape
        class_num = int(np.max(data_gt))
        print(f"[DATA] dataset={args.dataset} HWC={data.shape} classes={class_num}")

        data, _ = split_data.apply_PCA(data, num_components=pca_components)
        print(f"[PCA] components={pca_components} => data={data.shape}")

        gt_flat = data_gt.reshape(-1)

        train_index, val_index, test_index, split_msg = get_train_val_test_indices(
            args.dataset, data_gt, meta, class_num, args.train_num, args.seed
        )
        print(split_msg)

        train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(
            gt_flat, train_index.astype(int), val_index.astype(int), test_index.astype(int)
        )
        test_gt = test_samples_gt.reshape(H0, W0)
        split_desc = "official_TRTE" if args.dataset.upper() == "HU" else f"train_num_per_class={args.train_num}"
        dataset_tag = args.dataset
        pca_mode = "full_scene_transductive_HyperSIGMA_legacy"
        extra_meta = {}

    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
    test_samples_gt_cpu = torch.from_numpy(test_samples_gt.astype(np.float32)).to(cpu_device)
    test_gt_onehot_cpu = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(cpu_device)
    input_channels = int(data.shape[-1])

    # --------------------------
    # tiling
    # --------------------------
    img_tiles, num_H, num_W, data_gt_pad, data_pad = utils.Get_train_and_test_data(
        img_size, data, data_gt
    )
    Hpad, Wpad, _ = data_pad.shape
    num_tiles = int(num_H * num_W)
    print(f"[TILING] num_H={num_H} num_W={num_W} num_tiles={num_tiles} | padded={data_pad.shape}")

    img_tiles_t = torch.from_numpy(img_tiles.transpose(0, 3, 1, 2)).float()
    tile_ds = Data.TensorDataset(img_tiles_t)

    train_loader = Data.DataLoader(
        tile_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = Data.DataLoader(
        tile_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------
    # Build official / sparse train map
    # --------------------------
    if args.dataset.upper() == "EL":
        train_map_hw = train_map_hw_el
    elif args.dataset.upper() == "HU":
        train_map_hw = np.asarray(meta["TR"]).astype(np.int32)
        if train_map_hw.ndim == 3 and train_map_hw.shape[-1] == 1:
            train_map_hw = train_map_hw[..., 0]
    else:
        train_map_hw = np.zeros((H0, W0), dtype=np.int32)
        gt_hw = data_gt.astype(np.int32)
        train_map_hw.reshape(-1)[train_index] = gt_hw.reshape(-1)[train_index]

    TR_pad, label_pad0 = build_tr_pad_and_label0(data_gt_pad, train_map_hw)
    tile_base_flat = make_tile_base_flat_indices(img_size, Wpad)
    P = img_size * img_size

    # --------------------------
    # results dir / txt
    # --------------------------
    mode_name = "spectral" if args.use_spectral else "spatial"
    hsba_name = "noHSBA" if args.disable_hsba else "HSBA"
    original_result_name = (
        f"{dataset_tag}_results_pca{pca_components}_img{img_size}_patch{patch_size}"
        f"_bs{args.batch_size}_streamTR_{mode_name}_{hsba_name}"
    )
    RESULT_DIR = args.output_dir.strip() if args.output_dir.strip() else "MRMAE_" + original_result_name
    os.makedirs(RESULT_DIR, exist_ok=True)

    RESULT_FILE = os.path.join(RESULT_DIR, "results.txt")
    TRAIN_LOG = os.path.join(RESULT_DIR, "train_log.txt")
    VIS_DIR = os.path.join(RESULT_DIR, "visualizations")
    SPEC_ATTN_DIR = os.path.join(RESULT_DIR, "spectral_attention")

    if args.save_vis:
        os.makedirs(VIS_DIR, exist_ok=True)
    if args.save_spec_attn_vis and (not args.disable_hsba):
        os.makedirs(SPEC_ATTN_DIR, exist_ok=True)

    write_text(
        RESULT_FILE,
        f"=== RUN START ===\n"
        f"seed={args.seed}\n"
        f"dataset={dataset_tag}\n"
        f"mode={mode_name}\n"
        f"hsba={'disabled' if args.disable_hsba else 'enabled'}\n"
        f"skip_hsba_ckpt={args.skip_hsba_ckpt}\n"
        f"split={split_desc}\n"
        f"img_size={img_size} patch_size={patch_size} pca={pca_components} "
        f"bs={args.batch_size} accum={accum_steps}\n"
        f"eval_start={args.eval_start} eval_every={args.eval_every}\n"
        f"ckpt={args.ckpt}\n"
        f"spectral_depth={args.spectral_depth}\n"
        f"extra_meta={extra_meta}\n"
        f"PCA={pca_mode}\n"
        f"ckpt_loading={'disabled/random_init' if ckpt_is_disabled else 'enabled'}\n"
        f"save_weights=NO\n"
        f"save_vis={args.save_vis}\n"
        f"save_spec_attn_vis={args.save_spec_attn_vis}\n"
        f"spec_attn_epochs={sorted(list(spec_attn_epochs))}\n"
        f"spec_attn_block_idx={args.spec_attn_block_idx}\n"
        f"spec_attn_tile_index={args.spec_attn_tile_index}\n"
        f"spec_attn_pixel_y={args.spec_attn_pixel_y}\n"
        f"spec_attn_pixel_x={args.spec_attn_pixel_x}\n"
        f"measure_latency={args.measure_latency}\n"
        f"latency_warmup={args.latency_warmup}\n"
        f"latency_iters={args.latency_iters}\n"
        f"latency_tile_index={args.latency_tile_index}\n"
        f"measure_macs={args.measure_macs}\n"
        f"macs_tile_index={args.macs_tile_index}\n"
        f"save_stripped_ckpt={args.save_stripped_ckpt}\n"
        f"stripped_ckpt_path={args.stripped_ckpt_path}\n"
        f"=================\n"
    )
    write_text(
        TRAIN_LOG,
        f"=== TRAIN LOG START ===\n"
        f"seed={args.seed}\n"
        f"dataset={dataset_tag}\n"
        f"mode={mode_name}\n"
        f"hsba={'disabled' if args.disable_hsba else 'enabled'}\n"
        f"skip_hsba_ckpt={args.skip_hsba_ckpt}\n"
        f"split={split_desc}\n"
        f"ckpt={args.ckpt}\n"
        f"ckpt_loading={'disabled/random_init' if ckpt_is_disabled else 'enabled'}\n"
        f"save_weights=NO\n"
        f"save_vis={args.save_vis}\n"
        f"save_spec_attn_vis={args.save_spec_attn_vis}\n"
        f"measure_latency={args.measure_latency}\n"
        f"measure_macs={args.measure_macs}\n"
        f"=======================\n"
    )

    # --------------------------
    # Optional stripped checkpoint
    # --------------------------
    if args.save_stripped_ckpt:
        if ckpt_is_disabled:
            print("[STRIP HSBA CKPT] skipped: checkpoint loading is disabled")
            append_text(RESULT_FILE, "[STRIP HSBA CKPT] skipped: checkpoint loading is disabled\n")
            append_text(TRAIN_LOG, "[STRIP HSBA CKPT] skipped: checkpoint loading is disabled\n")
        else:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {ckpt_path}\n"
                    f"Cannot strip HSBA keys from a missing checkpoint.\n"
                    f"Use --ckpt none if you want to run without pretrained weights."
                )

            if args.stripped_ckpt_path.strip() == "":
                base, ext = os.path.splitext(ckpt_path)
                stripped_out = base + "_no_hsba" + ext
            else:
                stripped_out = args.stripped_ckpt_path

            strip_hsba_from_checkpoint_file(ckpt_path, stripped_out)

    # --------------------------
    # Build backbone
    # --------------------------
    mae_backbone = mae_vit_tiny_patch16(
        pretrained=False,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=input_channels,
        norm_pix_loss=False,
        spectral_d_model=64,
        spectral_depth=args.spectral_depth,
        use_hsba=(not args.disable_hsba),
    ).to(device)

    if args.remove_decoder:
        remove_decoder_parts(mae_backbone)

    if ckpt_is_disabled:
        print("[Load MAE ckpt] skipped: training/evaluating from random initialization")
        append_text(
            RESULT_FILE,
            "[Load MAE ckpt] skipped: training/evaluating from random initialization\n",
        )
        append_text(
            TRAIN_LOG,
            "[Load MAE ckpt] skipped: training/evaluating from random initialization\n",
        )
    else:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt_path}\n"
                f"Use --ckpt none if you want to train/evaluate without pretrained weights."
            )

        print(f"[Load MAE ckpt] {ckpt_path}")
        ckpt_load_stats = load_student_mae_ckpt(
            mae_model=mae_backbone,
            ckpt_path=ckpt_path,
            skip_hsba_ckpt=args.skip_hsba_ckpt,
            use_spectral=args.use_spectral,
        )

        append_text(
            RESULT_FILE,
            "[CKPT LOAD STATS]\n"
            f"used={ckpt_load_stats['used']}\n"
            f"skipped={ckpt_load_stats['skipped']}\n"
            f"missing={ckpt_load_stats['missing']}\n"
            f"unexpected={ckpt_load_stats['unexpected']}\n"
            f"skipped_hsba={ckpt_load_stats['skipped_hsba']}\n"
            f"loaded_hsba={ckpt_load_stats['loaded_hsba']}\n"
            + "-" * 40 + "\n",
        )
        append_text(
            TRAIN_LOG,
            "[CKPT LOAD STATS]\n"
            f"used={ckpt_load_stats['used']}\n"
            f"skipped={ckpt_load_stats['skipped']}\n"
            f"missing={ckpt_load_stats['missing']}\n"
            f"unexpected={ckpt_load_stats['unexpected']}\n"
            f"skipped_hsba={ckpt_load_stats['skipped_hsba']}\n"
            f"loaded_hsba={ckpt_load_stats['loaded_hsba']}\n"
            + "-" * 40 + "\n",
        )

    if (not args.use_spectral) and (not args.disable_hsba):
        for name, p in mae_backbone.named_parameters():
            if (
                name.startswith("spectral_adapter.")
                or name.startswith("spec_patch_embed.")
                or name.startswith("spec_ln.")
                or name == "spectral_alpha"
            ):
                p.requires_grad = False

    model = MAESegWrapper(
        mae_backbone,
        num_classes=class_num,
        img_size=img_size,
        use_spectral=args.use_spectral,
    ).to(device)

    # --------------------------
    # Batch=1 MACs measurement
    # --------------------------
    macs_pack = None
    if args.measure_macs:
        macs_tile_idx = int(np.clip(args.macs_tile_index, 0, num_tiles - 1))
        macs_sample_b1 = img_tiles_t[macs_tile_idx:macs_tile_idx + 1]

        print(
            f"[MACS] measuring batch=1 MACs | "
            f"tile_index={macs_tile_idx} | input_shape={tuple(macs_sample_b1.shape)}"
        )

        macs_pack = measure_batch1_macs(
            model=model,
            sample_b1=macs_sample_b1,
            device=device,
        )

        if macs_pack["available"]:
            macs_msg = (
                "[MACS - BATCH 1 FORWARD]\n"
                f"device={device}\n"
                f"tile_index={macs_tile_idx}\n"
                f"input_shape={tuple(macs_sample_b1.shape)}\n"
                f"macs={macs_pack['macs']:.0f} ({format_macs(macs_pack['macs'])})\n"
                f"thop_params={macs_pack['params']:.0f} ({format_param_count(int(macs_pack['params']))})\n"
            )
        else:
            macs_msg = (
                "[MACS - BATCH 1 FORWARD]\n"
                f"device={device}\n"
                f"tile_index={macs_tile_idx}\n"
                f"input_shape={tuple(macs_sample_b1.shape)}\n"
                f"status=FAILED\n"
                f"reason={macs_pack['reason']}\n"
            )

        print(macs_msg)
        append_text(RESULT_FILE, macs_msg + "-" * 40 + "\n")
        append_text(TRAIN_LOG, macs_msg + "-" * 40 + "\n")

    # --------------------------
    # Model parameter counting
    # --------------------------
    stats = count_parameters(model, print_detail=args.print_param_detail)
    total_p = stats["total_params"]
    train_p = stats["trainable_params"]
    frozen_p = stats["frozen_params"]
    total_buf = stats["total_buffers"]

    spectral_stats = count_spectral_parameters(
        model,
        print_detail=args.print_spectral_param_detail,
    )
    spectral_total_p = spectral_stats["total_params"]
    spectral_train_p = spectral_stats["trainable_params"]
    spectral_frozen_p = spectral_stats["frozen_params"]

    size_mb_fp32 = estimate_model_size_mb(model, dtype_bytes=4, include_buffers=True)
    size_mb_fp16 = estimate_model_size_mb(model, dtype_bytes=2, include_buffers=True)

    msg_params = (
        "[PARAMS - TOTAL MODEL]\n"
        f"  total params      : {total_p} ({format_param_count(total_p)})\n"
        f"  trainable params  : {train_p} ({format_param_count(train_p)})\n"
        f"  frozen params     : {frozen_p} ({format_param_count(frozen_p)})\n"
        f"  total buffers     : {total_buf} ({format_param_count(total_buf)})\n"
        f"  est. size         : {size_mb_fp32:.2f} MB (fp32, incl buffers)\n"
        f"                     {size_mb_fp16:.2f} MB (fp16/bf16, incl buffers)\n"
    )

    msg_spectral = (
        "[PARAMS - SPECTRAL ADAPTER PART]\n"
        f"  total params      : {spectral_total_p} ({format_param_count(spectral_total_p)})\n"
        f"  trainable params  : {spectral_train_p} ({format_param_count(spectral_train_p)})\n"
        f"  frozen params     : {spectral_frozen_p} ({format_param_count(spectral_frozen_p)})\n"
    )

    print(msg_params)
    print(msg_spectral)

    append_text(TRAIN_LOG, msg_params + "\n")
    append_text(TRAIN_LOG, msg_spectral + "\n")
    append_text(RESULT_FILE, msg_params + "\n")
    append_text(RESULT_FILE, msg_spectral + "\n")

    if args.print_spectral_param_detail:
        detail_lines = ["[SPECTRAL PARAM DETAIL]"]
        for name, n, req in spectral_stats["matched"]:
            tag = "trainable" if req else "frozen"
            detail_lines.append(f"  [{tag}] {name} : {n}")
        detail_text = "\n".join(detail_lines) + "\n"
        append_text(TRAIN_LOG, detail_text + "\n")
        append_text(RESULT_FILE, detail_text + "\n")

    if args.dry_run:
        print("[DRY RUN] Model/checkpoint/inputs validated; training not started")
        append_text(RESULT_FILE, "[DRY RUN] complete: no training or evaluation\n")
        append_text(TRAIN_LOG, "[DRY RUN] complete: no training or evaluation\n")
        return

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    eval_epochs = set(range(args.eval_start, args.max_epoch + 1, args.eval_every))
    best_loss = 1e18

    train_curve_epochs = []
    train_curve_losses = []
    eval_curve_epochs = []
    eval_curve_oas = []

    last_eval_pack = None

    # --------------------------
    # Eval core
    # --------------------------
    @torch.no_grad()
    def run_fullscene_prediction_cpu():
        model.eval()

        pred_flat_cpu = torch.empty((Hpad * Wpad, class_num), dtype=torch.float32, device=cpu_device)

        tile_ptr = 0
        for (batch_data,) in test_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            Bsz = batch_data.size(0)

            batch_logits = model(batch_data)
            batch_logits_cpu = batch_logits.float().cpu()

            for b in range(Bsz):
                k = tile_ptr + b
                i = k // num_W
                j = k % num_W
                y0 = i * img_size
                x0 = j * img_size
                offset = int(y0 * Wpad + x0)
                idx = tile_base_flat + offset
                pred_flat_cpu[idx, :] = batch_logits_cpu[b, :, :]

            tile_ptr += Bsz

        y_orgin_cpu = utils.image_reshape(
            pred_flat_cpu, Hpad, Wpad, H0, W0, class_num
        )
        if isinstance(y_orgin_cpu, np.ndarray):
            y_orgin_cpu = torch.from_numpy(y_orgin_cpu).float()
        y_orgin_cpu = y_orgin_cpu.to(cpu_device)

        pred_map_all = torch.argmax(y_orgin_cpu, dim=-1).cpu().numpy().astype(np.int64) + 1
        logits_all = y_orgin_cpu.cpu().numpy().astype(np.float32)

        model.train()
        return logits_all, pred_map_all

    @torch.no_grad()
    def evaluate_and_append(epoch_num: int, return_pack: bool = False):
        logits_all, pred_map_all = run_fullscene_prediction_cpu()

        overall_acc, _, average_acc, kappa, each_acc = utils.evaluate_performance_all(
            torch.from_numpy(logits_all).float().to(cpu_device),
            test_samples_gt_cpu,
            test_gt_onehot_cpu,
            H0,
            W0,
            class_num,
            test_gt,
            cpu_device,
            require_AA_KPP=True,
            printFlag=False,
        )

        print(f"[Eval @ epoch {epoch_num}] test OA={overall_acc:.4f} | kappa={kappa} | AA={average_acc}")

        gt_flat_local = data_gt.reshape(-1)
        pred_flat_local = pred_map_all.reshape(-1)

        test_gt_vec = gt_flat_local[test_index] - 1
        test_pred_vec = pred_flat_local[test_index] - 1

        cm = confusion_matrix(
            test_gt_vec,
            test_pred_vec,
            labels=list(range(class_num)),
        )
        each_acc_np, average_acc_np = compute_each_acc_from_cm(cm)
        kappa_np = cohen_kappa_score(test_gt_vec, test_pred_vec)
        oa_np = accuracy_score(test_gt_vec, test_pred_vec)

        append_text(
            RESULT_FILE,
            f"epoch={epoch_num}\n"
            f"seed={args.seed}\n"
            f"test_OA={oa_np}\n"
            f"kappa={kappa_np}\n"
            f"average_acc={average_acc_np}\n"
            f"each_acc=[" + ", ".join([f"{v:.6f}" for v in each_acc_np]) + "]\n"
            + "-" * 40 + "\n",
        )

        pack = {
            "epoch": int(epoch_num),
            "oa": float(oa_np),
            "aa": float(average_acc_np),
            "kappa": float(kappa_np),
            "each_acc": each_acc_np.astype(np.float64),
            "cm": cm.astype(np.int64),
            "pred_map_all": pred_map_all.astype(np.int64),
            "logits_all": logits_all.astype(np.float32),
        }

        if return_pack:
            return pack
        return float(oa_np)

    @torch.no_grad()
    def save_spectral_attention_visualization(epoch_num: int):
        if args.disable_hsba:
            print("[SPEC-ATTN] skip: HSBA is disabled")
            return

        if not args.use_spectral:
            print("[SPEC-ATTN] skip: --use_spectral is off")
            return

        model_was_training = model.training
        model.eval()

        tile_idx = int(np.clip(args.spec_attn_tile_index, 0, num_tiles - 1))
        tile_tensor = img_tiles_t[tile_idx:tile_idx + 1].to(device, non_blocking=True)

        py = args.spec_attn_pixel_y if args.spec_attn_pixel_y >= 0 else (img_size // 2)
        px = args.spec_attn_pixel_x if args.spec_attn_pixel_x >= 0 else (img_size // 2)
        py = int(np.clip(py, 0, img_size - 1))
        px = int(np.clip(px, 0, img_size - 1))

        attn_heads = model.backbone.spectral_adapter.extract_attention_map_for_pixel(
            tile_tensor,
            pixel_y=py,
            pixel_x=px,
            block_idx=args.spec_attn_block_idx,
            average_attn_weights=False,
        )

        attn_heads_np = attn_heads.numpy().astype(np.float32)
        attn_mean_np = attn_heads_np.mean(axis=0)

        L = attn_mean_np.shape[0]
        q_band = args.spec_attn_query_band_idx if args.spec_attn_query_band_idx >= 0 else (L // 2)
        q_band = int(np.clip(q_band, 0, L - 1))

        prefix = (
            f"epoch{epoch_num:03d}"
            f"_tile{tile_idx:04d}"
            f"_py{py:02d}_px{px:02d}"
            f"_blk{args.spec_attn_block_idx}"
        )

        raw_pt_path = os.path.join(SPEC_ATTN_DIR, f"{prefix}_raw_attention.pt")
        mean_npy_path = os.path.join(SPEC_ATTN_DIR, f"{prefix}_mean_attention.npy")
        mean_png_path = os.path.join(SPEC_ATTN_DIR, f"{prefix}_mean_attention.png")
        profile_png_path = os.path.join(SPEC_ATTN_DIR, f"{prefix}_queryband{q_band}_profile.png")

        torch.save(attn_heads, raw_pt_path)
        np.save(mean_npy_path, attn_mean_np)

        save_spectral_attention_heatmap(
            attn_mean_np,
            mean_png_path,
            title=(
                f"{args.dataset} spectral self-attn | epoch={epoch_num} | "
                f"tile={tile_idx} | pixel=({py},{px}) | block={args.spec_attn_block_idx}"
            ),
        )

        save_spectral_attention_profile(
            attn_mean_np,
            query_band_idx=q_band,
            out_path=profile_png_path,
            title=(
                f"{args.dataset} attention profile | epoch={epoch_num} | "
                f"query band={q_band}"
            ),
        )

        save_multihead_spectral_attention(
            attn_heads_np,
            SPEC_ATTN_DIR,
            prefix=prefix,
        )

        append_text(
            RESULT_FILE,
            "[SPECTRAL ATTENTION VIS]\n"
            f"epoch={epoch_num}\n"
            f"tile_index={tile_idx}\n"
            f"pixel_y={py}\n"
            f"pixel_x={px}\n"
            f"block_idx={args.spec_attn_block_idx}\n"
            f"raw_attention_pt={raw_pt_path}\n"
            f"mean_attention_npy={mean_npy_path}\n"
            f"mean_attention_png={mean_png_path}\n"
            f"profile_png={profile_png_path}\n"
            + "-" * 40 + "\n",
        )

        if model_was_training:
            model.train()

    def save_visualizations(eval_pack):
        gt_map = data_gt.astype(np.int64)
        pred_map_all = eval_pack["pred_map_all"].astype(np.int64)

        gt_flat_local = gt_map.reshape(-1)
        pred_flat_local = pred_map_all.reshape(-1)

        test_ids = test_index.astype(np.int64)
        labeled_ids = np.where(gt_flat_local > 0)[0].astype(np.int64)

        pred_test_only_map = np.zeros((H0 * W0,), dtype=np.int64)
        pred_test_only_map[test_ids] = pred_flat_local[test_ids]
        pred_test_only_map = pred_test_only_map.reshape(H0, W0)

        pred_labeled_map = np.zeros((H0 * W0,), dtype=np.int64)
        pred_labeled_map[labeled_ids] = pred_flat_local[labeled_ids]
        pred_labeled_map = pred_labeled_map.reshape(H0, W0)

        labeled_err_map = build_error_map_array(gt_map, pred_labeled_map, labeled_ids)

        class_names = [str(i + 1) for i in range(class_num)]

        save_training_curve(
            train_curve_epochs,
            train_curve_losses,
            eval_curve_epochs,
            eval_curve_oas,
            os.path.join(VIS_DIR, f"{dataset_tag}_training_curve.png"),
        )

        save_label_map(
            gt_map,
            os.path.join(VIS_DIR, f"{dataset_tag}_gt.png"),
            title=f"{args.dataset} Ground Truth",
        )

        save_label_map(
            pred_test_only_map,
            os.path.join(VIS_DIR, f"{dataset_tag}_pred_test_only.png"),
            title=f"{args.dataset} Prediction (Test pixels only)",
        )

        save_error_map(
            gt_map=gt_map,
            pred_map=pred_test_only_map,
            eval_ids=test_ids,
            H=H0,
            W=W0,
            out_path=os.path.join(VIS_DIR, f"{dataset_tag}_error_test_only.png"),
            title=f"{args.dataset} Error Map (Test pixels only)",
        )

        save_label_map(
            pred_labeled_map,
            os.path.join(VIS_DIR, f"{dataset_tag}_pred_labeled.png"),
            title=f"{args.dataset} Prediction (Labeled pixels)",
        )

        save_error_map(
            gt_map=gt_map,
            pred_map=pred_labeled_map,
            eval_ids=labeled_ids,
            H=H0,
            W=W0,
            out_path=os.path.join(VIS_DIR, f"{dataset_tag}_error_labeled.png"),
            title=f"{args.dataset} Error Map (Labeled pixels)",
        )

        save_confusion_matrix_fig(
            eval_pack["cm"],
            os.path.join(VIS_DIR, f"{dataset_tag}_confusion_matrix.png"),
            class_names=class_names,
            normalize=False,
            title=f"{args.dataset} Confusion Matrix",
        )

        save_confusion_matrix_fig(
            eval_pack["cm"],
            os.path.join(VIS_DIR, f"{dataset_tag}_confusion_matrix_norm.png"),
            class_names=class_names,
            normalize=True,
            title=f"{args.dataset} Confusion Matrix (Normalized)",
        )

        save_overview_figure(
            gt_map=gt_map,
            pred_map=pred_labeled_map,
            err_map=labeled_err_map,
            out_path=os.path.join(VIS_DIR, f"{dataset_tag}_overview.png"),
            suptitle=f"{args.dataset} - GT / Prediction / Error",
        )

        append_text(
            RESULT_FILE,
            "[VISUALIZATION FILES]\n"
            f"dir={VIS_DIR}\n"
            f"training_curve={os.path.join(VIS_DIR, f'{args.dataset}_training_curve.png')}\n"
            f"gt={os.path.join(VIS_DIR, f'{args.dataset}_gt.png')}\n"
            f"pred_test_only={os.path.join(VIS_DIR, f'{args.dataset}_pred_test_only.png')}\n"
            f"error_test_only={os.path.join(VIS_DIR, f'{args.dataset}_error_test_only.png')}\n"
            f"pred_labeled={os.path.join(VIS_DIR, f'{args.dataset}_pred_labeled.png')}\n"
            f"error_labeled={os.path.join(VIS_DIR, f'{args.dataset}_error_labeled.png')}\n"
            f"cm={os.path.join(VIS_DIR, f'{args.dataset}_confusion_matrix.png')}\n"
            f"cm_norm={os.path.join(VIS_DIR, f'{args.dataset}_confusion_matrix_norm.png')}\n"
            f"overview={os.path.join(VIS_DIR, f'{args.dataset}_overview.png')}\n"
            + "-" * 40 + "\n",
        )

    # --------------------------
    # Training loop
    # --------------------------
    print("[TRAIN] start")
    global_step = 0

    if args.save_spec_attn_vis and (0 in spec_attn_epochs):
        print("[SPEC-ATTN] saving epoch 0 attention...")
        save_spectral_attention_visualization(epoch_num=0)

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss_sum = 0.0
        epoch_tr_pixels = 0

        global_tile_idx = 0

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device, non_blocking=True)
            Bsz = batch_data.size(0)

            y0_list = []
            x0_list = []
            for b in range(Bsz):
                k = global_tile_idx + b
                i = k // num_W
                j = k % num_W
                y0_list.append(i * img_size)
                x0_list.append(j * img_size)

            tr_masks = []
            labels = []
            has_tr = False

            for b in range(Bsz):
                y0, x0 = y0_list[b], x0_list[b]
                tr_tile = TR_pad[y0:y0 + img_size, x0:x0 + img_size]
                lb_tile = label_pad0[y0:y0 + img_size, x0:x0 + img_size]

                m = tr_tile > 0
                tr_masks.append(m.reshape(-1))
                labels.append(lb_tile.reshape(-1))
                if m.any():
                    has_tr = True

            global_tile_idx += Bsz
            if not has_tr:
                continue

            tr_masks_np = np.stack(tr_masks, axis=0)
            labels_np = np.stack(labels, axis=0)

            tr_mask_flat = tr_masks_np.reshape(-1)
            tr_idx = np.where(tr_mask_flat)[0].astype(np.int64)
            if tr_idx.size == 0:
                continue

            logits = model(batch_data).reshape(Bsz * P, class_num)

            idx_t = torch.from_numpy(tr_idx).to(device)
            logits_tr = logits.index_select(0, idx_t)

            labels_flat = labels_np.reshape(-1)[tr_idx]
            labels_tr = torch.from_numpy(labels_flat).long().to(device)

            loss = criterion(logits_tr, labels_tr)
            (loss / float(accum_steps)).backward()

            epoch_loss_sum += float(loss.item()) * int(tr_idx.size)
            epoch_tr_pixels += int(tr_idx.size)

            global_step += 1
            if global_step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if global_step % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        avg_loss = epoch_loss_sum / max(1, epoch_tr_pixels)
        print(f"[Epoch {epoch:03d}] train loss={avg_loss:.6f} (per TR-pixel) | TR_pixels={epoch_tr_pixels}")

        train_curve_epochs.append(int(epoch))
        train_curve_losses.append(float(avg_loss))

        if args.log_train_each_epoch == 1:
            append_text(
                TRAIN_LOG,
                f"epoch={epoch}\tseed={args.seed}\ttrain_loss={avg_loss:.8f}\tTR_pixels={epoch_tr_pixels}\n",
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"[BEST] best loss updated: {best_loss:.6f} (saving skipped)")
            append_text(
                TRAIN_LOG,
                f"[BEST]\tepoch={epoch}\tseed={args.seed}\tbest_loss={best_loss:.8f}\tpath=SKIP_SAVE\n",
            )

        if epoch in eval_epochs:
            pack = evaluate_and_append(epoch, return_pack=True)
            eval_curve_epochs.append(int(epoch))
            eval_curve_oas.append(float(pack["oa"]))
            last_eval_pack = pack

        if args.save_spec_attn_vis and (epoch in spec_attn_epochs):
            print(f"[SPEC-ATTN] saving attention at epoch {epoch} ...")
            save_spectral_attention_visualization(epoch_num=epoch)

        if device.type == "cuda" and (epoch % 10 == 0):
            torch.cuda.empty_cache()

    # --------------------------
    # Final visualization
    # --------------------------
    if args.save_vis:
        if args.vis_use_last_eval and (last_eval_pack is not None):
            final_pack = last_eval_pack
            print(f"[VIS] using last scheduled eval result at epoch {final_pack['epoch']}")
        else:
            print("[VIS] running final full-scene evaluation for visualization ...")
            final_pack = evaluate_and_append(args.max_epoch, return_pack=True)
            if len(eval_curve_epochs) == 0 or eval_curve_epochs[-1] != args.max_epoch:
                eval_curve_epochs.append(int(args.max_epoch))
                eval_curve_oas.append(float(final_pack["oa"]))

        save_visualizations(final_pack)

    # --------------------------
    # Batch=1 latency measurement
    # --------------------------
    if args.measure_latency:
        tile_idx = int(np.clip(args.latency_tile_index, 0, num_tiles - 1))
        sample_b1 = img_tiles_t[tile_idx:tile_idx + 1]

        print(
            f"[LATENCY] measuring batch=1 inference latency | "
            f"tile_index={tile_idx} | warmup={args.latency_warmup} | iters={args.latency_iters}"
        )

        latency_pack = measure_batch1_latency(
            model=model,
            sample_b1=sample_b1,
            device=device,
            warmup=args.latency_warmup,
            iters=args.latency_iters,
        )

        latency_msg = (
            "[LATENCY - BATCH 1 FORWARD]\n"
            f"device={device}\n"
            f"tile_index={tile_idx}\n"
            f"input_shape={tuple(sample_b1.shape)}\n"
            f"warmup={args.latency_warmup}\n"
            f"iters={latency_pack['num_iters']}\n"
            f"mean_ms={latency_pack['mean_ms']:.6f}\n"
            f"std_ms={latency_pack['std_ms']:.6f}\n"
            f"min_ms={latency_pack['min_ms']:.6f}\n"
            f"max_ms={latency_pack['max_ms']:.6f}\n"
            f"p50_ms={latency_pack['p50_ms']:.6f}\n"
            f"p95_ms={latency_pack['p95_ms']:.6f}\n"
        )

        print(latency_msg)
        append_text(RESULT_FILE, latency_msg + "-" * 40 + "\n")
        append_text(TRAIN_LOG, latency_msg + "-" * 40 + "\n")

    append_text(RESULT_FILE, "=== RUN END ===\n")
    append_text(TRAIN_LOG, "=== TRAIN LOG END ===\n")

    print(f"[DONE] results saved at: {RESULT_FILE}")
    print(f"[DONE] train log saved at: {TRAIN_LOG}")
    if args.save_vis:
        print(f"[DONE] visualizations saved at: {VIS_DIR}")
    if args.save_spec_attn_vis and (not args.disable_hsba):
        print(f"[DONE] spectral attention saved at: {SPEC_ATTN_DIR}")
    print("[DONE] weights saving: DISABLED")


if __name__ == "__main__":
    main()
