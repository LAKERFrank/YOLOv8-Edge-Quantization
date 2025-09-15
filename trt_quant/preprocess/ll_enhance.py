from __future__ import annotations

# -*- coding: utf-8 -*-
"""
灰階 1ch 前處理：Gamma + CLAHE（可開關）
提供給 TensorRT 路徑：回傳 float32 NCHW [1,1,H,W]，值域 0~1
"""
import argparse
from typing import Optional
try:
    import cv2 as cv
except Exception:  # pragma: no cover
    cv = None  # type: ignore
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def add_ll_flags(ap: argparse.ArgumentParser):
    ap.add_argument("--ll-enhance", action="store_true",
                    help="啟用低亮度強化 (Gamma + CLAHE)")
    ap.add_argument("--ll-gamma", type=float, default=0.85,
                    help="gamma < 1 會提亮暗部；=1 不變")
    ap.add_argument(
        "--ll-clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否啟用 CLAHE",
    )
    ap.add_argument("--ll-clip", type=float, default=2.0,
                    help="CLAHE clipLimit")
    ap.add_argument("--ll-grid", type=int, default=8,
                    help="CLAHE tileGridSize 的邊長")


def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    """
    將輸入統一為灰階 uint8: (H, W)
    支援 BGR/Gray、uint8/float32/float64
    """
    if img is None:
        raise ValueError("ensure_gray_u8: input is None")
    if cv is None:
        raise ImportError("opencv-python is required for ensure_gray_u8")
    x = img
    if x.ndim == 3 and x.shape[2] == 3:
        x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
    if x.dtype in (np.float32, np.float64):
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return x


def ll_enhance_u8(gray_u8: np.ndarray,
                  gamma: Optional[float] = 0.85,
                  use_clahe: bool = True,
                  clahe_clip: float = 2.0,
                  clahe_grid: int = 8) -> np.ndarray:
    """
    回傳增強後的灰階 uint8 (H, W)
    1) gamma 校正（<1 提亮暗部）
    2) CLAHE 局部對比增強
    """
    if np is None:
        raise ImportError("numpy is required for ll_enhance_u8")
    g = ensure_gray_u8(gray_u8).astype(np.float32) / 255.0
    if gamma is not None:
        g = np.power(g, float(gamma))
    g = (g * 255.0).clip(0, 255).astype(np.uint8)

    if use_clahe:
        if cv is None:
            raise ImportError("opencv-python is required for CLAHE")
        clahe = cv.createCLAHE(clipLimit=float(clahe_clip),
                               tileGridSize=(int(clahe_grid), int(clahe_grid)))
        g = clahe.apply(g)
    return g


def _to_nchw_float01(gray_u8: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    轉成 float32 NCHW [1,1,H,W]，0~1
    """
    if np is None:
        raise ImportError("numpy is required for _to_nchw_float01")
    g = ensure_gray_u8(gray_u8).astype(np.float32)
    if normalize:
        g /= 255.0
    return g[None, None, ...]  # [1,1,H,W]


def preprocess_for_trt(img: np.ndarray,
                       enable: bool,
                       gamma: float,
                       use_clahe: bool,
                       clahe_clip: float,
                       clahe_grid: int) -> np.ndarray:
    """
    提供給 predict_trt.py：一行搞定前處理 + 打包成 NCHW tensor
    """
    if np is None:
        raise ImportError("numpy is required for preprocess_for_trt")
    g = ensure_gray_u8(img)
    if enable:
        g = ll_enhance_u8(g, gamma, use_clahe, clahe_clip, clahe_grid)
    return _to_nchw_float01(g, normalize=True)
