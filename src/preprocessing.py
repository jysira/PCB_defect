from __future__ import annotations

import cv2
import numpy as np


def apply_preprocessing(bgr: np.ndarray, method: str = "clahe") -> np.ndarray:
    """工业图像预处理；输入输出均为 BGR uint8。"""
    if method == "none":
        return bgr
    if method == "gaussian":
        return cv2.GaussianBlur(bgr, (5, 5), 0)
    if method == "median":
        return cv2.medianBlur(bgr, 5)
    if method == "bilateral":
        # 轻度保边平滑；保持图像尺寸及 BGR 通道不变。
        return cv2.bilateralFilter(bgr, d=5, sigmaColor=20, sigmaSpace=3)
    if method == "clahe":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if method == "canny":
        result = cv2.Canny(gray, 50, 150)
    elif method == "threshold":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "morphology":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        raise ValueError(f"未知预处理方法: {method}")
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
