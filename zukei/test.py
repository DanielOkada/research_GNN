import numpy as np

from preprocess import *

img = cv2.imread("data/ori_star.png", cv2.IMREAD_GRAYSCALE)
contours = extract_contours(img)

epsilon_factor=0.001
simplified_contours = []
for contour in contours:
    # 輪郭の周長を計算
    perimeter = cv2.arcLength(contour, True)

    # 周長に基づいて epsilon を設定
    epsilon = epsilon_factor * perimeter

    # 輪郭を近似（簡略化）
    approx = cv2.approxPolyDP(contour, epsilon, True)

    simplified_contours.append(approx)
result = draw_contours(np.ones_like(img) * 255, simplified_contours)
# cv2.imwrite("c_g.png", result)