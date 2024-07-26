import cv2
import numpy as np
import torch
from torch_geometric.data import Data


def extract_contours(img):
    # ノイズ除去のためのぼかし処理
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 輪郭の検出
    contours, _ = cv2.findContours(cv2.bitwise_not(blurred), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    epsilon_factor = 0.001
    simplified_contours = []
    for contour in contours:
        # 輪郭の周長を計算
        perimeter = cv2.arcLength(contour, True)

        # 周長に基づいて epsilon を設定
        epsilon = epsilon_factor * perimeter

        # 輪郭を近似（簡略化）
        approx = cv2.approxPolyDP(contour, epsilon, True)

        simplified_contours.append(approx)
    return contours


def draw_contours(img, contours):
    # 元の画像に輪郭を描画
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # 輪郭の点を赤い点で表示
    for contour in contours:
        for point in contour:
            x, y = point[0]
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

    # 結果の表示
    cv2.imshow('Contours and Points', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


def create_graph_from_contour(img_path, label):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    contours = extract_contours(img)

    # 輪郭点を収集
    all_points = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            all_points.append([x, y])

    # NumPy配列に変換
    all_points = np.array(all_points)

    # 最小座標と最大座標を計算
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    # 正規化された点を格納するリスト
    normalized_points = []

    # 各点を正規化
    for point in all_points:
        x, y = point
        # 0除算を避けるためのチェック
        normalized_x = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
        normalized_y = (y - min_y) / (max_y - min_y) if max_y != min_y else 0.5
        normalized_points.append([normalized_x, normalized_y])

    x = torch.tensor(normalized_points, dtype=torch.float)
    edge_index = []
    for i in range(len(normalized_points)):
        # (i + 1) % len(all_points) 最後の点のとき0になる（最初と最後を繋げる）
        edge_index.append([i, (i + 1) % len(normalized_points)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))