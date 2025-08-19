

import torch


def non_max_suppression(pred_conf, pred_x, pred_y, conf_threshold=0.5, dis_tolerance=10, stride=8):
        """
        Apply non-maximum suppression (NMS) to filter out overlapping balls based on distance.
        
        Args:
        - pred_conf (torch.Tensor): 80x80 confidence tensor.
        - pred_x (torch.Tensor): 80x80 x-coordinate tensor.
        - pred_y (torch.Tensor): 80x80 y-coordinate tensor.
        - conf_threshold (float): Confidence threshold.
        - dis_tolerance (float): Minimum allowable distance between two detections.
        
        Returns:
        - keep (list): List of (x, y, confidence) tuples for retained detections.
        """
        # 1. 篩選置信度 >= conf_threshold 的 cell
        mask = pred_conf >= conf_threshold
        conf_values = pred_conf[mask]
        indices = torch.nonzero(mask)

        # 2. 依照置信度降序排序
        sorted_indices = torch.argsort(conf_values, descending=True)
        sorted_conf_values = conf_values[sorted_indices]
        sorted_positions = indices[sorted_indices]

        # 3. 初始化結果列表
        keep = []
        result = []
        center = stride/2
        # 4. 應用距離檢查的 NMS
        for i in range(len(sorted_conf_values)):

            y_coordinates, x_coordinates = sorted_positions[i].tolist()
            x1 = center*stride - pred_x[y_coordinates][x_coordinates][0]+pred_x[y_coordinates][x_coordinates][1]
            y1 = center*stride - pred_y[y_coordinates][x_coordinates][0]+pred_y[y_coordinates][x_coordinates][1]
            conf = sorted_conf_values[i].item()

            x_coordinates *= stride
            y_coordinates *= stride
            current_x = x_coordinates+x1
            current_y = y_coordinates+y1

            # 只保留與所有已保留框距離大於 dis_tolerance 的框
            is_far_enough = True
            for x2, y2, _ in keep:
                distance = torch.sqrt((current_x - x2) ** 2 + (current_y - y2) ** 2)
                if distance < dis_tolerance:
                    is_far_enough = False
                    break

            if is_far_enough:
                keep.append((current_x, current_y, conf))
                result.append((x_coordinates/stride, y_coordinates/stride, conf))

        return result