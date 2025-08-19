from datetime import datetime
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.utils.nms import non_max_suppression
from ultralytics.tracknet.utils.plotting import display_predict_image
from ultralytics.tracknet.utils.transform import calculate_angle, calculate_dist, target_grid
from ultralytics.tracknet.val_dataset import TrackNetValDataset
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.metrics import DetMetrics
from sklearn.metrics import confusion_matrix
from collections import deque

class TrackNetValidatorV3(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)

        if self.args.half and self.device.type == "cuda":
            batch['img'] = batch['img'].half() / 255.0  # `float16`
        else:
            batch['img'] = batch['img'].float() / 255.0  # `float32`

        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.
        self.stride = 32
        self.num_groups = 10

        self.total_loss = 0.0
        self.num_samples = 0
        self.conf_TP = 0
        self.conf_TN = 0
        self.conf_FP = 0
        self.conf_FN = 0
        self.conf_acc = 0
        self.conf_precision = 0
        self.pos_TP = 0
        self.pos_TN = 0
        self.pos_FP = 0
        self.pos_FN = 0
        self.pos_acc = 0
        self.pos_precision = 0
        self.ball_count = 0
        self.pred_ball_count = 0
        device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_max = 16
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.no = 35
        self.feat_no = 2
        self.nc = 1
        self.dxdy_no = 2
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_size = preds.shape[0]
        if preds.shape == (350, 20, 20):
            self.update_metrics_once(0, preds, batch_target[0])
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx])
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target):
        # pred = [330 * 20 * 20]
        # batch_target = [10*7]
        feats = pred
        pred_distri, pred_scores, pred_dxdy = feats.view(self.no, -1).split(
            (self.reg_max * self.feat_no, self.nc, self.dxdy_no), 0)
        
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()
        pred_dxdy = pred_dxdy.permute(1, 0).contiguous()
        pred_dxdy = torch.tanh(pred_dxdy)

        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*20*20]
        
        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, self.feat_no, c // self.feat_no).softmax(2).matmul(
            self.proj.type(pred_distri.dtype))
        
        target_pos_distri = torch.zeros(self.num_groups, 20, 20, self.feat_no, device=self.device)
        mask_has_ball = torch.zeros(self.num_groups, 20, 20, device=self.device)
        cls_targets = torch.zeros(self.num_groups, 20, 20, 1, device=self.device)
        mask_has_next_ball = torch.zeros(self.num_groups, 20, 20, device=self.device)
        target_mov = torch.zeros(self.num_groups, 20, 20, 2, device=self.device)

        for target_idx, target in enumerate(batch_target):
            if target[1] == 1:
                # xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], self.stride)
                mask_has_ball[target_idx, grid_y, grid_x] = 1
                
                target_pos_distri[target_idx, grid_y, grid_x, 0] = offset_x*(self.reg_max-1)/self.stride
                target_pos_distri[target_idx, grid_y, grid_x, 1] = offset_y*(self.reg_max-1)/self.stride

                ## cls
                cls_targets[target_idx, grid_y, grid_x, 0] = 1

                if target_idx != len(batch_target)-1 and batch_target[target_idx+1][1] == 1:
                        mask_has_next_ball[target_idx, grid_y, grid_x] = 1
                        target_mov[target_idx, grid_y, grid_x, 0] = target[4]/640
                        target_mov[target_idx, grid_y, grid_x, 1] = target[5]/640
        
        target_pos_distri = target_pos_distri.view(self.num_groups*20*20, self.feat_no)
        cls_targets = cls_targets.view(self.num_groups*20*20, 1)
        mask_has_ball = mask_has_ball.view(self.num_groups*20*20).bool()
        target_mov = target_mov.view(self.num_groups*20*20, 2)
        mask_has_next_ball = mask_has_next_ball.view(self.num_groups*20*20).bool()

        # 計算 conf 的 confusion matrix
        threshold = 0.8
        pred_binary = (pred_probs >= threshold)
        self.pred_ball_count += pred_binary.int().sum()

        unique_classes = torch.unique(cls_targets.bool())
        if len(unique_classes) == 1:
            if unique_classes.item() == 1:
                # All targets are 1 (positive class)
                self.conf_TP += (pred_binary == 1).sum().item()  # Count of true positives
                self.conf_FN += (pred_binary == 0).sum().item()  # Count of false negatives
                self.conf_TN += 0  # No true negatives
                self.conf_FP += 0  # No false positives
            else:
                # All targets are 0 (negative class)
                self.conf_TN += (pred_binary == 0).sum().item()  # Count of true negatives
                self.conf_FP += (pred_binary == 1).sum().item()  # Count of false positives
                self.conf_TP += 0  # No true positives
                self.conf_FN += 0  # No false negatives
        else:
            # Compute confusion matrix normally
            conf_matrix = confusion_matrix(cls_targets.bool().cpu().numpy(), pred_binary.cpu().numpy())
            self.conf_TN += conf_matrix[0][0]
            self.conf_FP += conf_matrix[0][1]
            self.conf_FN += conf_matrix[1][0]
            self.conf_TP += conf_matrix[1][1]

        # 計算 x, y 的 confusion matrix
        pred_tensor = pred_pos[mask_has_ball]
        ground_truth_tensor = target_pos_distri[mask_has_ball]
        ball_count = mask_has_ball.sum()
        self.ball_count += ball_count
        

        tolerance = 1
        x_tensor_correct = (torch.abs(pred_tensor[:, 0] - ground_truth_tensor[:, 0]) <= tolerance).int()
        y_tensor_correct = (torch.abs(pred_tensor[:, 1] - ground_truth_tensor[:, 1]) <= tolerance).int()

        tensor_combined_correct = (x_tensor_correct & y_tensor_correct).int()

        ground_truth_binary_tensor = torch.ones(ball_count).int()

        unique_classes = torch.unique(ground_truth_binary_tensor)
        if ball_count == 0:
            pass
        elif len(unique_classes) == 1:
            if unique_classes.item() == 1:
                # All targets are 1 (positive class)
                self.pos_TP += (tensor_combined_correct == 1).sum().item()  # Count of true positives
                self.pos_FN += (tensor_combined_correct == 0).sum().item()  # Count of false negatives
                self.pos_TN += 0  # No true negatives
                self.pos_FP += 0  # No false positives
            else:
                # All targets are 0 (negative class)
                self.pos_TN += (tensor_combined_correct == 0).sum().item()  # Count of true negatives
                self.pos_FP += (tensor_combined_correct == 1).sum().item()  # Count of false positives
                self.pos_TP += 0  # No true positives
                self.pos_FN += 0  # No false negatives
        else:
            # Compute confusion matrix normally
            pos_matrix = confusion_matrix(ground_truth_binary_tensor.cpu().numpy(), tensor_combined_correct.cpu().numpy())
            self.pos_TN += pos_matrix[0][0]
            self.pos_FP += pos_matrix[0][1]
            self.pos_FN += pos_matrix[1][0]
            self.pos_TP += pos_matrix[1][1]

        
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        if (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP) != 0:
            self.pos_acc = (self.pos_TN + self.pos_TP) / (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP)
        if (self.conf_FN+self.conf_FP+self.conf_TN + self.conf_TP) != 0:
            self.conf_acc = (self.conf_TN + self.conf_TP) / (self.conf_FN+self.conf_FP+self.conf_TN + self.conf_TP)
        if (self.conf_TP+self.conf_FP) != 0:
            self.conf_precision = self.conf_TP/(self.conf_TP+self.conf_FP)
        if (self.pos_TP+self.pos_FP) != 0:
            self.pos_precision = self.pos_TP/(self.pos_TP+self.pos_FP)

    def get_stats(self):
        """Return the stats."""
        return {'pos_FN': self.pos_FN, 'pos_FP': self.pos_FP, 'pos_TN': self.pos_TN, 
                'pos_TP': self.pos_TP, 'pos_acc': self.pos_acc, 'pos_precision': self.pos_precision,
                'conf_FN': self.conf_FN, 'conf_FP': self.conf_FP, 'conf_TN': self.conf_TN, 
                'conf_TP': self.conf_TP, 'conf_acc': self.conf_acc, 'conf_precision': self.conf_precision,
                'threshold>0.8 rate':self.pred_ball_count/self.ball_count}
    
    def print_results(self):
        """Print the results."""
        # precision = 0
        # recall = 0
        # f1 = 0
        # if self.TP > 0:
        #     precision = self.TP/(self.TP+self.FP)
        #     recall = self.TP/(self.TP+self.FN)
        #     f1 = (2*precision*recall)/(precision+recall)
        # print(f"Validation Accuracy: {self.acc:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, , Validation F1-Score: {f1:.4f}")

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"

# use original input image and output predict result as csv file
class TrackNetValidatorV4(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetValDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255

        # if self.args.half and self.device.type == "cuda":
        #     batch['img'] = batch['img'].half() / 255.0  # `float16`
        # else:
        #     batch['img'] = batch['img'].float() / 255.0  # `float32`

        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.

        # TODO val 時，stride 取得異常
        if isinstance(model.stride, torch.Tensor):
            self.stride = model.stride[0]
        else:
            self.stride = model.model.stride[0]
        self.cell_num = int(640/self.stride)
        self.num_groups = 10

        self.total_loss = 0.0
        self.num_samples = 0
        self.conf_TP = 0
        self.conf_TN = 0
        self.conf_FP = 0
        self.conf_FN = 0
        self.conf_acc = 0
        self.conf_precision = 0
        self.pos_TP = 0
        self.pos_TN = 0
        self.pos_FP = 0
        self.pos_FN = 0
        self.pos_FP_dis = 0
        self.fast_TP = 0
        self.hit_TP = 0
        self.hit_FP = 0
        self.fast_FN = 0
        self.hit_FN = 0
        self.fast_hit_TP = 0
        self.fast_hit_FN = 0
        self.pos_acc = 0
        self.pos_precision = 0
        self.ball_count = 0
        self.pred_ball_count = 0
        device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_max = 16
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.feat_no = 8
        self.nc = 1
        self.no = 16*self.feat_no+self.nc

        self.fast_count = 0
        self.hit_count = 0
        self.fast_hit_count = 0

        self.target_hit_count = 0
        self.hitV2_TP = 0  # True Positives
        self.hitV2_FP = 0  # False Positives
        self.hitV2_TN = 0  # True Negatives
        self.hitV2_FN = 0  # False Negatives
        self.hitV1_TP = 0  # True Positives
        self.hitV1_FP = 0  # False Positives
        self.hitV1_TN = 0  # True Negatives
        self.hitV1_FN = 0  # False Negatives

        # 一顆球半徑 = 2 pixel (640*640)
        self.tolerance2 = 2.0 # 50% 距離容忍度
        self.tolerance3 = 3.0
        self.tolerance5 = 2.0
        self.conf_thresholds = [i * 0.05 for i in range(1, 20)]  # [0.5, 0.55, ..., 0.95]
        self.iou_dist_thresholds = [i * 1 for i in range(1, 6)]  # [1, 2, ..., 5]
        
        self.cumulative_TP = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_FP = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_FN = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_TN = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.fitness = 0

        self.frame_10_metrics = deque(maxlen=10)
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_img = batch['img']
        batch_img_file = batch['img_files']
        if preds.shape == (1290, self.cell_num, self.cell_num):
            self.update_metrics_once(0, preds, batch_target[0], batch_img[0], batch_img_file)
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx], batch_img[idx], batch_img_file)
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target, batch_img, batch_img_file):
        self.save_pred_results(batch_img_file, pred, batch_target)
        
    def save_pred_results(self, img_path, pred, target):
        # open img and get image shape
        img = cv2.imread(img_path[0][0])
        h, w = img.shape[:2] # y, x

        assert w > h, "w must be greater than h"
        x_n = w/640
        y_offset = (w - h)/2
            


        feats = pred.clone()
        pred_distri, pred_scores = feats.view(self.no, -1).split(
            (self.reg_max * self.feat_no, self.nc), 0)
        
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()

        pred_probs = torch.sigmoid(pred_scores)

        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, self.feat_no, c // self.feat_no).softmax(2).matmul(
            self.proj.type(pred_distri.dtype))
        
        each_probs = pred_probs.view(10, self.cell_num, self.cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, self.cell_num, self.cell_num, self.feat_no).split([2, 2, 2, 2], dim=3)

        for frame_idx in range(10):
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            center = self.stride/2
            metrics = []
            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]

            ############## MAX ##############
            conf_threshold = 0.7
            p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
            max_position = torch.argmax(p_conf_masked)
            # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
            max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
            max_conf = p_conf[max_y, max_x]
            
            ############# 多球 #############

            ### 只拿最大值
            preds = [(max_x, max_y, max_conf)]

            ### 拿多顆球
            # preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, dis_tolerance=30)

            for (x, y, conf) in preds:
                if len(metrics) > 5 :
                    break
                metric = {}
                x_coordinate = x*self.stride
                y_coordinate = y*self.stride
                
                x = x_coordinate + (center-p_cell_x[int(y)][int(x)][0]+p_cell_x[int(y)][int(x)][1]*self.stride)
                y = y_coordinate + (center-p_cell_y[int(y)][int(x)][0]+p_cell_y[int(y)][int(x)][1]*self.stride)
                nx = x_coordinate + (center-p_cell_nx[int(y)][int(x)][0]+p_cell_nx[int(y)][int(x)][1]*self.stride)
                ny = y_coordinate + (center-p_cell_ny[int(y)][int(x)][0]+p_cell_ny[int(y)][int(x)][1]*self.stride)
                
                metric["x"] = x * x_n
                metric["y"] = y + y_offset
                metric["conf"] = conf

                metric["nx"] = nx * x_n
                metric["ny"] = ny + y_offset

                metrics.append(metric)
                self.frame_10_metrics.append(metric)
            
            # 畫出來
            # for metric in metrics:
            #     cv2.circle(img, (int(metric["x"]), int(metric["y"])), 5, (0, 0, 255), -1)
            #     cv2.circle(img, (int(metric["nx"]), int(metric["ny"])), 5, (0, 255, 0), -1)

            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            # save to csv on self.save_dir, if csv not exist, create it
            csv_path = self.save_dir / "val_output.csv"
            if not csv_path.exists():
                with open(csv_path, "w") as f:
                    f.write("img_path,x,y,nx,ny,conf\n")

            with open(csv_path, "a") as f:
                f.write(f"{img_path[frame_idx][0]},{metrics[0]['x']},{metrics[0]['y']},{metrics[0]['nx']},{metrics[0]['ny']},{metrics[0]['conf']}\n")


# stable version
class TrackNetValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetValDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255

        # if self.args.half and self.device.type == "cuda":
        #     batch['img'] = batch['img'].half() / 255.0  # `float16`
        # else:
        #     batch['img'] = batch['img'].float() / 255.0  # `float32`

        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.

        # TODO val 時，stride 取得異常
        if isinstance(model.stride, torch.Tensor):
            self.stride = model.stride[0]
        else:
            self.stride = model.model.stride[0]
        self.cell_num = int(640/self.stride)
        self.num_groups = 10

        self.total_loss = 0.0
        self.num_samples = 0
        self.conf_precision = 0
        self.pos_TP = 0
        self.pos_TN = 0
        self.pos_FP = 0
        self.pos_FN = 0
        self.pos_FP_dis = 0
        self.pos_acc = 0
        self.pos_precision = 0
        self.ball_count = 0
        self.pred_ball_count = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_max = 16
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.feat_no = 8
        self.nc = 1
        self.no = 16*self.feat_no+self.nc

        # 一顆球半徑 = 2 pixel (640*640)
        self.tolerance2 = 2.0 # 50% 距離容忍度
        self.tolerance3 = 3.0
        self.tolerance4 = 4.0
        self.tolerance5 = 5.0
        self.conf_thresholds = [i * 0.05 for i in range(1, 20)]  # [0.5, 0.55, ..., 0.95]
        self.iou_dist_thresholds = [i * 1 for i in range(1, 6)]  # [1, 2, ..., 5]
        
        self.cumulative_TP = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_FP = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_FN = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_TN = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.fitness = 0
        self.avg_ap = 0

        self.frame_10_metrics = deque(maxlen=10)
    
    def update_metrics(self, preds, batch, loss):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[1][0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_img = batch['img']
        batch_img_file = batch['img_files']
        if len(preds.shape) == 3:
            self.update_metrics_once(0, preds, batch_target[0], batch_img[0], loss)
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx], batch_img[idx], loss)
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target, batch_img, loss):
        # pred = [330 * self.cell_num * self.cell_num]
        # batch_target = [10*7]
        feats = pred.clone()
        pred_distri, pred_scores = feats.view(self.no, -1).split(
            (self.reg_max * self.feat_no, self.nc), 0)
        
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()

        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*self.cell_num*self.cell_num]
        
        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, self.feat_no, c // self.feat_no).softmax(2).matmul(
            self.proj.type(pred_distri.dtype))
        
        mask_has_ball = torch.zeros(self.num_groups, self.cell_num, self.cell_num, device=self.device)
        cls_targets = torch.zeros(self.num_groups, self.cell_num, self.cell_num, 1, device=self.device)
        
        for target_idx, target in enumerate(batch_target):
            grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], self.stride)
            if grid_x >= 80:
                print(grid_x, grid_y, offset_x, offset_y)
                grid_x = 79
            if grid_y >= 80:
                print(grid_x, grid_y, offset_x, offset_y)
                grid_y = 79

            if target[1] == 1:
                # xy
                mask_has_ball[target_idx, grid_y, grid_x] = 1

                ## cls
                cls_targets[target_idx, grid_y, grid_x, 0] = 1
        
        cls_targets = cls_targets.view(self.num_groups*self.cell_num*self.cell_num, 1)
        mask_has_ball = mask_has_ball.view(self.num_groups*self.cell_num*self.cell_num).bool()

        each_probs = pred_probs.view(10, self.cell_num, self.cell_num)
        each_pos_x, each_pos_y, each_pos_nx, each_pos_ny = pred_pos.view(10, self.cell_num, self.cell_num, self.feat_no).split([2, 2, 2, 2], dim=3)

        # 計算 hit v2 效果
        # 先填充 hit 前後兩幀
        frame_idx = 0
        while frame_idx < 10:
            if batch_target[frame_idx][6] == 1:
                # 檢查並設定範圍內的相鄰元素
                if frame_idx - 2 >= 0:
                    batch_target[frame_idx - 2][6] = 1
                if frame_idx - 1 >= 0:
                    batch_target[frame_idx - 1][6] = 1
                if frame_idx + 1 < len(batch_target):
                    batch_target[frame_idx + 1][6] = 1
                if frame_idx + 2 < len(batch_target):
                    batch_target[frame_idx + 2][6] = 1
                # 跳過已處理過的範圍
                frame_idx += 3
            else:
                frame_idx += 1

        
        for frame_idx in range(10):
            label = ''
            
            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            p_cell_nx = each_pos_nx[frame_idx]
            p_cell_ny = each_pos_ny[frame_idx]
            center = 0.5
            metrics = []
            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]

            ############## MAX ##############
            conf_threshold = 0.5
            p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
            max_position = torch.argmax(p_conf_masked)
            # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
            max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
            max_conf = p_conf[max_y, max_x]
            
            ############# 多球 #############

            ### max_conf 版本 ###
            preds = [(max_x, max_y, max_conf)]
            ### max_conf 版本 ###

            ### nms 版本 ###
            # preds = non_max_suppression(p_conf, p_cell_x, p_cell_y, conf_threshold=conf_threshold, dis_tolerance=12)
            ### nms 版本 ###

            for (x, y, conf) in preds:
                if len(metrics) > 5 :
                    break
                # 全部都小於 conf_threshold 還是會選最大的一筆
                if conf < conf_threshold:
                    continue
                metric = {}
                metric["grid_x"] = x
                metric["grid_y"] = y
                
                metric["x"] = (center*self.stride-p_cell_x[int(y)][int(x)][0]+p_cell_x[int(y)][int(x)][1])/self.stride
                metric["y"] = (center*self.stride-p_cell_y[int(y)][int(x)][0]+p_cell_y[int(y)][int(x)][1])/self.stride
                metric["conf"] = conf

                metric["nx"] = (center*self.stride-p_cell_nx[int(y)][int(x)][0]+p_cell_nx[int(y)][int(x)][1])/self.stride
                metric["ny"] = (center*self.stride-p_cell_ny[int(y)][int(x)][0]+p_cell_ny[int(y)][int(x)][1])/self.stride
                metric["n_conf"] = conf


                metrics.append(metric)
                self.frame_10_metrics.append(metric)

            # confusion metrics
            
            pred_x = max_x*self.stride + (center*self.stride-p_cell_x[max_y][max_x][0]+p_cell_x[max_y][max_x][1])
            pred_y = max_y*self.stride + (center*self.stride-p_cell_y[max_y][max_x][0]+p_cell_y[max_y][max_x][1])
            
            target_x = batch_target[frame_idx][2]
            target_y = batch_target[frame_idx][3]

            ball_count = mask_has_ball.sum()
            self.ball_count += ball_count   

            
            distance = torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)
            
            box_color = 'red'

            ### max_conf 版本 ###
            if batch_target[frame_idx][1] == 0:
                if max_conf >= conf_threshold:
                    self.pos_FP += 1
                    box_color = 'blue'
                else:
                    self.pos_TN += 1
            else:
                if max_conf >= conf_threshold:
                    if distance <= self.tolerance5:
                        self.pos_TP += 1
                    else:
                        self.pos_FP_dis += 1
                        self.pos_FP += 1
                        box_color = 'blue'
                else:
                    self.pos_FN += 1
                    box_color = 'yellow'
            ### max_conf 版本 ###

            ### nms 版本 ###
            # gt_has_ball = batch_target[frame_idx][1] == 1
            # if len(preds) > 0:
            #     # 找出距離 ground truth 最近的一顆預測
            #     distances = [torch.sqrt((px - target_x)**2 + (py - target_y)**2) for _, px, py in preds]
            #     min_dist = min(distances)
            #     min_idx = distances.index(min_dist)

            #     if gt_has_ball:
            #         if min_dist <= self.tolerance5:
            #             self.pos_TP += 1
            #         else:
            #             self.pos_FN += 1
            #         # 其餘預測視為 FP
            #         self.pos_FP += len(preds) - 1
            #         self.pos_FP_dis += sum([1 for i, d in enumerate(distances) if i != min_idx])
            #         box_color = 'blue'
            #     else:
            #         # 沒有球，全部預測都是 false positive
            #         self.pos_FP += len(preds)
            #         box_color = 'blue'

            # else:
            #     if gt_has_ball:
            #         self.pos_FN += 1
            #         box_color = 'yellow'
            #     else:
            #         self.pos_TN += 1
            ### nms 版本 ###

            # threshold = 0.5 ~ 0.95
            # threshold_idx = 0 ~ 9
            # iou_dist = 1~5 (pixel 容忍距離)
            for iou_dist_idx in range(len(self.iou_dist_thresholds)):
                for threshold_idx in range(len(self.conf_thresholds)):
                    conf_threshold = self.conf_thresholds[threshold_idx]
                    if batch_target[frame_idx][1] == 0:
                        if max_conf >= conf_threshold:
                            self.cumulative_FP[iou_dist_idx][threshold_idx] += 1
                        else:
                            self.cumulative_TN[iou_dist_idx][threshold_idx] += 1
                    else:
                        if max_conf >= conf_threshold:
                            if distance <= self.iou_dist_thresholds[iou_dist_idx]:
                                self.cumulative_TP[iou_dist_idx][threshold_idx] += 1
                            else:
                                self.cumulative_FP[iou_dist_idx][threshold_idx] += 1
                        else:
                            self.cumulative_FN[iou_dist_idx][threshold_idx] += 1
            
            now = datetime.now()
            # Format the datetime object as a string
            formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
            

            display_interval = 1
            if self.args.mode == 'train':
                display_interval = 10

            if frame_idx%display_interval==0:
                if frame_idx != len(batch_target)-1:
                    target_xy = (batch_target[frame_idx][2], batch_target[frame_idx][3], batch_target[frame_idx+1][2], batch_target[frame_idx+1][3])
                else:
                    target_xy = (batch_target[frame_idx][2], batch_target[frame_idx][3], batch_target[frame_idx][2], batch_target[frame_idx][3])

                display_predict_image(
                        batch_img[frame_idx],  
                        metrics, 
                        'val_'+formatted_date+'_'+ str(int(batch_target[frame_idx][0])),
                        box_color=box_color,
                        label=label,
                        save_dir=self.metrics.save_dir,
                        stride = self.stride,
                        next=True,
                        loss=loss
                        ) 
            
                if box_color == 'blue':
                    display_predict_image(
                        batch_img[frame_idx],  
                        metrics, 
                        'val_'+formatted_date+'_'+ str(int(batch_target[frame_idx][0])),
                        box_color=box_color,
                        label=label,
                        save_dir=self.metrics.save_dir,
                        stride = self.stride,
                        target=target_xy,
                        path='predict_val_FP_img',
                        next=False,
                        loss=loss
                        ) 
                if box_color == 'yellow':
                    display_predict_image(
                        batch_img[frame_idx],  
                        metrics, 
                        'val_'+formatted_date+'_'+ str(int(batch_target[frame_idx][0])),
                        box_color=box_color,
                        label=label,
                        save_dir=self.metrics.save_dir,
                        stride = self.stride,
                        target=target_xy,
                        path='predict_val_FN_img',
                        next=False,
                        loss=loss
                        ) 

                display_predict_image(
                            batch_img[frame_idx],  
                            list(self.frame_10_metrics), 
                            'val_'+formatted_date+'_'+ str(int(batch_target[frame_idx][0])),
                            box_color=box_color,
                            label=label,
                            save_dir=self.metrics.save_dir,
                            stride = self.stride,
                            path='predict_val_10_frame_img',
                            next=False,
                            only_ball=True,
                            loss=loss
                            )

                # display_predict_image(
                #             batch_img[frame_idx],  
                #             list(self.frame_10_metrics), 
                #             'val_'+formatted_date+'_'+ str(int(batch_target[frame_idx][0])),
                #             box_color=box_color,
                #             label=label,
                #             save_dir=self.metrics.save_dir,
                #             stride = self.stride,
                #             path='predict_val_10_next_frame_img',
                #             next=False,
                #             only_ball=True,
                #             only_next=True
                #             )
        
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        if (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP) != 0:
            self.pos_acc = (self.pos_TN + self.pos_TP) / (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP)
        if (self.pos_TP+self.pos_FP) != 0:
            self.pos_precision = self.pos_TP/(self.pos_TP+self.pos_FP)

        self.calculate_precision_recall(True)

    def calculate_precision_recall(self, plot=False):
        # 確保維度一致性
        assert len(self.cumulative_TP) == len(self.iou_dist_thresholds), "TP and IoU thresholds length mismatch"
        assert len(self.cumulative_FP) == len(self.iou_dist_thresholds), "FP and IoU thresholds length mismatch"
        assert len(self.cumulative_FN) == len(self.iou_dist_thresholds), "FN and IoU thresholds length mismatch"
        assert all(len(self.cumulative_TP[i]) == len(self.conf_thresholds) for i in range(len(self.iou_dist_thresholds))), \
            "TP and confidence thresholds length mismatch in nested lists"
        assert all(len(self.cumulative_FP[i]) == len(self.conf_thresholds) for i in range(len(self.iou_dist_thresholds))), \
            "FP and confidence thresholds length mismatch in nested lists"
        assert all(len(self.cumulative_FN[i]) == len(self.conf_thresholds) for i in range(len(self.iou_dist_thresholds))), \
            "FN and confidence thresholds length mismatch in nested lists"

        # Precision-Recall 曲線結果儲存
        precision_recall_by_iou = []

        for iou_idx in range(len(self.iou_dist_thresholds)):
            precision_list = []
            recall_list = []

            for conf_idx in range(len(self.conf_thresholds)):
                TP = self.cumulative_TP[iou_idx][conf_idx]
                FP = self.cumulative_FP[iou_idx][conf_idx]
                FN = self.cumulative_FN[iou_idx][conf_idx]

                # 計算 Precision 和 Recall，處理分母為 0 的情況
                if (TP + FP) == 0:
                    print(f"Warning: TP + FP is 0 at IoU index {iou_idx}, Conf index {conf_idx}. Precision set to 0.")
                    precision = 0
                else:
                    precision = TP / (TP + FP)

                if (TP + FN) == 0:
                    print(f"Warning: TP + FN is 0 at IoU index {iou_idx}, Conf index {conf_idx}. Recall set to 0.")
                    recall = 0
                else:
                    recall = TP / (TP + FN)

                precision_list.append(precision)
                recall_list.append(recall)
            precision_list.reverse()
            recall_list.reverse()
            # 確保 Recall 是單調遞減的 (conf_threshold 小到大)
            for i in range(1, len(recall_list)):
                if recall_list[i] < recall_list[i - 1]:
                    print(f"Warning: Recall list is not non-decreasing at IoU index {iou_idx}, Conf index {i}. "
                        f"Recall[i-1]={recall_list[i - 1]:.6f}, Recall[i]={recall_list[i]:.6f}")

            # assert all(recall_list[i] >= recall_list[i - 1] for i in range(1, len(recall_list))), \
            #     f"Recall list must be non-decreasing for IoU index {iou_idx}: {recall_list}"

            # 確保 Precision 在 [0, 1] 範圍內
            assert all(0 <= p <= 1 for p in precision_list), f"Invalid precision values for IoU index {iou_idx}: {precision_list}"

            # 儲存 Precision-Recall 結果
            precision_recall_by_iou.append((precision_list, recall_list))

        # 平均 Precision-Recall 曲線繪製
        if plot:
            for iou_idx, (precision_list, recall_list) in enumerate(precision_recall_by_iou):
                plt.figure()
                for idx, (recall, precision) in enumerate(zip(recall_list, precision_list)):
                    plt.plot(recall, precision, marker='o', label=f"IoU={self.iou_dist_thresholds[iou_idx]:.2f}")
                    plt.text(recall, precision, str(idx), fontsize=8, ha='right', va='bottom')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve for IoU={self.iou_dist_thresholds[iou_idx]:.2f}')
                plt.legend()
                now = datetime.now()
                formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
                output_dir = self.metrics.save_dir / 'precision_recall'
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f'precision_recall_curve_iou_{iou_idx}_{formatted_date}.png')
                plt.close()

        # 計算平均 Precision (AP) 作為 fitness
        ap_list = []
        for iou_idx, (precision_list, recall_list) in enumerate(precision_recall_by_iou):
            ap = 0.0
            for i in range(1, len(recall_list)):
                ap += (recall_list[i] - recall_list[i - 1]) * precision_list[i]
            ap_list.append(ap)

            # 檢查 AP 是否合理
            assert ap >= 0, f"AP is negative for IoU index {iou_idx}: {ap}"

            print(f"Average Precision (AP) for IoU={self.iou_dist_thresholds[iou_idx]:.2f}: {ap}")
                # 最終的平均 AP
        
        self.avg_ap = sum(ap_list) / len(ap_list) if ap_list else 0
        print("Overall Average Precision (AP):", self.avg_ap)

        # 使用 f1
        self.fitness = self.calculate_weighted_f1(self.pos_TP, self.pos_FP, self.pos_FN, 2.0)
        print("Overall weighted_f1:", self.fitness)
    def calculate_weighted_f1(self, tp, fp, fn, beta=1.0):
        """
        计算 Weighted F1 Score
        Args:
            tp (int): True Positives 数量
            fp (int): False Positives 数量
            fn (int): False Negatives 数量
            beta (float): Recall 的权重参数
        
        Returns:
            float: Weighted F1 Score
        """
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Weighted F1 Score
        beta_sq = beta ** 2
        if precision + recall == 0:
            return 0.0
        weighted_f1 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        return weighted_f1

    def get_stats(self):
        # 繪製 Precision-Recall 曲線
        self.calculate_precision_recall(False)

        if (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP) != 0:
            self.pos_acc = (self.pos_TN + self.pos_TP) / (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP)
        if (self.pos_TP+self.pos_FP) != 0:
            self.pos_precision = self.pos_TP/(self.pos_TP+self.pos_FP)

        """Return the stats."""
        return {'self.avg_ap': self.avg_ap, 'fitness': self.fitness, 'pos_FN': self.pos_FN, 'pos_FP_dis': self.pos_FP_dis, 'pos_FP': self.pos_FP, 'pos_TN': self.pos_TN, 
                'pos_TP': self.pos_TP, 'pos_acc': self.pos_acc, 'pos_precision': self.pos_precision,
                'threshold>0.8 rate':self.pred_ball_count/self.ball_count}
    
    def print_results(self):
        """Print the results."""
        print(self.get_stats())

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"


# p3 v1
class TrackNetValidatorV2(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetValDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)

        if self.args.half and self.device.type == "cuda":
            batch['img'] = batch['img'].half() / 255.0  # `float16`
        else:
            batch['img'] = batch['img'].float() / 255.0  # `float32`

        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.

        # TODO val 時，stride 取得異常
        if isinstance(model.stride, torch.Tensor):
            self.stride = model.stride[0]
        else:
            self.stride = model.model.stride[0]
        self.cell_num = int(640/self.stride)
        self.num_groups = 10

        self.total_loss = 0.0
        self.num_samples = 0
        self.conf_TP = 0
        self.conf_TN = 0
        self.conf_FP = 0
        self.conf_FN = 0
        self.conf_acc = 0
        self.conf_precision = 0
        self.pos_TP = 0
        self.pos_TN = 0
        self.pos_FP = 0
        self.pos_FN = 0
        self.pos_FP_dis = 0
        self.fast_TP = 0
        self.hit_TP = 0
        self.hit_FP = 0
        self.fast_FN = 0
        self.hit_FN = 0
        self.fast_hit_TP = 0
        self.fast_hit_FN = 0
        self.pos_acc = 0
        self.pos_precision = 0
        self.ball_count = 0
        self.pred_ball_count = 0
        device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_max = 20
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.no = 33
        self.feat_no = 2
        self.nc = 1

        self.fast_count = 0
        self.hit_count = 0
        self.fast_hit_count = 0

        self.target_hit_count = 0
        self.hitV2_TP = 0  # True Positives
        self.hitV2_FP = 0  # False Positives
        self.hitV2_TN = 0  # True Negatives
        self.hitV2_FN = 0  # False Negatives
        self.hitV1_TP = 0  # True Positives
        self.hitV1_FP = 0  # False Positives
        self.hitV1_TN = 0  # True Negatives
        self.hitV1_FN = 0  # False Negatives

        # 一顆球半徑 = 3 pixel
        self.tolerance3 = 3.0 # 50% 距離容忍度
        self.conf_thresholds = [i * 0.05 for i in range(1, 20)]  # [0.5, 0.55, ..., 0.95]
        self.iou_dist_thresholds = [i * 1 for i in range(1, 6)]  # [1, 2, ..., 5]
        
        self.cumulative_TP = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_FP = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_FN = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.cumulative_TN = [[0 for _ in self.conf_thresholds] for _ in self.iou_dist_thresholds]
        self.fitness = 0
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_img = batch['img']
        batch_img_file = batch['img_files']
        if preds.shape == (330, self.cell_num, self.cell_num):
            self.update_metrics_once(0, preds, batch_target[0], batch_img[0])
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx], batch_img[idx])
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target, batch_img):
        # pred = [330 * self.cell_num * self.cell_num]
        # batch_target = [10*7]
        feats = pred.clone()
        pred_distri, pred_scores = feats.view(self.no, -1).split(
            (self.reg_max * self.feat_no, self.nc), 0)
        
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()

        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*self.cell_num*self.cell_num]
        
        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, self.feat_no, c // self.feat_no).softmax(2).matmul(
            self.proj.type(pred_distri.dtype))
        
        target_pos_distri = torch.zeros(self.num_groups, self.cell_num, self.cell_num, self.feat_no, device=self.device)
        mask_has_ball = torch.zeros(self.num_groups, self.cell_num, self.cell_num, device=self.device)
        cls_targets = torch.zeros(self.num_groups, self.cell_num, self.cell_num, 1, device=self.device)
        mask_fast_ball = torch.zeros(self.num_groups, 1, device=self.device)
        mask_hit_ball = torch.zeros(self.num_groups, 1, device=self.device)
        mask_hit_ball_v2 = torch.zeros(self.num_groups, 1, device=self.device)

        for target_idx, target in enumerate(batch_target):
            grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], self.stride)

            # 找出快球 => 慢球, 慢球 => 快球
            if target_idx > 1 and target_idx < len(batch_target)-2 and \
                batch_target[target_idx-2][1] == 1 and batch_target[target_idx][1] == 1 and batch_target[target_idx+2][1] == 1:
                before_hit2 = [batch_target[target_idx-2][2], batch_target[target_idx-2][3]]
                hit = [batch_target[target_idx][2], batch_target[target_idx][3]]
                after_hit2 = [batch_target[target_idx+2][2], batch_target[target_idx+2][3]]

                before_dist = calculate_dist(before_hit2, hit)
                after_dist = calculate_dist(hit, after_hit2)
                angle = calculate_angle(before_hit2, hit, after_hit2)

                if (angle and angle > 30 and (before_dist > 10 or after_dist > 10)) or \
                    ((before_dist > self.stride or after_dist > self.stride) and (before_dist > after_dist*2 or before_dist*2 < after_dist)):
                    mask_hit_ball_v2[target_idx-2] = 1
                    mask_hit_ball_v2[target_idx-1] = 1
                    mask_hit_ball_v2[target_idx] = 1
                    mask_hit_ball_v2[target_idx+1] = 1
                    mask_hit_ball_v2[target_idx+2] = 1

            if target_idx < len(batch_target)-1 and batch_target[target_idx+1][1] == 1 and \
                batch_target[target_idx][1] == 1 and target[4]**2 + target[5]**2 >= self.cell_num**2:
                mask_fast_ball[target_idx] = 1
                mask_fast_ball[target_idx+1] = 1

            if target_idx < len(batch_target)-2 \
                and batch_target[target_idx][1] == 1 and batch_target[target_idx+1][1] == 1 \
                and batch_target[target_idx+2][1] == 1:
                
                first = [batch_target[target_idx][2], batch_target[target_idx][3]]
                second = [batch_target[target_idx+1][2], batch_target[target_idx+1][3]]
                third = [batch_target[target_idx+2][2], batch_target[target_idx+2][3]]
                angle = calculate_angle(first, second, third)
                dist1 = calculate_dist(first, second)
                dist2 = calculate_dist(second, third)
                if angle and angle > 30 and (dist1 > 10 or dist2 > 10):
                    mask_hit_ball[target_idx] = 1
                    mask_hit_ball[target_idx+1] = 1
                    mask_hit_ball[target_idx+2] = 1
            if target[1] == 1:
                # xy
                mask_has_ball[target_idx, grid_y, grid_x] = 1
                
                target_pos_distri[target_idx, grid_y, grid_x, 0] = offset_x*(self.reg_max)/self.stride
                target_pos_distri[target_idx, grid_y, grid_x, 1] = offset_y*(self.reg_max)/self.stride

                ## cls
                cls_targets[target_idx, grid_y, grid_x, 0] = 1
        
        target_pos_distri = target_pos_distri.view(self.num_groups*self.cell_num*self.cell_num, self.feat_no)
        cls_targets = cls_targets.view(self.num_groups*self.cell_num*self.cell_num, 1)
        mask_has_ball = mask_has_ball.view(self.num_groups*self.cell_num*self.cell_num).bool()

        self.fast_count += mask_fast_ball.sum()
        self.hit_count += mask_hit_ball_v2.sum()
        mask_fast_hit_ball = (mask_fast_ball.bool()|mask_hit_ball_v2.bool()).float()
        self.fast_hit_count += mask_fast_hit_ball.sum()

        each_probs = pred_probs.view(10, self.cell_num, self.cell_num)
        each_pos_x, each_pos_y = pred_pos.view(10, self.cell_num, self.cell_num, 2).split([1, 1], dim=3)

        # 計算 hit v2 效果
        # 先填充 hit 前後兩幀
        frame_idx = 0
        while frame_idx < 10:
            if batch_target[frame_idx][6] == 1:
                # 檢查並設定範圍內的相鄰元素
                if frame_idx - 2 >= 0:
                    batch_target[frame_idx - 2][6] = 1
                if frame_idx - 1 >= 0:
                    batch_target[frame_idx - 1][6] = 1
                if frame_idx + 1 < len(batch_target):
                    batch_target[frame_idx + 1][6] = 1
                if frame_idx + 2 < len(batch_target):
                    batch_target[frame_idx + 2][6] = 1
                # 跳過已處理過的範圍
                frame_idx += 3
            else:
                frame_idx += 1

        
        
        for frame_idx in range(10):
            label = ''
            if mask_fast_ball[frame_idx] == 1:
                label += '_fast_'
            if mask_hit_ball[frame_idx] == 1:
                label += '_hit_'
            if mask_hit_ball_v2[frame_idx] == 1:
                label += '_hitV2_'
            if batch_target[frame_idx][6] == 1:
                label += '_targetHit_'
                self.target_hit_count+=1

            if batch_target[frame_idx][6] == 1 and mask_hit_ball_v2[frame_idx] == 1:
                self.hitV2_TP += 1
            elif batch_target[frame_idx][6] == 0 and mask_hit_ball_v2[frame_idx] == 1:
                self.hitV2_FP += 1
            elif batch_target[frame_idx][6] == 0 and mask_hit_ball_v2[frame_idx] == 0:
                self.hitV2_TN += 1
            elif batch_target[frame_idx][6] == 1 and mask_hit_ball_v2[frame_idx] == 0:
                self.hitV2_FN += 1

            if batch_target[frame_idx][6] == 1 and mask_hit_ball[frame_idx] == 1:
                self.hitV1_TP += 1
            elif batch_target[frame_idx][6] == 0 and mask_hit_ball[frame_idx] == 1:
                self.hitV1_FP += 1
            elif batch_target[frame_idx][6] == 0 and mask_hit_ball[frame_idx] == 0:
                self.hitV1_TN += 1
            elif batch_target[frame_idx][6] == 1 and mask_hit_ball[frame_idx] == 0:
                self.hitV1_FN += 1
            
             

            p_cell_x = each_pos_x[frame_idx]
            p_cell_y = each_pos_y[frame_idx]
            metrics = []
            # 獲取當前圖片的 conf
            p_conf = each_probs[frame_idx]


            ############## MAX ##############
            conf_threshold = 0.6
            p_conf_masked = p_conf * (p_conf >= conf_threshold).float()
            max_position = torch.argmax(p_conf_masked)
            # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
            max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
            max_conf = p_conf[max_y, max_x]
            
            metric = {}
            metric["grid_x"] = max_x
            metric["grid_y"] = max_y
            metric["x"] = p_cell_x[max_y][max_x]/self.reg_max
            metric["y"] = p_cell_y[max_y][max_x]/self.reg_max
            metric["conf"] = max_conf
            metrics = []
            if max_conf >= conf_threshold:
                metrics.append(metric)

            # confusion metrics
            pred_x = max_x*self.stride + (p_cell_x[max_y][max_x]/self.reg_max)*self.stride
            pred_y = max_y*self.stride + (p_cell_y[max_y][max_x]/self.reg_max)*self.stride
            target_x = batch_target[frame_idx][2]
            target_y = batch_target[frame_idx][3]

            ball_count = mask_has_ball.sum()
            self.ball_count += ball_count   

            
            distance = torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)
            
            box_color = 'red'
            if batch_target[frame_idx][1] == 0:
                if max_conf >= conf_threshold:
                    self.pos_FP += 1
                    box_color = 'blue'
                else:
                    self.pos_TN += 1
            else:
                if max_conf >= conf_threshold:
                    if distance <= self.tolerance3:
                        self.pos_TP += 1

                        if mask_hit_ball_v2[frame_idx] == 1:
                            self.hit_TP += 1
                    else:
                        self.pos_FP += 1
                        self.pos_FP_dis += 1
                        box_color = 'blue'

                        if mask_hit_ball_v2[frame_idx] == 1:
                            self.hit_FP += 1
                    if mask_fast_ball[frame_idx] == 1:
                        self.fast_TP += 1
                    if mask_fast_hit_ball[frame_idx] == 1:
                        self.fast_hit_TP += 1
                else:
                    self.pos_FN += 1
                    if mask_fast_ball[frame_idx] == 1:
                        self.fast_FN += 1
                    if mask_hit_ball_v2[frame_idx] == 1:
                        self.hit_FN += 1
                    if mask_fast_hit_ball[frame_idx] == 1:
                        self.fast_hit_FN += 1
            
            # threshold = 0.5 ~ 0.95
            # threshold_idx = 0 ~ 9
            # iou_dist = 1~5 (pixel 容忍距離)
            for iou_dist_idx in range(len(self.iou_dist_thresholds)):
                for threshold_idx in range(len(self.conf_thresholds)):
                    conf_threshold = self.conf_thresholds[threshold_idx]
                    if batch_target[frame_idx][1] == 0:
                        if max_conf >= conf_threshold:
                            self.cumulative_FP[iou_dist_idx][threshold_idx] += 1
                        else:
                            self.cumulative_TN[iou_dist_idx][threshold_idx] += 1
                    else:
                        if max_conf >= conf_threshold:
                            if distance <= self.iou_dist_thresholds[iou_dist_idx]:
                                self.cumulative_TP[iou_dist_idx][threshold_idx] += 1
                            else:
                                self.cumulative_FP[iou_dist_idx][threshold_idx] += 1
                        else:
                            self.cumulative_FN[iou_dist_idx][threshold_idx] += 1
            
            now = datetime.now()
            # Format the datetime object as a string
            formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # display_predict_image(
            #         batch_img[frame_idx],  
            #         metrics, 
            #         'val_'+formatted_date+'_'+ str(int(batch_target[frame_idx][0])),
            #         box_color=box_color,
            #         label=label,
            #         save_dir=self.metrics.save_dir,
            #         stride = self.stride,
            #         target=(batch_target[frame_idx][2], batch_target[frame_idx][3])
            #         ) 
            
            # if box_color == 'blue':
            #     display_predict_image(
            #         batch_img[frame_idx],  
            #         metrics, 
            #         'val_'+formatted_date+'_'+ str(int(batch_target[frame_idx][0])),
            #         box_color=box_color,
            #         label=label,
            #         save_dir=self.metrics.save_dir,
            #         stride = self.stride,
            #         path='predict_val_error_img'
            #         ) 

        # 計算 conf 的 confusion matrix
        threshold = 0.6
        pred_binary = (pred_probs >= threshold)
        self.pred_ball_count += pred_binary.int().sum()

        unique_classes = torch.unique(cls_targets.bool())
        if len(unique_classes) == 1:
            if unique_classes.item() == 1:
                # All targets are 1 (positive class)
                self.conf_TP += (pred_binary == 1).sum().item()  # Count of true positives
                self.conf_FN += (pred_binary == 0).sum().item()  # Count of false negatives
                self.conf_TN += 0  # No true negatives
                self.conf_FP += 0  # No false positives
            else:
                # All targets are 0 (negative class)
                self.conf_TN += (pred_binary == 0).sum().item()  # Count of true negatives
                self.conf_FP += (pred_binary == 1).sum().item()  # Count of false positives
                self.conf_TP += 0  # No true positives
                self.conf_FN += 0  # No false negatives
        else:
            # Compute confusion matrix normally
            conf_matrix = confusion_matrix(cls_targets.bool().cpu().numpy(), pred_binary.cpu().numpy())
            self.conf_TN += conf_matrix[0][0]
            self.conf_FP += conf_matrix[0][1]
            self.conf_FN += conf_matrix[1][0]
            self.conf_TP += conf_matrix[1][1]
        
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        if (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP) != 0:
            self.pos_acc = (self.pos_TN + self.pos_TP) / (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP)
        if (self.pos_TP+self.pos_FP) != 0:
            self.pos_precision = self.pos_TP/(self.pos_TP+self.pos_FP)

        if (self.conf_FN+self.conf_FP+self.conf_TN + self.conf_TP) != 0:
            self.conf_acc = (self.conf_TN + self.conf_TP) / (self.conf_FN+self.conf_FP+self.conf_TN + self.conf_TP)
        if (self.conf_TP+self.conf_FP) != 0:
            self.conf_precision = self.conf_TP/(self.conf_TP+self.conf_FP)

        self.calculate_precision_recall(True)

    def calculate_precision_recall(self, plot=False):
        # 確保維度一致性
        assert len(self.cumulative_TP) == len(self.iou_dist_thresholds), "TP and IoU thresholds length mismatch"
        assert len(self.cumulative_FP) == len(self.iou_dist_thresholds), "FP and IoU thresholds length mismatch"
        assert len(self.cumulative_FN) == len(self.iou_dist_thresholds), "FN and IoU thresholds length mismatch"
        assert all(len(self.cumulative_TP[i]) == len(self.conf_thresholds) for i in range(len(self.iou_dist_thresholds))), \
            "TP and confidence thresholds length mismatch in nested lists"
        assert all(len(self.cumulative_FP[i]) == len(self.conf_thresholds) for i in range(len(self.iou_dist_thresholds))), \
            "FP and confidence thresholds length mismatch in nested lists"
        assert all(len(self.cumulative_FN[i]) == len(self.conf_thresholds) for i in range(len(self.iou_dist_thresholds))), \
            "FN and confidence thresholds length mismatch in nested lists"

        # Precision-Recall 曲線結果儲存
        precision_recall_by_iou = []

        for iou_idx in range(len(self.iou_dist_thresholds)):
            precision_list = []
            recall_list = []

            for conf_idx in range(len(self.conf_thresholds)):
                TP = self.cumulative_TP[iou_idx][conf_idx]
                FP = self.cumulative_FP[iou_idx][conf_idx]
                FN = self.cumulative_FN[iou_idx][conf_idx]

                # 計算 Precision 和 Recall，處理分母為 0 的情況
                if (TP + FP) == 0:
                    print(f"Warning: TP + FP is 0 at IoU index {iou_idx}, Conf index {conf_idx}. Precision set to 0.")
                    precision = 0
                else:
                    precision = TP / (TP + FP)

                if (TP + FN) == 0:
                    print(f"Warning: TP + FN is 0 at IoU index {iou_idx}, Conf index {conf_idx}. Recall set to 0.")
                    recall = 0
                else:
                    recall = TP / (TP + FN)

                precision_list.append(precision)
                recall_list.append(recall)
            precision_list.reverse()
            recall_list.reverse()
            # 確保 Recall 是單調遞減的 (conf_threshold 小到大)
            for i in range(1, len(recall_list)):
                if recall_list[i] < recall_list[i - 1]:
                    print(f"Warning: Recall list is not non-decreasing at IoU index {iou_idx}, Conf index {i}. "
                        f"Recall[i-1]={recall_list[i - 1]:.6f}, Recall[i]={recall_list[i]:.6f}")

            # assert all(recall_list[i] >= recall_list[i - 1] for i in range(1, len(recall_list))), \
            #     f"Recall list must be non-decreasing for IoU index {iou_idx}: {recall_list}"

            # 確保 Precision 在 [0, 1] 範圍內
            assert all(0 <= p <= 1 for p in precision_list), f"Invalid precision values for IoU index {iou_idx}: {precision_list}"

            # 儲存 Precision-Recall 結果
            precision_recall_by_iou.append((precision_list, recall_list))

        # 平均 Precision-Recall 曲線繪製
        if plot:
            for iou_idx, (precision_list, recall_list) in enumerate(precision_recall_by_iou):
                plt.figure()
                for idx, (recall, precision) in enumerate(zip(recall_list, precision_list)):
                    plt.plot(recall, precision, marker='o', label=f"IoU={self.iou_dist_thresholds[iou_idx]:.2f}")
                    plt.text(recall, precision, str(idx), fontsize=8, ha='right', va='bottom')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve for IoU={self.iou_dist_thresholds[iou_idx]:.2f}')
                plt.legend()
                now = datetime.now()
                formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
                output_dir = self.metrics.save_dir / 'precision_recall'
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f'precision_recall_curve_iou_{iou_idx}_{formatted_date}.png')
                plt.close()

        # 計算平均 Precision (AP) 作為 fitness
        ap_list = []
        for iou_idx, (precision_list, recall_list) in enumerate(precision_recall_by_iou):
            ap = 0.0
            for i in range(1, len(recall_list)):
                ap += (recall_list[i] - recall_list[i - 1]) * precision_list[i]
            ap_list.append(ap)

            # 檢查 AP 是否合理
            assert ap >= 0, f"AP is negative for IoU index {iou_idx}: {ap}"

            print(f"Average Precision (AP) for IoU={self.iou_dist_thresholds[iou_idx]:.2f}: {ap}")

        # 最終的平均 AP
        self.fitness = sum(ap_list) / len(ap_list) if ap_list else 0
        print("Overall Average Precision (AP):", self.fitness)

    def get_stats(self):
        # 繪製 Precision-Recall 曲線
        self.calculate_precision_recall(False)

        if (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP) != 0:
            self.pos_acc = (self.pos_TN + self.pos_TP) / (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP)
        if (self.pos_TP+self.pos_FP) != 0:
            self.pos_precision = self.pos_TP/(self.pos_TP+self.pos_FP)

        """Return the stats."""
        return {'fitness': self.fitness, 'pos_FN': self.pos_FN, 'pos_FP_dis': self.pos_FP_dis, 'pos_FP': self.pos_FP, 'pos_TN': self.pos_TN, 
                'pos_TP': self.pos_TP, 'pos_acc': self.pos_acc, 'pos_precision': self.pos_precision,
                "fast_TP": self.fast_TP, "fast_FN": self.fast_FN, 
                "hit_TP": self.hit_TP, "hit_FP": self.hit_FP, "hit_FN": self.hit_FN, 
                "fast_hit_TP": self.fast_hit_TP, "fast_hit_FN": self.fast_hit_FN, 
                'conf_FN': self.conf_FN, 'conf_FP': self.conf_FP, 'conf_TN': self.conf_TN, 
                'conf_TP': self.conf_TP, 'conf_acc': self.conf_acc, 'conf_precision': self.conf_precision,
                'threshold>0.8 rate':self.pred_ball_count/self.ball_count}
    
    def print_results(self):
        """Print the results."""
        print(f'fast count: {self.fast_count}, hit count: {self.hit_count}')
        print(f'fast acc: {self.fast_TP/self.fast_count}, hit acc: {self.hit_TP/self.hit_count}')
        print(f'fast or hit count: {self.fast_hit_count}, fast or hit acc: {self.fast_hit_TP/self.fast_hit_count}')
        
        # print(f'target hit count: {self.target_hit_count}')
        # print(f"hitV2- TP: {self.hitV2_TP}, FP: {self.hitV2_FP}, TN: {self.hitV2_TN}, FN: {self.hitV2_FN}")
        # print(f"hitV1- TP: {self.hitV1_TP}, FP: {self.hitV1_FP}, TN: {self.hitV1_TN}, FN: {self.hitV1_FN}")

        print(self.get_stats())
        # precision = 0
        # recall = 0
        # f1 = 0
        # if self.TP > 0:
        #     precision = self.TP/(self.TP+self.FP)
        #     recall = self.TP/(self.TP+self.FN)
        #     f1 = (2*precision*recall)/(precision+recall)
        # print(f"Validation Accuracy: {self.acc:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, , Validation F1-Score: {f1:.4f}")

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"


class TrackNetValidatorWithHit(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)

        if self.args.half and self.device.type == "cuda":
            batch['img'] = batch['img'].half() / 255.0  # `float16`
        else:
            batch['img'] = batch['img'].float() / 255.0  # `float32`

        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.
        self.total_loss = 0.0
        self.num_samples = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
        self.hasMax = 0
        self.hasBall = 0
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_size = preds.shape[0]
        if preds.shape == (60, 20, 20):
            self.update_metrics_once(0, preds, batch_target)
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx])
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target):
        # pred = [50 * 20 * 20]
        # batch_target = [10*6]
        pred_distri, pred_scores, pred_hits = torch.split(pred, [40, 10, 10], dim=0)
        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*20*20]
        
        pred_pos, pred_mov = torch.split(pred_distri, [20, 20], dim=0)
        # pred_pos = torch.sigmoid(pred_pos)
        # pred_mov = torch.tanh(pred_mov)

        max_values_dim1, max_indices_dim1 = pred_probs.max(dim=2)
        final_max_values, max_indices_dim2 = max_values_dim1.max(dim=1)
        max_positions = [(index.item(), max_indices_dim1[i, index].item()) for i, index in enumerate(max_indices_dim2)]

        #targets = pred_distri.clone().detach()
        #cls_targets = torch.zeros(10, pred_scores.shape[1], pred_scores.shape[2])
        stride = 32
        if len(batch_target.shape) == 3:
            batch_target = batch_target[0]
        for idx, target in enumerate(batch_target):
            if target[1] == 1:
                # xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                if (grid_x > 20 or grid_y > 20):
                    LOGGER.Warning("target grid transform error")
                if (pred_probs[idx][grid_x][grid_y] > 0.5):
                    self.hasBall += 1
                
                # print(f"target: {(grid_x, grid_y, offset_x, offset_y)}, ")
                # print(f"predict_conf: {pred_probs[idx][grid_x][grid_y]}, ")
                # print(f"pred_pos: {pred_pos[idx][grid_x][grid_y]}")
                # print(pred_probs[idx][max_positions[idx]])
                # print(max_positions[idx])
                if pred_probs[idx][max_positions[idx]] > 0.5:
                    self.hasMax += 1
                    x, y = max_positions[idx]
                    real_x = x*stride + pred_pos[idx][x][y] #*stride
                    real_y = y*stride + pred_pos[idx][x][y] #*stride
                    if (grid_x, grid_y) == max_positions[idx]:
                        self.TP+=1
                    else:
                        self.FN+=1
                else:
                    self.FN+=1
            elif pred_probs[idx][max_positions[idx]] > 0.5:
                self.FP+=1
            else:
                self.TN+=1
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        self.acc = (self.TN + self.TP) / (self.FN+self.FP+self.TN + self.TP)

    def get_stats(self):
        """Return the stats."""
        return {'FN': self.FN, 'FP': self.FP, 'TN': self.TN, 'TP': self.TP, 'acc': self.acc, 'max_conf>0.5': self.hasMax, 'correct_cell>0.5':self.hasBall}
    
    def print_results(self):
        """Print the results."""
        precision = 0
        recall = 0
        f1 = 0
        if self.TP > 0:
            precision = self.TP/(self.TP+self.FP)
            recall = self.TP/(self.TP+self.FN)
            f1 = (2*precision*recall)/(precision+recall)
        print(f"Validation Accuracy: {self.acc:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, , Validation F1-Score: {f1:.4f}")

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"
