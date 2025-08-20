import csv
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.tracknet.utils.confusion_matrix import ConfConfusionMatrix
from ultralytics.tracknet.utils.plotting import display_image_with_coordinates, display_predict_in_checkerboard
from ultralytics.tracknet.utils.transform import calculate_angle, calculate_dist, target_grid

from ultralytics.yolo.utils import LOGGER

# check_training_img_path = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\check_training_img\img_'
# check_training_img_path = r'/usr/src/datasets/tracknet/visualize_train_img/img_'
# check_val_img_path = r'/usr/src/datasets/tracknet/visualize_val_img/img_'

class TrackNetLossWithHit:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.hyp = h
        print(self.hyp.weight_conf, self.hyp.weight_mov, self.hyp.weight_pos, self.hyp.use_dxdy_loss)

        m = model.model[-1]  # Detect() module
        pos_weight = torch.tensor(self.hyp.weight_hit).to(device)
        self.hit_bce = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
        self.mse = nn.MSELoss(reduction='sum')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.batch_count = 0
        self.train_count = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.sample_path = os.path.join(self.hyp.save_dir, "training_samples")

    def __call__(self, preds, batch):
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0].to(self.device) # only pick first (stride = 16)

        if preds.requires_grad:
            self.train_count += 1
        batch_target = batch['target'].to(self.device)
        batch_img = batch['img'].to(self.device)

        loss = torch.zeros(4, device=self.device)
            
        batch_size = preds.shape[0]

        # for each batch
        for idx, pred in enumerate(preds):
            # pred = [60 * 20 * 20]
            stride = self.stride[0]
            pred_distri, pred_scores, pred_hits = torch.split(pred, [40, 10, 10], dim=0)
            pred_distri = pred_distri.reshape(4, 10, 20, 20)
            pred_pos, pred_mov = torch.split(pred_distri, [2, 2], dim=0)

            pred_pos = pred_pos.permute(1, 0, 2, 3).contiguous()
            pred_mov = pred_mov.permute(1, 0, 2, 3).contiguous()

            pred_pos = torch.sigmoid(pred_pos)
            target_pos = pred_pos.detach().clone()
            pred_mov = torch.tanh(pred_mov)
            target_mov = pred_mov.detach().clone()
            
            cls_targets = torch.zeros(pred_scores.shape, device=self.device)
            hit_targets = torch.zeros(pred_hits.shape, device=self.device)
            
            position_loss = torch.tensor(0.0, device=self.device)
            move_loss = torch.tensor(0.0, device=self.device)
            
            mask_has_ball = torch.zeros_like(target_pos)
            for target_idx, target in enumerate(batch_target[idx]):
                if target[6] == 1:
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    hit_targets[target_idx, grid_y, grid_x] = 1
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    mask_has_ball[target_idx, :, grid_y, grid_x] = 1
                    
                    target_pos[target_idx, 0, grid_y, grid_x] = offset_x/stride
                    target_pos[target_idx, 1, grid_y, grid_x] = offset_y/stride

                    target_mov[target_idx, 0, grid_y, grid_x] = target[4]/640
                    target_mov[target_idx, 1, grid_y, grid_x] = target[5]/640

                    ## cls
                    cls_targets[target_idx, grid_y, grid_x] = 1
            position_loss = 32*self.mse(pred_pos, target_pos)

            move_loss = 640*self.mse(pred_mov, target_mov) # / (1 if mask_has_ball.sum() == 0 else mask_has_ball.sum())

            conf_loss = tracknet_conf_loss(cls_targets, pred_scores, [1, self.hyp.weight_conf], self.batch_count)
            hit_loss = self.hit_bce(pred_hits, hit_targets)
            # conf_loss = focal_loss(pred_scores, cls_targets, alpha=[0.998, 0.002], weight=weight_conf)

            if torch.isnan(position_loss).any() or torch.isinf(position_loss).any():
                LOGGER.warning("NaN or Inf values in position_loss!")
            if torch.isnan(conf_loss).any() or torch.isinf(conf_loss).any():
                LOGGER.warning("NaN or Inf values in conf_loss!")

            if pred_scores.requires_grad:
                pred_binary = pred_scores >= 0.5
                self.TP += torch.sum((pred_binary == 1) & (cls_targets == 1))
                self.FP += torch.sum((pred_binary == 1) & (cls_targets == 0))
                self.TN += torch.sum((pred_binary == 0) & (cls_targets == 0))
                self.FN += torch.sum((pred_binary == 0) & (cls_targets == 1))
                

            # check
            # if (self.batch_count%400 == 0 and pred_scores.requires_grad and idx == 15) or (self.batch_count%20 == 0 and not pred_scores.requires_grad and idx == 15):
            #     pred_conf_all = torch.sigmoid(pred_scores.detach()).cpu()
            #     pred_hits_all = torch.sigmoid(pred_hits.detach()).cpu()
            #     pred_mov_all = pred_mov.detach().clone()
            #     pred_pos_all = pred_pos.detach().clone()
            #     for rand_idx in range(10):
            #         pred_conf = pred_conf_all[rand_idx]
            #         pred__hit = pred_hits_all[rand_idx]
            #         img = batch_img[idx][rand_idx]
            #         x = int(batch_target[idx][rand_idx][2].item())
            #         y = int(batch_target[idx][rand_idx][3].item())
            #         dx = int(batch_target[idx][rand_idx][4].item())
            #         dy = int(batch_target[idx][rand_idx][5].item())
            #         hit = int(batch_target[idx][rand_idx][6].item())

            #         pred_conf_np = pred_conf.numpy()
            #         y_positions, x_positions = np.where(pred_conf_np >= 0.5)
            #         pred_coordinates = list(zip(x_positions, y_positions))

            #         pred_list = []
            #         for pred_x_coordinates, pred_y_coordinates in pred_coordinates:
            #             pred_conf_format = "{:.2f}".format(pred_conf[pred_y_coordinates][pred_x_coordinates].item())
            #             pred_hit_format = "{:.2f}".format(pred__hit[pred_y_coordinates][pred_x_coordinates].item())
            #             pred_list.append((pred_x_coordinates, 
            #                               pred_y_coordinates, 
            #                               pred_pos_all[rand_idx][0][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_pos_all[rand_idx][1][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_mov_all[rand_idx][0][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_mov_all[rand_idx][1][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_conf_format,
            #                               pred_hit_format))

            #         filename = f'{self.batch_count//1185}_{int(self.batch_count%1185)}_{rand_idx}_{pred_scores.requires_grad}'

            #         count_ge_05 = np.count_nonzero(pred_conf >= 0.5)
            #         count_lt_05 = np.count_nonzero(pred_conf < 0.5)
            #         loss_dict = {}
            #         loss_dict['conf_loss'] = conf_loss.item()
            #         loss_dict['position_loss'] = position_loss.item()
            #         loss_dict['moving_loss'] = move_loss.item()
            #         loss_dict['pred_conf >= 0.5 count'] = count_ge_05
            #         loss_dict['pred_conf < 0.5 count'] = count_lt_05
            #         loss_dict['x, y'] = (x%32, y%32)
            #         loss_dict['pred_x, pred_y'] = (pred_pos_all[rand_idx][0][int(y//32)][int(x//32)].item()*32, pred_pos_all[rand_idx][1][int(y//32)][int(x//32)].item()*32)
            #         loss_dict['dx, dy'] = (dx, dy)
            #         loss_dict['pred_dx, pred_dy'] = (pred_mov_all[rand_idx][0][int(y//32)][int(x//32)].item()*640, pred_mov_all[rand_idx][1][int(y//32)][int(x//32)].item()*640)

            #         display_predict_in_checkerboard([(x, y, dx, dy, hit)], pred_list, 'board_'+filename, loss_dict)
            #         display_image_with_coordinates(img, [(x, y, dx, dy)], pred_list, filename, loss_dict)

            loss[0] += position_loss * self.hyp.weight_pos
            loss[1] += move_loss * self.hyp.weight_mov
            loss[2] += conf_loss
            loss[3] += hit_loss

        tlose = loss.sum() * batch_size
        tlose_item = loss.detach()
        self.batch_count+=1

        if preds.requires_grad and self.train_count >= 1185 and self.train_count%1185 == 0:
            if self.TP > 0:
                precision = self.TP/(self.TP+self.FP)
                recall = self.TP/(self.TP+self.FN)
                f1 = (2*precision*recall)/(precision+recall)
            acc = (self.TN + self.TP) / (self.FN+self.FP+self.TN + self.TP)
            print(f"\nTraining Accuracy: {acc:.4f}, Training Precision: {precision:.4f}, Training Recall: {recall:.4f}, , Training F1-Score: {f1:.4f}\n")
            self.TP = 0
            self.FP = 0
            self.TN = 0
            self.FN = 0
        return tlose, tlose_item

# use p3 p4 p5
class TrackNetLoss:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.hyp = h

        m = model.model[-1]  # Detect() module
        self.mse = nn.MSELoss(reduction='sum')
        self.FLM = FocalLossWithMask()
        self.stride = m.stride  # model strides
        self.cell_size = 640/self.stride
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.feat_no = m.feat_no
        self.num_groups = 10
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.xy_loss = XYLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)

        self.sample_path = os.path.join(self.hyp.save_dir, "training_samples")

        self.confusion_class = ConfConfusionMatrix()

    def init_conf_confusion(self, confusion_class):
        self.confusion_class = confusion_class

    def __call__(self, preds, batch):
        loss = torch.zeros(2, device=self.device)

        feats = preds[1] if isinstance(preds, tuple) else preds

        for i, feat in enumerate(feats):
            pred_distri, pred_scores = feat.view(feat.shape[0], self.no, -1).split(
                (self.reg_max * self.feat_no, self.nc), 1)
            
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            b, a, c = pred_distri.shape  # batch, anchors, channels
            pred_pos_distri, n_pred_pos_distri = pred_distri.split([c//2, c//2], 2)
            pred_pos = pred_pos_distri.view(b, a, self.feat_no//2, self.reg_max).softmax(3).matmul(
                self.proj.type(pred_pos_distri.dtype))
            n_pred_pos = n_pred_pos_distri.view(b, a, self.feat_no//2, self.reg_max).softmax(3).matmul(
                self.proj.type(n_pred_pos_distri.dtype))

            batch_target = batch['target'].to(self.device)

            cell_num = int(640/self.stride[i])
            target_pos_distri = torch.zeros(b, self.num_groups, cell_num, cell_num, self.feat_no//2, device=self.device)
            mask_has_ball = torch.zeros(b, self.num_groups, cell_num, cell_num, device=self.device)
            cls_targets = torch.zeros(b, self.num_groups, cell_num, cell_num, 1, device=self.device)

            n_target_pos_distri = torch.zeros(b, self.num_groups, cell_num, cell_num, self.feat_no//2, device=self.device)
            mask_has_next_ball = torch.zeros(b, self.num_groups, cell_num, cell_num, device=self.device)
            n_cls_targets = torch.zeros(b, self.num_groups, cell_num, cell_num, 1, device=self.device)

            for idx, _ in enumerate(batch_target):
                # pred = [330 * cell_num * cell_num]
                stride = self.stride[i]
                
                for target_idx, target in enumerate(batch_target[idx]):
                    # target xy
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    if grid_x >= 80 or grid_y >= 80:
                        print(grid_x, grid_y, offset_x, offset_y)

                    if target[1] == 1:
                        mask_has_ball[idx, target_idx, grid_y, grid_x] = 1
                        center = 0.5
                        def clamp(x, min_value, max_value):
                            return max(min_value, min(x, max_value))
                        t_x = (grid_x*stride+center*stride-target[2])*8/stride
                        t_y = (grid_y*stride+center*stride-target[3])*8/stride
                        if abs(t_x) >= self.reg_max - 1 or abs(t_y) >= self.reg_max - 1:
                            print(f"warning 超過可預測範圍: t_x: {t_x}, t_y: {t_y}")
                        if t_x >= 0:
                            target_pos_distri[idx, target_idx, grid_y, grid_x, 0] = clamp(t_x, 0, self.reg_max-1 - 0.01)
                        else:
                            target_pos_distri[idx, target_idx, grid_y, grid_x, 1] = clamp(-t_x, 0, self.reg_max-1 - 0.01)

                        if t_y >= 0:
                            target_pos_distri[idx, target_idx, grid_y, grid_x, 2] = clamp(t_y, 0, self.reg_max-1 - 0.01)
                        else:
                            target_pos_distri[idx, target_idx, grid_y, grid_x, 3] = clamp(-t_y, 0, self.reg_max-1 - 0.01)

                        ## cls
                        cls_targets[idx, target_idx, grid_y, grid_x, 0] = 1

                        # 當前有球 & 下一個 target 也有球 & 下一個 target 是具有值的，才會計算下一個球的 loss
                        if (target[4] != 0 or target[5] != 0) and target_idx < len(batch_target[idx]) - 1 and batch_target[idx][target_idx + 1][1] == 1:
                            next_gtx = target[4]
                            next_gty = target[5]
                            next_t_x = (grid_x*stride+center*stride-next_gtx)*8/stride
                            next_t_y = (grid_y*stride+center*stride-next_gty)*8/stride

                            if abs(next_t_x) >= self.reg_max - 1 or abs(next_t_y) >= self.reg_max - 1:
                                # print(f"warning 超過可預測範圍: stride: {stride} next_t_x: {next_t_x}, next_t_y: {next_t_y}")
                                continue

                            mask_has_next_ball[idx, target_idx, grid_y, grid_x] = 1
                            n_cls_targets[idx, target_idx, grid_y, grid_x, 0] = 1

                            if next_t_x >= 0:
                                n_target_pos_distri[idx, target_idx, grid_y, grid_x, 0] = clamp(next_t_x, 0, self.reg_max-1 - 0.01)
                            else:
                                n_target_pos_distri[idx, target_idx, grid_y, grid_x, 1] = clamp(-next_t_x, 0, self.reg_max-1 - 0.01)

                            if next_t_y >= 0:
                                n_target_pos_distri[idx, target_idx, grid_y, grid_x, 2] = clamp(next_t_y, 0, self.reg_max-1 - 0.01)
                            else:
                                n_target_pos_distri[idx, target_idx, grid_y, grid_x, 3] = clamp(-next_t_y, 0, self.reg_max-1 - 0.01)

            target_scores_sum = max(cls_targets.sum(), 1)
            n_target_scores_sum = max(n_cls_targets.sum(), 1)

            target_pos_distri = target_pos_distri.view(b, self.num_groups*cell_num*cell_num, self.feat_no//2)
            cls_targets = cls_targets.view(b, self.num_groups*cell_num*cell_num, 1)
            mask_has_ball = mask_has_ball.view(b, self.num_groups*cell_num*cell_num).bool()

            n_target_pos_distri = n_target_pos_distri.view(b, self.num_groups*cell_num*cell_num, self.feat_no//2)
            n_cls_targets = n_cls_targets.view(b, self.num_groups*cell_num*cell_num, 1)
            mask_has_next_ball = mask_has_next_ball.view(b, self.num_groups*cell_num*cell_num).bool()
            
            _, xy_loss = self.xy_loss(pred_pos_distri, pred_pos, target_pos_distri, cls_targets, target_scores_sum, mask_has_ball)
            _, nxny_loss = self.xy_loss(n_pred_pos_distri, n_pred_pos, n_target_pos_distri, n_cls_targets, n_target_scores_sum, mask_has_next_ball)
            loss[0] += xy_loss
            loss[0] += nxny_loss
            cls_targets = cls_targets.to(pred_scores.dtype)

            self.confusion_class.confusion_matrix(pred_scores.sigmoid(), cls_targets)
            loss[1] += self.FLM(pred_scores, cls_targets, 2, 0.75)

        loss[0] *= 1  # dfl gain
        loss[1] *= 20  # cls gain

        tlose = loss.sum() * b
        tlose_item = loss.detach()

        return tlose, tlose_item

# use p3
class TrackNetLossV5:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.hyp = h

        m = model.model[-1]  # Detect() module
        self.mse = nn.MSELoss(reduction='sum')
        self.FLM = FocalLossWithMask()
        self.stride = m.stride  # model strides
        self.cell_size = 640/self.stride
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.feat_no = m.feat_no
        self.num_groups = 10
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.xy_loss = XYLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)

        self.sample_path = os.path.join(self.hyp.save_dir, "training_samples")

        self.confusion_class = ConfConfusionMatrix()

    def init_conf_confusion(self, confusion_class):
        self.confusion_class = confusion_class

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * self.feat_no, self.nc), 1)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        b, a, c = pred_distri.shape  # batch, anchors, channels
        pred_pos_distri = pred_distri
        pred_pos = pred_pos_distri.view(b, a, self.feat_no, c // self.feat_no).softmax(3).matmul(
            self.proj.type(pred_distri.dtype))

        batch_target = batch['target'].to(self.device)

        cell_num = int(640/self.stride[0])
        target_pos_distri = torch.zeros(b, self.num_groups, cell_num, cell_num, self.feat_no, device=self.device)
        mask_has_ball = torch.zeros(b, self.num_groups, cell_num, cell_num, device=self.device)
        cls_targets = torch.zeros(b, self.num_groups, cell_num, cell_num, 1, device=self.device)
        mask_has_next_ball = torch.zeros(b, self.num_groups, cell_num, cell_num, device=self.device)

        mask_fast_ball = torch.zeros(b, self.num_groups, cell_num, cell_num, device=self.device)
        mask_hit_ball = torch.zeros(b, self.num_groups, cell_num, cell_num, device=self.device)
        mask_hit_ball_v2 = torch.zeros(b, self.num_groups, cell_num, cell_num, device=self.device)

        fast_ball_count = 0
        hit_ball_count = 0
        for idx, _ in enumerate(batch_target):
            # pred = [330 * cell_num * cell_num]
            stride = self.stride[0]
            
            for target_idx, target in enumerate(batch_target[idx]):
                # target xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                # 找出快球 => 慢球, 慢球 => 快球
                if target_idx > 1 and target_idx < len(batch_target[idx])-2 and \
                    batch_target[idx][target_idx-2][1] == 1 and batch_target[idx][target_idx][1] == 1 and batch_target[idx][target_idx+2][1] == 1 and\
                    batch_target[idx][target_idx-1][1] == 1 and batch_target[idx][target_idx+1][1] == 1:
                    
                    before_hit2 = [batch_target[idx][target_idx-2][2], batch_target[idx][target_idx-2][3]]
                    hit = [batch_target[idx][target_idx][2], batch_target[idx][target_idx][3]]
                    after_hit2 = [batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3]]

                    before_dist = calculate_dist(before_hit2, hit)
                    after_dist = calculate_dist(hit, after_hit2)
                    angle = calculate_angle(before_hit2, hit, after_hit2)

                    if (angle and angle > 30 and (before_dist > 10 or after_dist > 10)) or \
                        ((before_dist > stride or after_dist > stride) and (before_dist > after_dist*2 or before_dist*2 < after_dist)):

                        grid_x_1, grid_y_1, _, _ = target_grid(batch_target[idx][target_idx-2][2], batch_target[idx][target_idx-2][3], stride)
                        grid_x_2, grid_y_2, _, _ = target_grid(batch_target[idx][target_idx-1][2], batch_target[idx][target_idx-1][3], stride)
                        grid_x_3, grid_y_3, _, _ = target_grid(batch_target[idx][target_idx][2], batch_target[idx][target_idx][3], stride)
                        grid_x_4, grid_y_4, _, _ = target_grid(batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3], stride)
                        grid_x_5, grid_y_5, _, _ = target_grid(batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3], stride)
                        
                        mask_hit_ball_v2[idx, target_idx-2, grid_y_1, grid_x_1] = 1
                        mask_hit_ball_v2[idx, target_idx-1, grid_y_2, grid_x_2] = 1
                        mask_hit_ball_v2[idx, target_idx, grid_y_3, grid_x_3] = 1
                        mask_hit_ball_v2[idx, target_idx+1, grid_y_4, grid_x_4] = 1
                        mask_hit_ball_v2[idx, target_idx+2, grid_y_5, grid_x_5] = 1
                if target_idx < len(batch_target[idx])-1 and batch_target[idx][target_idx+1][1] == 1 and\
                    batch_target[idx][target_idx][1] == 1 and target[4]**2 + target[5]**2 >= cell_num**2:

                    mask_fast_ball[idx, target_idx, grid_y, grid_x] = 1

                    next_grid_x, next_grid_y, _, _ = target_grid(batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3], stride)
                    mask_fast_ball[idx, target_idx+1, next_grid_y, next_grid_x] = 1
                    
                    fast_ball_count+=1

                if target_idx < len(batch_target[idx])-2 \
                    and batch_target[idx][target_idx][1] == 1 and batch_target[idx][target_idx+1][1] == 1 \
                    and batch_target[idx][target_idx+2][1] == 1:
                    
                    first = [batch_target[idx][target_idx][2], batch_target[idx][target_idx][3]]
                    second = [batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3]]
                    third = [batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3]]
                    angle = calculate_angle(first, second, third)
                    dist1 = calculate_dist(first, second)
                    dist2 = calculate_dist(second, third)
                    if angle and angle > 30 and (dist1 > 10 or dist2 > 10):
                        second_grid_x, second_grid_y, _, _ = target_grid(batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3], stride)
                        third_grid_x, third_grid_y, _, _ = target_grid(batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3], stride)
                        mask_hit_ball[idx, target_idx, grid_y, grid_x] = 1
                        mask_hit_ball[idx, target_idx+1, second_grid_y, second_grid_x] = 1
                        mask_hit_ball[idx, target_idx+2, third_grid_y, third_grid_x] = 1
                        hit_ball_count+=1

                if target[1] == 1:
                    mask_has_ball[idx, target_idx, grid_y, grid_x] = 1
                    
                    target_pos_distri[idx, target_idx, grid_y, grid_x, 0] = offset_x*(self.reg_max)/stride
                    target_pos_distri[idx, target_idx, grid_y, grid_x, 1] = offset_y*(self.reg_max)/stride

                    ## cls
                    cls_targets[idx, target_idx, grid_y, grid_x, 0] = 1

                    if target_idx != len(batch_target[idx])-1 and batch_target[idx][target_idx+1][1] == 1:
                        mask_has_next_ball[idx, target_idx, grid_y, grid_x] = 1
        
        mask_may_has_ball = F.max_pool2d(mask_has_ball, kernel_size=3, stride=1, padding=1)

        target_scores_sum = max(cls_targets.sum(), 1)

        target_pos_distri = target_pos_distri.view(b, self.num_groups*cell_num*cell_num, self.feat_no)
        cls_targets = cls_targets.view(b, self.num_groups*cell_num*cell_num, 1)
        mask_has_ball = mask_has_ball.view(b, self.num_groups*cell_num*cell_num).bool()
        mask_may_has_ball = mask_may_has_ball.view(b, self.num_groups*cell_num*cell_num, 1).bool()
        mask_fast_ball = mask_fast_ball.view(b, self.num_groups*cell_num*cell_num, 1).bool()
        mask_hit_ball = mask_hit_ball.view(b, self.num_groups*cell_num*cell_num, 1).bool()
        mask_hit_ball_v2 = mask_hit_ball_v2.view(b, self.num_groups*cell_num*cell_num, 1).bool()
        
        loss = torch.zeros(2, device=self.device)
        _, loss[0] = self.xy_loss(pred_pos_distri, pred_pos, target_pos_distri, cls_targets, target_scores_sum, mask_has_ball, mask_hit_ball_v2)
        
        cls_targets = cls_targets.to(pred_scores.dtype)

        # fp_additional_penalty = 4000
        # fn_additional_penalty = 400
        # cls_weight = torch.where(cls_targets == 1, w_pos + false_negative*fn_additional_penalty, 
        #                          w_neg + false_positive * fp_additional_penalty)
        # bce = nn.BCEWithLogitsLoss(reduction='none', weight=cls_weight)

        self.confusion_class.confusion_matrix(pred_scores.sigmoid(), cls_targets)
        loss[1] = self.FLM(pred_scores, cls_targets, mask_may_has_ball, mask_fast_ball, mask_hit_ball_v2, 2, 0.75)

        # print(f'conf loss: {fp_loss_weighted, fn_loss_weighted, tp_loss_weighted}\n')
        # print(f'fast ball count: {fast_ball_count}, total ball: {target_scores_sum}\n')
        # print(f'hit_ball_count: {hit_ball_count}, total ball: {target_scores_sum}\n')

        loss[0] *= 3  # dfl gain
        loss[1] *= 15  # cls gain
        # loss[2] *= 1  # iou gain

        tlose = loss.sum() * b
        tlose_item = loss.detach()

        return tlose, tlose_item


# tracknet loss without hit and dxdy loss
class TrackNetLossV4:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.hyp = h

        m = model.model[-1]  # Detect() module
        self.mse = nn.MSELoss(reduction='sum')
        self.FLM = FocalLossWithMask()
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.feat_no = m.feat_no
        self.num_groups = 10
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.xy_loss = XYLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)

        self.sample_path = os.path.join(self.hyp.save_dir, "training_samples")

        self.confusion_class = ConfConfusionMatrix()

    def init_conf_confusion(self, confusion_class):
        self.confusion_class = confusion_class

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * self.feat_no, self.nc), 1)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        b, a, c = pred_distri.shape  # batch, anchors, channels
        pred_pos_distri = pred_distri
        pred_pos = pred_pos_distri.view(b, a, self.feat_no, c // self.feat_no).softmax(3).matmul(
            self.proj.type(pred_distri.dtype))

        batch_target = batch['target'].to(self.device)

        target_pos_distri = torch.zeros(b, self.num_groups, 20, 20, self.feat_no, device=self.device)
        mask_has_ball = torch.zeros(b, self.num_groups, 20, 20, device=self.device)
        cls_targets = torch.zeros(b, self.num_groups, 20, 20, 1, device=self.device)
        mask_has_next_ball = torch.zeros(b, self.num_groups, 20, 20, device=self.device)

        mask_fast_ball = torch.zeros(b, self.num_groups, 20, 20, device=self.device)
        mask_hit_ball = torch.zeros(b, self.num_groups, 20, 20, device=self.device)
        mask_hit_ball_v2 = torch.zeros(b, self.num_groups, 20, 20, device=self.device)

        fast_ball_count = 0
        hit_ball_count = 0
        for idx, _ in enumerate(batch_target):
            # pred = [330 * 20 * 20]
            stride = self.stride[0]
            
            for target_idx, target in enumerate(batch_target[idx]):
                # target xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                # 找出快球 => 慢球, 慢球 => 快球
                if target_idx > 1 and target_idx < len(batch_target[idx])-2 and \
                    batch_target[idx][target_idx-2][1] == 1 and batch_target[idx][target_idx][1] == 1 and batch_target[idx][target_idx+2][1] == 1 and\
                    batch_target[idx][target_idx-1][1] == 1 and batch_target[idx][target_idx+1][1] == 1:
                    
                    before_hit2 = [batch_target[idx][target_idx-2][2], batch_target[idx][target_idx-2][3]]
                    hit = [batch_target[idx][target_idx][2], batch_target[idx][target_idx][3]]
                    after_hit2 = [batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3]]

                    before_dist = calculate_dist(before_hit2, hit)
                    after_dist = calculate_dist(hit, after_hit2)
                    angle = calculate_angle(before_hit2, hit, after_hit2)

                    if (angle and angle > 30 and (before_dist > 10 or after_dist > 10)) or \
                        ((before_dist > 32 or after_dist > 32) and (before_dist > after_dist*2 or before_dist*2 < after_dist)):

                        grid_x_1, grid_y_1, _, _ = target_grid(batch_target[idx][target_idx-2][2], batch_target[idx][target_idx-2][3], stride)
                        grid_x_2, grid_y_2, _, _ = target_grid(batch_target[idx][target_idx-1][2], batch_target[idx][target_idx-1][3], stride)
                        grid_x_3, grid_y_3, _, _ = target_grid(batch_target[idx][target_idx][2], batch_target[idx][target_idx][3], stride)
                        grid_x_4, grid_y_4, _, _ = target_grid(batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3], stride)
                        grid_x_5, grid_y_5, _, _ = target_grid(batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3], stride)
                        
                        mask_hit_ball_v2[idx, target_idx-2, grid_y_1, grid_x_1] = 1
                        mask_hit_ball_v2[idx, target_idx-1, grid_y_2, grid_x_2] = 1
                        mask_hit_ball_v2[idx, target_idx, grid_y_3, grid_x_3] = 1
                        mask_hit_ball_v2[idx, target_idx+1, grid_y_4, grid_x_4] = 1
                        mask_hit_ball_v2[idx, target_idx+2, grid_y_5, grid_x_5] = 1
                if target_idx < len(batch_target[idx])-1 and batch_target[idx][target_idx+1][1] == 1 and\
                    batch_target[idx][target_idx][1] == 1 and target[4]**2 + target[5]**2 >= 20**2:

                    mask_fast_ball[idx, target_idx, grid_y, grid_x] = 1

                    next_grid_x, next_grid_y, _, _ = target_grid(batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3], stride)
                    mask_fast_ball[idx, target_idx+1, next_grid_y, next_grid_x] = 1
                    
                    fast_ball_count+=1

                if target_idx < len(batch_target[idx])-2 \
                    and batch_target[idx][target_idx][1] == 1 and batch_target[idx][target_idx+1][1] == 1 \
                    and batch_target[idx][target_idx+2][1] == 1:
                    
                    first = [batch_target[idx][target_idx][2], batch_target[idx][target_idx][3]]
                    second = [batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3]]
                    third = [batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3]]
                    angle = calculate_angle(first, second, third)
                    dist1 = calculate_dist(first, second)
                    dist2 = calculate_dist(second, third)
                    if angle and angle > 30 and (dist1 > 10 or dist2 > 10):
                        second_grid_x, second_grid_y, _, _ = target_grid(batch_target[idx][target_idx+1][2], batch_target[idx][target_idx+1][3], stride)
                        third_grid_x, third_grid_y, _, _ = target_grid(batch_target[idx][target_idx+2][2], batch_target[idx][target_idx+2][3], stride)
                        mask_hit_ball[idx, target_idx, grid_y, grid_x] = 1
                        mask_hit_ball[idx, target_idx+1, second_grid_y, second_grid_x] = 1
                        mask_hit_ball[idx, target_idx+2, third_grid_y, third_grid_x] = 1
                        hit_ball_count+=1

                if target[1] == 1:
                    mask_has_ball[idx, target_idx, grid_y, grid_x] = 1
                    
                    target_pos_distri[idx, target_idx, grid_y, grid_x, 0] = offset_x*(self.reg_max-1)/stride
                    target_pos_distri[idx, target_idx, grid_y, grid_x, 1] = offset_y*(self.reg_max-1)/stride

                    ## cls
                    cls_targets[idx, target_idx, grid_y, grid_x, 0] = 1

                    if target_idx != len(batch_target[idx])-1 and batch_target[idx][target_idx+1][1] == 1:
                        mask_has_next_ball[idx, target_idx, grid_y, grid_x] = 1
        
        mask_may_has_ball = F.max_pool2d(mask_has_ball, kernel_size=3, stride=1, padding=1)

        target_scores_sum = max(cls_targets.sum(), 1)

        target_pos_distri = target_pos_distri.view(b, self.num_groups*20*20, self.feat_no)
        cls_targets = cls_targets.view(b, self.num_groups*20*20, 1)
        mask_has_ball = mask_has_ball.view(b, self.num_groups*20*20).bool()
        mask_may_has_ball = mask_may_has_ball.view(b, self.num_groups*20*20, 1).bool()
        mask_fast_ball = mask_fast_ball.view(b, self.num_groups*20*20, 1).bool()
        mask_hit_ball = mask_hit_ball.view(b, self.num_groups*20*20, 1).bool()
        mask_hit_ball_v2 = mask_hit_ball_v2.view(b, self.num_groups*20*20, 1).bool()
        
        loss = torch.zeros(2, device=self.device)
        _, loss[0] = self.xy_loss(pred_pos_distri, pred_pos, target_pos_distri, cls_targets, target_scores_sum, mask_has_ball, mask_hit_ball_v2)
        
        cls_targets = cls_targets.to(pred_scores.dtype)

        # fp_additional_penalty = 4000
        # fn_additional_penalty = 400
        # cls_weight = torch.where(cls_targets == 1, w_pos + false_negative*fn_additional_penalty, 
        #                          w_neg + false_positive * fp_additional_penalty)
        # bce = nn.BCEWithLogitsLoss(reduction='none', weight=cls_weight)

        self.confusion_class.confusion_matrix(pred_scores.sigmoid(), cls_targets)
        loss[1] = self.FLM(pred_scores, cls_targets, mask_may_has_ball, mask_fast_ball, mask_hit_ball_v2, 2, 0.75)

        # print(f'conf loss: {fp_loss_weighted, fn_loss_weighted, tp_loss_weighted}\n')
        # print(f'fast ball count: {fast_ball_count}, total ball: {target_scores_sum}\n')
        # print(f'hit_ball_count: {hit_ball_count}, total ball: {target_scores_sum}\n')

        loss[0] *= 3  # dfl gain
        loss[1] *= 15  # cls gain
        # loss[2] *= 1  # iou gain

        tlose = loss.sum() * b
        tlose_item = loss.detach()

        return tlose, tlose_item

# tracknet loss without hit loss
class TrackNetLossV3:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.hyp = h
        # print(self.hyp.weight_conf, self.hyp.weight_mov, self.hyp.weight_pos, self.hyp.use_dxdy_loss)

        m = model.model[-1]  # Detect() module
        self.mse = nn.MSELoss(reduction='mean')
        self.FLM = FocalLossWithMask()
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.dxdy_no = m.dxdy_no
        self.no = m.no
        self.reg_max = m.reg_max
        self.feat_no = m.feat_no
        self.num_groups = 10
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.xy_loss = XYLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)

        self.sample_path = os.path.join(self.hyp.save_dir, "training_samples")

        self.confusion_class = ConfConfusionMatrix()

    def init_conf_confusion(self, confusion_class):
        self.confusion_class = confusion_class

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores, pred_dxdy = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * self.feat_no, self.nc, self.dxdy_no), 1)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_dxdy = pred_dxdy.permute(0, 2, 1).contiguous()
        pred_dxdy = torch.tanh(pred_dxdy)

        b, a, c = pred_distri.shape  # batch, anchors, channels
        pred_pos_distri = pred_distri
        pred_pos = pred_pos_distri.view(b, a, self.feat_no, c // self.feat_no).softmax(3).matmul(
            self.proj.type(pred_distri.dtype))

        batch_target = batch['target'].to(self.device)

        target_pos_distri = torch.zeros(b, self.num_groups, 20, 20, self.feat_no, device=self.device)
        mask_has_ball = torch.zeros(b, self.num_groups, 20, 20, device=self.device)
        mask_may_has_ball = torch.zeros(b, self.num_groups, 20, 20, device=self.device)
        cls_targets = torch.zeros(b, self.num_groups, 20, 20, 1, device=self.device)
        mask_has_next_ball = torch.zeros(b, self.num_groups, 20, 20, device=self.device)
        target_mov = torch.zeros(b, self.num_groups, 20, 20, 2, device=self.device)
        for idx, _ in enumerate(batch_target):
            # pred = [330 * 20 * 20]
            stride = self.stride[0]
            
            for target_idx, target in enumerate(batch_target[idx]):
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    mask_has_ball[idx, target_idx, grid_y, grid_x] = 1
                    
                    target_pos_distri[idx, target_idx, grid_y, grid_x, 0] = offset_x*(self.reg_max-1)/stride
                    target_pos_distri[idx, target_idx, grid_y, grid_x, 1] = offset_y*(self.reg_max-1)/stride

                    ## cls
                    cls_targets[idx, target_idx, grid_y, grid_x, 0] = 1

                    if target_idx != len(batch_target[idx])-1 and batch_target[idx][target_idx+1][1] == 1:
                        mask_has_next_ball[idx, target_idx, grid_y, grid_x] = 1
                        target_mov[idx, target_idx, grid_y, grid_x, 0] = target[4]/640
                        target_mov[idx, target_idx, grid_y, grid_x, 1] = target[5]/640
        mask_may_has_ball = F.max_pool2d(mask_has_ball, kernel_size=3, stride=1, padding=1)
        
        target_scores_sum = max(cls_targets.sum(), 1)

        target_pos_distri = target_pos_distri.view(b, self.num_groups*20*20, self.feat_no)
        cls_targets = cls_targets.view(b, self.num_groups*20*20, 1)
        mask_has_ball = mask_has_ball.view(b, self.num_groups*20*20).bool()
        mask_may_has_ball = mask_may_has_ball.view(b, self.num_groups*20*20, 1).bool()
        target_mov = target_mov.view(b, self.num_groups*20*20, 2)
        mask_has_next_ball = mask_has_next_ball.view(b, self.num_groups*20*20).bool()
        
        loss = torch.zeros(3, device=self.device)
        a, loss[0] = self.xy_loss(pred_pos_distri, pred_pos, target_pos_distri, cls_targets, target_scores_sum, mask_has_ball)
        
        cls_targets = cls_targets.to(pred_scores.dtype)

        self.confusion_class.confusion_matrix(pred_scores.sigmoid(), cls_targets)
        loss[1] = self.FLM(pred_scores, cls_targets, mask_may_has_ball, 2, 0.8)
        loss[2] = self.mse(pred_dxdy[mask_has_next_ball], target_mov[mask_has_next_ball]) if mask_has_next_ball.sum() > 0 else 0

        # print(f'conf loss: {fp_loss_weighted, fn_loss_weighted, tp_loss_weighted}\n')

        loss[0] *= 3  # dfl gain
        loss[1] *= 320  # cls gain
        # loss[3] *= 1  # iou gain
        loss[2] *= 10000  # dxdy gain

        tlose = loss.sum() * b
        tlose_item = loss.detach()

        return tlose, tlose_item
class FocalLossWithMask(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    # OHEM
    def top_k_sampling(self, loss, labels, negative_ratio=3.0):
        """
        Hard Negative Mining: Selects the hardest negative examples based on the loss.
        """
        pos_mask = labels > 0
        num_pos = pos_mask.sum(dim=1, keepdim=True)
        num_neg = negative_ratio * num_pos

        loss_for_sort = loss.clone()
        loss_for_sort[pos_mask] = float('-inf')
        _, indices = loss_for_sort.sort(dim=1, descending=True)

        neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        for i in range(loss.size(0)):  
            num_neg_samples = int(num_neg[i].item()) if int(num_neg[i].item()) != 0 else int(negative_ratio)
            # num_neg_samples = 640
            # num_neg_samples = int(num_neg[i].item())
            neg_mask[i, indices[i, :num_neg_samples]] = True 

        return pos_mask | neg_mask

    def forward(self, pred, label, gamma=2, alpha=0.75, negative_ratio=3.0):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        assert torch.all((label == 0) | (label == 1)), f"`label` contains invalid values: {label.unique()}"
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        assert (loss >= 0).all(), f"`loss` contains negative values. Min: {loss.min()}, Max: {loss.max()}"

        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        
        #避免梯度爆炸
        #min_value = torch.finfo(pred.dtype).eps  # 取當前數據類型的機器精度
        #p_t = torch.clamp(p_t, min=min_value, max=1 - min_value)
        modulating_factor = (1.000001 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            if (alpha_factor < 0).any():
                print("Negative alpha_factor detected:", alpha_factor[alpha_factor < 0])
            loss *= alpha_factor
        
        TP_mask = (pred_prob >= 0.5) & (label == 1)  # True Positive
        FN_mask = (pred_prob < 0.5) & (label == 1)   # False Negative
        FP_mask = (pred_prob >= 0.5) & (label == 0)  # False Positive

        # Combine the masks (we only care about TP, FN, FP)
        relevant_mask = self.top_k_sampling(loss, label, negative_ratio)

        pos_no = label.sum() if label.sum() != 0 else 1

        w = (alpha/(1-alpha))

        loss = loss * relevant_mask.float()

        # loss[FN_mask] *= negative_ratio*20*w
        # loss[FP_mask & ~may_has_ball] *= negative_ratio*15*w
        # loss[TP_mask] *= negative_ratio*10
        # loss[mask_hit_ball] *= negative_ratio*10*w*3

        # print(f'fast and hit count: {(mask_fast_ball|mask_hit_ball).sum()}')
        # print(f'fast and hit with relevant count: {(loss[mask_fast_ball|mask_hit_ball] > 0).sum()}')
        # print(f'relevant loss count: {(loss > 0).sum()}')

        # Apply the mask to the loss
        loss_sum = (loss).sum()
        loss = loss_sum / max(relevant_mask.float().sum(), 1) if loss_sum != 0 else 0
        # count = (pred_prob >= 0.5) | (label == 1)
        # loss = loss_sum / max(count.sum(), 1) if loss_sum != 0 else 0

        return loss

class XYLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_pos, target_pos_distri, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        target_scores_clone = target_scores.clone()
        # target_scores_clone[hit_weight] *= 30
        weight = target_scores_clone.sum(-1)[fg_mask].unsqueeze(-1)
        # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # iou = gaussian_iou(pred_pos[fg_mask], target_pos_distri[fg_mask], sigma=0.7)
        # loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            assert target_pos_distri.min() >= 0 and target_pos_distri.max() <= self.reg_max+1, \
                f"Target out of range: min={target_pos_distri.min()}, max={target_pos_distri.max()}"

            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_pos_distri[fg_mask]) * weight
            assert (loss_dfl >= 0).all(), f"`loss` contains negative values. Min: {loss_dfl.min()}, Max: {loss_dfl.max()}"
            loss_dfl_sum = loss_dfl.sum()
                
            loss_dfl = (loss_dfl_sum / target_scores_sum) if loss_dfl_sum != 0 else 0
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_dfl, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

def gaussian_iou(pred_pos, target_pos, sigma=1.0):
    """
    Compute Gaussian IoU for two sets of points in (x, y) format.
    Each point is assumed to be the center of a Gaussian distribution with fixed variance.
    
    Args:
        pred_pos: Predicted positions (N, 2), where each row is (x, y).
        target_pos: Target positions (N, 2), where each row is (x, y).
        sigma: Variance (same for both x and y) for the Gaussian distribution.
    
    Returns:
        giou: Gaussian IoU values for each pair of predicted and target positions (N,).
    """
    # Calculate the squared Euclidean distance between predicted and target points
    dist_squared = ((pred_pos - target_pos) ** 2).sum(dim=-1)

    # Gaussian IoU based on distance and variance (sigma)
    giou = torch.exp(-dist_squared / (2 * sigma ** 2))

    return giou

def save_pred_and_loss(predictions, loss, filename, t_xy):
    """
    Save the predictions and loss into a CSV file.
    :param predictions: A tensor of shape [10, 20, 20] containing the predictions with gradient information.
    :param loss: A scalar representing the loss value.
    :param filename: The name of the file to save the data.
    """
    # Detach predictions from the current graph and convert to numpy
    if predictions.requires_grad:
        predictions = predictions.detach()
        loss = loss.detach()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    with open(filename+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each 20x20 prediction along with the loss
        i=0
        for pred in predictions:
            max_position = torch.argmax(t_xy[i])
            max_y, max_x = np.unravel_index(max_position, t_xy[i].shape)
            # Flatten the 20x20 prediction to a single row
            flattened_pred = pred.flatten()
            # Append the loss and write to the file
            writer.writerow(list(flattened_pred) + [loss] +[(max_x, max_y)])
            i+=1
def save_pos_mov_loss(predictions, loss, filename, target):
    """
    Save the predictions and loss into a CSV file.

    :param predictions: A tensor of shape [10, 20, 20] containing the predictions with gradient information.
    :param loss: A scalar representing the loss value.
    :param filename: The name of the file to save the data.
    """
    # Detach predictions from the current graph and convert to numpy
    if predictions.requires_grad:
        predictions = predictions.detach()
        loss = loss.detach()

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    with open(filename+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each 20x20 prediction along with the loss
        i=0
        for pred in predictions:
            # Flatten the 20x20 prediction to a single row
            flattened_pred = pred.flatten()
            # Append the loss and write to the file
            writer.writerow(list(flattened_pred) + [loss])
            i+=1
        for pred in target:
            # Flatten the 20x20 prediction to a single row
            flattened_pred = pred.flatten()
            # Append the loss and write to the file
            writer.writerow(list(flattened_pred))
            i+=1
            


def tracknet_conf_loss(y_true, y_pred, class_weight, batch_count):
    y_pred = torch.sigmoid(y_pred)  

    custom_weights = torch.square(1 - y_pred) * y_true + torch.square(y_pred) * (1 - y_true)
    # custom_weights = (1 - y_pred) * y_true + (y_pred) * (1 - y_true)

    # y_true_squeeze  = y_true.any(dim=0, keepdim=True).repeat(10, 1, 1).int()

    class_weights = class_weight[0] * (1 - y_true) + class_weight[1] * y_true

    loss = (-1) * class_weights * custom_weights * (y_true * torch.log(torch.clamp(y_pred, min=torch.finfo(y_pred.dtype).eps, max=1)) + 
                                                    (1 - y_true) * torch.log(torch.clamp(1 - y_pred, min=torch.finfo(y_pred.dtype).eps, max=1)))
    
    loss = torch.sum(loss)
    
    # save loss detail
    # if (batch_count%400 == 0 and y_pred.requires_grad) or (batch_count%20 == 0 and not y_pred.requires_grad):
    #     filename = f'{self.sample_path}/{batch_count//1185}_{int(batch_count%1185)}_{y_pred.requires_grad}'
    #     y_true_cpu = y_true.cpu()
    #     save_pred_and_loss(y_pred, loss, filename, y_true_cpu)
    return loss

def focal_loss(pred_logits, targets, alpha=0.95, gamma=2.0, epsilon=1e-3, weight=10):
    """
    :param pred_logits: 預測的logits, shape [batch_size, 1, H, W]
    :param targets: 真實標籤, shape [batch_size, 1, H, W]
    :param alpha: 用於平衡正、負樣本的權重。這裡可以是一個scalar或一個list[alpha_neg, alpha_pos]。
    :param gamma: 用於調節著重於正確或錯誤預測的程度
    :return: focal loss
    """
    pred_probs = torch.sigmoid(pred_logits)

    pred_probs = torch.clamp(pred_probs, epsilon, 1.0-epsilon)  # log(0) 會導致無窮大
    if isinstance(alpha, (list, tuple)):
        alpha_neg = alpha[0]
        alpha_pos = alpha[1]
    else:
        alpha_neg = (1 - alpha)
        alpha_pos = alpha

    pt = torch.where(targets == 1, pred_probs, 1 - pred_probs)
    alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
    
    ce_loss = -torch.log(pt)
    # if torch.isinf(ce_loss).any():
    #     LOGGER.warning("ce_loss value is infinite!")
    fl = alpha_t * (1 - pt) ** gamma * ce_loss
    test = fl.mean() * weight_conf
    return test