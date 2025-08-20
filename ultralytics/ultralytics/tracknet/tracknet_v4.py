import copy
import argparse

import torch

from ultralytics.tracknet.utils.confusion_matrix import ConfConfusionMatrix
from .val import TrackNetValidator
from .dataset import TrackNetDataset
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from .utils.loss import TrackNetLoss
from ultralytics.nn.tasks import DetectionModel

class TrackNetV4Model(DetectionModel):
    def init_criterion(self):
        if not hasattr(self, 'track_net_loss'):
            self.track_net_loss = TrackNetLoss(self)
        return self.track_net_loss
    def init_conf_confusion(self):
        if not hasattr(self, 'track_net_loss'):
            self.track_net_loss = TrackNetLoss(self)
        self.track_net_loss.init_conf_confusion(ConfConfusionMatrix())
    def print_confusion_matrix(self):
        if not hasattr(self, 'track_net_loss'):
            self.track_net_loss = TrackNetLoss(self)
        self.track_net_loss.confusion_class.print_confusion_matrix()