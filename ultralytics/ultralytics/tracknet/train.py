from ultralytics.tracknet.configurable_dataset import TrackNetConfigurableDataset
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.tracknet_v4 import TrackNetV4Model
from ultralytics.tracknet.val import TrackNetValidator
from ultralytics.tracknet.val_dataset import TrackNetValDataset
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from copy import copy
from torch.utils.data import random_split
import torch

class TrackNetTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        # generator = torch.Generator().manual_seed(42)
        # dataset = TrackNetConfigurableDataset(root_dir=img_path)
        # train_size = int(0.8 * len(dataset))  # 70% 的數據作為訓練集
        # val_size = len(dataset) - train_size  # 剩下的 30% 作為驗證集
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)

        if mode == 'train':
            dataset = TrackNetConfigurableDataset(root_dir=img_path)
            return dataset
        else:
            dataset = TrackNetValDataset(root_dir=img_path)
            return dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        self.tracknet_model = TrackNetV4Model(cfg, ch=10, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            self.tracknet_model.load(weights)
        return self.tracknet_model
    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

        # batch['img'] = batch['img'].to(self.device, non_blocking=True)

        # if self.args.half and self.device.type == "cuda":
        #     batch['img'] = batch['img'].half() / 255.0  # `float16`
        # else:
        #     batch['img'] = batch['img'].float() / 255.0  # `float32`

        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch

    def get_validator(self):
        # self.loss_names = 'pos_loss', 'mov_loss', 'conf_loss', 'hit_loss'
        self.loss_names = 'pos_loss', 'conf_loss'
        return TrackNetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        self.add_callback("print_confusion_matrix", self.tracknet_model.print_confusion_matrix())
        self.add_callback("init_conf_confusion", self.tracknet_model.init_conf_confusion())
        return ('\n' + '%11s' *
                (3 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Size')
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLOv5 training."""
        pass
    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass