import csv
import torch

class ConfConfusionMatrix:
    def __init__(self):
        self.conf_TP = 0
        self.conf_TN = 0
        self.conf_FP = 0
        self.conf_FN = 0
        self.conf_acc = 0
        self.conf_precision = 0
        self.recall = 0
        self.threshold = 0.8
    def confusion_matrix(self, pred_probs, cls_targets):
        pred_binary = (pred_probs >= self.threshold).int()

        unique_classes = torch.unique(cls_targets)
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
            conf_matrix = confusion_matrix_gpu(cls_targets, pred_binary)
            self.conf_TN += conf_matrix[0][0]
            self.conf_FP += conf_matrix[0][1]
            self.conf_FN += conf_matrix[1][0]
            self.conf_TP += conf_matrix[1][1]
    def get_acc(self):
        if (self.conf_FN + self.conf_FP + self.conf_TN + self.conf_TP) != 0:
            self.conf_acc = (self.conf_TN + self.conf_TP) / (self.conf_FN + self.conf_FP + self.conf_TN + self.conf_TP)
        return self.conf_acc
    def get_precision(self):
        if (self.conf_TP+self.conf_FP) != 0:
            self.conf_precision = self.conf_TP/(self.conf_TP+self.conf_FP)
        return self.conf_precision
    def get_recall(self):
        if (self.conf_TP+self.conf_FN) != 0:
            self.recall = self.conf_TP/(self.conf_TP+self.conf_FN)
        return self.conf_precision
    def print_confusion_matrix(self):
        header = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        self.get_acc()
        self.get_precision()
        self.get_recall()
        f1 = 0
        if (self.conf_precision+self.recall) != 0:
            f1 = 2*self.conf_precision*self.recall/(self.conf_precision+self.recall)
        data_row = [self.conf_TN, self.conf_FP, self.conf_FN, self.conf_TP, self.conf_acc, self.conf_precision, self.recall, f1]
        csv_file_path = r'/Users/bartek/git/BartekTao/datasets/tracknet/val_confusion_matrix/conf_matrix.csv'
        csv_file_path = r'/usr/src/datasets/tracknet/val_confusion_matrix/conf_matrix.csv'
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if file.tell() == 0:
                writer.writerow(header)
            
            # Write the data row
            writer.writerow(data_row)
        print(f"\nTN: {self.conf_TN}, FP: {self.conf_FP}, FN: {self.conf_FN}, TP: {self.conf_TP}")
        print(f"acc: {self.conf_acc}, precision: {self.conf_precision}, recall: {self.recall}, f1 score: {f1}\n")


def confusion_matrix_gpu(y_true, y_pred):
    conf_matrix = torch.zeros(2, 2, dtype=torch.int64, device=y_true.device)
    
    conf_matrix[0, 0] = torch.sum((y_true == 0) & (y_pred == 0))  # True Negative (TN)
    conf_matrix[0, 1] = torch.sum((y_true == 0) & (y_pred == 1))  # False Positive (FP)
    conf_matrix[1, 0] = torch.sum((y_true == 1) & (y_pred == 0))  # False Negative (FN)
    conf_matrix[1, 1] = torch.sum((y_true == 1) & (y_pred == 1))  # True Positive (TP)
    
    return conf_matrix