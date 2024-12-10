import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
from torch.utils.data import Subset
import mne
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau
from tqdm import tqdm
import optuna
from torch_geometric.nn import GCNConv, ChebConv
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from functools import partial
import json
import os

def dataset_list_set():
    eegset_list = []
    for i in range(1, 66):
        if i <= 9 or (10 <= i <= 36):
            path = f'F:/dataset_all_object/AD_HC_dataset/sample/sub-{i}'
            path_label = 1
        elif 37 <= i <= 65:
            path = f'F:/dataset_all_object/AD_HC_dataset/sample/sub-{i}'
            path_label = 0
        eegset_list.append([path, path_label])
    print(eegset_list)
    print(len(eegset_list))
    return eegset_list

def k_select_data(data):
    data_df = pd.DataFrame(data, columns=['file_path', 'label'])
    train_data, test_data = train_test_split(data_df, test_size=0.2, stratify=data_df['label'])
    all_folds = []
    all_folds.append({
        'train': train_data[['file_path', 'label']].values.tolist(),
        'test': test_data[['file_path', 'label']].values.tolist()})
    return all_folds

def calculate_file_number(path):
    folder_path = Path(path)
    num_files = sum(1 for item in folder_path.iterdir() if item.is_file())
    print(f'There are {num_files} files in the directory "{folder_path}".')
    return num_files
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cnn_18 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3),padding=(0,0),stride=1)
        self.cnn_21 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,6),padding=(0,0),stride=1)
        self.cnn_26 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,11),padding=(0,0),stride=1)
        self.cnn_76 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,61),padding=(0,0),stride=1)
        self.cnn_86 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,71),padding=(0,0),stride=1)

    def forward(self,data):
        feature_dim = data.x.size(1)
        x = data.x
        x= x.view(-1,1,19,feature_dim)

        x=x.to(torch.float)
        if feature_dim == 18:
            x = self.cnn_18(x)

        if feature_dim == 21:
            x = self.cnn_21(x)

        if feature_dim == 26:
            x = self.cnn_26(x)

        if feature_dim == 76:
            x = self.cnn_76(x)

        if feature_dim == 86:
            x = self.cnn_86(x)

        data_batch=data.batch.tolist()
        batch_size_model=len(set(data_batch))
        x = x.view(batch_size_model*19,-1).to(torch.float32)
        data.x = x
        return data

class GCN(nn.Module):
    def __init__(self, max_features, GCN1_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(max_features, GCN1_channels)
        self.conv2 = GCNConv(GCN1_channels, out_channels)
        self.cnn=SimpleCNN()
        self.max_features = max_features
        self.out_channels = out_channels

    def forward(self, data):
        data=self.cnn(data)
        x, edge_index, edge_weight, batch, edge_index_two, edge_weight_two = data.x, data.edge_index, data.edge_attr, data.batch, data.edge_index_two, data.edge_weight_two
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.long)
        edge_weight = edge_weight.to(torch.float32)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = x.to(torch.float32)
        edge_index_two = edge_index_two.to(torch.long)
        edge_weight_two = edge_weight_two.to(torch.float32)
        x = self.conv2(x, edge_index=edge_index_two, edge_weight=edge_weight_two)

        x = x.view(-1, 19 * self.out_channels)
        x = x.to(torch.float32)

        return x

class FiveGraphsModel(nn.Module):
    def __init__(self, max_features, GCN1_channels, out_channels, lin1_channels, lin2_channels, fc_out_channels, dropout_lin1_rate, dropout_lin2_rate):
        super(FiveGraphsModel, self).__init__()
        self.gcn = GCN(max_features, GCN1_channels, out_channels)
        self.fc = nn.Linear(5 * 19 * out_channels, lin1_channels)
        self.lin2 = nn.Linear(lin1_channels, lin2_channels)
        self.lin3 = nn.Linear(lin2_channels, fc_out_channels)
        self.dropout_lin1_rate = dropout_lin1_rate
        self.dropout_lin2_rate = dropout_lin2_rate

    def forward(self, data_list):
        concat = torch.cat([self.gcn(data) for data in data_list], dim=1)
        concat = concat.to(torch.float32)
        out = self.fc(concat)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout_lin1_rate, training=self.training)
        out = self.lin2(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout_lin2_rate, training=self.training)
        out = self.lin3(out)
        return out

class MyCustomDataset(Dataset):
    def __init__(self, data_list, root=None, transform=None, pre_transform=None):
        super(MyCustomDataset, self).__init__(root, transform, pre_transform)
        self.data_list = data_list
        self.labels = [data[0].y.item() for data in data_list]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def train_use(train_loader):
    model.train()
    correct = 0
    t_loss = 0
    for data_train in tqdm(train_loader, desc="Train Processing"):
        data = [data_gpu.to(device) for data_gpu in data_train]
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data[0].y)
        pred = out.softmax(dim=1).argmax(dim=1)
        correct += (pred == data[0].y).sum().item()
        t_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_evaluate['train_loss'].append(t_loss)
    cor = correct / len(inner_train)
    train_evaluate['train_accuracy'].append(cor)
    return t_loss, cor

def val_use(val_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    test_loss = 0
    correct = 0
    for data_test in tqdm(val_loader, desc="Val Processing"):
        data = [data_gpu.to(device) for data_gpu in data_test]
        out = model(data)
        y_scores_temporary = out.softmax(dim=1)[:, 1]
        y_scores_temporary = y_scores_temporary.detach().cpu().numpy()
        y_scores.append(y_scores_temporary)
        pred = out.softmax(dim=1).argmax(dim=1)
        y_pred.append(pred.cpu().numpy())
        y_true.append(data[0].y.cpu().numpy())
        loss = criterion(out, data[0].y)
        test_loss += loss.cpu().item()
        correct += int((pred == data[0].y).sum().item())
    y_true_list = [element for sublist in y_true for element in sublist]
    y_pred_list = [element for sublist in y_pred for element in sublist]
    y_scores_list = [element for sublist in y_scores for element in sublist]
    sensitivity, specificity, f1, auc_values = cal_evaluation_indicator(y_true=y_true_list, y_pred=y_pred_list,
                                                                        y_scores=y_scores_list)
    cor = correct / len(val)
    val_evaluate['val_loss'].append(test_loss)
    val_evaluate['val_accuracy'].append(cor)
    val_evaluate['val_sensitivity'].append(sensitivity)
    val_evaluate['val_specificity'].append(specificity)
    val_evaluate['val_f1'].append(f1)
    val_evaluate['val_auc'].append(auc_values)

    print(f'本次验证集指标，如下:')
    print('=======================')
    print(f'准确率:{cor:.4f}')
    print(f'灵敏度:{sensitivity:.4f}')
    print(f'特异度:{specificity:.4f}')
    print(f'f1分数:{f1:.4f}')
    print(f'AUC值:{auc_values:.4f}')
    print('=======================')
    return cor, test_loss, sensitivity, specificity, f1, auc_values

def test_use(test_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    test_loss = 0
    correct = 0
    for data_test in tqdm(test_loader, desc="Test Processing"):
        data = [data_gpu.to(device) for data_gpu in data_test]
        out = model(data)
        y_scores_temporary = out.softmax(dim=1)[:, 1]
        y_scores_temporary = y_scores_temporary.detach().cpu().numpy()
        y_scores.append(y_scores_temporary)
        pred = out.softmax(dim=1).argmax(dim=1)
        y_pred.append(pred.cpu().numpy())
        y_true.append(data[0].y.cpu().numpy())
        loss = criterion(out, data[0].y)
        test_loss += loss.cpu().item()
        correct += int((pred == data[0].y).sum().item())
    y_true_list = [element for sublist in y_true for element in sublist]
    y_pred_list = [element for sublist in y_pred for element in sublist]
    y_scores_list = [element for sublist in y_scores for element in sublist]
    sensitivity,specificity, f1, auc_values = cal_evaluation_indicator(y_true=y_true_list, y_pred=y_pred_list, y_scores=y_scores_list)
    cor = correct / len(test_loader.dataset)
    test_evaluate['test_loss'].append(test_loss)
    test_evaluate['test_accuracy'].append(cor)
    test_evaluate['test_sensitivity'].append(sensitivity)
    test_evaluate['test_specificity'].append(specificity)
    test_evaluate['test_f1'].append(f1)
    test_evaluate['test_auc'].append(auc_values)

    print(f'本次测试集指标，如下:')
    print('=======================')
    print(f'准确率:{cor:.4f}')
    print(f'灵敏度:{sensitivity:.4f}')
    print(f'特异度:{specificity:.4f}')
    print(f'f1分数:{f1:.4f}')
    print(f'AUC值:{auc_values:.4f}')
    print('=======================')
    return cor, test_loss,sensitivity, specificity, f1, auc_values

def cal_evaluation_indicator(y_true, y_pred, y_scores):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    f1 = f1_score(y_true, y_pred, pos_label=1)

    roc_auc = roc_auc_score(y_true, y_scores)

    return sensitivity, specificity, f1, roc_auc

def find_number_train_test_list(fold):
    train_name_list = fold['train']
    test_name_list = fold['test']
    train_figure_name_list = []
    for train_i in train_name_list:
        num_train_path = train_i[0]
        last_number = os.path.basename(num_train_path).split('-')[-1]
        train_figure_name_list.append(last_number)
    test_figure_name_list = []
    for test_i in test_name_list:
        num_test_path = test_i[0]
        last_number = os.path.basename(num_test_path).split('-')[-1]
        test_figure_name_list.append(last_number)
    return train_figure_name_list, test_figure_name_list

def load_pt_dataset(num_list):
    pt_list = []
    for pt_num in num_list:
        pt_path = f'F:/dataset_all_object/AD_HC_dataset/dataset_pt/sub-{pt_num}/graph_{pt_num}.pt'
        pt_list.append(pt_path)
    dataset_pt_list = []
    for path_data_pt in pt_list:
        loaded_data = torch.load(path_data_pt)
        for item in loaded_data:
            dataset_pt_list.append(item)
    pt_data_all = MyCustomDataset(dataset_pt_list)
    pt_dataset = pt_data_all[:]
    return pt_dataset

def inner_k_fold_data_preparation(data, k, random_state):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    all_folds = []
    for train_idx, val_idx in skf.split(data,data.labels):
        train_fold = Subset(data, train_idx)
        val_fold = Subset(data, val_idx)
        all_folds.append((train_fold, val_fold))
    return all_folds

def save_initial_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Initial model saved to {save_path}")

def save_best_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to {save_path}")

def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Model loaded from {load_path}")
    return model

def test_best_model(model,path):
    model = load_model(model,path, device)
    test_acc, test_sub_loss, test_sensitivity, test_specificity, test_f1scores, test_auc = test_use(test_loader)
    print(
        f'Loaded Best AUC Model Test Acc:{test_acc:.4f}, Test Sensitivity:{test_sensitivity:.4f}, Test Specificity:{test_specificity:.4f}, Test f1:{test_f1scores:.4f}, Test AUC:{test_auc:.4f}')

if __name__ == '__main__':
    data = dataset_list_set()
    all_folds = k_select_data(data)
    best_model_save_path = 'F:/dataset_all_object/AD_HC_dataset/best_model/best_auc_model.pth'

    for i, fold in enumerate(all_folds):
        if i+1==1:
            print(f"Fold {i + 1}:")
            print(f"Train size: {len(fold['train'])}, Sample: {fold['train']}")
            print(f"Test size: {len(fold['test'])}, Sample: {fold['test']}")
            df_train = pd.DataFrame(fold['train'], columns=['path', 'label'])
            df_test = pd.DataFrame(fold['test'], columns=['path', 'label'])
            file_path_train = f'F:/dataset_all_object/AD_HC_dataset/kfold_raw/fold_{i + 1}/train_fold_{i + 1}.csv'
            file_path_test = f'F:/dataset_all_object/AD_HC_dataset/kfold_raw/fold_{i + 1}/test_fold_{i + 1}.csv'
            df_train.to_csv(file_path_train)
            df_test.to_csv(file_path_test)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            train_num, test_num = find_number_train_test_list(fold)
            train_dataset = load_pt_dataset(train_num)
            test_dataset = load_pt_dataset(test_num)
            print(f'训练集的长度{len(train_dataset)}')
            print(f'测试集的长度{len(test_dataset)}')

            inner_folds = inner_k_fold_data_preparation(train_dataset, k=15, random_state=1)
            for j, (inner_train, val) in enumerate(inner_folds):
                if j+1==1:
                    print(f'Inner Fold {j + 1}:')
                    print(f'Inner Train size: {len(inner_train)}, Validation size: {len(val)}')
                    train_loader = DataLoader(inner_train, batch_size=64, shuffle=True)
                    val_loader = DataLoader(val, batch_size=128, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
                    criterion = torch.nn.CrossEntropyLoss()
                    model = FiveGraphsModel(
                        max_features=16,
                        GCN1_channels=16,
                        out_channels=5,
                        lin1_channels=30,
                        lin2_channels=14,
                        fc_out_channels=2,
                        dropout_lin1_rate=0.8,
                        dropout_lin2_rate=0.5
                    ).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.00012, weight_decay=1e-4)
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

                    val_best_auc = 0

                    train_evaluate = {'train_loss': [], 'train_accuracy': []}
                    val_evaluate = {'val_loss': [], 'val_accuracy': [], 'val_sensitivity': [], 'val_specificity': [], 'val_f1': [], 'val_auc': []}

                    test_evaluate = {'test_loss': [], 'test_accuracy': [], 'test_sensitivity': [],
                                     'test_specificity': [], 'test_f1': [], 'test_auc': []}
                    for epoch in tqdm(range(200)):
                        tra_los, train_zhuque = train_use(train_loader)
                        val_acc, val_sub_loss, val_sensitivity, val_specificity, val_f1scores, val_auc = val_use(val_loader)
                        print('训练集指标')
                        print(f'Epoch: {epoch:03d}, Train Loss: {tra_los:.4f}, Train Acc:{train_zhuque:.4f},Valid Loss:{val_sub_loss:.4f}')
                        print('验证集指标')
                        print(f'Val Acc:{val_acc:.4f},Val Sensitivity:{val_sensitivity:.4f},Valid Specificity:{val_specificity:.4f},Valid f1:{val_f1scores:.4f},Valid AUC:{val_auc:.4f}')

                        test_acc, test_sub_loss, test_sensitivity, test_specificity, test_f1scores, test_auc = test_use(test_loader)
                        print(f'Test Acc:{test_acc:.4f},Test Sensitivity:{test_sensitivity:.4f},Test Specificity:{test_specificity:.4f},Test f1:{test_f1scores:.4f},Tese AUC:{test_auc:.4f}')
                        if val_auc > val_best_auc:
                            val_best_auc = val_auc
                            save_best_model(model, best_model_save_path)
                        scheduler.step(test_sub_loss)
                        print(f'==========本{epoch:03d}轮训练结束============')
                    print('程序运行结束')
                    df_train_evaluate = pd.DataFrame(train_evaluate)
                    df_val_evaluate = pd.DataFrame(val_evaluate)
                    df_test_evaluate=pd.DataFrame(test_evaluate)
                    path_train = f'F:/dataset_all_object/AD_HC_dataset/evaluate/fold_{i + 1}/train_evaluate_{j + 1}.csv'
                    path_val = f'F:/dataset_all_object/AD_HC_dataset/evaluate/fold_{i + 1}/val_evaluate_{j + 1}.csv'
                    path_test = f'F:/dataset_all_object/AD_HC_dataset/evaluate/fold_{i + 1}/test_evaluate_{j + 1}.csv'
                    df_train_evaluate.to_csv(path_train)
                    df_val_evaluate.to_csv(path_val)
                    df_test_evaluate.to_csv(path_test)

                else:
                    break

        else:
            break

    model = FiveGraphsModel(
        max_features=16,
        GCN1_channels=16,
        out_channels=5,
        lin1_channels=30,
        lin2_channels=14,
        fc_out_channels=2,
        dropout_lin1_rate=0.8,
        dropout_lin2_rate=0.5
    ).to(device)

    test_best_model(model,best_model_save_path)
