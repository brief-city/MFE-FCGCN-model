import torch
from pathlib import Path
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data

def dataset_list_set():
    eegset_list = []
    for i in range(1,66):
        if i <= 9:
            path = f'F:/dataset_all_object/AD_HC_dataset/sample/sub-00{i}'
            path_label = 1
            eegset_list.append([path, path_label,i])
        if 10 <= i <= 36:
            path = f'F:/dataset_all_object/AD_HC_dataset/sample/sub-0{i}'
            path_label = 1
            eegset_list.append([path, path_label,i])
        if 37 <= i <= 65:
            path = f'F:/dataset_all_object/AD_HC_dataset/sample/sub-0{i}'
            path_label = 0
            eegset_list.append([path, path_label,i])
    print(eegset_list)
    print(len(eegset_list))
    return eegset_list


def calculate_file_number(path):
    folder_path = Path(path)
    num_files = sum(1 for item in folder_path.iterdir() if item.is_file())
    print(f'There are {num_files} files in the directory "{folder_path}".')
    return num_files

def create_graphs(sample_list,fold):
    data_graph = []
    ad_num = 0
    nc_num = 0
    frequency_name = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    sample_path=sample_list[0]
    label=sample_list[1]

    sub_id = int(sample_path.split('-')[-1])

    if sub_id <= 9:
        sub_folder = f"sub-00{sub_id}"
    else:
        sub_folder = f"sub-0{sub_id}"

    path_file_number = f'F:/dataset_all_object/AD_HC_dataset/sample/{sub_folder}/feature_matrix'
    files_number = calculate_file_number(path_file_number)
    files_number = int(files_number / 5)

    for idx in range(1, files_number + 1):
        graph = []
        for fr_name in frequency_name:

            file_path = f'F:/dataset_all_object/AD_HC_dataset/sample/{sub_folder}/feature_matrix/sub-{fr_name}-{idx}-sample-{sub_id}.csv'
            path_mat = f'F:/dataset_all_object/AD_HC_dataset/sample/{sub_folder}/adjacent_matrix/mul-mat-{fr_name}-{idx}-sample-{sub_id}.csv'
            path_person=f'F:/dataset_all_object/AD_HC_dataset/sample/{sub_folder}/adjacent_matrix/person-mat-{fr_name}-{idx}-sample-{sub_id}.csv'

            columns = pd.read_csv(file_path, nrows=0).columns.tolist()
            use_columns = columns[1:]
            nrows = len(pd.read_csv(file_path)) - 1
            df_csv = pd.read_csv(file_path, nrows=nrows, usecols=use_columns).values.T
            features = torch.tensor(df_csv)
            df_label = pd.read_csv(file_path, skiprows=nrows, nrows=1, usecols=[i for i in range(1, 20)]).values[0, 1]
            df_label = torch.tensor(df_label, dtype=torch.long)
            columns = pd.read_csv(path_mat, nrows=0).columns.tolist()
            use_columns = columns[1:]
            adjacency_raw = pd.read_csv(path_mat, usecols=use_columns)
            adjacency = torch.tensor(adjacency_raw.values)
            edge_index = torch.tensor(np.transpose(np.nonzero(adjacency)), dtype=torch.int64)
            edge_weight = torch.tensor(adjacency[edge_index[0], edge_index[1]])
            edge_weight = edge_weight.unsqueeze(-1)

            columns_two = pd.read_csv(path_person, nrows=0).columns.tolist()
            use_columns_two = columns_two[1:]
            adjacency_raw_two = pd.read_csv(path_person, usecols=use_columns_two)
            adjacency_two = torch.tensor(adjacency_raw_two.values)

            edge_index_two = torch.tensor(np.transpose(np.nonzero(adjacency_two)), dtype=torch.int64)
            edge_weight_two = torch.tensor(adjacency_two[edge_index_two[0], edge_index_two[1]])
            edge_weight_two = edge_weight_two.unsqueeze(-1)

            data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight, y=df_label)

            data.edge_index_two=edge_index_two
            data.edge_weight_two=edge_weight_two

            graph.append(data)
        data_graph.append(graph)
        if label == 1:
            ad_num += 1
        elif label == 0:
            nc_num += 1

    print(f'AD 患者的数目为:{ad_num}')
    print(f'NC 的数目为:{nc_num}')

    path_pt = f'F:/dataset_all_object/AD_HC_dataset/dataset_pt/sub-{fold}/graph_{fold}.pt'
    torch.save(data_graph, path_pt)
    return ad_num,nc_num


if __name__ == '__main__':
    ad_number = 0
    hc_number = 0
    all_number = 0
    data_list=dataset_list_set()
    for i in range(65):
        sample=data_list[i][0:2]
        fold_num=data_list[i][2]
        ad_num,hc_num=create_graphs(sample_list=sample,fold=fold_num)
        ad_number=ad_number+ad_num
        hc_number=hc_number+hc_num

    print(f'AD患者的数量为：{ad_number}')
    print(f'HC患者的数量为：{hc_number}')
    print(f'样本总数为：{ad_number+hc_number}')

