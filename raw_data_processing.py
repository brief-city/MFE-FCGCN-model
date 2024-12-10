import mne
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Data
import torch
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression


def cal_mutual_information(df):
    mi_matrix = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            mi = mutual_info_regression(df[[col1]], df[col2], discrete_features='auto')
            mi_matrix.loc[col1, col2] = mi[0]

    mi_matrix_np = mi_matrix.to_numpy()
    row_maxes = mi_matrix_np.max(axis=1, keepdims=True)
    final_matrix = np.divide(mi_matrix_np, row_maxes, where=row_maxes != 0)
    df_final_matrix = pd.DataFrame(final_matrix, columns=df.columns, index=df.columns)

    return df_final_matrix


def create_new_adjacent_matrix(df):
    df_processed = df.copy()
    df_processed = df_processed.abs()
    np.fill_diagonal(df_processed.values, np.nan)
    for i in range(len(df_processed)):
        row = df_processed.iloc[i]
        row_sum = row.sum()
        if row_sum != 0:
            df_processed.iloc[i] = row.apply(lambda x: x / row_sum)

    for i in range(len(df)):
        df_processed.iloc[i, i] = df.iloc[i, i]

    return df_processed


def is_element_in_list(element, list_to_check):
    return element in list_to_check

def eeglist_set():
    eegset_list = []
    for i in range(1, 66):
        if i <= 9:
            path = f'F:/gcn_main/derivatives/sub-00{i}/eeg/sub-00{i}_task-eyesclosed_eeg.set'
            path_label = 1
            eegset_list.append([path, path_label,i])
        if 10 <= i <= 36:
            path = f'F:/gcn_main/derivatives/sub-0{i}/eeg/sub-0{i}_task-eyesclosed_eeg.set'
            path_label = 1
            eegset_list.append([path, path_label,i])
        if 37 <= i <= 65:
            path = f'F:/gcn_main/derivatives/sub-0{i}/eeg/sub-0{i}_task-eyesclosed_eeg.set'
            path_label = 0
            eegset_list.append([path, path_label,i])

    return eegset_list

def create_five_sample_extension(data_list):
    number=0
    sample_num=0
    ad_sample_num=0
    hc_sample_num=0
    class_sample_num={}

    for path_set,label,pattern in data_list:
        interested_channel = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5',
                              'T6', 'Fz', 'Cz', 'Pz']

        if label==1:
            max_sample_num = 80

        if label==0:
            max_sample_num = 100

        label=[label]*19
        df_label = pd.DataFrame([label], columns=interested_channel)
        raw = mne.io.read_raw_eeglab(path_set, preload=True)
        sfreq = raw.info['sfreq']

        hz_min = [0.5, 4, 8, 13, 30]
        hz_max = [4, 8, 13, 30, 45]
        frequency_name = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

        for fr_name, (fmin, fmax) in enumerate(zip(hz_min, hz_max)):
            psd_data = {}
            for name in interested_channel:
                data, times = raw[name, :]
                psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq, fmin=fmin, fmax=fmax, n_fft=2500,average=None,n_overlap=500)
                psds_normalized=psds[0].T
                psd_data[name] = psds_normalized
            i = 0
            for num in range(len(psd_data['Fp1'])):
                if i>=max_sample_num:
                    break
                if i>len(psd_data['Fp1'])-4:
                    break
                psd_temporary_values = {}

                for ch_name in interested_channel:
                    psds_temporary = psd_data[ch_name][i:i+4]
                    psds_temporary_mean = np.mean(psds_temporary, axis=0)
                    sub_psds_normalized = (psds_temporary_mean - psds_temporary_mean.mean()) / psds_temporary_mean.std()
                    psd_temporary_values[ch_name] = sub_psds_normalized

                df_psd = pd.DataFrame(psd_temporary_values, columns=interested_channel)
                df_correlation = pd.DataFrame(psd_temporary_values, columns=interested_channel)

                df_mul_information = cal_mutual_information(df_correlation)
                df_person_corr = pd.DataFrame(psd_temporary_values, columns=interested_channel)

                corr_matrix = df_person_corr.corr()
                df_person= create_new_adjacent_matrix(corr_matrix)

                if df_label.values[0,0]==1:
                    ad_sample_num+=1
                if df_label.values[0,0]==0:
                    hc_sample_num+=1

                all_psd_csv = pd.concat([df_psd, df_label], axis=0, ignore_index=True)

                if pattern<=9:
                    path_csv = 'F:/dataset_all_object/AD_HC_dataset/sample/sub-00{}/feature_matrix/sub-{}-{}-sample-{}.csv'.format(pattern,frequency_name[fr_name], i+1,pattern)
                    path_mul_matrix = 'F:/dataset_all_object/AD_HC_dataset/sample/sub-00{}/adjacent_matrix/mul-mat-{}-{}-sample-{}.csv'.format(pattern,frequency_name[fr_name],i+1,pattern)
                    path_person_matrix = 'F:/dataset_all_object/AD_HC_dataset/sample/sub-00{}/adjacent_matrix/person-mat-{}-{}-sample-{}.csv'.format(pattern, frequency_name[fr_name], i + 1, pattern)
                if pattern>=10:
                    path_csv = 'F:/dataset_all_object/AD_HC_dataset/sample/sub-0{}/feature_matrix/sub-{}-{}-sample-{}.csv'.format(pattern,frequency_name[fr_name],i+1,pattern)
                    path_mul_matrix = 'F:/dataset_all_object/AD_HC_dataset/sample/sub-0{}/adjacent_matrix/mul-mat-{}-{}-sample-{}.csv'.format(pattern, frequency_name[fr_name], i + 1, pattern)
                    path_person_matrix = 'F:/dataset_all_object/AD_HC_dataset/sample/sub-0{}/adjacent_matrix/person-mat-{}-{}-sample-{}.csv'.format(pattern, frequency_name[fr_name], i + 1, pattern)


                df_mul_information.to_csv(path_mul_matrix)
                df_person.to_csv(path_person_matrix)
                all_psd_csv.to_csv(path_csv)

                sample_num+=1
                number+=1
                i += 1

    all_number=int(number/5)
    class_sample_num['AD']=[int(ad_sample_num/5)]
    class_sample_num['HC']=[int(hc_sample_num/5)]
    print(f'The total number of samples is {all_number},'
          f'the total number of AD samples is {int(ad_sample_num/5)},'
          f'the total number of HC samples is {int(hc_sample_num/5)}.')


if __name__ == '__main__':
    eeg_list=eeglist_set()
    create_five_sample_extension(eeg_list)



