# MFE-FCGCN-model
The repository is for the code and data of the MFE-FCGCN model.

The dataset is sourced from the paper 'A Dataset of Scalp EEG Recordings of Alzheimer’s Disease, Frontotemporal Dementia, and Healthy Subjects from Routine EEG'. The dataset can be downloaded from the following URL: https://openneuro.org/datasets/ds004504/versions/1.0.8.The raw dataset can be downloaded from the website, so it will not be uploaded here.

This code consists of three parts: the raw_data_processing section calculates PSD features and two types of functional connectivity relationships from the raw EEG dataset files; the generate_graph_data section converts the extracted data into graph data format and saves it; the model_and_train section includes the model architecture and training procedure. The main Python libraries used are PyTorch and PYG, and the Python version used is 3.11.
