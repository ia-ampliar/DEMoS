#!python
# -*- coding: UTF-8 -*-

"""
Function description:
    Split the dataset into training/testing/validation;
    To generate the configuration for the model training.
"""

import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import multiprocessing
import re
from itertools import product


def save_variable(var, filename):
    """Save a variable to a file using pickle serialization.

    This function takes a variable and a filename as input, serializes the variable using 
    the pickle module, and saves it to the specified file. The function returns the filename 
    where the variable has been saved.

    Args:
        var (any): The variable to be saved to a file.
        filename (str): The name of the file where the variable will be stored.

    Returns:
        str: The filename where the variable has been saved.
    """
    pickle_f = open(filename, 'wb')
    pickle.dump(var, pickle_f)
    pickle_f.close()
    return filename


def customized_drop(_df):
    """Remove rows from the DataFrame based on the existence of corresponding directories.

    This function checks each row in the provided DataFrame to determine if the directory 
    specified in the first column exists and is not empty. Rows corresponding to non-existent 
    or empty directories are dropped from the DataFrame.

    Args:
        _df (pandas.DataFrame): The DataFrame containing paths to be validated.

    Returns:
        pandas.DataFrame: A new DataFrame with rows removed based on directory checks.
    """

    drop_idx = []
    for i in range(len(_df)):
        _ori_folder_path = _df.iloc[i, 0]
        _folder_path = re.sub('tiling', 'tiling_macenko', _ori_folder_path)
        if not os.path.exists(_folder_path):
            print(_folder_path)
            drop_idx.append(i)
        elif not os.listdir(_folder_path):
            print(_folder_path)
            drop_idx.append(i)
    _df = _df.drop(index=drop_idx)
    _df = _df.reset_index(drop = True)
    return _df


def read_files(train_label_file, test_label_file=None):
    """Read training and testing label files and process their contents.

    Args:
        train_label_file (str): Path to the training label CSV file.
        test_label_file (str, optional): Path to the testing label CSV file.

    Returns:
        tuple: (train_df, test_df, MLMC)
    """
    # Verificar se o arquivo de treinamento existe
    if not os.path.exists(train_label_file):
        raise FileNotFoundError(f"O arquivo de treinamento '{train_label_file}' não foi encontrado.")
    
    train_df = pd.read_csv(train_label_file)
    
    # Verificar se o DataFrame está vazio ou se a coluna 'Label' está presente e preenchida
    if train_df.empty or 'Label' not in train_df.columns or train_df['Label'].isnull().all():
        raise ValueError("O arquivo de treinamento está vazio ou a coluna 'Label' está ausente/vazia.")
    
    MLMC = False  # Flag para classificação multi-label
    _tmp_label = train_df['Label'].astype(str).tolist()  # Garantir que todos os valores sejam strings
    _class_category = set()  # Usar set para evitar duplicatas

    for _item in _tmp_label:
        _item = _item.split(",")  # Dividir string em lista, se for multi-label
        _class_category.update(_item)  # Atualizar categorias únicas
        if len(_item) > 1:
            MLMC = True  # Verificar multi-label
    
    _class_category = list(_class_category)  # Converter para lista
    tmp_dict = {"Mode": MLMC, "Classes": _class_category, "Class_number": len(_class_category)}
    save_variable(tmp_dict, "HEAL_Workspace/outputs/parameter.conf")
    
    # Tentar ler o arquivo de teste, se fornecido
    test_df = None
    if test_label_file:
        if not os.path.exists(test_label_file):
            print(f"O arquivo de teste '{test_label_file}' não foi encontrado. Usando 20% do treinamento como teste.")
        else:
            test_df = pd.read_csv(test_label_file)
            print(f"O comprimento original do arquivo de teste é {len(test_df)}.")
    
    # Mostrar a distribuição de classes se for single-label
    if not MLMC:
        visualize_class_distribution(train_df, _class_category)
    
    return train_df, test_df, MLMC

def visualize_class_distribution(df, class_category):
    """Visualizar a distribuição de classes."""
    output_dir = "HEAL_Workspace/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.close('all')
    plt.style.use("ggplot")
    plt.xlabel("Class")
    plt.ylabel("Number of WSIs")
    plt.title("Data distribution of the Whole Slide Images (WSIs)")
    num_class = len(class_category)
    plt.hist(x=df['Label'], bins=np.arange(num_class+1)-0.5, color='#0504aa', alpha=0.7, rwidth=0.5)
    plt.grid(alpha=0.5, axis='y')
    plt.savefig(os.path.join(output_dir, "patient_vs_class_distribution.png"), dpi=400)
    plt.close()


def find_files(ori_folder_path, label):
    """Find image files in a specified folder and associate them with a label.

    This function searches for JPEG image files in a given directory, constructs their file paths, 
    and creates a DataFrame that associates each image path with a specified label. The resulting 
    DataFrame contains two columns: 'Image_path' and 'Label'.

    Args:
        ori_folder_path (str): The original folder path where images are located.
        label (str): The label to associate with the found images.

    Returns:
        pandas.DataFrame: A DataFrame containing the paths of the found images and their associated label.
    """
    img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
    #print(img_label_df)
    folder_path = re.sub('tiling', 'tiling_macenko', ori_folder_path)

    for files in os.listdir(folder_path):
        #print(files)
        if (files.split('.')[-1] == 'jpeg' or files.split('.')[-1] == 'jpg') and not files[0] == '.':
            img_path = os.path.join(folder_path, files)
            tmp = pd.DataFrame([[img_path, label]], columns=['Image_path', 'Label'])
            img_label_df = pd.concat([img_label_df, tmp])

    return img_label_df


result_df = pd.DataFrame(columns=['Image_path', 'Label'])

def log_result(result):
    """Log the result by appending it to a global DataFrame.

    This function takes a result DataFrame and concatenates it to a global DataFrame named 
    `result_df`. This allows for the accumulation of results over multiple calls to this function.

    Args:
        result (pandas.DataFrame): The DataFrame containing results to be logged.

    Returns:
        None
    """
    global result_df
    result_df = pd.concat([result_df, result])


def show_tiles_distribution(df):
    """Display the distribution of tiles across different classes.

    This function utilizes multiprocessing to find and log image files associated with each class 
    in the provided DataFrame. It then generates a histogram to visualize the distribution of tiles 
    for each class, providing insights into the dataset's composition.

    Args:
        df (pandas.DataFrame): A DataFrame containing image folder paths and their associated labels.

    Returns:
        None
    """
    #img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
    cpu_num = multiprocessing.cpu_count()
    print("The CPU number of this machine is %d" % cpu_num)
    pool = multiprocessing.Pool(int(cpu_num))
    #pool = multiprocessing.Pool(16)
    for idx in tqdm.tqdm(range(len(df))):
        label = df.iloc[idx, 1]
        folder_path = df.iloc[idx, 0]
        pool.apply_async(find_files, (folder_path, label), callback = log_result)
        #log_result(find_files(folder_path, label))
    pool.close()
    pool.join()
    plt.close('all')
    plt.style.use("ggplot")
    #matplotlib.rcParams['font.family'] = "Arial"
    plt.xlabel("Class")
    plt.ylabel("Number of tiles")
    plt.title("Data distribution of the tiles")
    num_class = len(set(list(result_df.iloc[:, 1])))
    plt.hist(x=list(result_df.iloc[:, 1]), bins=np.arange(num_class+1)-0.5,
             color='#a0040a', alpha=0.7, rwidth=0.5, align='mid')
    
    
def split_data(train_df, test_df, test_ratio, MLMC):
    """Split training data into training, testing, and validation sets.

    This function divides the provided training DataFrame into training, testing, and validation 
    subsets based on the specified test ratio and whether multi-label classification is being used. 
    It handles both cases where a separate test DataFrame is provided or needs to be created from 
    the training data, ensuring that the splits maintain the distribution of labels.

    Args:
        train_df (pandas.DataFrame): The DataFrame containing training data with image paths and labels.
        test_df (pandas.DataFrame or None): The DataFrame containing testing data or None if to be created from train_df.
        test_ratio (float): The proportion of the training data to be used for testing.
        MLMC (bool): A flag indicating whether multi-label classification is being used.

    Returns:
        tuple: A tuple containing the temporary training DataFrame, lists of training indices, 
               testing indices, and validation indices.

    Raises:
        ValueError: If the training DataFrame is empty or required columns are missing.
    """
    train_indices, test_indices, val_indices = [], [], []
    tmp_df = None

    # Verificar se train_df tem dados
    if train_df.empty:
        raise ValueError("O DataFrame de treinamento está vazio.")

    if test_df is None:
        if MLMC:
            split_train_vs_test = ShuffleSplit(n_splits=1, test_size=test_ratio)
            for train_index, test_index in split_train_vs_test.split(train_df):
                train_indices = list(train_index)  # Certificar-se de que seja uma lista
                test_indices = list(test_index)  # Certificar-se de que seja uma lista
        else:
            if 'Image_path' not in train_df.columns or 'Label' not in train_df.columns:
                raise ValueError("Colunas 'Image_path' ou 'Label' ausentes em train_df.")
            
            split_train_vs_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio)
            for train_index, test_index in split_train_vs_test.split(train_df['Image_path'], train_df['Label']):
                train_indices = list(train_index)
                test_indices = list(test_index)

        tmp_df = train_df.iloc[train_indices].reset_index(drop=True)
        test_tmp_df = train_df.iloc[test_indices]
        test_tmp_df.to_csv("HEAL_Workspace/outputs/test_label_file_tiled.csv",
                           encoding='utf-8', index=False)
        
        if MLMC:
            split_train_vs_val = ShuffleSplit(n_splits=10, test_size=0.2)
            for a, b in split_train_vs_val.split(tmp_df):
                train_indices.append(list(a))  # Append para manter as divisões
                val_indices.append(list(b))    # Append para manter as divisões
        else:
            if 'Image_path' not in tmp_df.columns or 'Label' not in tmp_df.columns:
                raise ValueError("Colunas 'Image_path' ou 'Label' ausentes em tmp_df.")
                
            split_train_vs_val = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
            for a, b in split_train_vs_val.split(tmp_df['Image_path'], tmp_df['Label']):
                train_indices.append(list(a))  # Append para manter as divisões
                val_indices.append(list(b))    # Append para manter as divisões

    else:
        # Verificar se train_df e test_df têm dados válidos
        if train_df.empty or (not MLMC and ('Image_path' not in train_df.columns or 'Label' not in train_df.columns)):
            raise ValueError("O DataFrame de treinamento está vazio ou colunas ausentes.")
        
        try:
            if MLMC:
                split_train_vs_val = ShuffleSplit(n_splits=10, test_size=0.2)
                for a, b in split_train_vs_val.split(train_df):
                    train_indices.append(list(a))  # Append para manter as divisões
                    val_indices.append(list(b))    # Append para manter as divisões
            else:
                if 'Image_path' not in train_df.columns or 'Label' not in train_df.columns:
                    raise ValueError("Colunas 'Image_path' ou 'Label' ausentes em train_df.")
                
                split_train_vs_val = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
                for a, b in split_train_vs_val.split(train_df['Image_path'], train_df['Label']):
                    train_indices.append(list(a))  # Append para manter as divisões
                    val_indices.append(list(b))    # Append para manter as divisões

        except Exception as e:
            print(e)
        test_indices = None

    return tmp_df, train_indices, test_indices, val_indices



train_img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
val_img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])


def process_fold(data_df, indices, fold_number, output_file):
    results_df = pd.DataFrame(columns=['Image_path', 'Label', 'type', 'fold'])

    cpu_num = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(int(cpu_num))

    if isinstance(indices, (list, np.ndarray)):
        for idx in tqdm.tqdm(range(len(indices))):
            label = data_df.iloc[idx, 1]
            folder_path = data_df.iloc[idx, 0]
            pool.apply_async(find_files, args=(folder_path, label), callback=lambda result: results_df.append(result, ignore_index=True))
    else:
        print(f"Índices não são listas ou arrays: {indices}")

    pool.close()
    pool.join()

    # Obter o caminho absoluto
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(output_file, encoding='utf-8', index=False)


def write_train_file(train_df, train_indices, val_indices):
    """Write training and validation data to CSV files for each fold."""
    
    for i in range(10):
        if i < len(train_indices) and i < len(val_indices):
            train_index = train_indices[i]
            val_index = val_indices[i]

            if isinstance(train_index, (list, np.ndarray)) and isinstance(val_index, (list, np.ndarray)):
                train = train_df.iloc[train_index]
                val = train_df.iloc[val_index]

                # Gera o caminho absoluto para evitar erros de diretório
                process_fold(train, train_index, i, f"{os.path.abspath('HEAL_Workspace/outputs/train_fold_' + str(i) + '.csv')}")
                process_fold(val, val_index, i, f"{os.path.abspath('HEAL_Workspace/outputs/val_fold_' + str(i) + '.csv')}")
            else:
                print(f"Índices de treinamento ou validação não são listas ou arrays para o fold {i}: "
                      f"train_index={train_index}, val_index={val_index}")
        else:
            print(f"Índice fora dos limites para o fold: {i}")


def data_split(train_label_file, test_label_file=None, test_ratio=0.2):
    train_df, test_df, MLMC = read_files(train_label_file, test_label_file)
    
    # Verificar se train_df não está vazio
    if train_df.empty:
        raise ValueError("O DataFrame de treinamento está vazio após a leitura do arquivo.")
    
    if not MLMC:
        show_tiles_distribution(train_df)
    
    tmp_df, train_indices, test_indices, val_indices = split_data(train_df, test_df, test_ratio, MLMC)
    
    # Verificar se os índices estão vazios
    if not train_indices or (test_indices is None and not val_indices):
        raise ValueError("Os índices resultantes da divisão de dados estão vazios.")

    write_train_file(tmp_df, train_indices, val_indices)