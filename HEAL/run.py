import argparse
import pickle
import os  # Importação para verificar a existência dos arquivos
from Tiling import tiling
from Pre_processing import pre_processing
from Data_split import data_split
from Training import train
from Independent_test import independent_test
from Hyperparameter_optimisation import hyperparameter_optimisation
from Data_visualisation import data_visualisation
from Grad_Cam import grad_cam


def save_variable(var, filename):
    with open(filename, 'wb') as pickle_f:
        pickle.dump(var, pickle_f)
    return filename


def load_variable(filename):
    with open(filename, 'rb') as pickle_f:
        var = pickle.load(pickle_f)
    return var


def run(label_file=None, testing_label_file=None, models=None, training_mode=None, procedure=None, tile_info=None):
    # Verifique se label_file não é None
    if label_file is None:
        raise ValueError("O argumento 'label_file' é obrigatório e não pode ser None.")
    if testing_label_file is None:
        print("Nenhum arquivo de teste fornecido. Usando divisão do arquivo de rótulos principal.")

    # Checar se o arquivo de rótulos existe
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"O arquivo de rótulos '{label_file}' não foi encontrado.")
    
    # Checar se o arquivo de teste existe (se fornecido)
    if testing_label_file and not os.path.exists(testing_label_file):
        raise FileNotFoundError(f"O arquivo de teste '{testing_label_file}' não foi encontrado.")
    
    _label_file_tiled = "HEAL_Workspace/outputs/label_file_tiled.csv" if label_file else None
    _test_label_file_tiled = "HEAL_Workspace/outputs/test_label_file_tiled.csv" if testing_label_file else None

    # Loop para executar os procedimentos especificados
    for _proc in procedure:
        if _proc == "Tiling":
            print("[INFO] Start tiling ...")
            tiling.tiling(label_file, testing_label_file, _tile_size=tile_info[0], _tile_level=tile_info[1])
        elif _proc == "Pre_processing":
            print("[INFO] Image pre-processing ...")
            pre_processing.pre_processing()
        elif _proc == "Data_split":
            print("[INFO] Data split ...")
            data_split.data_split(_label_file_tiled, _test_label_file_tiled)
        elif _proc == "Hyperparameter_optimisation":
            print("[INFO] Hyperparameter optimization ...")
            hyperparameter_optimisation.tuning()
        elif _proc == "Training":
            print("[INFO] Training ...")
            train.train(models, tile_size=tile_info[0], CV_Enable=(training_mode == "Cross_validation"))
        elif _proc == "Testing":
            print("[INFO] Testing ...")
            independent_test.independent_test(models, tile_info)
        elif _proc == "Data_visualisation":
            print("[INFO] Data visualization ...")
            data_visualisation.data_visualisation(tile_info)
        elif _proc == "Grad_CAM":
            print("[INFO] Grad-CAM visualization ...")
            grad_cam.grad_cam()

    print(label_file, testing_label_file, models, training_mode, procedure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HEAL Pipeline")
    
    # Definindo os argumentos
    parser.add_argument("--label_file", type=str, required=True, help="Path to the label file")
    parser.add_argument("--testing_label_file", type=str, help="Path to the testing label file")
    parser.add_argument("--models", type=str, nargs='+', required=True, help="List of models to use")
    parser.add_argument("--training_mode", type=str, choices=["single_round", "Cross_validation"], default="single_round", help="Training mode")
    parser.add_argument("--procedure", type=str, nargs='+', required=True, help="Procedure steps to run")
    parser.add_argument("--tile_info", type=int, nargs=2, default=[1000, 0], help="Tile size and level")

    args = parser.parse_args()

    # Verificar se o arquivo de rótulos existe e é legível
    if not os.path.isfile(args.label_file):
        raise FileNotFoundError(f"O arquivo de rótulos '{args.label_file}' não foi encontrado ou não é um arquivo válido.")
    
    if args.testing_label_file and not os.path.isfile(args.testing_label_file):
        raise FileNotFoundError(f"O arquivo de rótulos de teste '{args.testing_label_file}' não foi encontrado ou não é um arquivo válido.")

    # Executando a função com os argumentos
    run(label_file=args.label_file,
        testing_label_file=args.testing_label_file,
        models=args.models,
        training_mode=args.training_mode,
        procedure=args.procedure,
        tile_info=args.tile_info)

