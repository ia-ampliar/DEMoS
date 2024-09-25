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
    """Save a variable to a file using pickle serialization.

    This function serializes a given variable and writes it to a specified file in binary format 
    using the pickle module. It returns the filename where the variable has been saved.

    Args:
        var (any): The variable to be saved to a file.
        filename (str): The name of the file where the variable will be stored.

    Returns:
        str: The filename where the variable has been saved.
    """
    with open(filename, 'wb') as pickle_f:
        pickle.dump(var, pickle_f)
    return filename


def load_variable(filename):
    """Load a variable from a file using pickle serialization.

    This function opens a specified file in binary read mode and uses the pickle module to 
    deserialize the contents into a Python variable. It returns the loaded variable for further use.

    Args:
        filename (str): The path to the file from which the variable will be loaded.

    Returns:
        any: The deserialized variable loaded from the specified file.
    """
    with open(filename, 'rb') as pickle_f:
        var = pickle.load(pickle_f)
    return var


def run(label_file=None, testing_label_file=None, models=None, training_mode=None, procedure=None, tile_info=None):
    """Execute a series of procedures for model training and evaluation.

    This function orchestrates the execution of various procedures such as tiling, pre-processing, 
    data splitting, hyperparameter optimization, training, testing, and data visualization based on 
    the provided arguments. It ensures that the necessary files and parameters are available before 
    proceeding with each step.

    Args:
        label_file (str): The path to the main label file, which is required.
        testing_label_file (str, optional): The path to the testing label file. Defaults to None.
        models (list, optional): A list of models to be used in training and testing. Defaults to None.
        training_mode (str, optional): The mode of training, such as "Cross_validation". Defaults to None.
        procedure (list): A list of procedures to execute in order.
        tile_info (tuple, optional): Information about tile size and level. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If the required 'label_file' argument is None.
        FileNotFoundError: If the specified label or testing label files do not exist.
    """
    # Verify if the required arguments are provided
    if label_file is None:
        raise ValueError("O argumento 'label_file' é obrigatório e não pode ser None.")
    if testing_label_file is None:
        print("Nenhum arquivo de teste fornecido. Usando divisão do arquivo de rótulos principal.")

    # Check if the models are provided
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"O arquivo de rótulos '{label_file}' não foi encontrado.")
    
    # Check if the testing label file exists
    if testing_label_file and not os.path.exists(testing_label_file):
        raise FileNotFoundError(f"O arquivo de teste '{testing_label_file}' não foi encontrado.")
    
    _label_file_tiled = "HEAL_Workspace/outputs/label_file_tiled.csv" if label_file else None
    _test_label_file_tiled = "HEAL_Workspace/outputs/test_label_file_tiled.csv" if testing_label_file else None

    # Loop for each procedure
    for _proc in procedure:
        if _proc == "Tiling":
            print("[INFO] Start tiling ...")
            tiling.tiling(label_file, testing_label_file, _tile_size=tile_info[0], _tile_level=tile_info[1])
        elif _proc == "Pre_processing":
            print("[INFO] Image pre-processing ...")
            pre_processing.pre_processing()
        elif _proc == "Data_split":
            print("[INFO] Data split ...")
            data_split.data_split(_label_file_tiled, _test_label_file_tiled, test_ratio=0.5)
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
    
    # Defining the arguments
    parser.add_argument("--label_file", type=str, required=True, help="(required) Path to the label file containing the training data")
    parser.add_argument("--testing_label_file", type=str, help="(optional) Path to the testing label file. If not provided, the main label file will be used for testing.")
    parser.add_argument(
        "--models", 
        type=str, 
        nargs='+', 
        required=True, 
        help=(
            "AlexNet, DenseNet161, EFF-NET, GoogleNet, InceptionV3, MNASNET, "
            "MobileNetV2, ResNet101, ResNet152, ResNet18, ResNet34, ResNet50, "
            "ShuffleNetV2, Vgg16, Vgg16_BN, Vgg19, Vgg19_BN, WideResNet101, "
            "WideResNet50"
        )
    )
    parser.add_argument("--training_mode", type=str, choices=["single_round", "Cross_validation"], default="single_round", help="(optional) Training mode, either single_round or Cross_validation. Default is single_round.")
    parser.add_argument("--procedure", type=str, nargs='+', required=True, help=("Tiling, Pre_processing, Data_split, Hyperparameter_optimisation, Training, Testing, Data_visualisation, Grad_CAM"))
    parser.add_argument("--tile_info", type=int, nargs=2, default=[1000, 0], help="(optional) Tile size and level, specified as two integers. Default is [1000, 0]")

    args = parser.parse_args()

    # Verify if the files exist
    if not os.path.isfile(args.label_file):
        raise FileNotFoundError(f"O arquivo de rótulos '{args.label_file}' não foi encontrado ou não é um arquivo válido.")
    
    if args.testing_label_file and not os.path.isfile(args.testing_label_file):
        raise FileNotFoundError(f"O arquivo de rótulos de teste '{args.testing_label_file}' não foi encontrado ou não é um arquivo válido.")

    # execute the pipeline
    run(label_file=args.label_file,
        testing_label_file=args.testing_label_file,
        models=args.models,
        training_mode=args.training_mode,
        procedure=args.procedure,
        tile_info=args.tile_info)

