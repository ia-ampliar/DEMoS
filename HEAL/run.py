import sys
import os
import pickle
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


def run(**kwargs):
    """
    Main entry point of the HEAL package.
    """
    print(kwargs.items())

    # Parse the input parameters
    _label_file = kwargs.get("label_file", None)
    _testing_label_file = kwargs.get("testing_label_file", None)
    _models = kwargs.get("models", None)
    _training_mode = kwargs.get("training_mode", "Single_round")
    _procedure = kwargs.get("procedure", None)
    _tile_info = kwargs.get("tile_info", [256, 0])  # Default values for tile size and level
    _filter_model = kwargs.get("filter_model", None)
    _extra_test_file = kwargs.get("extra_testing_label_file", None)
    _extra_testing_pre_processing_enable = kwargs.get("extra_testing_pre_processing_enable", False)

    _label_file_tiled = "HEAL_Workspace/outputs/label_file_tiled.csv" if _label_file else None
    _test_label_file_tiled = "HEAL_Workspace/outputs/test_label_file_tiled.csv" if _testing_label_file else None

    # Create necessary directories if they do not exist
    os.makedirs('HEAL_Workspace/tiling/train', exist_ok=True)
    os.makedirs('HEAL_Workspace/outputs', exist_ok=True)

    # Call the corresponding functions based on the specified procedures
    for _proc in _procedure:
        if _proc == "Tiling":
            print("[INFO] Start tiling ...")
            tiling.tiling(_label_file, _testing_label_file, _tile_size=_tile_info[0], _tile_level=_tile_info[1])
        elif _proc == "Pre_processing":
            print("[INFO] Image pre-processing: color correction and blur detection ...")
            pre_processing.pre_processing()
        elif _proc == "Data_split":
            print("[INFO] Data split ...")
            data_split.data_split(_label_file_tiled, _test_label_file_tiled)
        elif _proc == "Hyperparameter_optimisation":
            print("[INFO] Using HyperOpt to optimise the parameters ...")
            hyperparameter_optimisation.tuning()
        elif _proc == "Training":
            if _training_mode == "Single_round":
                print("[INFO] Training the model in single round mode ...")
                train.train(_models, tile_size=_tile_info[0])
            elif _training_mode == "Cross_validation":
                print("[INFO] Training the model in 10-fold cross-validation mode ...")
                train.train(_models, tile_size=_tile_info[0], CV_Enable=True)
        elif _proc == "Testing":
            print("[INFO] Running the independent test ...")
            independent_test.independent_test(_models, _tile_info, extra_test_set=_extra_test_file,
                                              pre_processing_enable=_extra_testing_pre_processing_enable)
        elif _proc == "Data_visualisation":
            print("[INFO] Running the data visualisation ...")
            data_visualisation.data_visualisation(_tile_info)
        elif _proc == "Grad_CAM":
            print("[INFO] Using Grad-CAM to visualize the key regions ...")
            grad_cam.grad_cam()

    print(_label_file, _testing_label_file, _models, _training_mode, _procedure)

if __name__ == "__main__":
    run(
        label_file="/home/rsb6/Desktop/Trabalho/DEMoS/HEAL/datas/label_file.csv",
        testing_label_file=None,
        models=['ResNet50'],
        procedure=["Tiling"],
        tile_info=[256, 0]  # Exemplos de valores
    )
