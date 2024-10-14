import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from Hyperparameter_optimisation import Hyperopt_train
import numpy as np
import torch

# Liberar cache de memória antes de cada treinamento
torch.cuda.empty_cache()

BATCH_LIST = [4, 8, 12, 16]
STEP_SIZE_LIST = np.arange(3, 10, dtype=int)

def save_variable(var, filename):
    with open(filename, 'wb') as pickle_f:
        pickle.dump(var, pickle_f)
    return filename

space = hp.choice('model_architecture', [
    {
        'Model_name': 'InceptionV3',
        'Learning_rate': hp.uniform('LR1', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma1', 1e-7, 1e-1),
        'Batch_size': hp.choice('Batch_size1', list(range(len(BATCH_LIST)))),
        'Step_size': hp.choice('Step_size1', list(range(len(STEP_SIZE_LIST)))),
    },
    {
        'Model_name': 'ResNet50',
        'Learning_rate': hp.uniform('LR2', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma2', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size2', list(range(len(BATCH_LIST)))),
        'Step_size': hp.choice('Step_size2', list(range(len(STEP_SIZE_LIST)))),
    },
    {
        'Model_name': 'Vgg16',
        'Learning_rate': hp.uniform('LR3', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma3', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size3', list(range(len(BATCH_LIST)))),
        'Step_size': hp.choice('Step_size3', list(range(len(STEP_SIZE_LIST)))),
    },
    {
        'Model_name': 'ShuffleNetV2',
        'Learning_rate': hp.uniform('LR4', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma4', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size4', list(range(len(BATCH_LIST)))),
        'Step_size': hp.choice('Step_size4', list(range(len(STEP_SIZE_LIST)))),
    },
    {
        'Model_name': 'MobileNetV2',
        'Learning_rate': hp.uniform('LR5', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma5', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size5', list(range(len(BATCH_LIST)))),
        'Step_size': hp.choice('Step_size5', list(range(len(STEP_SIZE_LIST)))),
    },
    {
        'Model_name': 'MNASNET',
        'Learning_rate': hp.uniform('LR6', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma6', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size6', list(range(len(BATCH_LIST)))),
        'Step_size': hp.choice('Step_size6', list(range(len(STEP_SIZE_LIST)))),
    },
])

def objective(args):
    Model_name = args['Model_name']
    Learning_rate = args['Learning_rate']
    Gamma = args['Gamma']
    Batch_size = BATCH_LIST[args['Batch_size']]
    Step_size = STEP_SIZE_LIST[args['Step_size']]

    print(args)
    Train_object = Hyperopt_train.HyperoptTrain(Model_name, Learning_rate, Step_size, Gamma, Batch_size)

    return {
        'loss': Train_object.train_model(),
        'status': STATUS_OK
    }

trials = Trials()

def tuning():
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials,
    )
    print("The best configuration is: " + str(best))

    # Lista de modelos disponíveis
    model_list = ['InceptionV3', 'ResNet50', 'Vgg16', 'ShuffleNetV2', 'MobileNetV2', 'MNASNET']

    # Recupera o índice do modelo selecionado
    model_index = best['model_architecture']

    # Verifica e extrai os parâmetros corretos
    lr = best[f"LR{model_index + 1}"]
    gamma = best[f"Gamma{model_index + 1}"]
    batch_size_index = best[f"Batch_size{model_index + 1}"]
    step_size_index = best[f"Step_size{model_index + 1}"]

    # **Verificação de índices para evitar IndexError**
    if batch_size_index >= len(BATCH_LIST):
        raise ValueError(f"Batch size index {batch_size_index} is out of range for BATCH_LIST.")
    if step_size_index >= len(STEP_SIZE_LIST):
        raise ValueError(f"Step size index {step_size_index} is out of range for STEP_SIZE_LIST.")

    # Configuração final
    best_config = {
        'lr': lr,
        'gamma': gamma,
        'model_name': model_list[model_index],
        'step_size': STEP_SIZE_LIST[step_size_index],
        'batch_size': BATCH_LIST[batch_size_index]
    }

    # Salva a configuração no arquivo
    save_variable(best_config, "HEAL_Workspace/outputs/hyper_parameter.conf")
