# HEAL Pipeline

The HEAL Pipeline is a comprehensive framework designed for processing and analyzing image data in a structured manner, particularly for H&E-stained histopathological images. It integrates modules for tiling, pre-processing, data splitting, model training, hyperparameter optimization, independent testing, and visualizations such as Grad-CAM. The framework is designed to automate much of the process, enabling both experienced researchers and those with minimal programming expertise to efficiently analyze large datasets of whole-slide images (WSIs).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Procedure Steps](#procedure-steps)
- [Examples](#examples)
- [Methods](#methods)
- [License](#license)

## Installation

To run the HEAL Pipeline, ensure you have Python 3.8 version and the required libraries installed.

## Installing and Activating Python 3.8 Environment
You can set up a Python 3.8 environment using either `conda` or Python's `venv` module. Follow the steps below based on your preferred method.

### Option 1: Using `conda`
1. Create the Python 3.8 environment:

   ```bash
   conda create --name heal_env python=3.8
   ```

2. Activate the environment:

    ```bash
    conda activate heal_env
    ```
### Option 2: Using `venv`
1. Create the Python 3.8 environment:
    ```bash
    python3.8 -m venv heal_env
    ```
2. Activate the environment:
    ```bash
    source heal_env/bin/activate
    ```


### You can install the necessary packages using pip:


```bash
pip install -r requirements.txt
```

### Note: Issues with OpenSlide

If you encounter issues importing OpenSlide in Python 3.8, follow the steps below to properly install the system dependencies and the `openslide-python` library:

1. **Install system dependencies**: OpenSlide relies on system libraries that need to be installed. On Debian/Ubuntu-based distributions, use the following command:

   ```bash
   sudo apt-get install openslide-tools
   sudo apt-get install python-openslide
   ```

   For other operating systems like macOS or Windows, you can download the dependencies directly from the OpenSlide website: [OpenSlide Download](https://openslide.org/download/).

2. **Install the `openslide-python` library**: After installing the system dependencies, you can install the Python version of OpenSlide using `pip`:

   ```bash
   pip install openslide-python
   ```

3. If you have using aws and have troble happend, run this code below for to solve:

     ```bash
     conda install -c conda-forge openslide-python
     ```


## Usage

To execute the pipeline, use the following command in your terminal:

```bash
python run.py --label_file <path_to_label_file> --testing_label_file <path_to_testing_label_file> --models <model1> <model2> ... --training_mode <single_round|Cross_validation> --procedure <step1> <step2> ... --tile_info <tile_size> <tile_level>
```

### **Arguments**

- `--label_file`: (required) Path to the label file containing the training data.
- `--testing_label_file`: (optional) Path to the testing label file. If not provided, the main label file will be used for testing.
- `--models`: (required) List of models to use for training and testing.
- `--training_mode`: (optional) Training mode, either `single_round` or `Cross_validation`. Default is `single_round`.
- `--procedure`: (required) Procedure steps to run, specified as a list of strings.
- `--tile_info`: (optional) Tile size and level, specified as two integers. Default is `[1000, 0]`.

### **Procedure Steps**

The following procedure steps can be specified in the `--procedure` argument:

- **Tiling**: Divides the images into smaller tiles for processing. The images are segmented at a particular magnification level, and tiles with low tissue content (e.g., high background percentage) are removed.
- **Pre_processing**: Prepares the images by applying blur detection and color correction to improve data quality.
- **Data_split**: Splits the dataset into training, validation, and testing sets, ensuring that tiles from the same WSI do not appear in multiple splits.
- **Hyperparameter_optimisation**: Optimizes the training hyperparameters such as learning rate, batch size, and model architecture using Hyperopt.
- **Training**: Trains the specified models using either a single-round or cross-validation strategy.
- **Testing**: Tests the trained models on a held-out test set to evaluate performance.
- **Data_visualisation**: Visualizes the processed data, including training metrics and performance curves.
- **Grad_CAM**: Generates Grad-CAM visualizations to highlight the regions of the images that the model focuses on when making predictions.

## Examples

### Basic Example

```bash
python run.py --label_file path/to/label_file.csv --models model1 model2 --procedure Tiling Pre_processing Training --tile_info 1000 0
```

### Example with Testing

```bash
python run.py --label_file path/to/label_file.csv --testing_label_file path/to/testing_label_file.csv --models model1 --training_mode Cross_validation --procedure Tiling Pre_processing Data_split Testing
```

## Methods

### Overview of HEAL

HEAL is a deep learning framework for processing H&E-stained histopathological images. It supports classification tasks using convolutional neural networks (CNNs) and includes six core modules:

1. **Data Pre-processing**: 
   - *Image Tiling*: Segments WSIs into smaller patches at the desired magnification level (e.g., 10X).
   - *Blur Detection*: Identifies and removes blurry tiles using the Laplacian method.
   - *Color Correction*: Normalizes color variations in the slides using the Macenko method.

2. **Data Splitting**:
   - Splits data into training, validation, and test sets. No tiles from a single WSI are split across datasets to ensure independent evaluation.

3. **Model Training**:
   - Supports 16 CNN architectures, including ResNet, VGG, Inception-V3, and MobileNet. Training is performed using the Adam optimizer, and early stopping is implemented to prevent overfitting.

4. **Hyperparameter Optimization**:
   - Hyperopt is used to search for the best model hyperparameters, such as learning rate, batch size, and architecture.

5. **Independent Testing**:
   - Evaluates the model’s performance on a test set using AUC/ROC curves, confusion matrices, and Grad-CAM visualizations.

6. **Data Visualization**:
   - Includes performance metrics like accuracy, F1-score, and loss, along with visualizations such as ROC curves and Grad-CAM heatmaps.

### Image Tiling

The tiling module uses `Openslide` to extract tiles from WSIs. You can specify the magnification level and tile size (e.g., 512x512 pixels). Tiles with high background content are automatically excluded from the analysis.

### Blur Detection

Blur detection is based on the Laplacian operator. The variance of the Laplacian is computed for each tile, and a threshold (default: 1000) is used to remove blurry tiles. This step improves the dataset’s quality and reduces noise in the model’s training process.

### Color Correction

Color correction addresses variations in stain intensity and scanner differences. It uses the Macenko method, which transforms the images into a standardized color space. A template image is provided, but users can supply their own templates to match their dataset.

### Hyperparameter Optimization

The hyperparameter optimization module uses `Hyperopt` to explore different model architectures and training configurations. It searches through a predefined space for parameters such as learning rate, batch size, and model architecture. This process helps maximize performance and reduce training time.

### Independent Testing and Grad-CAM Visualization

After training, the optimal model is selected for independent testing on a reserved test set. A variety of visualizations are generated to assess performance, including:

- **AUC/ROC curves**: To measure the classification performance.
- **Confusion Matrix**: For comparing predicted labels against true labels.
- **Grad-CAM**: Heatmaps that visualize important regions in the images for the model’s classification decisions.

This ensures a comprehensive evaluation of the model's capabilities on unseen data.

## License

The HEAL Pipeline is distributed under the [LICENSE] file in this repository.
