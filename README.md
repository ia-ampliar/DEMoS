# HEAL Pipeline

The HEAL Pipeline is a comprehensive framework designed for processing and analyzing image data in a structured manner. It includes various stages such as tiling, pre-processing, data splitting, training, testing, hyperparameter optimization, data visualization, and Grad-CAM visualization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Procedure Steps](#procedure-steps)
- [Examples](#examples)
- [License](#license)

## Installation

To run the HEAL Pipeline, ensure you have Python and the required libraries installed. You can install the necessary packages using pip:

```bash
pip install -r [requirements.txt](VALID_FILE)

```

## Usage
To execute the pipeline, run the following command in your terminal:
```bash
python run.py --label_file <path_to_label_file> --testing_label_file <path_to_testing_label_file> --models <model1> <model2> ... --training_mode <single_round|Cross_validation> --procedure <step1> <step2> ... --tile_info <tile_size> <tile_level>
```

### **Arguments**
* --label_file: (required) Path to the label file containing the training data.
* --testing_label_file: (optional) Path to the testing label file. If not provided, the main label file will be used for testing.
* --models: (required) List of models to use for training and testing.
* --training_mode: (optional) Training mode, either single_round or Cross_validation. Default is single_round.
* --procedure: (required) Procedure steps to run, specified as a list of strings.
* --tile_info: (optional) Tile size and level, specified as two integers. Default is [1000, 0].

### Procedure Steps
The following procedure steps can be specified in the --procedure argument:

* **Tiling:** Divides the images into smaller tiles for processing.
* **Pre_processing:** Prepares the images for analysis.
* **Data_split:** Splits the data into training and testing sets.
* **Hyperparameter_optimisation:** Optimizes the hyperparameters for the models.
* **Training:** Trains the specified models on the training data.
* **Testing:** Tests the trained models on the testing data.
* **Data_visualisation:** Visualizes the processed data.
* **Grad_CAM:** Generates Grad-CAM visualizations for the models.
### Examples
Basic Example

```bash
python run.py --label_file path/to/label_file.csv --models model1 model2 --procedure Tiling Pre_processing Training --tile_info 1000 0
```

Example with Testing
```bash
python run.py --label_file path/to/label_file.csv --testing_label_file path/to/testing_label_file.csv --models model1 --training_mode Cross_validation --procedure Tiling Pre_processing Data_split Testing
```