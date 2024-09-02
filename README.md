# LaneNet-Nighttime-Performance

## Dataset
The dataset used in this project includes training, validation, and test data for nighttime lane detection. It is stored in Google Drive and can be accessed using the link below.

## Accessing the Dataset:
- You can download the dataset directly from [Google Drive](https://drive.google.com/drive/folders/11rHYcyQ0ZAZlAGV92FGSUXgOj3Mojnf1?usp=sharing).

## Usage:
- Download the dataset and place it in the appropriate directories in your project folder.
- Ensure the dataset paths in the code match your local setup.

## Structure:
- `train/`: Contains training images.
- `train_label/`: Contains the labels for the training images.
- `val/`: Contains validation images.
- `val_label/`: Contains the labels for the validation images.
- `test/`: Contains test images.
- `test_label/`: Contains the labels for the test images.

## CARLA Integration

### Prerequisites
- Python 3.7.10
- PyTorch
- CARLA 0.9.10: [Download CARLA](https://carla.org/)
- Ensure you have sufficient system resources (CPU, GPU) to run CARLA and your model simultaneously.

### Setting Up CARLA
1. Download and install CARLA from the official website.
2. Start the CARLA server by navigating to the CARLA root directory and running:
 ./CarlaUE4.sh
3. Ensure CARLA is running in the background before starting the simulation script.

### Loading the LaneNet Model in CARLA
1. Place your trained model `.pth` file in the appropriate directory.
2. Modify the CARLA Python API script to load your model:
```python
model = torch.load('path_to_your_model.pth')
3. Run the Script with:
python project-deployment1.py

### Running the Simulation
1. Ensure you have CARLA installed on your system.
2. Navigate to the `PythonAPI/examples/` directory within the CARLA folder. 3. Place the Deployment script; 'project-deployment1.py' into this directory.
4. Run the deployment script. Ensure CARLA is running before executing the Script

## Pretrained Weights
This project uses pretrained weights from the [LaneNet PyTorch implementation](https://github.com/IrohXu/lanenet-lane-detection-pytorch.git) by IrohXu. These weights were used as the starting point for fine-tuning the model to improve performance under nighttime driving conditions.

###How to Use the Pretrained Weights
1. Download the Weights: Clone the repository or download the pretrained weights from [here](https://github.com/IrohXu/lanenet-lane-detection-pytorch.git).
2. Load the Weights: Place the weights in the appropriate directory and specify the path in your configuration file or code.
