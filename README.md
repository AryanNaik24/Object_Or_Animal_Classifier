CIFAR-10 Image Classification using PyTorch in Jupyter Notebook
===============================================================

Overview:
---------
This project demonstrates how to perform image classification on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) implemented in PyTorch. The work was carried out in a Jupyter Notebook named "ObjectClassification.ipynb". The notebook covers data loading, model building, training, evaluation, and prediction on new images.

Project Structure:
------------------
- **ObjectClassification.ipynb**: The main Jupyter Notebook containing all code cells for data processing, model definition, training, evaluation, and predictions.
- **trained_net.pth**: The file where the trained model's weights are saved.
- **data/**: Directory where the CIFAR-10 dataset is automatically downloaded.
- **Image Files**: Sample image files (e.g., `catTest1.jpg`, `deertest11.jpg`, `Planetest1.jpg`) used for demonstrating model predictions.

Dependencies:
-------------
- Python 3.x
- PyTorch
- torchvision
- numpy
- Pillow

Installation:
-------------
You can install the required packages using pip:

    pip install torch torchvision numpy pillow

Usage:
------
1. **Running the Notebook:**
   - Open "ObjectClassification.ipynb" in Jupyter Notebook.
   - Run the notebook cells sequentially to:
     - Download and preprocess the CIFAR-10 dataset.
     - Build and train the CNN.
     - Evaluate model accuracy on the test set.
     - Save the trained model.
     - Make predictions on new images provided as sample files.

2. **Training and Evaluation:**
   - The notebook trains the model over several epochs while printing the loss for each epoch.
   - It then evaluates the model on the CIFAR-10 test set, printing the overall accuracy.

3. **Making Predictions on New Images:**
   - The notebook includes a section to load new images (with proper resizing and normalization) using a custom `load_image` function.
   - The model outputs a predicted class, mapped to one of the CIFAR-10 classes:
     `['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`.

Important Notes:
----------------
- **Model Architecture:**
  The CNN consists of:
    - Two convolutional layers with ReLU activation functions and max pooling.
    - Two fully connected (linear) layers.
    - A final output layer for class prediction.
  
- **Bug Fix:**
  Ensure that in the notebook's model `forward` method, the final layer used for prediction is `fc3` (i.e., `x = self.fc3(x)`) rather than reusing `fc1`, to avoid shape mismatch errors.

- **Adjustments:**
  Hyperparameters such as learning rate, momentum, and the number of epochs can be modified in the notebook as desired.

Running the Code:
-----------------
To run the code, simply open and execute the cells in "ObjectClassification.ipynb" within Jupyter Notebook. The notebook will guide you through the entire workflow from data loading to predictions.


