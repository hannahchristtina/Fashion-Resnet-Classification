# Fashion Image Classification with Transfer Learning using ResNet
This project demonstrates how to accurately classify images of clothing from the FashionMNIST dataset by fine-tuning a pre-trained deep learning model, avoiding the need to train a complex model from scratch.

# Technologies
- PyTorch/torchvision: The core framework used for building, training and evaluating the neural network.
- ResNet-18: The pre-trained convolutional neural network utilized for transfer learning.
- FashionMNIST Dataset: The dataset containing 70,000 grayscale images of 10 clothing categories used for training and testing.
- VS Code: The primary code editor used for development.
- GitHub: Used for version control and project collaboration.

# Project Goal
The project's main objective is to build a high-performing image classifier that can correctly identify one of the 10 clothing categories from a given image. This is achieved by:
- Leveraging Transfer Learning: Using the powerful feature extraction capabilities of a pre-trained ResNet-18 model.
- Model Fine-Tuning: Modifying and training only the final classification layer of the model to adapt it to the FashionMNIST dataset.
- Achieving High Accuracy: Demonstrating that transfer learning is an efficient method to achieve high classification accuracy with minimal training time.
