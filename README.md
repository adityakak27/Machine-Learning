# Machine Learning Projects

Welcome to my collection of machine learning projects! This repository contains implementations of two different neural networks developed from scratch using Python. The projects are designed to demonstrate fundamental concepts in machine learning, including neural network architecture and reinforcement learning.

## Projects

### 1. Neural Network for MNIST Digit Recognition

This project involves building a neural network from scratch, without using high-level libraries like PyTorch or TensorFlow. Instead, it relies solely on NumPy and Pandas to implement and train the network.

- **Objective:** Identify handwritten digits from the MNIST database.
- **Accuracy:** Achieves up to 90% accuracy.
- **Key Features:**
  - Manual implementation of forward and backward propagation.
  - Custom gradient descent optimization.
  - Use of ReLU and softmax activation functions.

#### Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/adityakak27/machine-learning-projects.git
   cd machine-learning-projects
   ```

2. Install the required libraries:
   ```sh
   pip install numpy pandas
   ```

3. Run the MNIST neural network script:
   ```sh
   python mnist_neural_network.py
   ```

### 2. Reinforcement Learning for Snake Game

This project showcases a neural network that learns to play the classic Snake game using reinforcement learning principles.

- **Objective:** Train the snake to maximize its score by eating food and avoiding collisions.
- **Scoring System:**
  - +10 for each piece of food eaten.
  - -10 for collisions (game over).
  - 0 for all other moves.
 
**NB:** For the snake game, you will require a conda virtual enviornment, so please make sure conda is installed in your system.

#### Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/adityakak27/machine-learning-projects.git
   cd machine-learning-projects
   ```

2. Install the required libraries:
   ```sh
   pip install numpy pandas pygame
   ```

3. Run the Snake game reinforcement learning script:
   ```sh
   python snake_reinforcement_learning.py
   ```


---

# 3. MediLife Pro

MediLife Pro is a machine learning-based web application designed to predict the disease of a patient based on their symptoms. It provides detailed information including medication, precautions, disease description, recommended workout, and diet plan. This project utilizes a Support Vector Classifier (SVC) and is built using Flask for the web interface. The backend is powered by pandas, numpy, and scikit-learn.

## Features

- Predicts diseases based on input symptoms.
- Provides detailed disease descriptions.
- Recommends appropriate medications.
- Suggests precautions and preventive measures.
- Offers workout and diet plans tailored to the predicted disease.
- User-friendly web interface built with Flask.

## Tech Stack

- **Python**
- **Flask**
- **pandas**
- **numpy**
- **scikit-learn**

## Installation

### Prerequisites

- Python 3.6+
- pip (Python package installer)

### Steps

1. **Clone the repository**

    ```bash
    git clone https://github.com/adityakak27/MediLife-Pro.git
    cd MediLife-Pro
    ```

2. **Install the required packages**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**

    ```bash
    python app.py
    ```

    The web application should now be running on `http://127.0.0.1:5000/`.

## Usage

1. Navigate to `http://127.0.0.1:5000/` in your web browser.
2. Input your symptoms in the provided text field.
3. Click on the "Predict" button to get the diagnosis and recommendations.
4. The application will display the predicted disease along with relevant medication, precautions, disease description, workout, and diet plan.




## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
