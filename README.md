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

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
