================================================================================
 MNIST DIGIT RECOGNIZER — FROM SCRATCH
 A two-layer neural network built using only NumPy
================================================================================

OVERVIEW
--------
This project implements a feedforward neural network from scratch to classify
handwritten digits (0-9) from the Kaggle MNIST Digit Recognizer dataset.
No TensorFlow, no PyTorch — just NumPy, math, and backpropagation.


RESULTS
-------
  Training Accuracy (iter 490) : ~79.0%
  Dev Set Accuracy             : 77.2%
  Iterations                   : 500
  Learning Rate (alpha)        : 0.10
  Training Set Size            : 41,000 samples
  Dev Set Size                 : 1,000 samples


NETWORK ARCHITECTURE
--------------------

  [Input Layer]        [Hidden Layer]       [Output Layer]
   784 neurons    -->   10 neurons      -->   10 neurons
  (28x28 pixels)       (ReLU)               (Softmax)
                    W1 [10x784], b1       W2 [10x10], b2

  - Input  : 784 nodes — flattened 28x28 pixel values, normalized to [0, 1]
  - Hidden : 10 neurons with ReLU activation
  - Output : 10 neurons with Softmax (one per digit class 0-9)


TRAINING ACCURACY PROGRESS
---------------------------

  Iter    0  |##                                        |   9.8%
  Iter   50  |############                              |  29.6%
  Iter  100  |#################                         |  43.9%
  Iter  150  |######################                    |  54.9%
  Iter  200  |##########################                |  65.1%
  Iter  250  |############################              |  69.5%
  Iter  300  |#############################             |  72.4%
  Iter  350  |##############################            |  74.5%
  Iter  400  |###############################           |  76.3%
  Iter  450  |################################          |  77.9%
  Iter  490  |#################################         |  79.0%
  Dev Set    |################################          |  77.2%  <-- final


SAMPLE INPUT (28x28 pixel grid — digit "7")
--------------------------------------------

  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . # # # # # # # # # # # # # # . . . . . .
  . . . . . . . . # # # # # # # # # # # # # # . . . . . .
  . . . . . . . . . . . . . . . . . . # # # . . . . . . .
  . . . . . . . . . . . . . . . . . # # # . . . . . . . .
  . . . . . . . . . . . . . . . . # # # . . . . . . . . .
  . . . . . . . . . . . . . . . # # # . . . . . . . . . .
  . . . . . . . . . . . . . . # # # . . . . . . . . . . .
  . . . . . . . . . . . . . # # # . . . . . . . . . . . .
  . . . . . . . . . . . . . # # # . . . . . . . . . . . .
  . . . . . . . . . . . . # # # . . . . . . . . . . . . .
  . . . . . . . . . . . # # # . . . . . . . . . . . . . .
  . . . . . . . . . . # # # . . . . . . . . . . . . . . .
  . . . . . . . . . # # # . . . . . . . . . . . . . . . .
  . . . . . . . . . # # # . . . . . . . . . . . . . . . .
  . . . . . . . . . # # . . . . . . . . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  . . . . . . . . . . . . . . . . . . . . . . . . . . . .


HOW IT WORKS
------------

Step 1 — Data Preparation
  Load train.csv, shuffle rows, split into dev (first 1,000) and train (rest).
  Normalize pixel values: X = X / 255.0

Step 2 — Weight Initialization
  W1 = np.random.rand(10, 784) - 0.5
  b1 = np.random.rand(10, 1)   - 0.5
  W2 = np.random.rand(10, 10)  - 0.5
  b2 = np.random.rand(10, 1)   - 0.5

Step 3 — Forward Propagation
  Z1 = W1 · X + b1
  A1 = ReLU(Z1)          # max(Z, 0)
  Z2 = W2 · A1 + b2
  A2 = softmax(Z2)       # exp(Z) / sum(exp(Z))

Step 4 — Backpropagation
  dZ2 = A2 - one_hot(Y)
  dW2 = (1/m) * dZ2 · A1.T
  db2 = (1/m) * sum(dZ2)
  dZ1 = W2.T · dZ2 * ReLU_deriv(Z1)
  dW1 = (1/m) * dZ1 · X.T
  db1 = (1/m) * sum(dZ1)

Step 5 — Parameter Update
  W1 -= alpha * dW1
  b1 -= alpha * db1
  W2 -= alpha * dW2
  b2 -= alpha * db2

Step 6 — Repeat for 500 iterations, print accuracy every 10 steps.


KEY FUNCTIONS
-------------

  init_params()         Random weight/bias initialization
  ReLU(Z)               Activation: max(Z, 0)
  softmax(Z)            Multi-class output: exp(Z) / sum(exp(Z))
  forward_prop(...)     Compute Z1, A1, Z2, A2
  ReLU_deriv(Z)         Gradient of ReLU: Z > 0
  one_hot(Y)            Convert labels to one-hot vectors
  backward_prop(...)    Compute gradients via chain rule
  update_params(...)    Apply gradient descent step
  gradient_descent(...) Full training loop
  make_predictions(...) Run inference
  test_prediction(...)  Visualize sample + model prediction


HYPERPARAMETERS
---------------

  Learning Rate (alpha)  : 0.10
  Iterations             : 500
  Hidden Layer Units     : 10
  Input Size             : 784  (28 x 28 pixels)
  Output Classes         : 10   (digits 0-9)
  Batch Type             : Full-batch gradient descent


SETUP & USAGE
-------------

  Requirements:
    pip install numpy pandas matplotlib

  Dataset:
    Download from https://www.kaggle.com/c/digit-recognizer
    Files needed: train.csv, test.csv

  Run (Kaggle):
    Open notebook, attach Digit Recognizer dataset, click Run All.

  Run (local):
    Update file paths from /kaggle/input/... to your local directory.


POTENTIAL IMPROVEMENTS
----------------------

  - Increase hidden layer size (e.g., 128 neurons) for more capacity
  - Use He/Xavier initialization for faster convergence
  - Add learning rate decay to fine-tune near the optimum
  - Switch to mini-batch gradient descent for faster updates
  - Add L2 regularization or dropout to reduce overfitting
  - Stack more hidden layers for deeper representations
  - Log cross-entropy loss alongside accuracy
  - Use Adam or momentum-based optimizer instead of vanilla GD


DATASET INFO
------------

  Source  : https://www.kaggle.com/c/digit-recognizer
  Format  : CSV — 1 label column + 784 pixel columns
  Classes : 10 (digits 0 through 9)
  Rows    : 42,000 training samples

================================================================================
 Built from scratch using NumPy — no deep learning frameworks used.
================================================================================
