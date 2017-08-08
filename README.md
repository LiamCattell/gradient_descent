# Gradient descent

Examples of first order gradient descent optimization methods commonly used in convolutional neural networks.

Optimzers currently include:
1. GD (vanilla gradient descent)
2. GD + momentum
3. Nesterov acclerated gradient (part of the GD class)
4. Adagrad
5. Adadelta
6. RMSprop
7. Adam

TODO: Adamax, Nadam


## Usage

All optimizers are children of the BaseOptimizer class. See minimise_rosenbrock.py for examples on how to set up a minimization problem using different optimizers.


## Credits

Liam Cattell
