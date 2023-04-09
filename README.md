# ML-experiments

This is my personal repository to experiment with different kinds of neural network achitectures and practice implementing them.

## mlp.py
*Code based off of Michael Nielson's code snippites from "[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)", uses MNIST data loaders from the snippit repo*

This is an implementation of a basic multi-layer perceptron. This implementation allows specification of a desired number of layers, each with a desired number of neurons. 

### Features/Notes
#### Backpropagation Algorithm
The backpropagation algorithm is implemented using numpy as the only external library. Algorithm is modeled after "Neural Networks and Deep Learning"

#### Regularization
To prevent overfitting, **L2** Regularizaiton is implemented in the cost function.

#### Weight Initialization
To avoid neuron saturation, weights are initialized as Gaussian Random Variables with $\text{mean}=0$ and $\text{standard deviation = }\frac{1}{\sqrt{n_{in}}}$ where $n_{in}$ is the number of input weights of the given neuron.

#### Stochastic Gradient Descent
Implemented Stochastic Gradient Descent as learning algorithm which can be perfomed with desired training data, number of ephocs, mini-batch size, learning rate (hyperparamater), and the L2 regularizaiton hyperparameter