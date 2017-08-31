# qnn-inference-examples
Jupyter notebook examples on image classification with quantized neural networks. The intent here is to give a better understanding of what kind of computations takes place when performing inference with a quantized neural network.

So far, the following notebooks are available:

1. [Basics](0-basics.ipynb) for a gentle warmup
2. [Binarized, fully-connected MNIST](1-fully-connected-binarized-mnist.ipynb) for a deep dive inside a binarized fully-connected network
3. [Binary weights, 2-bit activations, fully-connected MNIST](2-fully-connected-w1a2-mnist.ipynb) for demonstrating what happens when we go to 2-bit activations
4. [Binarized, convolutional GTSRB](3-convolutional-binarized-gtsrb.ipynb) for an introduction to convolutional and pooling layers
