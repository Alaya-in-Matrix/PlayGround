# mnist

# requirement

* tensorflow
* python-mnist
`pip install python-mnist #(before 'from mnist import MNIST')`


# usage

* download dataset
`bash get_mnist.sh`
* run the script:
`python mnist_softmax.py` or `python mnist_deep.py`


# experiment

* It turns out that mnist_softmax which has only one fully-connected layer, and it can be trained easily.
* mnist_deep should be trained for a long time, even if we use dropout for the first fully-connected layer
