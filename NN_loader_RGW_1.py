#
# This program runs the Python36 port of Michael Nielsen Neural Networks and Deep Learning by Ron Wellard
#

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#import NN_MDD
import NN_RGW_2

#net = NN_MDD.Network([784, 30, 10])
net = NN_RGW_2.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 2.5, test_data=test_data)