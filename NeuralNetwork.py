#!/usr/bin/env python3

# Make Your Own Neural Network
# (c) Tariq Rashid, 2016
# license is GPLv2

# Added functionalities
# (c) Vincent Mercier, 2018
# license is GPLv2

import numpy
import Plot
# scipy.special for the sigmoid function expit()
import scipy.special
import scipy.ndimage


# neural network class definition
class NeuralNetwork:
    
    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        
        self.lr = learning_rate
        
        # link weight matrices, wih and who (input and output weights of hidden layer nodes)
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def train_dataset(self, data_file, epochs=1, rotate_left=None, rotate_right=None):
        records_cnt = 0
        for e in range(epochs):
            with open(data_file) as df:
                for record in df:
                    # split the record by the ',' commas
                    all_values = record.split(',')
                    # scale and shift the inputs
                    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                    # create the target output values
                    targets = numpy.zeros(self.onodes) + 0.01
                    # all_values[0] is the target label for this record
                    targets[int(all_values[0])] = 0.99
                    self.train(inputs, targets)
                    
                    if rotate_left is not None:
                        inputs_left = scipy.ndimage.interpolation.rotate(
                            inputs.reshape(28,28), rotate_left, cval=0.01, order=1, reshape=False)
                        self.train(inputs_left.reshape(784), targets)
                        records_cnt += 1
                        
                    if rotate_right is not None:
                        inputs_right = scipy.ndimage.interpolation.rotate(
                            inputs.reshape(28,28), -rotate_right, cval=0.01, order=1, reshape=False)
                        self.train(inputs_right.reshape(784), targets)
                        records_cnt += 1
                    
                    records_cnt += 1
                    print("Epoch: {} ({})".format(e + 1, records_cnt), end='\r')

        print("Trained with a dataset of {} records for {} epochs.".format(int(records_cnt / epochs), epochs))
        
    def test_dataset(self, test_file, show_errors=False):
        records_cnt = 0
        good_cnt = 0
        with open(test_file) as f:
            for record in f:
                target = int(record[0])
                inputs = (numpy.asfarray(record.split(',')[1:]) / 255.0 * 0.99) + 0.01
                outputs = self.query(inputs)
                found = int(numpy.argmax(outputs))
                records_cnt += 1
                
                if found == target:
                    good_cnt += 1
                elif show_errors:
                    Plot.plot_record(inputs, target, outputs[found][0] * 100.0, found)
                    
        print("{} errors on {} records".format(records_cnt - good_cnt, records_cnt))
        print("Accuracy of {}%".format(float(good_cnt) / float(records_cnt) * 100.0))
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            numpy.save(f, self.wih)
            numpy.save(f, self.who)
        
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.wih = numpy.load(f)
            self.who = numpy.load(f)
