import numpy as np
import cPickle as pickle

def get_enb(filename):
    f = open(filename)
    contents = []
    for line in f:
        contents.append(line)       
    contents = contents[14:]
    dataset = []
    for l in contents:
        l = l.strip().split(',')
        for i in range(len(l)):
            l[i] = float(l[i])
        dataset.append(l)
    return np.array(dataset)

data = get_enb('enb.arff')

num_train = 700
num_test = 68
dim = 8

dataset = {}
dataset['train_x'] = data[:num_train, :dim].T
dataset['test_x'] = data[num_train:, :dim].T
dataset['train_y'] = data[:num_train, dim:].T
dataset['test_y'] = data[num_train:, dim:].T

# save dataset as pickle file
with open('enb.pickle','wb') as f:
    pickle.dump(dataset, f)

'''
# load dataset

with open('enb.pickle','rb') as f:
    dataset = pickle.load(f)

'''
