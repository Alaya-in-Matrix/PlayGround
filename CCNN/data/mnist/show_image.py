import numpy as np
import matplotlib.pyplot as plt

classes = ['basic','rand','rot','bg','bgrot']

examples_per_class = 8
for cls, cls_name in enumerate(classes):
    f = open('mnist_'+cls_name+'.image.npy','rb')
    X = np.load(f)
    X = np.reshape(X,[X.shape[0],28,28])*255
    idxs = np.random.choice(range(X.shape[0]), examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i*len(classes) + cls + 1)
        plt.imshow(X[idx].astype('uint8'), cmap=plt.cm.gray_r)
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()

