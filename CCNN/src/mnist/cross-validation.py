import sys
import os

lr = [0.09,0.07,0.05,0.03,0.01,0.009,0.007,0.005,0.003,0.001]
reg = [100.0,10.0,1.0,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]

k = sys.argv[1]
for i in lr:
    print '### learning_rate =', i
    print 
    sys.stdout.flush()
    for j in reg:
        os.system('python TFCNN.py '+str(i)+' '+str(j)+' '+k) 
    print


