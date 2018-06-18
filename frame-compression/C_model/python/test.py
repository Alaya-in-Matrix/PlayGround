from hardware_recompression import *
from pylab import *
from PIL import Image
import os

imglist = [f.split('.')[0] for f in os.listdir('./data/')]

d1 = load('distribution.npy')
d2 = load('distribution_hardware.npy')
d3 = load('distribution_mode_hardware.npy')
x = range(33)

print d1[0]
print d2[0]
print d3[0]

delta1 = d2[0] - d1[0]
delta2 = d3[0] - d1[0]
plot(x,delta1)
plot(x,delta2)
legend(['hardware+PSNR','hardware+max'],loc='lower right',fontsize=8)
show()



d1 = d1*1.0
for f in d1:
    f = f/f[-1]
    plot(x,f)
legend(imglist,loc='lower right',fontsize=8)
show()
d2 = d2*1.0
for f in d2:
    f = f/f[-1]
    plot(x,f)
legend(imglist,loc='lower right',fontsize=8)
show()

d3 = d3*1.0
for f in d3:
    f = f/f[-1]
    plot(x,f)
legend(imglist,loc='lower right',fontsize=8)
show()





