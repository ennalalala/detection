import matplotlib
matplotlib.use('Agg')
import matplotlib as plt
from pylab import *

f = open("./o",'r')

loss = []
for line in f.readlines():
        line = line.strip().split()
        loss.append(line[-1])

import matplotlib.pyplot as plt

x = [i+1 for i in range(len(loss))]

plt.figure()
plt.plot(x,loss)
ylim(0,0.5)
plt.savefig("./loss1.jpg")
