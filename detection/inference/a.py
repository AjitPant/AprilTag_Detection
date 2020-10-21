#  usage
#
#  python plot_file_of_floats.py  < /tmp/my_file_of_floats


import matplotlib.pyplot as plt
import sys

floats = map(float, sys.stdin.read().split())[3::4] # read input file from command line


plt.plot(floats, 'ro') # show floats as curve
plt.figure()           # permit another plot to get rendered

plt.hist(floats, bins =100)  #  show histogram of floats
plt.show()

