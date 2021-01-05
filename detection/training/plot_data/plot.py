import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('out-train.csv', sep=',',header=None, index_col =None)

data.plot(kind='hist', bins = 30)
plt.ylabel('Frequency')
plt.xlabel('Words')
plt.title('Title')

plt.show()
