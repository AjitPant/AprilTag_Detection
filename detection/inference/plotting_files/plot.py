import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./data_out/out.csv', sep=',',header=None, index_col =None)

# data = data.iloc[3::4, :]

for i in data.columns:
    plt.figure()
    plt.hist(data[i], bins = 100)
    # data.plot.hist(by = 0,bins = 100)
    # plt.ylabel('Frequency')
    # plt.xlabel('Words')
    # plt.title('Title')

    plt.show()
