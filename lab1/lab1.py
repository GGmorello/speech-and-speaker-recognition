import numpy as np
from lab1_proto import *
def main():
    # example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
    # data = np.load('lab1_data.npz', allow_pickle=True)['data']
    # mfcc(example['samples'])
    mfcc_data,mspec_data=get_all_data()
    #Then compute the correlation coefficients between features4 and display the result with pcolormesh
    # corr = np.corrcoef(mfcc_data, rowvar=False)
    # print(corr)
    mfcc_data_2=np.vstack(mfcc_data)
    # mspec_data_2=np.vstack(mspec_data)
    # corr_mfcc = np.corrcoef(mfcc_data_2, rowvar=False)
    # corr_mspec = np.corrcoef(mspec_data_2, rowvar=False)
    # plt.pcolormesh(corr_mfcc)
    # plt.show()
    # plt.pcolormesh(corr_mspec)
    # plt.show()
    GMM(mfcc_data_2, mfcc_data)
   

if __name__ == '__main__':
    main()