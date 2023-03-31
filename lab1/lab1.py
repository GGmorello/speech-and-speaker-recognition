import numpy as np
from lab1_proto import *
def main():
    example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
    data = np.load('lab1_data.npz', allow_pickle=True)['data']
    mfcc(example['samples'])

if __name__ == '__main__':
    main()