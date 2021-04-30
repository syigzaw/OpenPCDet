import numpy as np
from npy_append_array import NpyAppendArray
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--input_dir', type=str, help='The directory of the input Numpy files.')
    parser.add_argument('--output_file', type=str, help='The output Numpy file. To load this file, use np.load(filename, mmap_mode="r")')
    args = parser.parse_args()
    return args

def main(args):
    count = 0
    npaa = NpyAppendArray(args.output_file)
    for file in glob.glob(args.input_dir):
        count += 1
        if count % 100 == 0:
            print(count)
        frame = np.load(file)
        npaa.append(frame)

if __name__ == '__main__':
    args = parse_config()
    main(args)
