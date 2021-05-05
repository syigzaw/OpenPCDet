import sensor_msgs.point_cloud2 as pc2
import rosbag
import numpy as np
from npy_append_array import NpyAppendArray
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--bag', type=str, help='The input bag file.')
    parser.add_argument('--numpy', type=str, help='The output Numpy file. To load this file, use np.load(filename, mmap_mode="r")')
    args = parser.parse_args()
    return args

def main(args):
    bag = rosbag.Bag(args.bag)
    count = 0
    print(bag)
    npaa = NpyAppendArray(args.numpy)
    for topic, msg, t in bag.read_messages(topics=['/points_raw']):
        count += 1
        if count % 1000 == 0:
            print(count)
        frame = []
        for p in pc2.read_points(msg, field_names = ('x', 'y', 'z', 'intensity'), skip_nans=True):
            frame.append([p[1], -p[0], p[2], p[3]])
        frame = np.array(frame)
        frame[:, 3] = np.clip(frame[:, 3], 0, 40)
        frame[:, 3] /= np.max(frame[:, 3])
        npaa.append(frame)
    bag.close()

if __name__ == '__main__':
    args = parse_config()
    main(args)
