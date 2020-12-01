import sensor_msgs.point_cloud2 as pc2
import rosbag
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--input_dir', type=str, default='/home/autoware/shared_dir/P7/2020-05-27-00-00-04.bag', help='Location of the input bag file')
    parser.add_argument('--output_dir', type=str, help='Location of the output Numpy file. Same directory as the input if not specified.')
    args = parser.parse_args()

    bag = rosbag.Bag(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir != None else Path(args.input_dir)
    count = 0
    print(bag)
    for topic, msg, t in bag.read_messages(topics=['/points_raw']):
        count += 1
        if count > 11870:
            if count % 1000 == 0:
                print(count)
            frame = []
            for p in pc2.read_points(msg, field_names = ('x', 'y', 'z', 'intensity'), skip_nans=True):
                frame.append([p[1], -p[0], p[2], p[3]])
            frame = np.array(frame)
            frame[:, 3] = np.clip(frame[:, 3], 0, 40)
            frame[:, 3] /= np.max(frame[:, 3])
            np.save(f'{output_dir}/{count}.npy', frame)
    bag.close()

if __name__ == '__main__':
    main()
