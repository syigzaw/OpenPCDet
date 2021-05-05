import Extract_Numpy_File
import demo_compute1
import argparse
import glob
from pathlib import Path

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--lowest_P', type=int, default=0, help='Lowest value of range of selected participant drives.')
    parser.add_argument('--highest_P', type=int, default=100, help='Highest value of range of selected participant drives.')
    # parser.add_argument('--bag_dir', type=str, help='Location of the directory containing input bag files.')
    # parser.add_argument('--numpy_dir', type=str, help='Location of the directory containing output Numpy files.')    
    # parser.add_argument('--pred_dir', type=str, help='Location of the directory containing output object detection prediction files.')
    parser.add_argument('--make_numpy', type=bool, default=True)
    parser.add_argument('--make_pred', type=bool, default=False)
    parser.add_argument('--mnt_dir', type=str, default='/home/syigzaw/shared_dir/mnt', help='specify the directory that the ftp server is mounted at')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml', help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='/root/OpenPCDet/models/pointpillar_7728.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_config()
    for i in glob.glob('{}/*P[0-9]*/'.format(args.mnt_dir)):
        P_number = int(i.split('/')[-2].split('-')[-1][1:])
        if P_number >= args.lowest_P and P_number <= args.highest_P:
            processed = Path(i) / 'LIDAR/processed'
            processed.mkdir(exist_ok=True)
            for j in glob.glob(str(Path(i) / 'LIDAR/*.pcap')):
                args.bag = processed / j.split('/')[-1].replace('.pcap', '.bag')
                args.numpy = processed / j.split('/')[-1].replace('.pcap', '-numpy.npy')
                args.pred = processed / j.split('/')[-1].replace('.pcap', '-pred.npy')
                # if args.make_numpy:
                #     print('\nTransforming bag file to numpy file\n')
                #     Extract_Numpy_File.main(args)
                # if args.make_pred:
                #     print('\nCreating prediction file from numpy file\n')
                #     demo_compute1.main(args)
                print(args.pcap)
                print(args.bag)
                print(args.numpy)
                print(args.pred)
