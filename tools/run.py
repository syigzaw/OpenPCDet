import demo_compute1
import Extract_Numpy_File
import glob
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--lowest_P_num', type=int, help='Lowest value of range of selected participant drives.')
    parser.add_argument('--highest_P_num', type=int, help='Highest value of range of selected participant drives.')
    parser.add_argument('--bag_dir', type=str, help='Location of the directory containing input bag files.')
    parser.add_argument('--numpy_dir', type=str, help='Location of the directory containing output Numpy files.')    
    parser.add_argument('--pred_dir', type=str, help='Location of the directory containing output object detection prediction files.')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml', help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='/root/OpenPCDet/models/pointpillar_7728.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_config()
    list_p_dir = args.bag_dir if args.bag_dir else args.numpy_dir
    for i in glob.glob('{}/*P[0-9]*/'.format(list_p_dir)):
        P_number = int(i.split('/')[-2].split('-')[-1][1:])
        if P_number >= args.lowest_P_num and P_number <= args.highest_P_num:
            for j in glob.glob('{}*'.format(i)):
                args.bag_file = j
                args.numpy_file = j.replace('bag', 'numpy').split('.')[0] + '.npy'
                args.pred_file = j.replace('bag', 'prediction').split('.')[0] + '.npy'
                if args.bag_dir:
                    print('\nTransforming bag file to numpy files\n')
                    Extract_Numpy_File.main(args)
                if args.pred_dir:
                    print('\nCreating predictions from numpy files\n')
                    demo_compute1.main(args)
