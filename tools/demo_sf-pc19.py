import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / '*{}'.format(self.ext))) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index], mmap_mode='r')
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--numpy', type=str, help='The output Numpy file.')    
    parser.add_argument('--pred', type=str, help='The output object detection prediction file.')    
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml', help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='~/OpenPCDet/models/pointpillar_7728.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    return args

def main(args):
    cfg_from_yaml_file(args.cfg_file, cfg)

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.numpy_file), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    print('torch.cuda.is_available():', torch.cuda.is_available())

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Generating: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            pred_dicts, _ = model.forward(data_dict)
            np.save('{}/{:0>8}.npy'.format(Path(args.pred).mkdir(parents=True, exist_ok=True), idx), pred_dicts[0].numpy())

    logger.info('Demo done.')


if __name__ == '__main__':
    args = parse_config()
    main(args)
