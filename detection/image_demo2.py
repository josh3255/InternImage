# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os
import os.path as osp
import numpy as np

def get_all_files_in_folder(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('source', help='Source file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args

def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    num_classes = 4
    
    file_list = get_all_files_in_folder(args.source)
    fd_list = os.listdir(args.source)

    for fd in fd_list:
        fd_path = os.path.join(args.source, fd)
        fd_path = fd_path.replace('images', 'labels').replace('aihub', 'aihub2')
        if not os.path.exists(fd_path):
            os.makedirs(fd_path)

    for i, file in enumerate(file_list):
        txt_file = file.split('.')[0].replace('aihub', 'aihub2').replace('images', 'labels') + '.txt'

        img = mmcv.imread(file)
        im_h, im_w, _ = img.shape

        result = inference_detector(model, file)
        
        with open(txt_file, 'w') as wf:
            for j in range(num_classes):
                for res in result[j]:
                    x1, y1, x2, y2, conf = res
                    
                    cx = ((x1 + x2) / 2.0) / im_w
                    cy = ((y1 + y2) / 2.0) / im_h
                    w = (x2 - x1) / im_w
                    h = (y2 - y1) / im_h

                    wf.write('{} {} {} {} {}\n'.format(int(j), cx, cy, w, h))

        print('{}/{} {} file saved!'.format(i+1, len(file_list), txt_file))

if __name__ == '__main__':
    args = parse_args()
    main(args)