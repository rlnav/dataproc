# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import argparse
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from utils.constants import COCO_PANOPTIC_CLASSES
import os
import glob
from tqdm import tqdm


def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--imgs-path', metavar="FILE", help='path to data directory')
    parser.add_argument('--conf-files', default="configs/seem/seem_focall_lang.yaml", metavar="FILE",
                        help='path to config file', )
    parser.add_argument('--pretrained-pth', default="../seem_focall_v1.pt", metavar="FILE",
                        help='path to pretrained model')
    parser.add_argument('--resized-width', type=int, default=512, help='resized width')
    args = parser.parse_args()

    return args


'''
build args
'''
args = parse_option()
opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

# META DATA
cur_model = 'Focal-L'
pretrained_pth = args.pretrained_pth

transform = transforms.Resize(args.resized_width, interpolation=Image.BICUBIC)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

imgs_path = args.imgs_path
img_files = sorted(glob.glob(os.path.join(imgs_path, '*')))
print('found {} images'.format(len(img_files)))
unsegmented = 133

seg_path = os.path.join(imgs_path, 'seg')
vis_path = os.path.join(imgs_path, 'vis')
os.makedirs(seg_path, exist_ok=True)
os.makedirs(vis_path, exist_ok=True)
_ = os.stat(imgs_path).st_mtime  # to refresh the cache

'''
build model
'''
with torch.cuda.device(0):
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"],
                                                                            is_eval=True)
        for idx, img_fpath in tqdm(enumerate(img_files), total=len(img_files)):
            seg_fpath = os.path.join(seg_path, os.path.basename(img_fpath).replace('.png', '.npy'))
            vis_fpath = os.path.join(vis_path, os.path.basename(img_fpath))
            # if already exists, skip
            if os.path.exists(seg_fpath) and os.path.exists(vis_fpath):
                continue

            # if is not a file, skip
            if not os.path.isfile(img_fpath):
                continue

            image = Image.open(img_fpath)
            image = transform(image)
            width = image.size[0]
            height = image.size[1]
            image = np.asarray(image)
            visual = Visualizer(image, metadata=metadata)
            images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()

            data = {"image": images, "height": height, "width": width}

            # initialize task
            model.model.task_switch['spatial'] = False
            model.model.task_switch['visual'] = False
            model.model.task_switch['grounding'] = False
            model.model.task_switch['audio'] = False

            batch_inputs = [data]
            model.model.metadata = metadata
            results = model.model.evaluate(batch_inputs)
            pano_seg = results[-1]['panoptic_seg'][0]
            pano_seg_info = results[-1]['panoptic_seg'][1]

            _pano_seg = pano_seg.cpu().numpy().astype(np.uint8)
            seg_label = np.ones_like(_pano_seg) * unsegmented

            for idx in range(len(pano_seg_info)):
                seg_label[_pano_seg == idx + 1] = pano_seg_info[idx]['category_id']

            seg_vis = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info).get_image()  # rgb Image
            seg_vis = Image.fromarray(seg_vis)

            np.save(seg_fpath, seg_label)
            seg_vis.save(vis_fpath)

    print('Annotations Finished!')
