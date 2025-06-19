import copy
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from fusionforce.utils import load_calib
from PIL import Image


class Data(Dataset):
    """
    A dataset for depth correction.
    """

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.path = path
        self.max_depth = 10_000.0  # in mm
        self.image_files = {
            'left': sorted(glob(os.path.join(path, 'images', 'left', '*.png'))),
            'right': sorted(glob(os.path.join(path, 'images', 'right', '*.png'))),
        }
        self.depth_files = {
            'luxonis': sorted(glob(os.path.join(path, 'luxonis', 'depth', '*.png'))),
            'defom-stereo': sorted(glob(os.path.join(path, 'defom-stereo', 'depth', '*.npy'))),
        }
        self.calib_path = os.path.join(path, 'calibration')
        self.calib = load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64)):
            sample = self.get_sample(i)
            return sample
        ds = copy.deepcopy(self)
        if isinstance(i, (list, tuple, np.ndarray)):
            ds.ids = [self.ids[k] for k in i]
        else:
            assert isinstance(i, (slice, range))
            ds.ids = self.ids[i]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        ids = [f[:-4] for f in self.image_files['left']]
        ids = sorted(ids)
        return ids

    def ind_to_stamp(self, i):
        ind = self.ids[i]
        stamp = float(ind.replace('_', '.'))
        return stamp

    def get_image(self, i, camera='left'):
        img_path = self.image_files[camera][i]
        img = Image.open(img_path)
        return img

    def get_depth(self, i, source='luxonis'):
        depth_path = self.depth_files[source][i]
        if source == 'luxonis':
            depth = np.array(Image.open(depth_path))
        elif source == 'defom-stereo':
            depth = np.load(depth_path)
        else:
            raise ValueError("Unknown depth source: {}".format(source))
        return depth

    def get_sample(self, i):
        depth_input = self.get_depth(i, source='luxonis')
        depth_label = self.get_depth(i, source='defom-stereo')
        return depth_input[np.newaxis], depth_label[np.newaxis]


def demo():
    import cv2

    dataset_path = '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/helhest_2025_06_13-15_01_10'
    ds = Data(dataset_path)

    i = 150
    depth_in, depth_gt = ds[i]
    max_depth = 10_000.0  # in mm

    # visualize colored depth
    depth_scaled = cv2.convertScaleAbs(depth_in.squeeze(), alpha=255.0 / max_depth)
    depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
    cv2.imshow("Depth Input", depth_colored)

    depth_scaled_label = cv2.convertScaleAbs(depth_gt.squeeze(), alpha=255.0 / max_depth)
    depth_colored_label = cv2.applyColorMap(depth_scaled_label, cv2.COLORMAP_JET)
    cv2.imshow("Depth Label", depth_colored_label)

    mask_dist = depth_in > 0
    mask_nan = np.isnan(depth_gt) | np.isinf(depth_gt)
    mask_valid = np.ones(depth_gt.shape, dtype=bool)
    mask_valid[:, :7, :] = False
    mask_valid[:, :, :7] = False
    mask = mask_dist & mask_valid & (~mask_nan)
    # mask = mask_dist & (~mask_nan)
    cv2.imshow("Mask", mask.squeeze().astype(np.uint8) * 255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    dataset_path = '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/helhest_2025_06_13-15_01_10'
    ds = Data(dataset_path)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = smp.Unet(
        encoder_name='mobilenet_v2',  # choose encoder, e.g. mobilenet_v2 or resnet18
        encoder_weights='imagenet',  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for grayscale images)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    for epoch in range(10):  # number of epochs
        for depth_in, depth_gt in tqdm(loader):
            depth_in = depth_in.float().to(device)
            depth_gt = depth_gt.float().to(device)

            # preprocessing
            depth_in = depth_in / ds.max_depth
            depth_gt = depth_gt / ds.max_depth

            optimizer.zero_grad()
            depth_pred = model(depth_in)

            mask_dist = depth_in > 0
            mask_nan = torch.isnan(depth_gt) | torch.isinf(depth_gt)
            mask_valid = torch.ones(depth_gt.shape, dtype=torch.bool, device=device)
            mask_valid[..., :7, :] = False
            mask_valid[..., :, :7] = False
            mask = mask_dist & mask_valid & (~mask_nan)

            loss = criterion(depth_pred[mask], depth_gt[mask])

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

        # # visualize predictions
        # with torch.no_grad():
        #     depth_in, depth_gt = next(iter(loader))
        #     depth_in = depth_in.float().to(device)
        #     depth_pred = model(depth_in)
        #     depth_pred = depth_pred.cpu().numpy().squeeze()[0]
        #     depth_gt = depth_gt.cpu().numpy().squeeze()[0]
        #
        #     # visualize colored depth
        #     import cv2
        #
        #     max_depth = 10_000.0  # in mm
        #     depth_scaled = cv2.convertScaleAbs(depth_pred, alpha=255.0 / max_depth)
        #     depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        #     cv2.imshow("Depth Prediction", depth_colored)
        #
        #     depth_scaled_gt = cv2.convertScaleAbs(depth_gt, alpha=255.0 / max_depth)
        #     depth_colored_gt = cv2.applyColorMap(depth_scaled_gt, cv2.COLORMAP_JET)
        #     cv2.imshow("Depth Ground Truth", depth_colored_gt)
        #
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # demo()