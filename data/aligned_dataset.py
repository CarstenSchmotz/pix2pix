import os
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        # RGB- und Tiefenbilder laden
        rgb_path = os.path.join(self.dir_AB, 'rgb', AB_path)
        depth_path = os.path.join(self.dir_AB, 'depth', AB_path)
        lidar_path = os.path.join(self.dir_AB, 'lidar', AB_path)
        #added extra prints for debugging
        print("RGB Path:", rgb_path)
        print("Depth Path:", depth_path)
        print("LiDAR Path:", lidar_path)

        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        lidar = Image.open(lidar_path).convert('L')

        # RGB-Bild in NumPy-Array konvertieren und vierten Kanal hinzufügen
        rgb = np.array(rgb)
        depth = np.array(depth)
        rgbd = np.dstack((rgb, depth))

        # RGBD-Bild und LiDAR-Bild als PIL Image
        A = Image.fromarray(rgbd)
        B = lidar

        # Transformationen anwenden
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)
