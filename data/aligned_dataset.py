import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.rgb_dir = os.path.join(self.dir_AB, 'rgb')
        self.depth_dir = os.path.join(self.dir_AB, 'depth')
        self.lidar_dir = os.path.join(self.dir_AB, 'lidar')
        self.AB_paths = sorted(make_dataset(self.rgb_dir, opt.max_dataset_size))  # Assuming all modalities have the same filenames
        assert self.opt.load_size >= self.opt.crop_size
        
        # Set input and output channels based on direction
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        # Construct paths for the RGB, depth, and LiDAR images
        rgb_path = os.path.join(self.rgb_dir, AB_path)
        depth_path = os.path.join(self.depth_dir, AB_path)
        lidar_path = os.path.join(self.lidar_dir, AB_path)

        # Open images
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        lidar = Image.open(lidar_path).convert('L')

        # Convert to NumPy arrays
        rgb_np = np.array(rgb)
        depth_np = np.array(depth)

        # Stack RGB and depth to form RGB-D image
        rgbd_np = np.dstack((rgb_np, depth_np))

        # Convert NumPy array back to PIL Image
        A = Image.fromarray(rgbd_np)
        B = lidar

        # Apply transformations
        transform_params = get_params(self.opt, A.size)

        # Transform RGB-D image A
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A = A_transform(A)

        # Manually normalize RGB-D image A
        mean = [0.485, 0.456, 0.406, 0.5]  # RGB mean + depth mean
        std = [0.229, 0.224, 0.225, 0.5]   # RGB std + depth std
        A = F.to_tensor(A)
        A = F.normalize(A, mean=mean, std=std)

        # Transform LiDAR image B
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': rgb_path, 'B_paths': lidar_path}

    def __len__(self):
        return len(self.AB_paths)
