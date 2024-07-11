import os
from PIL import Image
import torch
from data.base_dataset import BaseDataset, make_dataset, get_transform

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_rgb = os.path.join(opt.dataroot, opt.phase, 'rgb')
        self.dir_depth = os.path.join(opt.dataroot, opt.phase, 'depth')
        self.dir_lidar = os.path.join(opt.dataroot, opt.phase, 'lidar')
        
        assert os.path.isdir(self.dir_rgb), '%s is not a valid directory' % self.dir_rgb
        assert os.path.isdir(self.dir_depth), '%s is not a valid directory' % self.dir_depth
        assert os.path.isdir(self.dir_lidar), '%s is not a valid directory' % self.dir_lidar
        
        self.rgb_paths = sorted(make_dataset(self.dir_rgb, opt.max_dataset_size))
        self.depth_paths = sorted(make_dataset(self.dir_depth, opt.max_dataset_size))
        self.lidar_paths = sorted(make_dataset(self.dir_lidar, opt.max_dataset_size))
        
        self.transform_rgb = get_transform(opt, grayscale=False)
        self.transform_depth_lidar = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        lidar_path = self.lidar_paths[index]
        
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')
        lidar_img = Image.open(lidar_path).convert('L')
        
        rgb = self.transform_rgb(rgb_img)
        depth = self.transform_depth_lidar(depth_img)
        lidar = self.transform_depth_lidar(lidar_img)
        
        combined_img = torch.cat((rgb, depth, lidar), 0)
        
        return {'rgb': rgb, 'depth': depth, 'lidar': lidar, 'combined': combined_img, 'A_paths': rgb_path, 'B_paths': rgb_path}

    def __len__(self):
        return len(self.rgb_paths)
    
    def name(self):
        return 'AlignedDataset'
