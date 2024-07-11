import os
from PIL import Image
from data.base_dataset import BaseDataset, make_dataset, get_transform

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'A')  # create a path '/path/to/data/train/A'
        self.dir_B = os.path.join(opt.dataroot, 'B')  # create a path '/path/to/data/train/B'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/train/A'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/train/B'
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform(A_img)
        B = self.transform(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)
    
    def name(self):
        return 'AlignedDataset'
