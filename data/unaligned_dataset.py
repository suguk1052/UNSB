import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.use_mask = getattr(opt, 'use_mask', False)
        if self.use_mask:
            self.dir_A_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')
            self.dir_B_mask = os.path.join(opt.dataroot, opt.phase + 'B_mask')

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")
            if self.use_mask:
                self.dir_A_mask = os.path.join(opt.dataroot, "valA_mask")
                self.dir_B_mask = os.path.join(opt.dataroot, "valB_mask")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        if self.use_mask:
            self.A_mask_paths = sorted(make_dataset(self.dir_A_mask, opt.max_dataset_size))
            self.B_mask_paths = sorted(make_dataset(self.dir_B_mask, opt.max_dataset_size))
            if len(self.A_mask_paths) != len(self.A_paths):
                raise ValueError(f"Mask directory {self.dir_A_mask} must contain the same number of files as {self.dir_A}.")
            if len(self.B_mask_paths) != len(self.B_paths):
                raise ValueError(f"Mask directory {self.dir_B_mask} must contain the same number of files as {self.dir_B}.")
        else:
            self.A_mask_paths = []
            self.B_mask_paths = []
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        if self.use_mask:
            A_mask_path = self.A_mask_paths[index % self.A_size]
            B_mask_path = self.B_mask_paths[index_B]
            A_mask_img = Image.open(A_mask_path).convert('L')
            B_mask_img = Image.open(B_mask_path).convert('L')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        params_A = get_params(modified_opt, A_img.size)
        params_B = get_params(modified_opt, B_img.size)
        A_transform = get_transform(modified_opt, params_A)
        B_transform = get_transform(modified_opt, params_B)
        A = A_transform(A_img)
        B = B_transform(B_img)
        sample = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        if self.use_mask:
            A_mask_transform = get_transform(modified_opt, params_A, grayscale=True, method=Image.NEAREST)
            B_mask_transform = get_transform(modified_opt, params_B, grayscale=True, method=Image.NEAREST)
            A_mask = A_mask_transform(A_mask_img)
            B_mask = B_mask_transform(B_mask_img)
            sample['A'] = torch.cat([sample['A'], A_mask], dim=0)
            sample['B'] = torch.cat([sample['B'], B_mask], dim=0)
            sample['A_mask'] = A_mask
            sample['B_mask'] = B_mask

        return sample

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
