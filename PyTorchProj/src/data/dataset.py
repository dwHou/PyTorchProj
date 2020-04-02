import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from util import common
import numpy as np
import pickle
import lmdb

from option import opt


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

# def load_img(filepath):
#  # img = Image.open(filepath)
#  # return img
#     img = Image.open(filepath).convert('YCbCr')
#     y, _, _ = img.split()
#     return y

class Demo_lmdb(data.Dataset):
    def __init__(self, db_path):
        super(Demo_lmdb, self).__init__()
        env = lmdb.open(db_path)
        txn = env.begin()
        self.txn = txn

    def __getitem__(self, index):
        self.index = index
        np_in, np_tar = self._load_lmdb(self.index)
        self.patch_size = opt.patchSize
        patch_in, patch_tar = common.get_patch(np_in, np_tar, self.patch_size)

        patch_in, patch_tar = common.np2Tensor([patch_in, patch_tar], opt.rgb_range)

        return patch_in, patch_tar

    def __len__(self):
        return self.txn.stat()['entries']


    def _load_lmdb(self, index):
        pairs = self.txn.get('{}'.format(index).encode())
        np_pairs = pickle.loads(pairs)
        return np_pairs[0], np_pairs[1]




class Demo(data.Dataset):
    def __init__(self, txt_path):
        super(Demo, self).__init__()
        fh = open(txt_path, 'r')
        pairs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            pairs.append((words[0], words[1]))
            self.pairs = pairs

    def __getitem__(self, index):
        # input, target = self.pairs[index]
        # img_in = Image.open(input).convert('RGB')
        # img_tar = Image.open(target).convert('RGB')
        self.index = index
        np_in, np_tar = self._load_file(index)
        self.patch_size = opt.patchSize
        patch_in, patch_tar = common.get_patch(np_in, np_tar, self.patch_size)

        patch_in, patch_tar = common.np2Tensor([patch_in, patch_tar], opt.rgb_range)

        return patch_in, patch_tar

    def __len__(self):
        return len(self.pairs)

    def _load_file(self, index):
        #index = self.index
        input, target = self.pairs[index]
        img_in = Image.open('/Applications/Programming/Dataset/VSR/youkudataset/'+input).convert('RGB')
        img_tar = Image.open('/Applications/Programming/Dataset/VSR/youkudataset/'+target).convert('RGB')
        np_in = np.asarray(img_in)
        np_tar = np.asarray(img_tar)

        return np_in, np_tar
