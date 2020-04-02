from os.path import exists, join, basename
# from os import makedirs, remove
from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
# 现在我只使用了VESR Demo数据集
from data.dataset import Demo, Demo_lmdb


# 手动transform的。没有用上。
def input_transform(crop_size):
    return Compose([
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        ToTensor(),
    ])


def get_training_set():
    #train_txt = '/Applications/Programming/Dataset/VSR/youkudataset/Demo/train.txt'
    #train_txt = '/Volumes/Samsung_T5/Datasets/youkuCAR/train.txt'
    #return Demo(train_txt)
    train_db = '/Applications/Programming/Dataset/VSR/youkudataset/Demo_lmdb/demo_train_lmdb'
    return Demo_lmdb(train_db)

def get_test_set():
    #val_txt = '/Applications/Programming/Dataset/VSR/youkudataset/Demo/valid.txt'
    #val_txt = '/Volumes/Samsung_T5/Datasets/youkuCAR/valid.txt'
    #return Demo(val_txt)
    val_db = '/Applications/Programming/Dataset/VSR/youkudataset/Demo_lmdb/demo_val_lmdb'
    return Demo_lmdb(val_db)
