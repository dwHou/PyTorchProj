from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Compose

import numpy as np

# Testing settings
parser = argparse.ArgumentParser(description='PyTorch CAR Solver')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('RGB')

model = torch.load(opt.model)
img_to_tensor = ToTensor()
input = img_to_tensor(img)
input = torch.unsqueeze(input, 0)


if opt.cuda:
    model = model.cuda()
    input = input.cuda()

with torch.no_grad():
    out = model(input)
#print(out.shape)
out = torch.squeeze(out, 0)
#print(out.shape)
out = out.cpu()
out = out.detach().numpy() * 255.0
out = out.clip(0, 255).transpose(1, 2, 0)
print(out.shape)
out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')
print(out_img.size)

'''
out = model(input)
out = out.cpu()
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
'''

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
