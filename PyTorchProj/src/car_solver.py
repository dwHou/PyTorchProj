# :set expandtab
# :%ret! 4

from __future__ import print_function
import os, fnmatch
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from util import common, brisque_v3
from model.VQBooster.VQBooster import RES
# from model.VQBooster.VQNLα_r4 import RES
from model.ArCAIN1 import ArCAIN
from model.LSTM.ByteCAR import LSTMIQANet
import numpy as np
import time
from tqdm import tqdm
import math
from contextlib import contextmanager

# solver settings
parser = argparse.ArgumentParser(description='PyTorch CAR Solver')
parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')
parser.add_argument('--input_video', type=str, default='./test/pipi003', help='input video to use')
parser.add_argument('--ckp', type=str, default='./experiment/vqtiny_37.43dB.pth', help='model file to use')
parser.add_argument('--output_video', type=str, default='./arout/VQtiny/pipi003loss', help='where to save the output video')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--step', type=int, default=10, help='input frames numbers of convlstm')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
args = parser.parse_args()
print(args)


def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[-2]
        w_x = x.size()[-1]
#     r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
#     l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
#     t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
#     b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        r = F.pad(x, (0, 1, 0, 0), mode='replicate')[:, :, :, :w_x]
        l = F.pad(x, (1, 0, 0, 0), mode='replicate')[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0), mode='replicate')[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1), mode='replicate')[:, :, :h_x, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    # xgrad = (r - l) * 0.5 + (t - b) * 0.5
    return xgrad

class Evaluator():
    def __init__(self, args, my_model):
        self.args = args
        self.ckp = args.ckp
        self.video_i = args.input_video
        self.video_o = args.output_video
        self.model = my_model

    def sharp_si_test(self, crf=24):
        img_list = fnmatch.filter(os.listdir(self.video_i), '*.png')
        img_list.sort(key= lambda x:int(x[:-4]))

        start = time.perf_counter()
        # avg_psnr = 0
        with torch.no_grad():
            for i in tqdm(range(len(img_list))):
                if not img_list[i].endswith('png'):
                    continue
                img_path = os.path.join(self.video_i, img_list[i])
                img = Image.open(img_path).convert('RGB')
                img = np.asarray(img)
	        
                # feats = brisque_v3.brisque_feats(img)
                # feats = np.append(feats, crf/50)
                # degradation = torch.from_numpy(feats).float().cuda()
	
                input = torch.from_numpy(img).permute(2,0,1).float()/255

                input = torch.unsqueeze(input, 0)
                input = input.cuda()

                model = self.model

                with self.timer('inference'):
                   #  if self.args.chop:
                        # out = forward_chop(input, model)
                   #  else:    
                   #  out = model(input, degradation)
                    out = model(input)

                   # sharp
                    grad = gradient_1order(out)
                    out = out + grad

                # B=1, C, H, W
                out = torch.squeeze(out, 0)
                out = out.cpu()
                out = out.detach().numpy() * 255.0
                out = out.clip(0, 255).transpose(1, 2, 0)
                out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')

                output_path = os.path.join(self.video_o, img_list[i])
                out_img.save(output_path)
                # print('output image saved to ', output_path)
        end = time.perf_counter()
        infer_time = end - start
        print('total inference per frame consume {:.6f} seconds'.format(infer_time / len(img_list)))

    def si_test(self, crf=24):
        img_list = fnmatch.filter(os.listdir(self.video_i), '*.png')
        img_list.sort(key=lambda x: int(x[:-4]))

        start = time.perf_counter()
        # avg_psnr = 0
        with torch.no_grad():
            for i in tqdm(range(len(img_list))):
                if not img_list[i].endswith('png'):
                    continue
                img_path = os.path.join(self.video_i, img_list[i])
                img = Image.open(img_path).convert('RGB')
                img = np.asarray(img)

                # feats = brisque_v3.brisque_feats(img)
                # feats = np.append(feats, crf/50)
                # degradation = torch.from_numpy(feats).float().cuda()

                input = torch.from_numpy(img).permute(2, 0, 1).float() / 255

                input = torch.unsqueeze(input, 0)
                input = input.cuda()

                model = self.model

                with self.timer('inference'):
                    #  if self.args.chop:
                    # out = forward_chop(input, model)
                    #  else:
                    #  out = model(input, degradation)
                    out = model(input)
                # B=1, C, H, W
                out = torch.squeeze(out, 0)
                out = out.cpu()
                out = out.detach().numpy() * 255.0
                out = out.clip(0, 255).transpose(1, 2, 0)
                out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')

                output_path = os.path.join(self.video_o, img_list[i])
                out_img.save(output_path)
                # print('output image saved to ', output_path)
        end = time.perf_counter()
        infer_time = end - start
        print('total inference per frame consume {:.6f} seconds'.format(infer_time / len(img_list)))

    def v_test(self):
        img_list = fnmatch.filter(os.listdir(self.video_i), '*.png')
        img_list.sort(key= lambda x:int(x[:-4]))

        run_times = int(len(img_list) / self.args.step)
        remains = len(img_list) - run_times * self.args.step

        with torch.no_grad():
            for clip in tqdm(range(run_times)):
                list_clip = []
                for i in range(clip * self.args.step, clip * self.args.step + self.args.step):
                    if not img_list[i].endswith('png'):
                        continue

                    img_path = os.path.join(self.video_i, img_list[i])
                    img = Image.open(img_path).convert('RGB')
                    img = np.asarray(img)
                    list_clip.append(img)
                with self.timer('inference'):
                    input = np.stack(list_clip, axis=0)
                    # print(input.shape)
                    feats = brisque_v3.brisque_feats(input)
                    feats = torch.from_numpy(feats).float().cuda()
                    input = input[np.newaxis,:]
                    input = common.np2Tensor(input, self.args.rgb_range)[0]
                    input = torch.unsqueeze(input, 0)
                    input = input.cuda()
                    model = self.model
                
                # with self.timer('inference'):
                    if self.args.chop:
                        out = forward_chop(input, model)
                    else:    
                        out = model(input, feats)

                out = torch.squeeze(out, 0)
                out = out.cpu() # 20, 3 , 540, 960
                out = out.detach().numpy() * 255.0
                out = out.clip(0, 255).transpose(0, 2, 3, 1)
                i = 0
                for out_img in out:
                    # print(out_img.shpe)
                    out_img = Image.fromarray(out_img.astype(np.uint8), mode='RGB')
                    output_path = os.path.join(self.video_o, '{}.png'.format(clip * self.args.step + i))
                    out_img.save(output_path)
                    i += 1

            if remains != 0:
                for i in range(len(img_list) - self.args.step, len(img_list)):
                    if not img_list[i].endswith('png'):
                        continue

                    img_path = os.path.join(self.video_i, img_list[i])
                    img = Image.open(img_path).convert('RGB')
                    img = np.asarray(img)
                    list_clip.append(img)

                input = np.stack(list_clip, axis=0)
                # print(input.shape)
                feats = brisque_v3.brisque_feats(input)
                feats = torch.from_numpy(feats).float().cuda()
                input = input[np.newaxis, :]
                input = common.np2Tensor(input, self.args.rgb_range)[0]
                input = torch.unsqueeze(input, 0)
                input = input.cuda()
                model = self.model

                with self.timer('inference'):
                    if self.args.chop:
                        out = forward_chop(input, model)
                    else:
                        out = model(input, feats)

                out = torch.squeeze(out, 0)
                out = out.cpu()  # 20, 3 , 540, 960
                out = out.detach().numpy() * 255.0
                out = out.clip(0, 255).transpose(0, 2, 3, 1)
                i = 0
                for out_img in out:
                    # print(out_img.shpe)
                    out_img = Image.fromarray(out_img.astype(np.uint8), mode='RGB')
                    output_path = os.path.join(self.video_o, '{}.png'.format(len(img_list) - self.args.step + i))
                    out_img.save(output_path)
                    i += 1

        

    def mf_test(self):
        img_list = fnmatch.filter(os.listdir(self.video_i), '*.png')
        img_list.sort(key= lambda x:int(x[:-4]))

        start = time.perf_counter()
        with torch.no_grad():
            for index in tqdm(range(len(img_list))):
                list_mf = []
                for i in [index-2 if (index-2)>0 else 0, 
                          index-1 if (index-1)>0 else 0, 
                          index, 
                          index+1 if (index+1)<len(img_list) else len(img_list)-1, 
                          index+2 if (index+2)<len(img_list) else len(img_list)-1]:
            
                    if not img_list[i].endswith('png'):
                        continue

                    img_path = os.path.join(self.video_i, img_list[i])
                    img = Image.open(img_path).convert('RGB')
                    img = np.asarray(img)
                    list_mf.append(img)

                input = np.stack(list_mf, axis=0)
                input = input[np.newaxis, :] # B N H W C
                input = common.np2Tensor(input, self.args.rgb_range)[0]
                input = torch.unsqueeze(input, 0)  # 因为np2Tensor输入是有batch，输出却没有了。

                input = input.cuda()
                model = self.model
               
                with self.timer('inference'):
                    if self.args.chop:
                        out = forward_chop(input, model)
                    else:    
                        out = model(input)

                out = torch.squeeze(out, 0)
                # out = out.squeeze(0)
                out = out.cpu()
                out = out.detach().numpy() * 255.0
                out = out.clip(0, 255).transpose(1, 2, 0)
                out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')

                output_path = os.path.join(self.video_o, img_list[index])
                out_img.save(output_path)
                # print('output image saved to ', output_path)
        end = time.perf_counter()
        infer_time = end - start
        print('total inference per frame consume {:.6f} seconds'.format(infer_time / len(img_list)))
        
        
    @contextmanager
    def timer(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        yield
        end.record()

        torch.cuda.synchronize()
        print(f'[{name}] done in {start.elapsed_time(end):.3f} ms')
        print('per frame consume {} ms'.format(start.elapsed_time(end) / self.args.step))



    def cal_psnr(self, img1, img2):
        img1 = np.asarray(img1)
        img2 = np.asarray(img2)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))
    
    def mean_psnr(self):
        avg_psnr = 0
        i_dir = os.path.join(self.video_o, 'si_test')
        i_label_dir = './test/GT'
        img_list = os.listdir(i_dir)  
        # img_list.sort(key=lambda x: int(x[:-4]))

        for i in range(len(img_list)):
            if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                continue

            img_i = os.path.join(i_dir, img_list[i])
            label_i = os.path.join(i_label_dir, img_list[i])
            img_i, label_i = Image.open(img_i), Image.open(label_i)
            # img_i, label_i = cv2.imread(img_i), cv2.imread(label_i)
            psnr_num = self.cal_psnr(img_i, label_i)
            # psnr_num = measure.compare_psnr(img_i, label_i, data_range=255)
            # list_psnr.append(psnr_num) 
            print(psnr_num)
            avg_psnr += psnr_num

        # print("平均PSNR:", np.mean(list_psnr))
        print("平均PSNR:", avg_psnr / len(img_list))
    
def forward_chop(x, forward_function, shave=8, min_size=150000):
    # min_size = 40000 chop到240x135
    # min_size = 150000 chop到480x270
    # min_size = self.chop_threshold
    # scale = self.scale[self.idx_scale]
    scale = 1 # 不是超分任务
    # n_GPUs = min(self.n_GPUs, 4)
    n_GPUs = 1
    multi_frame = len(x.size()) == 5
    if multi_frame:
        b, f, c, h, w = x.size()
    else:
        b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    if multi_frame:
        lr_list = [
            x[:, :, :, 0:h_size, 0:w_size],
            x[:, :, :, 0:h_size, (w - w_size):w],
            x[:, :, :, (h - h_size):h, 0:w_size],
            x[:, :, :, (h - h_size):h, (w - w_size):w]]
    else:
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
    # 240 x 135    
    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = forward_function(lr_batch)
            # https://zhuanlan.zhihu.com/p/59141209 chunk是和cat相反的过程。
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    # 如果显存还是不够，再切
    else:
        sr_list = [
            forward_chop(patch, forward_function, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale
 
    
    # output = x.new(b, c, h, w)
    # output[:, :, 0:h_half, 0:w_half] \
    #     = sr_list[0][:, :, 0:h_half, 0:w_half]
    # output[:, :, 0:h_half, w_half:w] \
    #     = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    # output[:, :, h_half:h, 0:w_half] \
    #     = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    # output[:, :, h_half:h, w_half:w] \
    #     = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        
    

    output = x.new(b, f, c, h, w)

    output[:, :, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, :, 0:h_half, 0:w_half]

    output[:, :, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, :, 0:h_half, (w_size - w + w_half):w_size] # 相当于shave:w_size
    output[:, :, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, :, (h_size - h + h_half):h_size, 0:w_half]
        
    output[:, :, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        
        
    return output

if __name__=='__main__': 
    print('===> Loading pretrained model')
    
    device = torch.device("cuda" if args.cuda else "cpu")
    # model = RES(6, 32, 1).to(device)
    # model = LSTMIQANet().to(device)
    model = RES(6, 32, 1).to(device)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.ckp).items()})
    
    print('load success')
    t = Evaluator(args, model)
    t.si_test()
    # t.mean_psnr()

            


