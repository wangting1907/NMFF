# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:56:43 2023

@author: 19652
"""

from __future__ import print_function

import argparse
import os
import random
import shutil
import scipy.io as scio
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.optimize import nnls
from sklearn.decomposition import NMF
import scipy.io as scio
from model_nerv_unmixing import *
from utils import *
from nmf_abundance import *
import os


def main():
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # dataset parameters
    parser.add_argument('--vid',  default=[None], type=int,  nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame_gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset',)
    parser.add_argument('--test_gap', default=1, type=int, help='evaluation gap')

# NERV architecture parameters
    # embedding parameters
    parser.add_argument('--embed', type=str, default='1.25_80', help='base value/embed length for position encoding')

    # FC + Conv parameters
  #  parser.add_argument('--stem_dim_num', type=str, default='1024_1', help='hidden dimension and length')
   # parser.add_argument('--fc_hw_dim', type=str, default='9_16_128', help='out size (h,w) for mlp')

    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
   # parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
   
    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')


    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    args = parser.parse_args()
        
    args.warmup = int(args.warmup * args.epochs)

    print(args)
    torch.set_printoptions(precision=4) 

    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    extra_str = '_dist' if args.distributed else '', f'_eval' if args.eval_only else ''
    norm_str = '' if args.norm == 'none' else args.norm

    exp_id = f'{args.dataset}_embed{args.embed}_cycle{args.cycles}' + \
            f'_gap{args.frame_gap}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_lr{args.lr}_{args.lr_type}' + \
            f'_{args.loss_type}{norm_str}{extra_str}'
    
    exp_id += f'_{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
    	print('Will overwrite the existing output dir!')
    	shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)


def train(local_rank, args):
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSetuniform
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

  #  train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
   # is_train_best, is_val_best = False, False

    PE = PositionalEncoding(args.embed)
    args.embed_length = PE.embed_length
    model = Siren(in_features=args.embed_length, out_features=3, hidden_features=64,
            hidden_layers=3, outermost_linear=True)

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda() #model.cuda() #
    else:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))
    data_h=scio.loadmat('./data/dataset_real_samson.mat')
    X=data_h.get("X")
   # X1=X
    iterNum = 1
    tol = 0.1
    tolObj=0.001; 
    maxIter=3
    alpha = 15
    fDelta =50
    args.start_epoch = 0
    maxIters=500
    endnum=3
    row=95
    end=scio.loadmat('./data/endmem_init.mat')
    #frame_idx=[2,4,5,6,8,12,14,15,18,21,28,32,36,40,45,48,52,54,57,60,64,70,75,79,85,89,90,95,98,101,103,105,108,112,114,116,120,123,124,128,130,139,144,152,156]
    #frame_idx=[1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 36, 40, 43, 47, 50, 54, 57, 61, 64, 68, 71, 75, 78, 82, 86, 89, 93, 96, 100, 103, 107, 110, 114, 117, 121, 124, 128, 131, 135, 138, 142, 145, 149, 152, 156]
    #frame_idx=[1,2,4,5,6,8,9,10,12,14,15,18,19,21,22,25,28,29,32,60,61,64,67,68,70,71,72,75,76,101,103,104,105,108,112,113,114,116,117,120,123,124,128,130,156]
    frame_idx= [15,18,19,21,22,25,28,29,32,101,103,104,105,108,112,113,114,116,117,120,123,124,128,130,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,150,151,152,153,155,156]
    frame_idx = [int(x - 1) for x in frame_idx]
    #total_idx = list(range(156))
   # missing=[1, 8, 32, 35, 50, 54, 76, 117, 120, 154]
    #frame_idx=set(total_idx)-set(missing)
    #frame_idx=sorted(list(frame_idx))
    X = X[frame_idx, :]
    endmember=end.get("HVca")
    endmember=endmember[frame_idx,:]
    #endmember = endmember / np.tile(np.max(endmember, axis=0), (endmember.shape[0], 1))
    abundance=FCLSU(X, endmember)

    #abundance=nmfAbundance(X.T, endnum, endmember.T, alpha, tol, maxIters)
    abundance=np.transpose(abundance)
    abundance = abundance/np.sum(abundance, axis=0)
    #model_init = NMF(n_components=3, init='random', random_state=0)
    #W = model_init.fit_transform(X)
    #abundance = model_init.components_
    #abundance = abundance/np.sum(abundance, axis=0)
    train_dataset = DataSet(X, frame_idx, img_transforms,vid_list=args.vid, frame_gap=args.frame_gap,  )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    val_dataset = DataSet(X, frame_idx, img_transforms, vid_list=args.vid, frame_gap=args.test_gap,  )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,  shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)
    data_size = len(train_dataset)
    while iterNum <= maxIter:
        #train_data_dir = f'./data/{args.dataset.lower()}'
        #val_data_dir = f'./data/{args.dataset.lower()}'
        # Training
        start = datetime.now()
        total_epochs = args.epochs * args.cycles
        abundance=torch.tensor(abundance)
        for epoch in range(args.start_epoch, total_epochs):
            model.train()
            epoch_start_time = datetime.now()
            psnr_list = []
            msssim_list = []
            output_list=[]
            # iterate over dataloader
            for i, (data,  norm_idx) in enumerate(train_dataloader):
                if i > 10 and args.debug:
                    break
                embed_input = PE(norm_idx)
                if local_rank is not None:
                    data = data.type(torch.float).cuda(local_rank, non_blocking=True)
                    embed_input = embed_input.type(torch.float).cuda(local_rank, non_blocking=True)
                    abundance=abundance.type(torch.float).cuda(local_rank, non_blocking=True)
                else:
                    data,  embed_input, abundance = data.type(torch.float).cuda(non_blocking=True), embed_input.type(torch.float).cuda(non_blocking=True), abundance.type(torch.float).cuda(non_blocking=True)
            # forward and backward
                Endmember = model(embed_input)
                output_list=endab(Endmember,abundance,endnum,row)                                                                       #######
                target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
                target_list=torch.stack(target_list)
                target_list=target_list.reshape((1, 1, row, row))
                loss_list = [loss_fn(output.float(), target.float(), args) for output, target in zip(output_list, target_list)]
                loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
                lambdas = [0.6 for _ in range(130)] + [0 for _ in range(26)]
                extra= lambdas[i] *vol(Endmember,endnum)
                loss_sum = sum(loss_list)
                loss_sum=loss_sum+extra
                lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                

                # compute psnr and msssim
                psnr_list.append(psnr_fn(output_list, target_list))
                msssim_list.append(msssim_fn(output_list, target_list))
                if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                    train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                    train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                    train_msssim = torch.cat(msssim_list, dim=0) #(batchsize, num_stage)
                    train_msssim = torch.mean(train_msssim.float(), dim=0) #(num_stage)
                    time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                        time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                        RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                    print(print_str, flush=True)
                    if local_rank in [0, None]:
                        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                            f.write(print_str + '\n')

            # collect numbers from other gpus
            if args.distributed and args.ngpus_per_node > 1:
                train_psnr = all_reduce([train_psnr.to(local_rank)])
                train_msssim = all_reduce([train_msssim.to(local_rank)])
            state_dict = model.state_dict()
            save_checkpoint={
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),   
            }    
             #evaluation
            if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
                val_start_time = datetime.now()
                val_psnr, val_msssim, endmember = evaluate(model, val_dataloader, PE, endnum,row,local_rank, abundance, args)
                val_end_time = datetime.now()
                if args.distributed and args.ngpus_per_node > 1:
                    val_psnr = all_reduce([val_psnr.to(local_rank)])
                    endmember=all_reduce([endmember.to(local_rank)])
            if local_rank in [0, None]:
                torch.save(save_checkpoint, '{}/trained_model.pth'.format(args.outf))
        #torch.save(Endmembers, '{}/Endmembers.pth'.format(args.outf))
        print("Training complete in: " + str(datetime.now() - start))
        endmember=torch.stack(endmember)
        endmember=endmember.view(45,endnum)
        endmember= endmember.detach().cpu().numpy()
        endmember = endmember / np.tile(np.max(endmember, axis=0), (endmember.shape[0], 1))
        abundance=abundance.detach().cpu().numpy()
        abundance=nmfAbundance(X.T, endnum, endmember.T, abundance.T,  alpha, tol, maxIters)
        abundance=np.transpose(abundance)
        abundance[abundance<=0.001]=0
       # abundance=hyperNmfASCL1_2(X, endmember, abundance, tolObj, 200, fDelta)
       # abundance[abundance<=0.001]=0
        abundance = abundance/np.sum(abundance, axis=0)
        iterNum += 1
        err = 0.5 * np.linalg.norm(X- np.dot(endmember, abundance), ord=2) ** 2
        dispStr = 'Iteration {}, loss = {}'.format(iterNum, err)
        print(dispStr)
    endmember1=endmember
    torch.save( abundance, '{}/abundance.pth'.format(args.outf))
    torch.save( endmember1, '{}/endmember.pth'.format(args.outf))
@torch.no_grad()
def evaluate(model, val_dataloader, pe, endnum, row,local_rank, abundance, args):
    psnr_list = []
    msssim_list = []
    if args.dump_images:
        from torchvision.utils import save_image
        visual_dir = f'{args.outf}/visualize'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    time_list = []
    output_list=[]
    Endmember_predict=[]
    model.eval()
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        if i > 10 and args.debug:
            break
        embed_input = pe(norm_idx)
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
            
        else:
            data,  embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)
            

        # compute psnr and msssim
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            # embed_input = embed_input.half()
            # model = model.half()
            start_time = datetime.now()
            Endmember = model(embed_input)
            Endmember_predict.append(Endmember)
            output_list=endab(Endmember,abundance,endnum,row)   
            
                                                                                  ##############################
            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        # dump predictions
        if args.dump_images:
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')
        ### save endmembers
        torch.save( Endmember_predict, '{}/Endmemberpredict.pth'.format(args.outf))
        
        
        
        
        # compute psnr and ms-ssim
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)        
        if i % args.print_freq == 0:
            fps = fwd_num * (i+1) * args.batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
    model.train()

    return val_psnr, val_msssim, Endmember_predict    

def predict(pre_dataloader, pe, local_rank, args):
    
    model= torch.load('save_checkpoint/trained_model.pth')
    Endmember_predict=[]
    with torch.no_grad():
       for i, (data,  norm_idx) in enumerate(pre_data):
             if i > 10 and args.debug:
                   break
             embed_input = pe(norm_idx)
             if local_rank is not None:
                   data = data.cuda(local_rank, non_blocking=True)
                   embed_input = embed_input.cuda(local_rank, non_blocking=True)
            
             else:
                   data,  embed_input = data.cuda(non_blocking=True), 
                   embed_input.cuda(non_blocking=True)        
             Endmember = model(embed_input)
             Endmember_predict.append(Endmember)
             return Endmember_predict

def predict(pre_dataloader, pe, local_rank, args):
    
    model= torch.load('save_checkpoint/trained_model.pth')
    Endmember_predict=[]
    with torch.no_grad():
       for i, norm_idx in enumerate(pre_data):
             if i > 10 and args.debug:
                   break
             embed_input = pe(norm_idx)
             if local_rank is not None:
                   data = data.cuda(local_rank, non_blocking=True)
                   embed_input = embed_input.cuda(local_rank, non_blocking=True)
            
             else:
                   data,  embed_input = data.cuda(non_blocking=True), 
                   embed_input.cuda(non_blocking=True)        
             Endmember = model(embed_input)
             Endmember_predict.append(Endmember)
             return Endmember_predict
if __name__ == '__main__':
    main()


      