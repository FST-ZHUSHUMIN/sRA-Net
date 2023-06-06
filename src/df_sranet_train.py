# python -m src.df_sranet_train --conf src.conf.df_sranet_cf

# from torch._C import float64
# from torch._C import float64
from torch.utils.data import sampler
from src.dataset import  DeepFashionCAPDataset
from src.const import base_df_path
import pandas as pd
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
from tensorboardX import SummaryWriter
import os
from skimage import io, transform
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import random


def train(train_dataloader, net, epoch, writer, step):
    net.train()
    if epoch + 1<= 6:    #epoch从0开始计数
        learning_rate = 0.0001
    elif epoch + 1 <= 12:
        learning_rate = 0.00001
    else:
        learning_rate = 0.000001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    total_step = len(train_dataloader)
    
    for i, sample in enumerate(train_dataloader):
        step += 1
        for key in sample:
            sample[key] = sample[key].to(const.device)

        # print(i, sample.keys())
        output = net(sample)
        loss = net.cal_loss(sample, output)
        optimizer.zero_grad()
        loss['all'].backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            if 'category_loss' in loss:
                writer.add_scalar('loss/category_loss', loss['category_loss'], step)
                writer.add_scalar('loss_weighted/category_loss', loss['weighted_category_loss'], step)
            if 'attr_loss' in loss:
                writer.add_scalar('loss/attr_loss', loss['attr_loss'], step)
                writer.add_scalar('loss_weighted/attr_loss', loss['weighted_attr_loss'], step)
            if 'lm_vis_loss' in loss:
                writer.add_scalar('loss/lm_vis_loss', loss['lm_vis_loss'], step)
                writer.add_scalar('loss_weighted/lm_vis_loss', loss['weighted_lm_vis_loss'], step)
            if 'lm_pos_loss' in loss:
                writer.add_scalar('loss/lm_pos_loss', loss['lm_pos_loss'], step)
                writer.add_scalar('loss_weighted/lm_pos_loss', loss['weighted_lm_pos_loss'], step)
            writer.add_scalar('loss_weighted/all', loss['all'], step)
            writer.add_scalar('global/learning_rate', learning_rate, step)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, 12, i + 1, total_step, loss['all'].item()))

        if (i + 1) % total_step == 0:
        # if (i + 1) % 1000 == 0:
            print('Saving Model....')
            # net.set_buffer('step', step)
            save_file = 'models/' + const.MODEL_NAME
            if not os.path.exists(save_file):
                os.mkdir(save_file)
            save_name = save_file + '/' + const.MODEL_NAME + str(epoch+1)+'.pkl'
            torch.save(net.state_dict(), save_name)
            print('OK.')
            break
            # evaluate(val_dataloader, net, writer, step)
    return net, step

# os.path.exists
# os.makedirs

def evaluate(val_dataloader, net, writer, step):
    val_step = len(val_dataloader)
    if const.VAL_WHILE_TRAIN:
        # print(const.EVALUATOR, const.mode)
        with torch.no_grad():
            net.eval()
            evaluator = const.EVALUATOR(mode = const.mode)
            for j, sample in enumerate(val_dataloader):
                for key in sample:
                    sample[key] = sample[key].to(const.device)
                output = net(sample)
                evaluator.add(output, sample)
                if (j + 1) % 100 == 0:
                    print('Val Step [{}/{}]'.format(j + 1, val_step))
                    
            ret = evaluator.evaluate()
        
            for topk, accuracy in ret['attr_group_recall'].items():
                for attr_type in range(1, 6):
                    print('metrics/attr_top{}_type_{}_{}_recall'.format(
                    topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1]
                    )
                    writer.add_scalar('metrics/attr_top{}_type_{}_{}_recall'.format(
                    topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1], step
                    )
                print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall_real'][topk])
                writer.add_scalar('metrics/attr_top{}_all_recall_real'.format(topk), ret['attr_recall_real'][topk], step)
          #      print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall_fake'][topk])
          #      writer.add_scalar('metrics/attr_top{}_all_recall_fake'.format(topk), ret['attr_recall_fake'][topk], step)

            print('mode: ', const.mode)
            if const.mode == 'addcategory':
                for topk, accuracy in ret['category_accuracy_topk'].items():
                    print('metrics/category_top{}'.format(topk), accuracy)
                    writer.add_scalar('metrics/category_top{}'.format(topk), accuracy, step)
            
            if const.mode == 'addlandmark':
                for i in range(8):
                    print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
                    writer.add_scalar('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i], step)
                print('metrics/dist_all', ret['lm_dist'])
                writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)

    return net, step


def test(path, net, test_dataloader, save_name):
    state_dict = torch.load(save_name)
    if 'attr_loss_func.pos_weight' in  state_dict:
        state_dict.pop('attr_loss_func.pos_weight')
    # net.load_state_dict(state_dict, strict = False)
    net.load_state_dict(state_dict)

    evaluator = const.EVALUATOR()
    net.eval()

    with torch.no_grad():

        evaluator = const.EVALUATOR(mode = const.mode)
        for j, sample in enumerate(test_dataloader):
            for key in sample:
                sample[key] = sample[key].to(const.device)

            output = net(sample)
            evaluator.add(output, sample)
            if (j + 1) % 50 == 0:
                print('Val Step [{}/{}]'.format(j + 1, val_step))
                
        ret = evaluator.evaluate()
    
        for topk, accuracy in ret['attr_group_recall'].items():
            for attr_type in range(1, 6):
                print('metrics/attr_top{}_type_{}_{}_recall'.format(
                topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1]
                )
            print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall_real'][topk])
            print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall_fake'][topk])

        if const.mode == 'addcategory':
            for topk, accuracy in ret['category_accuracy_topk'].items():
                print('metrics/category_top{}'.format(topk), accuracy)
        
        if const.mode == 'addlandmark':
            for i in range(8):
                print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
            print('metrics/dist_all', ret['lm_dist'])
    return net


if __name__ == '__main__':
    # preprocessxlsx()
    if os.path.exists('models') is False:
            os.makedirs('models')
    parse_args_and_merge_const()

    df_path = const.base_df_path + const.USE_df_CSV
    # print('deepfashion path', df_path)
    df = pd.read_csv( df_path )
    # print('columns', len(df.columns), df.columns)

    train_df = df[df['evaluation_status'] == 'train']
    train_dataset =  DeepFashionCAPDataset(train_df, training=True, mode=const.DATASET_PROC_METHOD_TRAIN)   
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4)
    val_df = df[df['evaluation_status'] == 'test']
    val_dataset =  DeepFashionCAPDataset(val_df, training=False, mode=const.DATASET_PROC_METHOD_TRAIN)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=const.VAL_BATCH_SIZE, shuffle=False, num_workers=4)
    val_step = len(val_dataloader)
    # # net  
    net = const.USE_NET()
    # print(net)
    net = net.to(const.device)
    
    # # for name, param in net.named_parameters():
    # #     print(name,param.requires_grad)

    
    # # # print(net)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    writer = SummaryWriter(const.TRAIN_DIR)

    total_step = len(train_dataloader)
    total_epoch = 12
    
    step = 0
    save_file = './models/' + const.MODEL_NAME
    if os.path.exists(save_file) is False:
        os.makedirs(save_file)
    
    for epoch in range(total_epoch):
        net, step = train(train_dataloader, net, epoch, writer, step)
        if const.VAL_WHILE_TRAIN:
            print('Now Evaluate..') 
            net, step = evaluate(val_dataloader, net, writer, step)

    # save_name = '/home/shumin/Documents/new_folder/zhushumin-project/DeepFashionmaster/TMM-RANet/models/df_ranetSA_1/df_ranetSA_112.pkl'
    # print(save_name)
    # print('Now Test......')
    # from backbone.ablation_Spacial_attention import df_RAnet_S
    # net = df_RAnet_S()
    # test(save_name, net, val_dataloader, save_name)



  
