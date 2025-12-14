import csv
import os
import copy
import torch
import argparse
import distutils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from utils import get_logger, test, train, save_val_result, seed_torch, show_time, save_test_result, save_val_result_regression, save_test_result_regression
from models.models import DR_crossatt_model
from Data.data import Load_Dataset
from omegaconf import OmegaConf
from collections import defaultdict
from sample import get_strategy
import time
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    torch.cuda.empty_cache()
    # Load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from_cfg', type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="Load from config?")
    parser.add_argument('--cfg_file', type=str, help="Experiment configuration file", default="config/config.yml")
    parser.add_argument('--al_runs', type=str, default="")
    parser.add_argument('--AL_START', type=str, default="cluster_Knn_uncertainty")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--run_times', type=str, default="0")

    args_cmd = parser.parse_args()

    if args_cmd.load_from_cfg:
        args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
        args_cmd = vars(args_cmd)
        for k in args_cmd.keys():
            if args_cmd[k] is not None: args_cfg[k] = args_cmd[k]
        args = OmegaConf.create(args_cfg)
    else: 
        args = args_cmd

    # logger
    # 设置是回归任务还是分类任务
    if args.predict_task == 'regression':
        model_task = '_regression'
    else:
        model_task = ''
    logger = get_logger(f'logs/{args.method}_{args.AL_START}{args.al_runs}{model_task}.log')
    logger.info(args)
    device = args.device
        
    if args.DATA.drug_feature_method == 'MACCS':
        source_model = DR_crossatt_model(args, gene_embed_size = args.DATA.variances_dim, drug_embed_size = 167,\
                                            attention_dim_size = args.MODEL.attention_dim_size).to(device)
    elif args.DATA.drug_feature_method == 'ECFP4': 
        source_model = DR_crossatt_model(args, gene_embed_size = args.DATA.variances_dim, drug_embed_size = 2048,\
                                            attention_dim_size = args.MODEL.attention_dim_size).to(device)
    elif args.DATA.drug_feature_method == 'unimol':
        source_model = DR_crossatt_model(args, gene_embed_size = args.DATA.variances_dim, drug_embed_size = 512,\
                                            attention_dim_size = args.MODEL.attention_dim_size).to(device)
    elif args.DATA.drug_feature_method == 'KPGT':
        source_model = DR_crossatt_model(args, gene_embed_size = args.DATA.variances_dim, drug_embed_size = 2304,\
                                            attention_dim_size = args.MODEL.attention_dim_size).to(device)
    elif args.DATA.drug_feature_method == 'Molformer':
        source_model = DR_crossatt_model(args, gene_embed_size = args.DATA.variances_dim, drug_embed_size = 768,\
                                            attention_dim_size = args.MODEL.attention_dim_size).to(device)
    elif args.DATA.drug_feature_method == 'GCN':
        source_model = DR_crossatt_model(args, gene_embed_size = args.DATA.variances_dim, drug_embed_size = 290,\
                                            attention_dim_size = args.MODEL.attention_dim_size).to(device)

    weight_p, bias_p = [], []
    for p in source_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in source_model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    logger.info('模型初始化参数:{}'.format(weight_p[0]))

    if args.AL.AL_TASK == False:
        source_file = f'{args.DATA.source_data_name}_source{args.al_runs}{model_task}.pth'
        source_path = os.path.join('checkpoints', 'source', source_file)

    else:
        # 先判断主动学习最后一轮模型是否存在，再判断主动学习前初始模型是否存在
        source_path = os.path.join('checkpoints', 'AL', f'{args.AL_START}_{args.al_runs}_{args.run_times}', f'{args.AL_START}_round{args.AL.num_rounds}{model_task}.pth')
        AL_start_path = os.path.join('checkpoints', 'source', f'{args.DATA.source_data_name}_AL_source{model_task}.pth')

    if os.path.exists(source_path): 
        logger.info('Loading source checkpoint: {}'.format(source_path))
        source_model.load_state_dict(torch.load(source_path, map_location=device), strict=True)
        best_source_model = source_model
        src_dset = Load_Dataset(args, logger=logger)
        _, val_loader, test_loader, _, _, _ = src_dset.get_loaders()

    else:
        if args.AL.AL_TASK == False:
            logger.info('不使用主动学习')
            print(f"start: {datetime.now()}")
            start_time = time.time()
            # 加载数据集
            if args.DATA.split_task=='AL_cold_split_20_20_START':
                src_dset = Load_Dataset(args, logger=logger, AL_start_part=True)
                train_loader, val_loader, test_loader,train_idx, _, AL_start_loader, _ = src_dset.get_loaders()
            else:
                src_dset = Load_Dataset(args, logger=logger)
                train_loader, val_loader, test_loader,train_idx, _, _ = src_dset.get_loaders()
            end_time = time.time()
            show_time(start_time, end_time, task = '加载数据')


            best_epoch, best_val_auc, best_mse, best_source_model = 0, 0.0, 114514, None
            # 源模型优化器
            source_optimizer = optim.AdamW([{'params': weight_p, 'weight_decay': args.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=args.Learning_rate)
            # 使用学习率衰减
            # 验证损失patience轮没改善时(最小下降1e-4)将学习率调整为原来的factor倍
            source_scheduler = ReduceLROnPlateau(source_optimizer, mode='min', factor=0.8, patience=4, verbose=True,\
                                threshold=1e-5, threshold_mode='rel', cooldown=0, min_lr=5e-8)
            # source_scheduler = optim.lr_scheduler.CyclicLR(source_optimizer, base_lr=args.Learning_rate, max_lr=args.Learning_rate*5, cycle_momentum=False,
            #                                         step_size_up=len(train_idx) // args.batch_size)
            epoch_len = len(str(args.Epoch))
            val_result_savepath = f'result/{args.method}_{args.al_runs}{model_task}'

            if not os.path.exists(val_result_savepath):
                os.makedirs(val_result_savepath)

            with open(val_result_savepath + '/val_results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                if args.predict_task == 'Classification':
                    writer.writerow(["epoch", "train_loss","HSIC_loss", "val_loss",  "valid_AUC", "valid_PRC", "valid_Accuracy", "valid_Precision", "valid_Sensitivity", "valid_Specificity", "valid_F1", "valid_Recall"])
                else:
                    writer.writerow(["epoch", "train_loss","HSIC_loss", "val_loss",  "valid_mse", "valid_rmse", "valid_mae", "valid_r2", "valid_pearsonr_rho", "valid_pearsonr_pval", "valid_spearmanr_rho", "valid_spearmanr_pval"])
            for epoch in range(1, args.Epoch + 1):
                train_loss_a_epoch, HSIC_loss_a_epoch = train(args, source_model, device, train_loader, val_loader, source_optimizer, epoch, logger, scheduler = source_scheduler)
                if args.predict_task == 'Classification':
                    valid_Y, valid_P, valid_loss_a_epoch, valid_Accuracy, valid_Precision, valid_Recall, valid_AUC, valid_PRC, valid_Sensitivity, valid_Specificity , valid_F1 = test(args, source_model, device, val_loader, split="val") 
                    save_val_result(args, val_result_savepath, epoch_len, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_AUC, valid_PRC, valid_Accuracy, valid_Precision, valid_Sensitivity, valid_Specificity, valid_F1, valid_Recall)
                    if (valid_AUC > best_val_auc):
                        best_epoch = epoch
                        best_val_auc = valid_AUC
                        best_source_model = source_model
                        torch.save(best_source_model.state_dict(), source_path)
                else:
                    valid_Y, valid_P, valid_loss_a_epoch, valid_mse, valid_rmse, valid_mae, valid_r2, valid_pearsonr_rho, valid_pearsonr_pval, valid_spearmanr_rho, valid_spearmanr_pval = test(args, source_model, device, val_loader, split="val") 
                    save_val_result_regression(args, val_result_savepath, epoch_len, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_mse, valid_rmse, valid_mae, valid_r2, valid_pearsonr_rho, valid_pearsonr_pval, valid_spearmanr_rho, valid_spearmanr_pval)
                    if (valid_mse < best_mse):
                        best_epoch = epoch
                        best_mse = valid_mse
                        best_source_model = source_model
                        torch.save(best_source_model.state_dict(), source_path)
                # source_scheduler.step(valid_loss_a_epoch)
                
            logger.info('保存的最佳模型轮数:{}'.format(best_epoch))

        elif args.AL.AL_TASK == True:

            logger.info(f'使用{args.AL_START}主动学习策略，总预算{args.AL.total_budget}，轮数{args.AL.num_rounds}')
            print(f"start: {datetime.now()}")
            start_time = time.time()

            if args.DATA.split_task=='AL_cold_split_20_20_START':
                src_dset = Load_Dataset(args, logger=logger, AL_start_part=True)
                train_loader, val_loader, test_loader, train_idx, AL_dataset, AL_start_loader, AL_start_dataset = src_dset.get_loaders()
            else:
                src_dset = Load_Dataset(args, logger=logger)
                train_loader, val_loader, test_loader, train_idx, AL_dataset, _ = src_dset.get_loaders()
            end_time = time.time()
            show_time(start_time, end_time, task = '加载数据')

            #保存结果用
            AL_selected_samples_path = f'checkpoints/AL/{args.AL_START}_{ args.al_runs}_{args.run_times}{model_task}/'
            if not os.path.exists(AL_selected_samples_path):
                os.makedirs(AL_selected_samples_path)
            #将主动学习挑选的样本保存到csv
            with open(AL_selected_samples_path+'AL_selected_samples.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['SMILES','COSMIC_ID','Classification','ID'])
            #主动学习结果记录
            with open(AL_selected_samples_path + 'AL_results.csv', mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["AL_ROUND", "AUC", "PRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Recall"])

            val_result_savepath = f'result/{args.method}_{args.AL_START}{args.al_runs}{model_task}'
            if not os.path.exists(val_result_savepath):
                os.makedirs(val_result_savepath)
            with open(val_result_savepath + '/val_results.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["epoch", "train_loss","HSIC_loss", "val_loss",  "valid_AUC", "valid_PRC", "valid_Accuracy", "valid_Precision", "valid_Sensitivity", "valid_Specificity", "valid_F1", "valid_Recall"])
            
            # 读取初始化的模型
            best_source_model = source_model
            torch.save(best_source_model.state_dict(), AL_start_path)

            #每轮增加采样量
            sampling_ratio = [(args.AL.total_budget / args.AL.num_rounds) * n for n in range(args.AL.num_rounds + 1)]
            tqdm_rat = trange(len(sampling_ratio[:]))

            # 每次运行开始从最佳模型状态重新开始，确保独立性
            curr_model = copy.deepcopy(best_source_model)
            # curr_source_model = curr_model
            # logger.info('主动学习实验开始时候的模型参数：{}'.format(curr_model.state_dict()['protein_embed.weight']))

            idxs_lb = np.zeros(len(train_idx), dtype=bool)  # 表示训练集中的哪些样本已经被标记（1）或未被标记（0），初始所有样本视为0

            # 初始化不确定度级别
            certainty_set_dtype = [('Positive_certainty_score', float), ('Uncertainty_score', float), ('Negative_certainty_score', float)]
            certainty_set = np.zeros(len(train_idx), dtype=certainty_set_dtype)
            certainty_set['Positive_certainty_score'] = 0
            certainty_set['Uncertainty_score'] = 0
            certainty_set['Negative_certainty_score'] = 0

            # 初始化主动学习策略
            sampling_strategy = get_strategy(args.AL_START, AL_dataset, train_idx, curr_model, device, args, certainty_set)


            for ix in tqdm_rat: # 主动学习迭代的轮次，采样数量循环
                with open(val_result_savepath + '/val_results.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([])     # 插入空行
                    writer.writerow([f"round{ix}"])
                    writer.writerow(["epoch", "train_loss","HSIC_loss", "val_loss",  "valid_AUC", "valid_PRC", "valid_Accuracy", "valid_Precision", "valid_Sensitivity", "valid_Specificity", "valid_F1", "valid_Recall"])
                ratio = sampling_ratio[ix]
                tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
                tqdm_rat.refresh()

                if ix == 0 and not args.AL.Cold_start:

                    if args.DATA.split_task=='split_data':
                        logger.info(f'使用AL_START作为启动数据(不包含训练集中的药和细胞系)')
                        logger.info(f'主动学习启动样本数{len(AL_start_dataset)}')
                        AL_start_idx = list(range(len(AL_start_dataset)))
                        # 将启动数据置为已标记
                        idxs_lb[AL_start_idx] = True
                        sampling_strategy.update(idxs_lb)           # 先把初始的启动数据更新了
                        # 将主动学习挑选的样本保存到csv
                        for idx in AL_start_idx:
                            with open(AL_selected_samples_path+'AL_selected_samples.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([AL_dataset[idx]['smiles'], AL_dataset[idx]['cell_line_id'], int(AL_dataset[idx]['label']), idx])
                        best_model, last_epoch_model = sampling_strategy.train(AL_dataset, val_loader, logger, AL_round=(ix))

                    else:
                        logger.info(f'第一轮随机挑选数据作为启动数据')
                        logger.info('Selecting instances...')
                        new_select_idxs = np.random.choice(np.where(idxs_lb==0)[0], args.AL.AL_START_DATA, replace=False)
                        idxs_lb[new_select_idxs] = True
                        sampling_strategy.update(idxs_lb)           # 先把初始的启动数据更新了
                        logger.info(f'主动学习前启动样本数{np.sum(idxs_lb)}')
                        # 将主动学习挑选的样本保存到csv
                        for idx in new_select_idxs:
                            with open(AL_selected_samples_path+'AL_selected_samples.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([AL_dataset[idx]['smiles'], AL_dataset[idx]['cell_line_id'], int(AL_dataset[idx]['label']), idx])
                        best_model, last_epoch_model = sampling_strategy.train(AL_dataset, val_loader, logger, AL_round=(ix))

 
                else:
                    logger.info(f'第{ix}轮主动学习前已标记样本数{np.sum(idxs_lb)}')
                    logger.info(f'开始第{ix}轮主动学习')
                    logger.info('Selecting instances...')
                    if ix > args.AL.certainty_confidence:
                        Pseudo_task = args.AL.Pseudo_task
                    else:
                        Pseudo_task = False

                    if args.AL_START == 'SUAD':
                        if Pseudo_task:
                            new_select_idxs, Pseudo_idx= sampling_strategy.query(int(sampling_ratio[1]), logger, Pseudo_task = True) # 选取样本
                        else:
                            new_select_idxs = sampling_strategy.query(int(sampling_ratio[1]), logger, Pseudo_task = False) # 选取样本

                    else:
                        new_select_idxs = sampling_strategy.query(int(sampling_ratio[1]), logger, ix) # 选取样本

                    # 将主动学习挑选的样本保存到csv
                    for idx in new_select_idxs:
                        with open(AL_selected_samples_path+'AL_selected_samples.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([AL_dataset[idx]['smiles'], AL_dataset[idx]['cell_line_id'], int(AL_dataset[idx]['label']), idx])
                    idxs_lb[new_select_idxs] = True
                    sampling_strategy.update(idxs_lb)
                    logger.info(f'第{ix}轮主动学习后已标记样本数{np.sum(idxs_lb)}')

                    if Pseudo_task:
                        best_model, last_epoch_model = sampling_strategy.train(AL_dataset, val_loader, logger, AL_round=(ix), Pseudo_task=True, Pseudo_idx = Pseudo_idx)
                    else:
                        best_model, last_epoch_model = sampling_strategy.train(AL_dataset, val_loader, logger, AL_round=(ix))

                if args.predict_task == 'Classification':
                    _, _, _, AL_test_Accuracy, AL_test_Precision, AL_test_Recall, AL_test_AUC, AL_test_PRC, AL_test_Sensitivity, AL_test_Specificity, AL_test_F1 = test(args, best_model, device, test_loader, split="test")
                    #主动学习结果记录
                    with open(AL_selected_samples_path + 'AL_results.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([ix, f"{AL_test_AUC:.5f}", f"{AL_test_PRC:.5f}", f"{AL_test_Accuracy:.5f}", f"{AL_test_Precision:.5f}", f"{AL_test_Sensitivity:.5f}", f"{AL_test_Specificity:.5f}", f"{AL_test_F1:.5f}", f"{AL_test_Recall:.5f}"])
                else:
                    _, _, _, AL_test_mse, AL_test_rmse, AL_test_mae, AL_test_r2, AL_test_pearsonr_rho, AL_test_pearsonr_pval, AL_test_spearmanr_rho, AL_test_spearmanr_pval = test(args, best_model, device, test_loader, split="test")
                    #主动学习结果记录
                    with open(AL_selected_samples_path + 'AL_results.csv', mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([ix, f"{AL_test_mse:.5f}", f"{AL_test_rmse:.5f}", f"{AL_test_mae:.5f}", f"{AL_test_r2:.5f}", f"{AL_test_pearsonr_rho:.5f}", f"{AL_test_pearsonr_pval:.5f}", f"{AL_test_spearmanr_rho:.5f}", f"{AL_test_spearmanr_pval:.5f}"])
               
    if args.AL.AL_TASK == True:
        test_result_savepath = f'result/{args.method}_{args.AL_START}{args.al_runs}{model_task}'
        best_source_model.load_state_dict(torch.load(source_path, map_location='cpu'), strict=True)
    else:
        test_result_savepath = f'result/{args.method}_{args.al_runs}{model_task}'
        best_source_model.load_state_dict(torch.load(source_path, map_location='cpu'), strict=True)

    if not os.path.exists(test_result_savepath):
        os.makedirs(test_result_savepath)

    if args.predict_task == 'Classification': 
        test_Y, test_P, _, test_Accuracy, test_Precision, test_Recall, test_AUC, test_PRC, test_Sensitivity, test_Specificity, test_F1 = test(args, best_source_model, device, test_loader, split="test")
        save_test_result(args, logger, test_result_savepath, test_AUC, test_PRC, test_Accuracy, test_Precision, test_Sensitivity, test_Specificity, test_F1, test_Recall)
    else:
        test_Y, test_P, _, test_mse, test_rmse, test_mae, test_r2, test_pearsonr_rho, test_pearsonr_pval, test_spearmanr_rho, test_spearmanr_pval = test(args, best_source_model, device, test_loader, split="test")
        save_test_result_regression(args, logger, test_result_savepath, test_Y, test_P, _, test_mse, test_rmse, test_mae, test_r2, test_pearsonr_rho, test_pearsonr_pval, test_spearmanr_rho, test_spearmanr_pval)

if __name__ == '__main__':
    print(f"start: {datetime.now()}")
    start_time = time.time()
    main()
    end_time = time.time()
    show_time(start_time,end_time, task = '训练')
    print(f"end: {datetime.now()}")
