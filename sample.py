import csv
import math
import os
import copy
import random
import numpy as np
import hnswlib
import pandas as pd
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from scipy.sparse import lil_matrix
from utils import ActualSequentialSampler, test, train, save_val_result, kmeans_plus_plus_opt, row_norms,save_val_result_regression
from torch.optim.lr_scheduler import ReduceLROnPlateau



save_path = "tensorboard/跨域收敛情况" 
note = ''

al_dict = {}
def register_strategy(name):
	def decorator(cls):
		al_dict[name] = cls
		return cls
	return decorator

def get_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	return al_dict[sample](*args)

class SamplingStrategy:
	""" 
	Sampling Strategy wrapper class
	"""
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		self.dset = dset
		self.num_classes = 2
		self.train_idx = np.array(train_idx)
		self.model = model
		self.device = device
		self.args = args
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)
		self.batch_size = args.batch_size
		self.certainty_set = certainty_set

		if self.args.predict_task == 'regression':
			self.model_task = '_regression'
		else:
			self.model_task = ''

	def query(self, n, logger, round_now):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb

	def train(self, AL_dataset, val_loader, logger, AL_round=1, Pseudo_task = False, Pseudo_idx = None):
		"""
		Driver train method
		"""
		best_val_auc, best_mse, best_model = 0.0, 114514, None
		AL_model_save_path = f'checkpoints/AL/{self.args.AL_START}_{self.args.al_runs}_{self.args.run_times}{self.model_task}/'
		os.makedirs(AL_model_save_path, exist_ok=True) 
		AL_model_path = os.path.join(AL_model_save_path, f'{self.args.AL_START}_round{AL_round}.pth')
		val_result_savepath = f'result/{self.args.method}_{self.args.AL_START}{self.args.al_runs}{self.model_task}'
		if AL_round>0:
			if self.args.AL.retrain:
				AL_model_path_lastround = os.path.join('checkpoints', 'source', f'{self.args.DATA.source_data_name}_AL_source{self.model_task}.pth')  #从头训练
			else:
				AL_model_path_lastround = os.path.join(AL_model_save_path, f'{self.args.AL_START}_round{AL_round-1}.pth')
				
			self.model.load_state_dict(torch.load(AL_model_path_lastround, map_location=torch.device('cpu')), strict=True)

		if os.path.exists(AL_model_path):
			logger.info(f'第{AL_round}轮主动学习模型已存在,加载当前模型')
			with open(val_result_savepath + '/val_results.csv', mode='a', newline='') as file:
				writer = csv.writer(file)
				# 写入表头
				writer.writerow(['已存在当前轮数模型，莫得验证'])
			self.model.load_state_dict(torch.load(AL_model_path, map_location=torch.device('cpu')), strict=True)
			best_model = copy.deepcopy(self.model)
		else:
			if Pseudo_task :
				train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
				Pseudo_sampler = SubsetRandomSampler(self.train_idx[Pseudo_idx])
				logger.info(f'使用的已标记训练数据量{len(train_sampler)}')
				logger.info(f'伪标记训练数据量{len(Pseudo_sampler)}')
				sup_loader = DataLoader(AL_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=0, drop_last=False)
				Pseudo_loader = DataLoader(AL_dataset, batch_size=self.batch_size, sampler=Pseudo_sampler, num_workers=0, drop_last=False)

				AL_optimizer = optim.AdamW(self.model.parameters(), lr=self.args.Learning_rate)
				Pseudo_optimizer = optim.AdamW(self.model.parameters(), lr=self.args.Learning_rate)

				# # 学习率衰减策略
				# AL_scheduler = optim.lr_scheduler.CyclicLR(AL_optimizer, base_lr=self.args.Learning_rate, max_lr=self.args.Learning_rate*5, cycle_momentum=False,
				# 										step_size_up = len(sup_loader))
				# Pseudo_scheduler = optim.lr_scheduler.CyclicLR(Pseudo_optimizer, base_lr=self.args.Learning_rate, max_lr=self.args.Learning_rate*5, cycle_momentum=False,
				# 										step_size_up = len(Pseudo_loader))
				
				# 使用学习率衰减
				# 验证损失patience=5轮没改善时(下降1e-5)将学习率调整为原来的0.5倍（学习率最小为5e-6）
				AL_scheduler = ReduceLROnPlateau(AL_optimizer, mode='min', factor=0.8, patience=4, verbose=True,\
								threshold=1e-5, threshold_mode='rel', cooldown=0, min_lr=5e-8)
				
				Pseudo_scheduler = ReduceLROnPlateau(AL_optimizer, mode='min', factor=0.8, patience=4, verbose=True,\
								threshold=1e-5, threshold_mode='rel', cooldown=0, min_lr=5e-8)
				
				# 保存验证集结果路径
				for epoch in range(self.args.AL.AL_START_Epoch):
					train_loss_a_epoch, HSIC_loss_a_epoch = train(self.args, self.model, self.args.device, sup_loader, val_loader, AL_optimizer, epoch, logger, scheduler = AL_scheduler, \
												Pseudo_task = True, Pseudo_loader = Pseudo_loader, Pseudo_scheduler = Pseudo_scheduler) 
					# 返回分类结果
					if self.args.predict_task == 'Classification':
						_ ,_ , valid_loss_a_epoch, valid_Accuracy, valid_Precision, valid_Recall, valid_AUC, valid_PRC, valid_Sensitivity, valid_Specificity, valid_F1 = test(self.args, self.model, self.device, val_loader, split="val") 
						save_val_result(self.args, val_result_savepath, self.args.AL.AL_START_Epoch, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_AUC, valid_PRC, valid_Accuracy, valid_Precision, valid_Sensitivity, valid_Specificity, valid_F1, valid_Recall)
						if epoch > 5:
							if (valid_AUC > best_val_auc):
								best_val_auc = valid_AUC
								best_model = copy.deepcopy(self.model)
								torch.save(best_model.state_dict(), AL_model_path)
								early_stop = 0
							else:
								early_stop += 1
							# lr_scheduler.step()
							if early_stop >= self.args.AL.AL_early_stop_num :break

					# 返回回归结果
					else:
						_, _, valid_loss_a_epoch, valid_mse, valid_rmse, valid_mae, valid_r2, valid_pearsonr_rho, valid_pearsonr_pval, valid_spearmanr_rho, valid_spearmanr_pval = test(self.args, self.model, self.device, val_loader, split="val") 
						save_val_result_regression(self.args, val_result_savepath, self.args.AL.AL_START_Epoch, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_mse, valid_rmse, valid_mae, valid_r2, valid_pearsonr_rho, valid_pearsonr_pval, valid_spearmanr_rho, valid_spearmanr_pval)
						if epoch > 5:
							if (valid_mse < best_mse):
								best_mse = valid_mse
								best_model = copy.deepcopy(self.model)
								torch.save(best_model.state_dict(), AL_model_path)
								early_stop = 0
							else:
								early_stop += 1
							# lr_scheduler.step()
							if early_stop >= self.args.AL.AL_early_stop_num :break
					AL_scheduler.step(valid_loss_a_epoch)
					Pseudo_scheduler.step(valid_loss_a_epoch)
			else:

				train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
				logger.info(f'使用的已标记训练数据量{len(train_sampler)}')
				sup_loader = DataLoader(AL_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=0, drop_last=False)

				AL_optimizer = optim.AdamW(self.model.parameters(), lr=self.args.Learning_rate)
				# 使用学习率衰减
				# 验证损失patience=4轮没改善时(下降1e-5)将学习率调整为原来的0.8倍（学习率最小为5e-8）
				AL_scheduler = ReduceLROnPlateau(AL_optimizer, mode='min', factor=0.8, patience=4, verbose=True,\
								threshold=1e-5, threshold_mode='rel', cooldown=0, min_lr=5e-8)
				
				# AL_scheduler = optim.lr_scheduler.CyclicLR(AL_optimizer, base_lr=self.args.Learning_rate, max_lr=self.args.Learning_rate*5, cycle_momentum=False,
				# 										step_size_up = len(sup_loader))
				# 保存验证集结果路径
				for epoch in range(self.args.AL.AL_START_Epoch):
					train_loss_a_epoch, HSIC_loss_a_epoch = train(self.args, self.model, self.args.device, sup_loader, val_loader, AL_optimizer, epoch, logger, scheduler = AL_scheduler) 
					
					# 返回分类结果
					if self.args.predict_task == 'Classification':
						_ ,_ , valid_loss_a_epoch, valid_Accuracy, valid_Precision, valid_Recall, valid_AUC, valid_PRC, valid_Sensitivity, valid_Specificity, valid_F1 = test(self.args, self.model, self.device, val_loader, split="val") 
						save_val_result(self.args, val_result_savepath, self.args.AL.AL_START_Epoch, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_AUC, valid_PRC, valid_Accuracy, valid_Precision, valid_Sensitivity, valid_Specificity, valid_F1, valid_Recall)
						if epoch > 5:
							if (valid_AUC > best_val_auc):
								best_val_auc = valid_AUC
								best_model = copy.deepcopy(self.model)
								torch.save(best_model.state_dict(), AL_model_path)
								early_stop = 0
							else:
								early_stop += 1
							# lr_scheduler.step()
							if early_stop >= self.args.AL.AL_early_stop_num :break
											# 返回回归结果
					else:
						_, _, valid_loss_a_epoch, valid_mse, valid_rmse, valid_mae, valid_r2, valid_pearsonr_rho, valid_pearsonr_pval, valid_spearmanr_rho, valid_spearmanr_pval = test(self.args, self.model, self.device, val_loader, split="val") 
						save_val_result_regression(self.args, val_result_savepath, self.args.AL.AL_START_Epoch, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_mse, valid_rmse, valid_mae, valid_r2, valid_pearsonr_rho, valid_pearsonr_pval, valid_spearmanr_rho, valid_spearmanr_pval)
						if epoch > 5:
							if (valid_mse < best_mse):
								best_mse = valid_mse
								best_model = copy.deepcopy(self.model)
								torch.save(best_model.state_dict(), AL_model_path)
								early_stop = 0 
							else:
								early_stop += 1
							# lr_scheduler.step()
							if early_stop >= self.args.AL.AL_early_stop_num :break
					AL_scheduler.step(valid_loss_a_epoch)
					current_lr = AL_scheduler.optimizer.param_groups[0]['lr']
					logger.info(f'当前学习率为{current_lr}')

		return best_model, self.model


@register_strategy('uniform')#随机采样策略
class RandomSampling1(SamplingStrategy):
	# np.random.seed(42)#固定随机方法挑选的样本

	"""
	Uniform sampling 
	"""
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(RandomSampling1, self).__init__(dset, train_idx, model, device, args, certainty_set,)

	def query(self, n, logger, round_now):
		# print("查询时的模型参数：{}".format(self.model.state_dict()['fc2.weight']))
		logger.info(f'返回随机选择样本{n}个')
		return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)
		

@register_strategy('margin')
class MarginSampling1(SamplingStrategy):
	"""
	Margin sampling 
	"""
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(MarginSampling1, self).__init__(dset, train_idx, model, device, args, certainty_set)
		self.dset = dset

	def query(self, n, logger, round_now):
		self.model.eval()
		print("查询时的模型参数：{}".format(self.model.state_dict()['fc2.weight']))
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=32, drop_last=False)
		
		all_log_probs, all_probs = [], []
		with torch.no_grad():
			for batch_idx, data in enumerate(tqdm(data_loader)):
				'''data preparation '''
				# if batch_idx > 500 :			#调试用
				# 	break

				labels = data['label']
				labels = labels.to(self.device)
				_, predicted_scores,  _ , _  = self.model(self.args, data, self.device)

				probs = F.softmax(predicted_scores, 1).to('cpu')
				log_probs = torch.log(probs)
				all_probs.append(probs)
				# all_log_probs.append(log_probs)

		all_probs = torch.cat(all_probs)	
		probs_sorted, idxs = all_probs.sort(descending=True)	#按预测概率降序排序
		uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]  

		return idxs_unlabeled[uncertainties.sort()[1][:n]]


# faiss实现KMeans
# faiss实现KMeans
# faiss实现KMeans
@register_strategy('KMeans')
class KMeansSampling1(SamplingStrategy):
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(KMeansSampling1, self).__init__(dset, train_idx, model, device, args, certainty_set)

	def query(self, n, logger, round_now):
		print("查询时的模型参数：{}".format(self.model.state_dict()['out.weight']))
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=32, drop_last=False, shuffle=False)
		self.model.eval()

		# Get embeddings
		embeddings = []
		with torch.no_grad():
			for batch_idx, data in tqdm(enumerate(data_loader)):
				# data = data.to(self.device)
				labels = data['label']
				labels = labels.to(self.device)
				features, predicted_scores, _ , _ = self.model(self.args, data, self.device)
				embeddings.append(features)

		embeddings = torch.cat(embeddings)
		embeddings = embeddings.cpu().numpy().astype('float32')

		# 使用 faiss 实现 KMeans
		kmeans = faiss.Kmeans(d=embeddings.shape[1], k=n, niter=20, verbose=False)
		kmeans.train(embeddings)
		_, cluster_idxs = kmeans.index.search(embeddings, 1)
		cluster_idxs = cluster_idxs.flatten()

		# 为每个簇选择一个距离簇中心最近的样本
		q_idxs = []
		for i in range(n):
			cluster_samples = np.where(cluster_idxs == i)[0]
			cluster_centers = kmeans.centroids[i]					# 计算簇内样本到簇中心的距离
			distances = np.linalg.norm(embeddings[cluster_samples] - cluster_centers, axis=1)
			q_idxs.append(cluster_samples[np.argmin(distances)])	# 选择距离最小的样本

		q_idxs = np.array(q_idxs)

		return idxs_unlabeled[q_idxs]



@register_strategy('Coreset')
class Coreset(SamplingStrategy):
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(Coreset, self).__init__(dset, train_idx, model, device, args, certainty_set)

	def query(self, n, logger, round_now):
		if np.count_nonzero(self.idxs_lb) < 1:  # 初始化没标注，将就 uniform 启动一下
			# np.random.seed(42)  # 固定随机方法挑选的样本
			print("查询时的模型参数：{}".format(self.model.state_dict()['fc2.weight']))
			return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)
		else:
			idxs_labeled = self.idxs_lb.copy()
			train_sampler = ActualSequentialSampler(self.train_idx[np.arange(len(self.train_idx))])  # 要同时获取已标记和未标记的数据信息
			data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=32, drop_last=False)
			self.model.eval()

			# Get embeddings
			embeddings = []
			with torch.no_grad():
				for batch_idx, data in tqdm(enumerate(data_loader)):
					features, _, _, _ = self.model(self.args, data, self.device)
					embeddings.append(features.cpu().numpy())
			embeddings = np.vstack(embeddings).astype(np.float32)  # 转换为 float32，faiss 要求
			# 从未标记样本中选择与已标记样本集合最远的样本
			selected_indices = []
			for _ in tqdm(range(n), ncols=100):

				# 初始化faiss索引
				d = embeddings.shape[1]  # 嵌入向量的维度
				index = faiss.IndexFlatL2(d)  # 使用 L2 距离（欧氏距离）
				# 将未标注样本添加到索引（在这些数据中找索引）
				index.add(embeddings[~idxs_labeled])
				# 计算到已标注数据集的距离
				distances, id = index.search(embeddings[idxs_labeled], k=1)
				farthest_idx = np.argmax(distances)  # 找到最远的样本索引
				farthest_sample_idx = id[farthest_idx]	# 对应到原始的索引
				selected_indices.append(farthest_sample_idx)
				# 更新已标记数据池
				idxs_labeled[farthest_sample_idx] = True
			selected_indices = [index.item() for index in selected_indices]
			return selected_indices
			

@register_strategy('entropy')
class EntropySampling1(SamplingStrategy):
	"""
	Entropy sampling 
	"""
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(EntropySampling1, self).__init__(dset, train_idx, model, device, args, certainty_set, balanced=False)
		self.dset = dset
		self.device = device

	def query(self, n, logger, round_now):
		self.model.eval()
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]#未标记样本索引
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=32, drop_last=False)
		
		all_log_probs, all_probs = [], []
		with torch.no_grad():
			for batch_idx, data in enumerate(tqdm(data_loader)):
				
				# if batch_idx > 500 :			#调试用
				# 	break
				# data = data.to(self.device)
				labels = data['label']
				labels = labels.to(self.device)
				features, predicted_scores, _ , _ = self.model(self.args, data, self.device)

				probs = F.softmax(predicted_scores, 1).to('cpu')
				log_probs = torch.log(probs)
				all_probs.append(probs)
				all_log_probs.append(log_probs)

		all_probs = torch.cat(all_probs) # 多个batch拼接
		all_log_probs = torch.cat(all_log_probs)
		uncertainties = (all_probs*all_log_probs).sum(1)  # 每个样本的熵，熵值越高其预测越不确定 p*logp
		return idxs_unlabeled[uncertainties.sort()[1][:n]] # sort升序，这里是最不确定的样本（商最大，负）在前
	

	
@register_strategy('BADGE')
class BADGESampling(SamplingStrategy):
	"""
	Implements BADGE: Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671)
	"""
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(BADGESampling, self).__init__(dset, train_idx, model, device, args, certainty_set, balanced=False)

	def query(self, n, logger, round_now):
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=self.args.batch_size, drop_last=False)
		self.model.eval()
		emb_dim = 256 	#emb_dim为feature的特征维度
		tgt_emb = torch.zeros([len(data_loader.sampler), self.num_classes])	#
		tgt_pen_emb = torch.zeros([len(data_loader.sampler), emb_dim])  #emb_dim为feature的特征维度  用于kmeans聚类
		tgt_lab = torch.zeros(len(data_loader.sampler))
		tgt_preds = torch.zeros(len(data_loader.sampler))	
		batch_sz = self.args.batch_size

		with torch.no_grad():
			for batch_idx, data in tqdm(enumerate(data_loader)):
				labels = data['label']
				labels = labels.to(self.device)
				labels = labels.squeeze()  # 将labels从[32, 1]变为[32]
				features, predicted_scores, _ , _ = self.model(self.args, data, self.device) #e2,e1
				#按batch存入相应的张量中
				tgt_pen_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, features.shape[0]), :] = features.cpu()
				tgt_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, predicted_scores.shape[0]), :] = predicted_scores.cpu()
				tgt_lab[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, labels.shape[0])] = labels
				tgt_preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, predicted_scores.shape[0])] = predicted_scores.argmax(dim=1, keepdim=True).squeeze()#每个样本最有可能的预测索引
		# Compute uncertainty gradient
		tgt_scores = nn.Softmax(dim=1)(tgt_emb)  #预测得分转换为概率分布
		tgt_scores_delta = torch.zeros_like(tgt_scores) 
		tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1   #.long()确保标签整形  对每个样本的预测类别
		# Uncertainty embedding
		badge_uncertainty = (tgt_scores-tgt_scores_delta)  #不确定性
		# Seed with maximum uncertainty example
		max_norm = row_norms(badge_uncertainty.cpu().numpy()).argmax() #计算不确定性向量每行的L2范数（各向量平方和的开放），找到范数最大的样本作为初始种子（即不确定性最高的样本）
		_, q_idxs = kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), n, init=[max_norm])  #使用初始种子聚类后选出最具代表性的样本索引
		return idxs_unlabeled[q_idxs]


from sklearn.neighbors import NearestNeighbors
from scipy.special import rel_entr
@register_strategy('contrastive_active_learning')
class ContrastiveActiveLearning(SamplingStrategy):
	"""
	Contrastive Active Learning selects instances whose k-nearest neighbours
	exhibit the largest mean Kullback-Leibler divergence.
	"""
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(ContrastiveActiveLearning, self).__init__(dset, train_idx, model, device, args, certainty_set, balanced=False)
		self.dset = dset
		self.device = device
		self.k = 500  # Number of nearest neighbors

	def query(self, n, logger, round_now):
		self.model.eval()
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  # 未标记样本索引
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=32, drop_last=False)

		all_embeddings, all_probs = [], []
		with torch.no_grad():
			for batch_idx, data in enumerate(tqdm(data_loader)):
				features, predicted_scores, _, _ = self.model(self.args, data, self.device)
				probs = F.softmax(predicted_scores, dim=1).cpu()
				all_probs.append(probs)
				all_embeddings.append(features.cpu())

		all_embeddings = torch.cat(all_embeddings).numpy()
		all_probs = torch.cat(all_probs).numpy()

		# 计算每个样本的 k-最近邻的平均 KL 散度
		nn_model = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
		nn_model.fit(all_embeddings)
		distances, indices = nn_model.kneighbors(all_embeddings)

		kl_divergences = []
		for i in range(len(all_embeddings)):
			neighbors_probs = all_probs[indices[i]]
			avg_kl_div = np.mean([np.sum(rel_entr(neighbors_probs[j], all_probs[i])) for j in range(self.k)])
			kl_divergences.append(avg_kl_div)

		kl_divergences = np.array(kl_divergences)
		selected_indices = idxs_unlabeled[np.argsort(-kl_divergences)[:n]]  # 选择 KL 散度最大的样本

		return selected_indices


import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_distances	
from sklearn.cluster import Birch
import faiss
@register_strategy('cluster_Knn_uncertainty')
class Birch_Knn_Uncertainty_Sampling(SamplingStrategy):
	def __init__(self, dset, train_idx, model, device, args, certainty_set, balanced=False):
		super(Birch_Knn_Uncertainty_Sampling, self).__init__(dset, train_idx, model, device, args, certainty_set, balanced=False)
		self.args = args
		self.dset = dset
		self.device = device
		self.Lambda = self.args.AL.Lambda		# 调节子池大小, 仅初始化

	def faiss_KMeans_cluster(self, logger, embeddings, n, round_now):
		embeddings = embeddings.numpy()
		total_round = self.args.AL.num_rounds

		balance = (total_round - round_now) / total_round 
		self.clusters = max(int(n * balance), int(n // total_round))
		logger.info(f"第{round_now}轮聚类数量{self.clusters} 个")
		# 使用 faiss 实现 KMeans
		kmeans = faiss.Kmeans(d=embeddings.shape[1], k = self.clusters , niter=20, verbose=False)
		kmeans.train(embeddings)
		_, cluster_idxs = kmeans.index.search(embeddings, 1)
		cluster_idxs = cluster_idxs.flatten()

		# 获取每个簇的样本数量
		self.cluster_sample_counts = np.bincount(cluster_idxs, minlength=kmeans.k)

		# 获取每个簇的平均特征
		cluster_centers = []
		for i in range(kmeans.k):
			cluster_samples = np.where(cluster_idxs == i)[0]
			# 计算簇内样本的平均特征
			cluster_center = embeddings[cluster_samples].mean(axis=0)
			cluster_centers.append(cluster_center)

		cluster_centers = np.array(cluster_centers)
		logger.info(f"返回 KMeans 获得的聚类簇 {len(cluster_centers)} 个")

		return cluster_centers, len(cluster_centers)

	def compute_dynamic_k(self):
		"""根据簇样本数量动态计算k值"""
		counts = self.cluster_sample_counts.astype(np.float32)
		# 使用平均簇数作为最大k近邻
		k_max = int(np.mean(counts)) // self.Lambda
		k_min = k_max // self.Lambda + 1

		# 归一化（防止全零）
		min_count = np.min(counts) if len(counts) > 0 else 0
		max_count = np.max(counts) if len(counts) > 0 else 1

		normalized = (counts - min_count) / (max_count - min_count)
		
		# 非线性映射（三次函数增小簇权重）
		dynamic_ks = k_min + (normalized ** 0.5) * (k_max - k_min)
		return np.clip(dynamic_ks.astype(int), k_min, k_max)

	def extract_feature(self, dset, idxs, batch_size=32):
		sampler = ActualSequentialSampler(idxs)
		data_loader = DataLoader(dset, sampler=sampler, num_workers=0, batch_size=batch_size, drop_last=False)
		labels, embeddings, probs = [], [], []
		with torch.no_grad():
			for batch_idx, data in enumerate(tqdm(data_loader)):
				label = data['label']
				features, predicted_scores, _, _ = self.model(self.args, data, self.device)
				prob = F.softmax(predicted_scores, 1).to('cpu')
				embeddings.append(features.cpu())
				labels.append(label.cpu())
				probs.append(prob)
		return torch.cat(embeddings), torch.cat(labels), torch.cat(probs)

	def select_candidate_samples(self, embeddings_unlabeled, combined_cluster, idxs_unlabeled, dynamic_ks):
		d = embeddings_unlabeled.shape[1]  # 嵌入维度
		index = faiss.IndexFlatL2(d)  							# L2距离
		index.add(embeddings_unlabeled.numpy())  				# 添加未标记样本嵌入
		# D表示距离，I表示每个样本最近k个邻居的索引

		# 按最大k值统一搜索
		k_max = int(np.max(dynamic_ks))
		D, I = index.search(combined_cluster, k_max)			# 搜索最近邻，返回的是未标记样本的索引
		candidate_idxs = []
		for i in range(len(combined_cluster)):
			k = dynamic_ks[i]
			# 取前k个近邻
			valid_indices = I[i, :k]
			candidate_idxs.extend(idxs_unlabeled[valid_indices])

		return D, I, np.unique(candidate_idxs)
	
	def query(self, n, logger, round_now):
		self.model.eval()
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  # 未标记样本索引

		# 提取未标记样本和已标记样本的嵌入
		all_embeddings, labels, probs = self.extract_feature(self.dset, self.train_idx)
		embeddings_unlabeled = all_embeddings[~self.idxs_lb]
		probs_unlabeled = probs[~self.idxs_lb]
		embeddings_labeled = all_embeddings[self.idxs_lb]

		# labeled_embedding_cluster, labeled_num_cluster = self.faiss_KMeans_cluster(logger, embeddings_labeled, n)
		# 聚类并获取动态k值
		unlabeled_embedding_cluster, unlabeled_num_cluster = self.faiss_KMeans_cluster(logger, embeddings_unlabeled, n, round_now)
		dynamic_ks = self.compute_dynamic_k()

		# D表示距离，I表示每个样本最近k个邻居的索引(未标记样本索引)
		D, I, candidate_idxs = self.select_candidate_samples(embeddings_unlabeled, unlabeled_embedding_cluster, idxs_unlabeled, dynamic_ks)
		logger.info(f"动态k值分布 | 最小:{dynamic_ks.min()} 平均:{dynamic_ks.mean():.1f} 最大:{dynamic_ks.max()}")
		logger.info(f'得到候选子集{len(candidate_idxs)}个, 聚类得到所有数据簇数{unlabeled_num_cluster}')
		
		self.uncertain_sets = []  # 创建一个列表来保存所有的候选子集
		self.cluster_center_samples = []  # 创建一个列表来保存聚类中心样本
		self.uncertain_samples = []  # 创建一个列表来保存最终挑选的不确定样本

		final_selected_id = np.array([])
		for i in range(len(I)):
			neighbors_id = I[i]
			# path = f'{self.args.al_runs}_select_samples'
			uncertain_set = np.atleast_1d(idxs_unlabeled[neighbors_id])		#每个簇的子集范围
			self.uncertain_sets.append(uncertain_set)	

			selected_id_cluster = np.atleast_1d(idxs_unlabeled[neighbors_id[0]])	#选择与簇中心最近的样本
			final_selected_id = np.concatenate((final_selected_id, selected_id_cluster), axis=0)
			self.cluster_center_samples.extend(selected_id_cluster)

		final_selected_id = np.unique(final_selected_id).astype(int)

		# 直接返回 candidate_idxs 中不在 final_selected_id 中的元素
		candidate_idxs = np.setdiff1d(candidate_idxs, final_selected_id)
		remain_num = n - len(final_selected_id)
		logger.info(f'已得到簇中心最近样本{len(candidate_idxs)}， 剩余不确定需要挑选数{remain_num}')
		# 计算候选样本的熵
		candidate_sampler = ActualSequentialSampler(candidate_idxs)
		candidate_loader = DataLoader(self.dset, sampler=candidate_sampler, num_workers=0, batch_size=32, drop_last=False)
		all_log_probs, all_probs = [], []
		with torch.no_grad():
			for batch_idx, data in enumerate(tqdm(candidate_loader)):
				labels = data['label']
				labels = labels.to(self.device)
				_, predicted_scores,  _ , _  = self.model(self.args, data, self.device)
				probs = F.softmax(predicted_scores, 1).to('cpu')
				log_probs = torch.log(probs)
				all_probs.append(probs)
				# all_log_probs.append(log_probs)
		all_probs = torch.cat(all_probs)	
		probs_sorted, idxs = all_probs.sort(descending=True)	#按预测概率降序排序
		uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
		# 不确定度最高的,概率差值越小不确定度越大（下面是升序排序）
		# atleast_1保证在一个样本时也能挑
		selected_id_uncertain1 = np.atleast_1d(candidate_idxs[uncertainties.sort()[1][:remain_num]])
		final_selected_id = np.concatenate((final_selected_id, selected_id_uncertain1), axis=0)
		self.uncertain_samples.extend(selected_id_uncertain1)

		selected_samples_save_path = f'checkpoints/AL/{self.args.AL_START}_{self.args.al_runs}_{self.args.run_times}{self.model_task}/selected_samples/{round_now}/'	
		if not os.path.exists(selected_samples_save_path):
			os.makedirs(selected_samples_save_path)

		np.save(f"{selected_samples_save_path}/uncertain_sets.npy", self.uncertain_sets)
		np.save(f"{selected_samples_save_path}/cluster_center_samples.npy", self.cluster_center_samples)
		np.save(f"{selected_samples_save_path}/uncertain_samples.npy", self.uncertain_samples)

		return final_selected_id
