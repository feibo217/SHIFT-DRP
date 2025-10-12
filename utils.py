import csv
import os
import random
import torch
import logging
import numpy as np

from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
from HSIC import HSIC_loss
from torch.utils.data.sampler import Sampler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
# 日志函数
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def train(args, model, device, train_loader, val_loader, optimizer, epoch, logger, scheduler = None, Pseudo_task = False, Pseudo_loader = None, Pseudo_scheduler = None):
    train_losses_a_epoch = 0.0
    HSIC_loss_a_epoch = 0.0
    
    model.train()
    if Pseudo_task:
        logger.info(f'更新自训练损失')
        for train_i, train_data in enumerate(tqdm(train_loader)):
            label = train_data['label'].squeeze()
            smiles = train_data['smiles']
            cell_line_id = train_data['cell_line_id']
            # 有注意力
            if args.MCAN.IFATT:
                _, predicted_interaction, crossatt_gene, crossatt_drug = model(args, train_data, device)
                batch_HSIC_loss = HSIC_loss(crossatt_gene, crossatt_drug, s_x=args.MODEL.Kernel_bandwidth, s_y=args.MODEL.Kernel_bandwidth) * args.MODEL.HSIC_loss_weight   # s_x, s_y为带宽
                # print(f'batch_HSIC_loss{batch_HSIC_loss}')
                if args.predict_task == 'Classification':
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    # print(f'计算分类损失')
                    if len(label)==1:
                        criterion = nn.CrossEntropyLoss(reduction='sum')
                        CE_loss = criterion(predicted_interaction, label.to(device))
                        
                    else:
                        CE_loss = nn.CrossEntropyLoss()(predicted_interaction, label.to(device))

                    train_loss = CE_loss  +  batch_HSIC_loss.to(device)
                else:
                    # print(f'计算回归损失')
                    predicted_values = predicted_interaction[:, 1].double()   # 提取预测值的第一列
                    MES_loss = nn.MSELoss()(predicted_values, label.to(device))
                    train_loss = MES_loss  +  batch_HSIC_loss.to(device)
                
                train_losses_a_epoch += train_loss.item()
                HSIC_loss_a_epoch += batch_HSIC_loss.item()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                # scheduler.step()

            else:
                _, predicted_interaction, _, _= model(args, train_data, device)
                if args.predict_task == 'Classification':
                    # print(f'计算分类损失')
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    if len(label)==1:
                        criterion = nn.CrossEntropyLoss(reduction='sum')
                        CE_loss = criterion(predicted_interaction, label.to(device))
                    else:
                        CE_loss = nn.CrossEntropyLoss()(predicted_interaction, label.to(device))
                    train_loss = CE_loss
                else:
                    # print(f'计算回归损失')
                    predicted_values = predicted_interaction[:, 1].double()   # 提取预测值的第一列
                    MES_loss = nn.MSELoss()(predicted_values, label.to(device))
                    train_loss = MES_loss  

                train_losses_a_epoch += train_loss.item()
                HSIC_loss_a_epoch += 0
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                # scheduler.step()  
        logger.info(f'更新伪标签损失')

        for _, Pseudo_data in enumerate(tqdm(Pseudo_loader)):
            if args.MCAN.IFATT:
                _, predicted_interaction, predict_Pseudo, _, _= model(args, Pseudo_data, device, Pseudo_task = True)
                predicted_interaction = F.softmax(predicted_interaction, dim=1)
                predict_Pseudo = F.softmax(predict_Pseudo, dim=1)
                Pseudo_loss = - args.AL.Pseudo_lambda * torch.mean(torch.sum(predicted_interaction * (torch.log(predict_Pseudo + 1e-6)), 1))
                optimizer.zero_grad()
                Pseudo_loss.backward()
                optimizer.step()
                # Pseudo_scheduler.step()
            else:
                _, predicted_interaction, predict_Pseudo, _, _= model(args, Pseudo_data, device, Pseudo_task = True)
                predicted_interaction = F.softmax(predicted_interaction, dim=1)
                predict_Pseudo = F.softmax(predict_Pseudo, dim=1)
                Pseudo_loss = - args.AL.Pseudo_lambda * torch.mean(torch.sum(predicted_interaction * (torch.log(predict_Pseudo + 1e-6)), 1))
                optimizer.zero_grad()
                Pseudo_loss.backward()
                optimizer.step()
                # Pseudo_scheduler.step()
    else:
        logger.info(f'更新自训练损失')
        for train_i, train_data in enumerate(tqdm(train_loader)):
            label = train_data['label'].squeeze()
            smiles = train_data['smiles']
            cell_line_id = train_data['cell_line_id']

            # 有注意力
            if args.MCAN.IFATT:
                _, predicted_interaction, crossatt_gene, crossatt_drug= model(args, train_data, device)
                batch_HSIC_loss = HSIC_loss(crossatt_gene, crossatt_drug, s_x=args.MODEL.Kernel_bandwidth, s_y=args.MODEL.Kernel_bandwidth) * args.MODEL.HSIC_loss_weight   # s_x, s_y为带宽
                # print(f'batch_HSIC_loss{batch_HSIC_loss}')
                if args.predict_task == 'Classification':
                    # print(f'计算分类损失')
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    if len(label)==1:
                        criterion = nn.CrossEntropyLoss(reduction='sum')
                        CE_loss = criterion(predicted_interaction, label.to(device))
                    else:
                        CE_loss = nn.CrossEntropyLoss()(predicted_interaction, label.to(device))
                    train_loss = CE_loss  +  batch_HSIC_loss.to(device)
                else:
                    # print(f'计算回归损失')
                    # 这里.double()是为了保证MSE损失计算时的输入内容格式一致，是哟交叉熵损失的话float32/64其实都可以
                    predicted_values = predicted_interaction[:, 1].double()  # 提取预测值的第一列
                    MES_loss = nn.MSELoss()(predicted_values, label.to(device))
                    train_loss = MES_loss  +  batch_HSIC_loss.to(device)
                train_losses_a_epoch += train_loss.item()
                HSIC_loss_a_epoch += batch_HSIC_loss.item()
                optimizer.zero_grad()
                train_loss.backward()
                # # 打印  loss 的梯度
                # if train_i ==0 :
                #     print("\nGradients from  loss:")
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             print(f"{name}: {param.grad.norm().item():.6f}")
                optimizer.step()
                if args.AL.AL_TASK == False:
                    scheduler.step()

            else:
                _, predicted_interaction, _, _= model(args, train_data, device)
                if args.predict_task == 'Classification':
                    # print(f'计算分类损失')
                    CE_loss = nn.CrossEntropyLoss()(predicted_interaction, label.to(device))
                    train_loss = CE_loss
                else:
                    # print(f'计算回归损失')
                    predicted_values = predicted_interaction[:, 1].double()   # 提取预测值的第一列
                    MES_loss = nn.MSELoss()(predicted_values, label.to(device))
                    train_loss = MES_loss  
                train_losses_a_epoch += train_loss.item()
                HSIC_loss_a_epoch += 0
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                if args.AL.AL_TASK == False:
                    scheduler.step()

    train_losses_a_epoch  /= len(train_loader)
    HSIC_loss_a_epoch  /= len(train_loader)
    return train_losses_a_epoch, HSIC_loss_a_epoch


def test(args, model, device, test_loader, split="test"):
    print('\nEvaluating model on {}...'.format(split))
    model.eval()
    test_losses = []
    test_losses_a_epoch = 0.0

    Y, P, S = [], [], []
    with torch.no_grad():
        for i, test_data in enumerate(tqdm(test_loader, mininterval = 1e-9)): #mininterval减小刷新时间进度条完全显示
            label = test_data['label'].squeeze()
            _, predicted_scores, _, _ = model(args, test_data, device)
           
            # loss = nn.CrossEntropyLoss()(predicted_scores, label.to(device))

    

            if args.predict_task == 'Classification':
                # 操作后对零维张量使用 len() 函数
                if label.dim() == 0:
                    label = label.unsqueeze(0)
                if len(label)==1:
                    criterion = nn.CrossEntropyLoss(reduction='sum')
                    loss = criterion(predicted_scores, label.to(device))
                else:
                    loss = nn.CrossEntropyLoss()(predicted_scores, label.to(device))
                
                # 如果是分类任务需要把预测值经过softmax（或者sigmod，因为是二分类两者等价）
                predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
                predicted_labels = np.argmax(predicted_scores, axis=1)
                predicted_scores = predicted_scores[:, 1]
            else:
                # print(f'计算回归损失')
                predicted_values = predicted_scores[:, 1].double()   # 提取预测值的第一列
                loss = nn.MSELoss()(predicted_values, label.to(device))
                predicted_scores = predicted_scores[:, 1].double().to('cpu').data.numpy()



            correct_labels = label.to('cpu')
            # test_losses.append(loss.item())
            test_losses_a_epoch += loss.item()
            Y.extend(correct_labels)
            # P.extend(predicted_labels)  # 按0.5阈值划分
            S.extend(predicted_scores)


    if args.predict_task == 'Classification':
        AUC = roc_auc_score(Y, S)
        PRC = average_precision_score(Y, S)  
        test_losses_a_epoch /= len(test_loader)

        fpr, tpr, thr = metrics.roc_curve(Y, S)
        optimal_idx = np.argmax(tpr-fpr)
        optimal_thr = thr[optimal_idx]
        P = (S > optimal_thr).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(Y, P).ravel()
        acc = metrics.accuracy_score(Y,P)
        precision = metrics.precision_score(Y,P)
        recall = metrics.recall_score(Y,P)
        f1 = metrics.f1_score(Y, P)
        specificity = tn / (tn + fp)
        return Y, P, test_losses_a_epoch, acc, precision, recall, AUC, PRC, recall, specificity, f1
    else:
        test_losses_a_epoch /= len(test_loader)
        mse = mean_squared_error(Y, S)
        rmse = np.sqrt(mse) 
        mae = mean_absolute_error(Y, S)
        r2 = r2_score(Y, S)
        # 皮尔逊相关系数和p值
        pearsonr_rho, pearsonr_pval = stats.pearsonr(Y, S)
        # 斯皮尔曼秩相关系数 和其p值
        spearmanr_rho, spearmanr_pval = stats.spearmanr(Y, S)

        return Y, S, test_losses_a_epoch, mse, rmse, mae, r2, pearsonr_rho, pearsonr_pval, spearmanr_rho, spearmanr_pval


def save_val_result(args, val_result_savepath, epoch_len, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_AUC, valid_PRC, valid_Accuracy, valid_Precision, valid_Sensitivity, valid_Specificity, valid_F1, valid_Recall):
    if args.AL.AL_TASK:
        total_Epoch = args.AL.AL_START_Epoch
    else:
        total_Epoch = args.Epoch
    
    logger.info(f'[{epoch}/{total_Epoch}] ' +
            f'train_loss: {train_loss_a_epoch:.5f} ' +
            f'HSIC_loss: {HSIC_loss_a_epoch:.10f} ' +
            f'valid_loss: {valid_loss_a_epoch:.5f} ' +
            f'valid_AUC: {valid_AUC:.5f} ' +
            f'valid_PRC: {valid_PRC:.5f} ' +
            f'valid_Accuracy: {valid_Accuracy:.5f} ' +
            f'valid_Pre cision: {valid_Precision:.5f} ' +
            f'valid_Sensitivity: {valid_Sensitivity:.5f} ' +
            f'valid_Specificity: {valid_Specificity:.5f} ' +
            f'valid_F1: {valid_F1:.5f} ' +
            f'valid_Reacll: {valid_Recall:.5f} ') 
    

    with open(val_result_savepath + '/val_results.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([epoch,
            f"{train_loss_a_epoch:.5f}",
            f"{HSIC_loss_a_epoch:.10f}",
            f"{valid_loss_a_epoch:.5f}",
            f"{valid_AUC:.5f}",
            f"{valid_PRC:.5f}",
            f"{valid_Accuracy:.5f}",
            f"{valid_Precision:.5f}",
            f"{valid_Sensitivity:.5f}",
            f"{valid_Specificity:.5f}",
            f"{valid_F1:.5f}",
            f"{valid_Recall:.5f}"
        ])

def save_val_result_regression(args, val_result_savepath, epoch_len, epoch, logger, train_loss_a_epoch, HSIC_loss_a_epoch, valid_loss_a_epoch, valid_mse, valid_rmse, valid_mae, valid_r2, valid_pearsonr_rho, valid_pearsonr_pval, valid_spearmanr_rho, valid_spearmanr_pval):
    if args.AL.AL_TASK:
        total_Epoch = args.AL.AL_START_Epoch
    else:
        total_Epoch = args.Epoch
    
    logger.info(f'[{epoch}/{total_Epoch}] ' +
            f'train_loss: {train_loss_a_epoch:.5f} ' +
            f'HSIC_loss: {HSIC_loss_a_epoch:.10f} ' +
            f'valid_loss: {valid_loss_a_epoch:.5f} ' +
            f'valid_mse: {valid_mse:.5f} ' +
            f'valid_rmse: {valid_rmse:.5f} ' +
            f'valid_mae: {valid_mae:.5f} ' +
            f'valid_r2: {valid_r2:.5f} ' +
            f'valid_pearsonr_rho: {valid_pearsonr_rho:.5f} ' +
            f'valid_pearsonr_pval: {valid_pearsonr_pval:.5f} ' +
            f'valid_spearmanr_rho: {valid_spearmanr_rho:.5f} ' +
            f'valid_spearmanr_pval: {valid_spearmanr_pval:.5f} ') 
    

    with open(val_result_savepath + '/val_results.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([epoch,
            f"{train_loss_a_epoch:.5f}",
            f"{HSIC_loss_a_epoch:.10f}",
            f"{valid_loss_a_epoch:.5f}",
            f"{valid_mse:.5f}",
            f"{valid_rmse:.5f}",
            f"{valid_mae:.5f}",
            f"{valid_r2:.5f}",
            f"{valid_pearsonr_rho:.5f}",
            f"{valid_pearsonr_pval:.5f}",
            f"{valid_spearmanr_rho:.5f}",
            f"{valid_spearmanr_pval:.5f}"
        ])



def save_test_result(args, logger, test_result_savepath, test_AUC, test_PRC, test_Accuracy, test_Precision, test_Sensitivity, test_Specificity, test_F1, test_Recall):
    logger.info(    f'source_test ' +
                    f'test_AUC: {test_AUC:.5f} ' +
                    f'test_PRC: {test_PRC:.5f} ' +
                    f'test_Accuracy: {test_Accuracy:.5f} ' +
                    f'test_Precision: {test_Precision:.5f} ' +
                    f'test_Sensitivity: {test_Sensitivity:.5f} ' +
                    f'test_Specificity: {test_Specificity:.5f} ' +
                    f'test_F1: {test_F1:.5f} ' +
                    f'test_Recall: {test_Recall:.5f} ') 
    with open(test_result_savepath + '/test_results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入表头
                writer.writerow(["test_AUC", "test_PRC", "test_Accuracy", "test_Precision", "test_Sensitivity", "test_Specificity", "test_F1", "test_Recall"])
                writer.writerow([
                        f"{test_AUC:.5f}", 
                        f"{test_PRC:.5f}", 
                        f"{test_Accuracy:.5f}", 
                        f"{test_Precision:.5f}", 
                        f"{test_Sensitivity:.5f}", 
                        f"{test_Specificity:.5f}", 
                        f"{test_F1:.5f}", 
                        f"{test_Recall:.5f}"
                    ])
    
def save_test_result_regression(args, logger, test_result_savepath, test_Y, test_P, _, test_mse, test_rmse, test_mae, test_r2, test_pearsonr_rho, test_pearsonr_pval, test_spearmanr_rho, test_spearmanr_pval):
    logger.info(    f'source_test ' +
                    f'test_mse: {test_mse:.5f} ' +
                    f'test_rmse: {test_rmse:.5f} ' +
                    f'test_mae: {test_mae:.5f} ' +
                    f'test_r2: {test_r2:.5f} ' +
                    f'test_pearsonr_rho: {test_pearsonr_rho:.5f} ' +
                    f'test_pearsonr_pval: {test_pearsonr_pval:.5f} ' +
                    f'test_spearmanr_rho: {test_spearmanr_rho:.5f} ' +
                    f'test_spearmanr_pval: {test_spearmanr_pval:.5f} ') 
    with open(test_result_savepath + '/test_results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入表头
                writer.writerow(["test_mse", "test_rmse", "test_mae", "test_r2", "test_pearsonr_rho", "test_pearsonr_pval", "test_spearmanr_rho", "test_spearmanr_pval"])
                writer.writerow([
                        f"{test_mse:.5f}", 
                        f"{test_rmse:.5f}", 
                        f"{test_mae:.5f}", 
                        f"{test_r2:.5f}", 
                        f"{test_pearsonr_rho:.5f}", 
                        f"{test_pearsonr_pval:.5f}", 
                        f"{test_spearmanr_rho:.5f}", 
                        f"{test_spearmanr_pval:.5f}"
                    ])


def show_time(start_time,end_time,task):
    total_time_seconds = end_time - start_time
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60
    seconds = total_time_seconds % 60
    print("{}耗时: {} hours {} minutes {} seconds".format(task, int(hours), int(minutes),int(seconds)))


class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source
        

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)



def row_norms(X, squared=False):
	"""Row-wise (squared) Euclidean norm of X.
	Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
	matrices and does not create an X.shape-sized temporary.
	Performs no input validation.
	Parameters
	----------
	X : array_like
		The input array
	squared : bool, optional (default = False)
		If True, return squared norms.
	Returns
	-------
	array_like
		The row-wise (squared) Euclidean norm of X.
	"""
	norms = np.einsum('ij,ij->i', X, X)

	if not squared:
		np.sqrt(norms, norms)
	return norms

def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(1234), n_local_trials=None):
	"""Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)
	Parameters
	----------
	X1, X2 : array or sparse matrix
		The data to pick seeds for. To avoid memory copy, the input data
		should be double precision (dtype=np.float64).
	n_clusters : integer
		The number of seeds to choose
	init : list
		List of points already picked
	random_state : int, RandomState instance
		The generator used to initialize the centers. Use an int to make the
		randomness deterministic.
		See :term:`Glossary <random_state>`.
	n_local_trials : integer, optional
		The number of seeding trials for each center (except the first),
		of which the one reducing inertia the most is greedily chosen.
		Set to None to make the number of trials depend logarithmically
		on the number of seeds (2+log(k)); this is the default.
	Notes
	-----
	Selects initial cluster centers for k-mean clustering in a smart way
	to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
	"k-means++: the advantages of careful seeding". ACM-SIAM symposium
	on Discrete algorithms. 2007
	Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
	which is the implementation used in the aforementioned paper.
	"""

	n_samples, n_feat1 = X1.shape
	_, n_feat2 = X2.shape
	# x_squared_norms = row_norms(X, squared=True)
	centers1 = np.empty((n_clusters+len(init)-1, n_feat1), dtype=X1.dtype)
	centers2 = np.empty((n_clusters+len(init)-1, n_feat2), dtype=X1.dtype)

	idxs = np.empty((n_clusters+len(init)-1,), dtype=np.int64)

	# Set the number of local seeding trials if none is given
	if n_local_trials is None:
		# This is what Arthur/Vassilvitskii tried, but did not report
		# specific results for other than mentioning in the conclusion
		# that it helped.
		n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = init

	centers1[:len(init)] = X1[center_id]
	centers2[:len(init)] = X2[center_id]
	idxs[:len(init)] = center_id

	# Initialize list of closest distances and calculate current potential
	distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init), -1)

	candidates_pot = distance_to_candidates.sum(axis=1)
	best_candidate = np.argmin(candidates_pot)
	current_pot = candidates_pot[best_candidate]
	closest_dist_sq = distance_to_candidates[best_candidate]

	# Pick the remaining n_clusters-1 points
	for c in range(len(init), len(init)+n_clusters-1):
		# Choose center candidates by sampling with probability proportional
		# to the squared distance to the closest existing center
		rand_vals = random_state.random_sample(n_local_trials) * current_pot
		candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
										rand_vals)
		# XXX: numerical imprecision can result in a candidate_id out of range
		np.clip(candidate_ids, None, closest_dist_sq.size - 1,
				out=candidate_ids)

		# Compute distances to center candidates
		distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(len(candidate_ids), -1)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates,
				   out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		idxs[c] = best_candidate

	return None, idxs[len(init)-1:]

def outer_product_opt(c1, d1, c2, d2):
	"""Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products
	"""
	B1, B2 = c1.shape[0], c2.shape[0]
	t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)), np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
	t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)), np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
	t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
	t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
	t2 = t2.reshape(1, B2).repeat(B1, axis=0)
	return t1 + t2 - 2*t3