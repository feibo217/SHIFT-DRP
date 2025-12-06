import csv
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
#分子指纹
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

#转分子图
from functools import partial
# from dgllife.utils import smiles_to_bigraph
# from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
# from dgllife.model.gnn import GCN

#并行加速
from joblib import Parallel, delayed



def load_smiles_and_exp_data(args, data_path):
    smiles_data, labels, exp_data, cosmic_id_list = [], [], [], []
    depmap_id_list = []

    # 读取 CSV 文件的总行数
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        total_rows = sum(1 for _ in csv_reader)

    # 确定表达谱数据文件路径
    if args.DATA.expdim_reduce_method == 'PCA':
        input_file = f'Data/processed_data/{args.DATA.source_data_name}/PCA_dim_reduce_expression_800.csv'
    elif args.DATA.expdim_reduce_method == 'Variance':
        input_file = f'Data/processed_data/{args.DATA.source_data_name}/variances_dim_reduce_expression_{args.DATA.variances_dim}.csv'
    elif args.DATA.expdim_reduce_method == 'AE':
        input_file = f'Data/processed_data/{args.DATA.source_data_name}/AE_dim_reduce_expression_800.csv'
    elif args.DATA.expdim_reduce_method == 'correlation+Variance':
        input_file = f'Data/processed_data/{args.DATA.source_data_name}/correlation+Variance_dim_reduce_expression_800.csv'
    elif args.DATA.expdim_reduce_method == 'correlation':
        input_file = f'Data/processed_data/{args.DATA.source_data_name}/correlation_dim_reduce_expression_{args.DATA.correlation_dim}.csv'
    else:
        raise ValueError("Choose from 'PCA' / 'Variance'/ 'AE'")
    
    if args.DATA.source_data_name == 'GDSC2_AUC_crossdomain':       # GDSC2_AUC_crossdomain  GDSC2_AUC
        # 读取表达谱数据文件，并构建索引
        exp_df = pd.read_csv(input_file)
        exp_index = {cosmic_id: idx for idx, cosmic_id in enumerate(exp_df.columns)}
        # 重新打开文件以逐块读取
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                print(f'Processing row: {i + 1}/{total_rows}')
                smiles = row['SMILES']
                # 对应分类任务和回归任务的标签
                if args.predict_task == 'Classification':
                    label = row['Classification']
                else:
                    label = row['AUC']
                cosmic_id = row['COSMIC_ID']
                # 检查 COSMIC_ID 是否存在于表达谱数据文件中
                if cosmic_id not in exp_index:                
                    raise ValueError(f"COSMIC_ID {cosmic_id} not found in the expression data file.")
                # 获取表达谱数据
                cosmic_id_column = exp_df.iloc[:, exp_index[cosmic_id]].values
                if args.predict_task == 'Classification':
                    labels.append(int(label))
                else:
                    labels.append(float(label))
                smiles_data.append(smiles)
                exp_data.append(cosmic_id_column)
                cosmic_id_list.append(cosmic_id)
        return smiles_data, exp_data, labels, cosmic_id_list
    
    # 如果是PRISM数据集。数据对应会有所变化
    elif args.DATA.source_data_name == 'PRISM_AUC':
        # 读取表达谱数据文件，并构建索引
        exp_df = pd.read_csv(input_file)
        exp_index = {depmap_id: idx for idx, depmap_id in enumerate(exp_df.columns)}
        # 重新打开文件以逐块读取
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                print(f'Processing row: {i + 1}/{total_rows}')
                smiles = row['SMILES']
                label = row['Classification']
                depmap_id = row['depmap_id']
                # 检查 depmap_id 是否存在于表达谱数据文件中
                if depmap_id not in exp_index:                
                    raise ValueError(f"depmap_id {depmap_id} not found in the expression data file.")
                # 获取表达谱数据
                cosmic_id_column = exp_df.iloc[:, exp_index[depmap_id]].values
                labels.append(int(label))
                smiles_data.append(smiles)
                exp_data.append(cosmic_id_column)
                depmap_id_list.append(depmap_id)
        return smiles_data, exp_data, labels, depmap_id_list
    
    elif args.DATA.source_data_name == 'Organoid_BLCA_GDSCEXP':
        # 读取表达谱数据文件，并构建索引
        exp_df = pd.read_csv(input_file)
        exp_index = {Organoid: idx for idx, Organoid in enumerate(exp_df.columns)}
        # 重新打开文件以逐块读取
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                print(f'Processing row: {i + 1}/{total_rows}')
                smiles = row['SMILES']
                label = row['Classification']
                Organoid = row['Organoid']
                # 检查 Organoid 是否存在于表达谱数据文件中
                if Organoid not in exp_index:                
                    raise ValueError(f"Organoid {Organoid} not found in the expression data file.")
                # 获取表达谱数据
                cosmic_id_column = exp_df.iloc[:, exp_index[Organoid]].values
                labels.append(int(label))
                smiles_data.append(smiles)
                exp_data.append(cosmic_id_column)
                depmap_id_list.append(Organoid)
        return smiles_data, exp_data, labels, depmap_id_list
    
    elif args.DATA.source_data_name == 'Organoid_COAD_GDSCEXP':
        # 读取表达谱数据文件，并构建索引
        exp_df = pd.read_csv(input_file)
        exp_index = {Organoid: idx for idx, Organoid in enumerate(exp_df.columns)}
        # 重新打开文件以逐块读取
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                print(f'Processing row: {i + 1}/{total_rows}')
                smiles = row['SMILES']
                label = row['Classification']
                Organoid = row['Organoid']  
                # 检查 Organoid 是否存在于表达谱数据文件中
                if Organoid not in exp_index:                
                    raise ValueError(f"Organoid {Organoid} not found in the expression data file.")
                # 获取表达谱数据
                cosmic_id_column = exp_df.iloc[:, exp_index[Organoid]].values
                labels.append(int(label))
                smiles_data.append(smiles)
                exp_data.append(cosmic_id_column)
                depmap_id_list.append(Organoid)
        return smiles_data, exp_data, labels, depmap_id_list
    
    else:
        # 读取表达谱数据文件，并构建索引
        exp_df = pd.read_csv(input_file)
        exp_index = {Organoid: idx for idx, Organoid in enumerate(exp_df.columns)}
        # 重新打开文件以逐块读取
        with open(data_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                print(f'Processing row: {i + 1}/{total_rows}')
                smiles = row['SMILES']
                label = row['Classification']
                Organoid = row['bcr_patient_barcode']
                # 检查 Organoid 是否存在于表达谱数据文件中
                if Organoid not in exp_index:                
                    raise ValueError(f"Organoid {Organoid} not found in the expression data file.")
                # 获取表达谱数据
                cosmic_id_column = exp_df.iloc[:, exp_index[Organoid]].values
                labels.append(int(label))
                smiles_data.append(smiles)
                exp_data.append(cosmic_id_column)
                depmap_id_list.append(Organoid)
        return smiles_data, exp_data, labels, depmap_id_list

# 加载数据集
class Load_Dataset:
    def __init__(self, args, logger, AL_start_part = False):
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.AL_start_part = AL_start_part

        self.src_dset = None
        self.train_size = None
        self.num_classes = None
        self.logger = logger
        self.num_workers = 0
        self.source_train_dataset = None
        self.source_valid_dataset = None
        self.source_test_dataset = None
        self.source_Patient_test_dataset = None
        self.full_dataset = None


        #划分好的源域数据
        self.source_train_path = f'Data/processed_data/{args.DATA.source_data_name}/{args.DATA.split_task}/train.csv'
        self.source_val_path = f'Data/processed_data/{args.DATA.source_data_name}/{args.DATA.split_task}/val.csv'
        self.source_test_path = f'Data/processed_data/{args.DATA.source_data_name}/{args.DATA.split_task}/test.csv'

        if AL_start_part:
            self.source_AL_start_path = f'Data/processed_data/{args.DATA.source_data_name}/{args.DATA.split_task}/AL_start.csv'         
            self.source_AL_start_dataset = None
            
        # if self.args.s1s2t1_crossdomain:
        #     self.train_labeled_path = f'Data/processed_data/{args.DATA.source_data_name}/{args.DATA.split_task}/train_labeled.csv'         
        #     self.train_labeled_dataset = None
        #     self.train_unlabeled_path = f'Data/processed_data/{args.DATA.source_data_name}/{args.DATA.split_task}/train_unlabeled.csv'         
        #     self.train_unlabeled_dataset = None


    def get_dsets(self):
                
        if self.args.Patient_test:
            self.source_test_dataset = transfer_feature(self.args, data_path = self.source_test_path, data_part = 'Patient_test', root='Data')
        else:
            self.source_test_dataset = transfer_feature(self.args, data_path = self.source_test_path, data_part = 'test', root='Data')
        self.source_train_dataset = transfer_feature(self.args, data_path = self.source_train_path, data_part = 'train', root='Data')
        self.source_valid_dataset = transfer_feature(self.args, data_path = self.source_val_path, data_part = 'valid', root='Data')

        # self.train_labeled_dataset = transfer_feature(self.args, data_path = self.train_labeled_path, data_part = 'train_labeled', root='Data')
        # self.train_unlabeled_dataset = transfer_feature(self.args, data_path = self.train_unlabeled_path, data_part = 'train_unlabeled', root='Data')
        
        # self.source_test_dataset = transfer_feature(self.args, data_path = self.source_test_path, data_part = 'test', root='Data')
        # self.source_valid_dataset = transfer_feature(self.args, data_path = self.source_val_path, data_part = 'valid', root='Data')


        if self.AL_start_part:
            self.source_AL_start_dataset = transfer_feature(self.args, data_path = self.source_AL_start_path, data_part = 'AL_start', root='Data')
            self.logger.info("AL_start大小{} ，总数据集大小：{} ".format(len(self.source_AL_start_dataset),len(self.source_AL_start_dataset) + len(self.source_train_dataset)+len(self.source_valid_dataset)+len(self.source_test_dataset)))
            # 合并AL_START 和 train
            self.source_train_dataset = ConcatDataset([self.source_AL_start_dataset, self.source_train_dataset])
            # return self.source_train_dataset,self.source_valid_dataset,self.source_test_dataset,self.source_AL_start_dataset
        else:
            self.logger.info("总数据集大小：{} ".format(len(self.source_train_dataset)+len(self.source_valid_dataset)+len(self.source_test_dataset)))
            # return self.source_train_dataset,self.source_valid_dataset,self.source_test_dataset

    
    def get_loaders(self):

        if not self.source_train_dataset: self.get_dsets()

        train_idx = list(range(len(self.source_train_dataset)))
        train_loader = DataLoader(
            self.source_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        val_loader = DataLoader(
            self.source_valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        test_loader = DataLoader(
            self.source_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        if self.AL_start_part:
            AL_start_loader = DataLoader(self.source_AL_start_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            return train_loader, val_loader, test_loader, train_idx, self.source_train_dataset, AL_start_loader, self.source_AL_start_dataset
        else:
            return train_loader, val_loader, test_loader, train_idx, self.source_train_dataset, None
        
    def get_full_loaders(self):
        if not self.source_train_dataset: self.get_dsets()
        self.full_dataset = ConcatDataset([self.source_train_dataset, self.source_test_dataset, self.source_valid_dataset])
        full_loader = DataLoader(self.source_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return self.full_dataset , full_loader

# 分子图特征提取
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        # self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats



class transfer_feature(Dataset):
    def __init__(self, args, data_path, data_part, root='/tmp'):
        self.args = args
        self.data_part = data_part
        # self.atom_featurizer = CanonicalAtomFeaturizer()  # 原子特征提取器
        # self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)  # 边特征提取器
        # self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.GCN_max_drug_nodes = args.DATA.GCN_max_drug_nodes
        self.drug_extractor = MolecularGCN(in_feats=args.DATA.GCN_node_in_feats, dim_embedding=args.DATA.GCN_node_in_embedding,
                                           padding=args.DATA.GCN_drug_padding,
                                           hidden_feats=args.DATA.GCN_drug_hidden_layer)
        # 设置是回归任务还是分类任务
        if self.args.predict_task == 'regression':
            model_task = '_regression'
        else:
            model_task = ''
        # 随机划分的数据存储路径
        if self.args.DATA.split_task == 'split_data':
            # 定义预处理文件的路径
            if not os.path.exists(f'{root}/processed_pt_file{model_task}'):
                os.makedirs(f'{root}/processed_pt_file{model_task}')
            if args.DATA.expdim_reduce_method == 'Variance':
                self.processed_file = os.path.join(root, f'processed_pt_file{model_task}/{args.DATA.source_data_name}_{args.DATA.expdim_reduce_method}{args.DATA.variances_dim}_{args.DATA.drug_feature_method}_{data_part}_seed{str(args.seed)}.pt')
            else:
                self.processed_file = os.path.join(root, f'processed_pt_file{model_task}/{args.DATA.source_data_name}_{args.DATA.expdim_reduce_method}_{args.DATA.drug_feature_method}_{data_part}_seed{str(args.seed)}.pt')
            
            if os.path.isfile(self.processed_file):
                print('Pre-processed data found: {}, loading ...'.format(self.processed_file))
                self.data_list = torch.load(self.processed_file)
            else:
                print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_file))
                self.smiles_data, self.exp_data, self.labels, self.cell_line_id = load_smiles_and_exp_data(self.args, data_path)
                self.data_list = list(self.process(self.args, self.smiles_data, self.exp_data, self.labels, self.cell_line_id))
                torch.save(self.data_list, self.processed_file)
        # 其他划分的数据存储路径（主动学习冷启动，冷启动）
        else:
            # 定义预处理文件的路径
            if args.DATA.expdim_reduce_method == 'Variance':
                self.processed_file = os.path.join(root, f'processed_pt_file{model_task}/{args.DATA.source_data_name}_{args.DATA.split_task}_{args.DATA.expdim_reduce_method}{args.DATA.variances_dim}_{args.DATA.drug_feature_method}_{data_part}_seed{str(args.seed)}.pt')
            else:
                self.processed_file = os.path.join(root, f'processed_pt_file{model_task}/{args.DATA.source_data_name}_{args.DATA.split_task}_{args.DATA.expdim_reduce_method}_{args.DATA.drug_feature_method}_{data_part}_seed{str(args.seed)}.pt')
            if os.path.isfile(self.processed_file):
                print('Pre-processed data found: {}, loading ...'.format(self.processed_file))
                self.data_list = torch.load(self.processed_file)
            else:
                print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_file))
                self.smiles_data, self.exp_data, self.labels, self.cell_line_id = load_smiles_and_exp_data(self.args, data_path)
                self.data_list = list(self.process(self.args, self.smiles_data, self.exp_data, self.labels, self.cell_line_id))
                torch.save(self.data_list, self.processed_file)

    def process(self, args, smiles_data, exp_data, labels, cell_line_id):
        SMILES_DATA = smiles_data
        if args.DATA.drug_feature_method == 'unimol':
            #unimol
            from unimol_tools import UniMolRepr #感觉初始化导入unimol有点慢，放在这
            clf = UniMolRepr(data_type='molecule', remove_hs=True)
            unimol_repr = clf.get_repr(smiles_data, return_atomic_reprs=False)
            # smiles_data = unimol_repr['atomic_reprs'] # [原子数,512]
            smiles_data = unimol_repr['cls_repr'] # [512]

        elif args.DATA.drug_feature_method == 'KPGT':
            #KPGT
            drug_feature_data = np.load(f'Data/KPGT_npz_file/{args.DATA.source_data_name}_{args.DATA.split_task}_processed/kpgt_{args.DATA.source_data_name}_{self.data_part}.npz')
            fps_array = drug_feature_data['fps']
            smiles_data = fps_array.tolist()

        elif args.DATA.drug_feature_method == 'Molformer':
            #Molformer
            drug_feature_data = np.load(f'Data/Molformer_npz_file/{args.DATA.source_data_name}_processed/{self.data_part}.npy')
            smiles_data = drug_feature_data.tolist()

        assert (len(smiles_data) == len(exp_data) and len(smiles_data) == len(labels)), "The three lists must be the same length!"

        data_len = len(smiles_data)
        for i in range(data_len):
            print(f'正在将特征保存: {i + 1}/{data_len}')
            smiles = smiles_data[i]  # SMILES
            exp_feature = exp_data[i]  # 表达谱特征
            label = labels[i]  # 标签
            cell_line_ids = cell_line_id[i]
            SMILES = SMILES_DATA[i]

            # 使用fc函数从SMILES字符串生成图特征
            if args.DATA.drug_feature_method == 'GCN':
                    print('GCN NOT READY')
                    # GCN_feature = self.fc(smiles=smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
                    # actual_node_feats = GCN_feature.ndata.pop('h') #从图 v_d 中取出节点特征
                    # num_actual_nodes = actual_node_feats.shape[0] #计算当前图中的实际节点数量
                    # num_virtual_nodes = self.GCN_max_drug_nodes - num_actual_nodes  #计算需要添加的虚拟节点数量
                    # virtual_node_bit = torch.zeros([num_actual_nodes, 1])  #创建一个虚拟节点标记 ， （通常是一个全零向量）
                    # actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1) # 将虚拟节点标记与实际节点特征拼接起来，形成新的节点特征
                    # GCN_feature.ndata['h'] = actual_node_feats #将更新后的节点特征放回图中
                    # virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)  #创建虚拟节点的特征，其中前 74 个元素是 0，最后一个元素是 1，以区别于实际节点
                    # GCN_feature.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})  #添加虚拟节点
                    # GCN_feature = GCN_feature.add_self_loop()  #添加自环(确保节点在进行消息传递时能考虑到自身的特征)
                    # smiles_feature = self.drug_extractor(GCN_feature)
                    # smiles_feature = torch.mean(smiles_feature, dim=2).detach().numpy().ravel()

            elif args.DATA.drug_feature_method in ['unimol', 'KPGT', 'Molformer']:
                    smiles_feature = np.array(smiles)

            elif args.DATA.drug_feature_method == 'ECFP4':
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_feature = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
                else:
                    print(f"转换ECFP4失败: {smiles}")
                    continue

            elif args.DATA.drug_feature_method == 'MACCS':
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_feature = np.array(list(MACCSkeys.GenMACCSKeys(mol)))   #(167,)
                else:
                    print(f"转换MACCS失败: {smiles}")
                    continue

            elif args.DATA.drug_feature_method == 'Pubchem':
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_feature = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=881)))
                else:
                    print(f"转换Pubchem失败: {smiles}")
                    continue

            else:
                print('重新选择  config.yml   DATA.drug_feature_method ')
                continue

            # 存储特征信息
            if args.predict_task == 'Classification':
                sample = {
                    'smiles_feature': torch.Tensor(smiles_feature),
                    'label': torch.LongTensor([label]),
                    'exp_feature': exp_feature,
                    'smiles': SMILES,
                    'cell_line_id': cell_line_ids
                }
            else:
                sample = {
                    'smiles_feature': torch.Tensor(smiles_feature),
                    'label': label,
                    'exp_feature': exp_feature,
                    'smiles': SMILES,
                    'cell_line_id': cell_line_ids
                }


            yield sample  # 生成器返回单个样本

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]