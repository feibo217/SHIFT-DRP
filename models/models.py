import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Function
from models.mcanet import MCA_ED


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""
    def __init__(self, dim=-1):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return torch.squeeze(data, dim=self.dim)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return torch.unsqueeze(data, self.dim)

def make_mask(feature):  # feature :[32, length, feature_dim]
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class DR_crossatt_model(nn.Module):

    def __init__(self, args,
                 gene_embed_size: int,
                 drug_embed_size: int,
                 attention_dim_size: int,
                 ):
        super(DR_crossatt_model, self).__init__()

        self.gene_unsqueeze = Unsqueeze(2) # only needed if doing at root level
        self.drug_unsqueeze = Unsqueeze(2)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

        self.leaky_relu = nn.LeakyReLU()
        #  基因长度+药物特征长度
        if args.DATA.drug_feature_method == 'MACCS':
            self.drugfeature_linear = nn.Linear(167, 256)

        elif args.DATA.drug_feature_method == 'ECFP4':
            self.drugfeature_linear = nn.Linear(2048, 256)

        elif args.DATA.drug_feature_method == 'unimol':
            self.drugfeature_linear = nn.Linear(512, 256)

        elif args.DATA.drug_feature_method == 'KPGT':
            self.drugfeature_linear = nn.Linear(2304, 256)

        elif args.DATA.drug_feature_method == 'Molformer':
            self.drugfeature_linear = nn.Linear(768, 256)

        elif args.DATA.drug_feature_method == 'GCN':
            self.drugfeature_linear = nn.Linear(290, 256)


        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(256)

        self.fc1 = nn.Linear(1056, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 2) 

        self.transfor_linear = nn.Linear(1056, 1056)

        # self.drugfeature_linear = nn.Linear(2304, 256)


        self.mca_linear = nn.Linear(1, args.MCAN.HIDDEN_SIZE)
        self.MCAN_ATT = MCA_ED(args.MCAN)

        self.ATT_alpha_Linear = nn.Linear(attention_dim_size, 1)
        self.ATT_squeeze = Squeeze()
        self.ATT_Softmax = nn.Softmax(dim=1)

    def classify(self, feature, output_f3 = False):
            """
            分类器部分的预测
            """
            fully1 = self.leaky_relu(self.norm1(self.fc1(feature)))
            fully1 = self.dropout2(fully1)
            fully2 = self.leaky_relu(self.norm2(self.fc2(fully1)))
            fully2 = self.dropout3(fully2)
            fully3 = self.leaky_relu(self.norm3(self.fc3(fully2)))
            predict = self.out(fully3)
            # fully1 = self.leaky_relu((self.fc1(feature)))
            # fully1 = self.dropout2(fully1)
            # fully2 = self.leaky_relu((self.fc2(fully1)))
            # fully2 = self.dropout3(fully2)
            # fully3 = self.leaky_relu((self.fc3(fully2)))
            # predict = self.out(fully3)
            if output_f3:
                return predict, fully3
            else:
                return predict

    def classify_input_f3(self, feature):
            predict = self.out(feature)
            return predict


    def forward(self, args, data, device, Pseudo_task = False, output_origin_feature = False): 
        
        # 有注意力
        if args.MCAN.IFATT:
            smiles_feature = data['smiles_feature'].to(device)
            smiles_feature = self.leaky_relu(self.drugfeature_linear(smiles_feature))
            exp_feature = data['exp_feature'].to(device, dtype=torch.float32)

            gene = self.gene_unsqueeze(exp_feature)  #([32, 800, 1])            
            drug = self.drug_unsqueeze(smiles_feature) #([32, 512, 1])

            gene_mask = make_mask(gene)  #([32, 1, 1, length])
            drug_mask = make_mask(drug)  #([32, 1, 1, length])

            # MCA
            mca_gene = self.mca_linear(gene)
            mca_drug = self.mca_linear(drug)
            _ , attended_drug, gene_shape_maps = self.MCAN_ATT(mca_gene, mca_drug, gene_mask, drug_mask)
            _ , attended_gene, drug_shape_maps = self.MCAN_ATT(mca_drug, mca_gene, drug_mask, gene_mask)
            attended_drug  = torch.mean(attended_drug, dim=2)
            attended_gene  = torch.mean(attended_gene, dim=2)
            att_feature = torch.cat([attended_gene, attended_drug], dim=1) 
            origin_feature = torch.cat([gene, drug], dim=1) 
            origin_feature = torch.squeeze(origin_feature, dim=2) 
            feature = self.dropout1(att_feature)
            predict, fully3 = self.classify(feature, output_f3 = True)

            if Pseudo_task:
                # 这里和fc1的维度对应即可
                # stddev = 0.1  # 添加噪声的标准差  在特征上添加零均值、小标准差的高斯噪声
                # noise = torch.randn_like(feature) * stddev
                # noisy_features = feature + noise

                transfor_feature = self.leaky_relu(self.transfor_linear(feature)).to(device)
                predict_Pseudo = self.classify(transfor_feature)
                return fully3, predict, predict_Pseudo, attended_gene, attended_drug
            else:
                if output_origin_feature:
                    return fully3, predict, attended_gene, attended_drug, origin_feature
                else:
                    return fully3, predict, attended_gene, attended_drug

        else:
            # 注意力消融
            smiles_feature = data['smiles_feature'].to(device)
            smiles_feature = self.leaky_relu(self.drugfeature_linear(smiles_feature))
            exp_feature = data['exp_feature'].to(device, dtype=torch.float32)

            gene = self.gene_unsqueeze(exp_feature)  #([32, 800, 1])            
            drug = self.drug_unsqueeze(smiles_feature) #([32, 512, 1])

            # 注意力消融
            drug  = torch.mean(drug, dim=2)
            gene  = torch.mean(gene, dim=2)
            origin_feature = torch.cat([gene, drug], dim=1) 
            feature = self.dropout1(origin_feature)
            predict = self.classify(feature)

            if Pseudo_task:
                # 这里和fc1的维度对应即可
                # stddev = 0.1  # 添加噪声的标准差  在特征上添加零均值、小标准差的高斯噪声
                # noise = torch.randn_like(feature) * stddev
                # noisy_features = feature + noise
                transfor_feature = self.leaky_relu(self.transfor_linear(feature)).to(device)
                predict_Pseudo = self.classify(transfor_feature)
                return fully3, predict, predict_Pseudo, None, None
            else:
                return fully3, predict, None, None