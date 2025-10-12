import pandas as pd
import requests

''' 
①读取Excel文件保存为 CSV 文件
①读取Excel文件保存为 CSV 文件
①读取Excel文件保存为 CSV 文件
''' 
# excel_file = 'Data/origin_data/GDSC2_AUC/GDSC2_exp.xlsx'
# df = pd.read_excel(excel_file, engine='openpyxl')
# csv_file = 'Data/origin_data/GDSC2_AUC/GDSC2_exp.csv'
# df.to_csv(csv_file, index=False)



''' 
②提取所需列
②提取所需列
②提取所需列
''' 
# input_file = 'Data/origin_data/GDSC2_AUC/GDSC2_response_deduplicate.csv'
# df = pd.read_csv(input_file)
# # 提取所需列
# columns_to_extract = ['NLME_RESULT_ID', 'COSMIC_ID', 'DRUG_ID', 'DRUG_NAME', 'Classification', 'SMILES']
# df_selected = df[columns_to_extract]
# output_file = 'Data/origin_data/GDSC2_AUC/classified_GDSC2_auc_processed.csv'
# df_selected.to_csv(output_file, index=False)



''' 
③根据DRUG_ID爬取smiles
③根据DRUG_ID爬取smiles
③根据DRUG_ID爬取smiles
''' 

# # 读取原始 CSV 文件
# input_file = 'Data/origin_data/GDSC2_AUC/classified_GDSC2_auc_processed.csv'
# df = pd.read_csv(input_file)
# # 定义一个函数来从 PubChem 获取 SMILES
# def get_smiles_from_pubchem(drug_name):
#     base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
#     search_url = f"{base_url}/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
#     try:
#         response = requests.get(search_url)
#         response.raise_for_status()
#         data = response.json()
#         if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
#             properties = data['PropertyTable']['Properties']
#             if properties:
#                 return properties[0]['CanonicalSMILES']
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching SMILES for {drug_name}: {e}")
#     return None
# # 获取所有唯一药物名称
# unique_drug_names = df['DRUG_NAME'].unique()
# # 创建一个字典来存储药物名称和对应的 SMILES
# smiles_dict = {}
# for drug_name in unique_drug_names:
#     smiles = get_smiles_from_pubchem(drug_name)
#     smiles_dict[drug_name] = smiles
# # 将 SMILES 添加到 DataFrame 中
# df['SMILES'] = df['DRUG_NAME'].map(smiles_dict)
# # 保存到新的 CSV 文件
# output_file = 'Data/origin_data/GDSC2_AUC/classified_GDSC2_auc_with_smiles.csv'
# df.to_csv(output_file, index=False)




''' 
为unique药物（SMILES）添加uniqueID
为unique药物（SMILES）添加uniqueID
为unique药物（SMILES）添加uniqueID
groupby('SMILES').ngroup()：
groupby('SMILES') 会根据 SMILES 列的值对数据进行分组。
ngroup() 会为每个分组分配一个从 0 开始的整数标识符，这样每个唯一的 SMILES 都会有一个唯一的 uniqueID。
''' 
# import pandas as pd

# response_file = 'Data/origin_data/GDSC2_AUC/GDSC2_response_deduplicate.csv'
# df = pd.read_csv(response_file)
# # 为每个唯一的 SMILES 分配一个 uniqueID
# df['uniqueID'] = df.groupby('SMILES').ngroup()
# # 保存修改后的数据框
# df.to_csv('Data/origin_data/GDSC2_AUC/GDSC2_response_with_druguniqueID.csv', index=False)




''' 
④合并表达谱数据
④合并表达谱数据
④合并表达谱数据
''' 

import pandas as pd
# 读取原始 CSV 文件
input_file = 'Data/origin_data/GDSC2_AUC/GDSC2_exp.csv'
df_exp = pd.read_csv(input_file)

# 读取包含 COSMIC_ID 的 CSV 文件
response_file = 'Data/origin_data/GDSC2_AUC/GDSC2_response_with_druguniqueID.csv'
df_response = pd.read_csv(response_file)

# 提取 COSMIC_ID 列
cosmic_ids = df_response['COSMIC_ID'].unique()

# 过滤出 GDSC2_exp.csv 中包含这些 COSMIC_ID 的列
filtered_columns = [col for col in df_exp.columns if str(col).isdigit() and int(col) in cosmic_ids]  #  str.isdigit() 方法检查这个字符串是否只包含数字

# 创建一个新的 DataFrame，只包含这些列

# 确保保留GENE_SYMBOLS列和GENE_title列
# 如果保存完后没有检查列名是否一致
if 'GENE_SYMBOLS' in df_exp.columns:
    filtered_columns.insert(0, 'GENE_SYMBOLS')

# 更新后的数据没有GENE_title
# if 'GENE_title' in df_exp.columns:
#     filtered_columns.insert(1, 'GENE_title')

df_filtered = df_exp[filtered_columns]

# 保存到新的 CSV 文件
output_file = 'Data/origin_data/GDSC2_AUC/GDSC2_exp_processed.csv'
df_filtered.to_csv(output_file, index=False)