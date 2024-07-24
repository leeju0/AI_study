import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import silhouette_samples, silhouette_score
import networkx as nx
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import seaborn as sns


# df : LUAD
file_path = '/BiO/lee/TCGAdata/LUAD.csv'

df = pd.read_csv(file_path)
df = pd.read_csv(file_path, index_col='gene_name').drop('Unnamed: 0', axis=1)

# df2 : LAUD_clinic
df2 = pd.read_csv('/home/lee/home/clinic_patient/LUAD_clinic_patient.csv')
df2= df2.iloc[:,1:]

# 분석하고자 하는 컬럼만 남기기
df2 = df2[['bcr_patient_barcode', 'histological_type']]

# 'bcr_patient_barcode' 값만 추출
LUAD_bar = list(df2['bcr_patient_barcode'])

df2.set_index('bcr_patient_barcode', inplace=True)

# 바코드와 "분석하고자 하는 컬럼"을 담을 딕셔너리 생성
LUAD_dict = {}

# LUAD 데이터프레임의 각 컬럼에 대해 "분석하고자 하는 컬럼"을 매칭하여 딕셔너리에 추가
for col in df.columns:  
    patient_barcode = col[:12]
    if patient_barcode in df2.index:
        histological_type = df2.at[patient_barcode, 'histological_type']
        LUAD_dict[col] = histological_type
   

# df(LAUD.csv) 에서 df2(LUAD_clinic_patinet.csv)에 존재하는 bcr_patient_barcode만 컬럼으로 가지는 subtype 데이터프레임 생성
subtype_LUAD = df.loc[:, df.columns.str[:12].isin(LUAD_bar)]


# 분석하고자 하는 컬럼에서 구분하고자 하는 서브타입 리스트
subtypes = [
       'Lung Papillary Adenocarcinoma', 'Lung Mucinous Adenocarcinoma',
       'Mucinous (Colloid) Carcinoma', 'Lung Clear Cell Adenocarcinoma',
       'Lung Acinar Adenocarcinoma',
       'Lung Bronchioloalveolar Carcinoma Nonmucinous',
       'Lung Bronchioloalveolar Carcinoma Mucinous',
       'Lung Solid Pattern Predominant Adenocarcinoma',
       'Lung Micropapillary Adenocarcinoma',
       'Lung Signet Ring Adenocarcinoma']


# 저장된 결과를 pickle 파일에서 불러오기
with open('/home/lee/home/clustering_results_0.5_LUAD.pkl', 'rb') as f:
    results = pickle.load(f)

print("Results loaded from clustering_results.pkl")

# 불러온 results를 사용하여 네트워크 구축 및 추가 분석 수행
G = nx.Graph()
for combo, result in results:
    if result is not None:
        for sample in result:
            G.add_edge(combo[0], combo[1], weight=sample[2], sample=sample[1])


# 샘플별 유전자 집합 생성
sample_to_genes = {}
for u, v, data in G.edges(data=True):
    for sample in data['sample']:
        if sample not in sample_to_genes:
            sample_to_genes[sample] = set()
        sample_to_genes[sample].add(u)
        sample_to_genes[sample].add(v)

# 서브타입별 샘플 분류 subtype_to_samples
subtype_to_samples = {subtype: [] for subtype in subtypes}
for sample, subtype in LUAD_dict.items():
    if subtype in subtypes:
        subtype_to_samples[subtype].append(sample)

# 자카드 인덱스 계산 함수
def calculate_jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

# 모든 샘플 쌍에 대해 자카드 인덱스 계산
jaccard_indices = {}
for sample1, sample2 in combinations(sample_to_genes.keys(), 2):
    jaccard_indices[(sample1, sample2)] = calculate_jaccard(sample_to_genes[sample1], sample_to_genes[sample2])

# 자카드 인덱스 값만 리스트로 추출 (전체)
jaccard_values = list(jaccard_indices.values())


# 서브타입별 자카드 인덱스 계산 및 저장
subtype_jaccard_values = {subtype: [] for subtype in subtypes}
different_subtype_jaccard_values = []
for (sample1, sample2), jaccard_index in jaccard_indices.items():
    subtype1 = LUAD_dict.get(sample1)
    subtype2 = LUAD_dict.get(sample2)
    
    if subtype1 == subtype2 and subtype1 in subtypes:
        subtype_jaccard_values[subtype1].append(jaccard_index)
        
    elif subtype1 in subtypes and subtype2 in subtypes and subtype1 != subtype2:
        different_subtype_jaccard_values.append(jaccard_index)

# 모든 서브타입에 대해 같은 색상 사용
for subtype in subtypes:
    if subtype_jaccard_values[subtype]:
        sns.histplot(subtype_jaccard_values[subtype], bins=20, kde=True, color='red', label=f'{subtype} (within)', alpha=0.5)
        
# 서로 다른 서브타입 간 자카드 인덱스 히스토그램 추가
sns.histplot(different_subtype_jaccard_values, bins=20, kde=True, color='peachpuff', label='Between Different Subtypes')

# 0.5 축에 세로선 추가
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Jaccard Index 0.5')

plt.xlim(0.1,0.9)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.title('Distribution of Jaccard Index : Different subtypes 1 & same subtype n')
plt.xlabel('Jaccard Index')
plt.ylabel('Count')
plt.legend(title='Subtype Jaccard Index', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# 각 서브타입의 샘플 개수를 계산하고 출력
print("Number of subtype samples")
for subtype in subtypes:
    count = len(subtype_to_samples[subtype])  # 해당 서브타입에 대응되는 샘플 리스트의 길이 계산
    print(f"{subtype}:", count)
