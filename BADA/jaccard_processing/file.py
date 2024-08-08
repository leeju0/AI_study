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
from scipy import stats

# 엣지 쌍을 포함하는 파일 로드 (예시 경로, 실제 경로로 변경해야 함)
edge_pairs = pd.read_csv('/BiO/lee/TCGAdata/integrated_network.csv')
valid_edges = set(map(tuple, edge_pairs[['src', 'dest']].values))


# 파일 로드
file_path = '/BiO/lee/TCGAdata/STAD.csv'

# gene : 18437 / sample : 1118
df = pd.read_csv(file_path)
# 파일 로드
df = pd.read_csv(file_path, index_col='gene_name').drop('Unnamed: 0', axis=1)


#이상치 제거
def remove_outliers(data, threshold=0.1):
    # 임계값보다 큰 값만 유지
    high_expression_mask = (data > threshold).all(axis=1)
    filtered_data = data[high_expression_mask]
    
    return filtered_data


#유전자 : 18437 개
all_genes = list(df.index)


def calculate_slope_intercept(cluster_data):
    
    x = cluster_data.iloc[:, 0].values
    y = cluster_data.iloc[:, 1].values
    n = len(x)

    # 필요한 합계 계산
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    # 기울기와 절편 계산
    denominator = n * sum_x2 - sum_x**2
    if denominator != 0:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
    else:
        # 모든 x 값이 같은 경우, 기울기는 정의할 수 없고 절편은 y 값의 평균을 사용
        slope = 0  # 기울기를 0이나 다른 특정 값으로 처리할 수 있음
        intercept = np.mean(y)  # y의 평균값을 절편으로 사용

    return slope, intercept


# 유효한 유전자 쌍만을 포함하는 새로운 리스트 생성
filtered_valid_edges = [combo for combo in valid_edges if combo[0] in all_genes and combo[1] in all_genes]

# 유효한 유전자 쌍 사전에 대해 데이터 전처리 및 저장
preprocessed_data = {}
for combo in filtered_valid_edges:
    try:
        selected_genes = df.loc[list(combo)]
        preprocessed_data[combo] = remove_outliers(selected_genes.T, 0.1)
    except KeyError as e:
        print(f"KeyError: {e} - Skipping this gene pair.")
        
        

# 클러스터링 및 네트워크 구축을 병렬로 수행하는 함수
def process_combo(combo, data):
    remaining_samples = len(data)
    
    if remaining_samples < 2:
        #print(f"Error: Not enough data points ({remaining_samples}) for {combo}")
        return None, None
    
    n_clusters = min(7, remaining_samples)
    bgm = BayesianGaussianMixture(n_components=n_clusters, max_iter=600, n_init=20,
                                  init_params='random', tol=1e-3, reg_covar=1e-4,
                                  covariance_type='diag')
    bgm.fit(data)
    labels = bgm.predict(data)
    
    unique_labels = np.unique(labels) #클러스터링 결과에서 각 샘플에 대한 labels를 사용하여 고유한 클러스터 라벨(개별 클러스터)을 추출
    
    results = [] 
    if len(unique_labels) > 1: #고유한 클러스터 라벨(개별 클러스터)의 개수가 1보다 큰지 확인함
        silhouette_vals = silhouette_samples(data, labels) #각 샘플에 대한 실루엣 값을 계산
        overall_silhouette = silhouette_score(data, labels) #전체 데이터셋에 대한 평균 실루엣 값을 계산
        
        if overall_silhouette >= 0.4:
            for idx, label in enumerate(unique_labels): #각 고유 클러스터 라벨(개별 클러스터)에 대해 반복
                cluster_data = data[labels == label] #현재 클러스터(ex, 개별 클러스터 1, 개별 클러스터 2, ...) 라벨에 속하는 샘플을 추출
                cluster_silhouette = silhouette_vals[labels == label].mean() #현재 개별 클러스터에 속하는 샘플들의 평균 실루엣 점수 계산
                slope, intercept = calculate_slope_intercept(cluster_data) #현재 개별 클러스터에 대해 기울기와 절편 계산

                if len(cluster_data) > 1: #개별 클러스터 데이터 포인트가 1보다 많은지 확인
                    x = cluster_data.iloc[:, 0] #개별 클러스터 데이터프레임 첫 번째 열(gene1)을 추출
                    y = cluster_data.iloc[:, 1] #개별 클러스터 데이터프레임 두 번째 열(gene2)을 추출
                    #row는 sample이다.
                    correlation, p_value = stats.pearsonr(x, y) #개별 클러스터의 pcc와 p-value 계산
                else:
                    correlation, p_value = np.nan, np.nan
                
                if cluster_silhouette > overall_silhouette and 0.1 < abs(slope) < 45 and abs(correlation) >= 0.3  and p_value <= 0.05:
                    #print(f"Overall Silhouette Score for {combo}: {overall_silhouette:.2f}")
                    #print(f"Cluster {label}: Silhouette Score: {cluster_silhouette:.2f}, Slope: {slope:.2f}, "
                    #      f"Intercept: {intercept:.2f}, Pearson Correlation: {average_correlation:.2f}")
                    
                    samples_in_cluster = cluster_data.index.tolist() #현재 클러스터에 속하는 샘플의 인덱스를 리스트로 변환
                    results.append((combo, samples_in_cluster)) #유전자쌍 , 유의미한 클러스터에 속하는 샘플 인덱스
    return combo, results


results = Parallel(n_jobs=5)(
    delayed(process_combo)(combo, data) for combo, data in preprocessed_data.items()
    )

# 결과를 pickle 파일로 저장
with open('pcc0.3_sil0.4_th0.1_pvalue0.05_STAD.pkl', 'wb') as f:
    pickle.dump(results, f)

