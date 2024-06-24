"""
 코드 요약 : TPM 데이터를 이용하여 bgmm클러스터링을 수행하고, 결과를 기반으로 네트워크를 구축하여 두 유전자 간 연관성을 그래프로 시각화
 
 이상치 처리 임계값 : 0.01
 모델 학습 : bgmm, 유효한 엣지 쌍에 대해서만 클러스터링 수행
 모델 하이퍼 파라미터 : max_iter : 600, ax_iter=600, n_init=20, init_params='random', tol=1e-3, reg_covar=1e-4, covariance_type='diag'
 모델 평가 : 실루엣 점수, pcc, slope
 유의미한 클러스터 조건 : 실루엣> 0.4, |corr| > 0.3, 0.1 < |slope| < 45
 유의미한 클러스터에 속하는 샘플을 엣지로 하고 이때 초기 가중치는 해당 샘플이 속하는 클러스터의 slope
 전체 네트워크 : sample을 edge로 하고 gene을 node로 하는 전체 네트워크
 서브 네트워크 : 전체 네트워크에서 edge가 "특정 샘플"인 경우에 대해서만 출력하는, 샘플별 네트워크


"""





#라이브러리 로드
import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import silhouette_samples, silhouette_score
import networkx as nx
from collections import Counter

#데이터 로드 및 전처리
dataframes = [pd.read_csv('/BiO/lee/TCGAdata/Overlap_node(LIHC,KIRC).csv', index_col='gene_name')] #간암과 신장암에 대한 TPM
edge_pairs = pd.read_csv('/BiO/lee/TCGAdata/integrated_network.csv') # 유효한 유전자 쌍
valid_edges = set(map(tuple, edge_pairs[['src', 'dest']].values)) #유효한 엣지 쌍을 튜플로 저장

# 유전자를 기준으로 데이터 프레임 병합 (공통 유전자만 유지)
merged_data = pd.concat(dataframes, axis=1, join='inner')
merged_data.drop('Unnamed: 0', axis=1, inplace=True) # 'Unnamed: 0' 열을 데이터프레임에서 제거


#이상치 제거
def remove_outliers(data, threshold=0.01):
    # 임계값 0.01 보다 큰 값만 유지
    high_expression_mask = (data > threshold).all(axis=1)
    filtered_data = data[high_expression_mask]
    
    return filtered_data

#유전자 목록 : 18037
all_genes = list(merged_data.index)

#로그2 변환 클러스터링 결과 출력
def plot_results2(data, labels, centers, combo):
    log_data = np.log2(1 + data) #로그2 변환
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(log_data.iloc[:, 0], log_data.iloc[:, 1], c=labels, cmap='viridis')
    
    plt.xlabel(f' {combo[0]}')
    plt.ylabel(f' {combo[1]}')
    plt.title(f'Filtered BGMM {combo[0]} and {combo[1]}')
    plt.colorbar(label='Cluster')
    plt.xlim(0.0, plt.xlim()[1])
    plt.ylim(0.0, plt.ylim()[1])

    
    handles, legend_labels = scatter.legend_elements()
    cluster_labels = [f"Cluster {i+1}" for i in range(len(np.unique(labels)))]
    legend1 = plt.legend(handles, cluster_labels, title="Clusters")
    plt.gca().add_artist(legend1)

    plt.show()

# orginal tpm에 대한 클러스터링 결과 출력  
def plot_results(data, labels, centers, combo):
   
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')

    plt.xlabel(f' {combo[0]}')
    plt.ylabel(f' {combo[1]}')
    plt.title(f'Filtered BGMM {combo[0]} and {combo[1]}')
    plt.colorbar(label='Cluster')
    plt.xlim(0.0, plt.xlim()[1])
    plt.ylim(0.0, plt.ylim()[1])

    # 범례 추가
    handles, legend_labels = scatter.legend_elements()
    cluster_labels = [f"Cluster {i+1}" for i in range(len(np.unique(labels)))]
    legend1 = plt.legend(handles, cluster_labels, title="Clusters")
    plt.gca().add_artist(legend1)

    plt.show()

#slope, intercept 계산 함수
def calculate_slope_intercept(cluster_data):
    
    x = cluster_data.iloc[:, 0].values
    y = cluster_data.iloc[:, 1].values
    n = len(x)

   
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
        slope = 0  
        intercept = np.mean(y)  

    return slope, intercept

# 유효한 유전자 쌍만을 포함하는 새로운 리스트 생성
filtered_valid_edges = [combo for combo in valid_edges if combo[0] in all_genes and combo[1] in all_genes]

# bgmm클러스터링 및 네트워크 구축
G = nx.Graph()
for combo in filtered_valid_edges:
    
    #유전자 쌍 정보를 list(combo)로 저장
    selected_genes = merged_data.loc[list(combo)].T
    
    #이상치 제거
    filtered_selected_genes = remove_outliers(selected_genes, 0.01)
    
    remaining_samples = len(filtered_selected_genes)
    
    #클러스터링 수행할 수 없는 경우, 에러 출력
    if remaining_samples < 2:
        print(f"Error: Not enough data points ({remaining_samples}) for {combo}")
        continue
    
    #초기 설정 클러스터 수 7
    n_clusters = min(7, remaining_samples)
    
    #bgmm 학습
    bgm = BayesianGaussianMixture(n_components=n_clusters, max_iter=600, n_init=20,
                                  init_params='random', tol=1e-3, reg_covar=1e-4,
                                  covariance_type='diag')
    bgm.fit(filtered_selected_genes)
    
    #cluster labeling
    labels = bgm.predict(filtered_selected_genes)
    
    #전체 클러스터링 결과의 실루엣 계수와, 각 클러스터에 대해 (실루엣,상관계수,slope intercept 계산)
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        silhouette_vals = silhouette_samples(filtered_selected_genes, labels)
        overall_silhouette = silhouette_score(filtered_selected_genes, labels)
        
        
        if overall_silhouette >= 0.4:
            for idx, label in enumerate(unique_labels):
                cluster_data = filtered_selected_genes[labels == label]
                cluster_silhouette = silhouette_vals[labels == label].mean()
                slope, intercept = calculate_slope_intercept(cluster_data)
                
                if len(cluster_data) > 1:
                    correlation_matrix = cluster_data.corr()
                    average_correlation = correlation_matrix.stack().mean()
                else:
                    average_correlation = np.nan
                
                #아래 조건을 만족하는 클러스터를 "유의미한 클러스터" 라 간주하고 , 클러스터 정보를 추가하고, 해당 클러스터에 속하는 샘플을 두 유전자의 엣지로 하여 추가함
                if cluster_silhouette > overall_silhouette and 0.1 < abs(slope) < 45 and abs(average_correlation) > 0.3:
                    print(f"Overall Silhouette Score for {combo}: {overall_silhouette:.2f}")
                    print(f"Cluster {label}: Silhouette Score: {cluster_silhouette:.2f}, Slope: {slope:.2f}, "
                          f"Intercept: {intercept:.2f}, Pearson Correlation: {average_correlation:.2f}")
                    
                    
                    samples_in_cluster = cluster_data.index.tolist()
                    for sample in samples_in_cluster:
                        G.add_edge(combo[0], combo[1], weight=abs(slope), sample=sample) #sample : 샘플 이름

# 모든 엣지 출력
for edge in G.edges(data=True):
    print(edge)

# 샘플별로, 유의미한 클러스터에  몇 번 속하는지 카운트하고 출력
edge_names = [d['edge_name'] for _, _, d in G.edges(data=True)]
edge_count = Counter(edge_names)

print("\nEdge Counts:")
for edge_name, count in sorted(edge_count.items(), key=lambda item: item[1], reverse=True):
    print(f"{edge_name}: {count}")
    


# 전체 네트워크 시각화
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42) 
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]

nx.draw(G, pos, with_labels=True, node_size=700, font_size=15, width=weights) #width : weights 간선의 두께를 가중치에 비례하게 설정
plt.title("Network Graph of Genes with Sample Edges")
plt.show()


# 특정 샘플 'TCGA-B8-4146-01B-11R-1672-07'에 대한 샘플별 네트워크 시각화
sample_edge_name = 'TCGA-GJ-A3OU-01A-31R-A38B-07'
filtered_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['edge_name'] == sample_edge_name]

sub_graph = nx.Graph()
sub_graph.add_edges_from(filtered_edges)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(sub_graph, seed=42)  # 노드 위치 결정
weights = [d['weight'] for _, _, d in sub_graph.edges(data=True)]
nx.draw(sub_graph, pos, with_labels=True, node_size=500, font_size=10, width=weights, edge_color='blue')
plt.title(f"Network Graph of Genes with Sample Edge Name: {sample_edge_name}")
plt.show()
