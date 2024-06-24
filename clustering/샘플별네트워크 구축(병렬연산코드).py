import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import silhouette_samples, silhouette_score
import networkx as nx
from joblib import Parallel, delayed


# 엣지 쌍을 포함하는 파일 로드 (예시 경로, 실제 경로로 변경해야 함)
edge_pairs = pd.read_csv('/BiO/lee/TCGAdata/integrated_network.csv')
valid_edges = set(map(tuple, edge_pairs[['src', 'dest']].values))

valid_edges

file_paths = [
    '/BiO/lee/TCGAdata/Overlap_node(LIHC,KIRC).csv',
    
    
]

dataframes = [pd.read_csv(file, index_col='gene_name') for file in file_paths]

# 유전자 기준 하나의 데이터프레임으로 병합 (inner join으로 공통 유전자만 포함)
merged_data = pd.concat(dataframes, axis=1, join='inner')
# 'Unnamed: 0' 열을 데이터프레임에서 제거
merged_data.drop('Unnamed: 0', axis=1, inplace=True)

def remove_outliers(data, threshold=0.01):
    # 임계값보다 큰 값만 유지
    high_expression_mask = (data > threshold).all(axis=1)
    filtered_data = data[high_expression_mask]
    
    # IQR 계산
    #Q1 = filtered_data.quantile(0.25)
    #Q3 = filtered_data.quantile(0.75)
    #IQR = Q3 - Q1

    # IQR 기준으로 이상치를 필터링
    #lower_bound = Q1 - 1.5 * IQR
    #upper_bound = Q3 + 1.5 * IQR 

    #return filtered_data[(filtered_data >= lower_bound) & (filtered_data <= upper_bound)].dropna()
    return filtered_data


# 랜덤으로 유전자 개수 선택
#random_genes = random.sample(list(merged_data.index), 5)
#random_genes =['ENSG00000261420', 'ENSG00000162006', 'ENSG00000164398', 'ENSG00000099985', 'ENSG00000141867', 'ENSG00000226703', 'ENSG00000135956', 'ENSG00000228767', 'ENSG00000111262', 'ENSG00000138152', 'ENSG00000237961', 'ENSG00000249028', 'ENSG00000160973', 'ENSG00000181552', 'ENSG00000270077', 'ENSG00000233250', 'ENSG00000274514', 'ENSG00000115232', 'ENSG00000188801', 'ENSG00000182327', 'ENSG00000246422', 'ENSG00000176165', 'ENSG00000260087', 'ENSG00000243845', 'ENSG00000272034', 'ENSG00000225978', 'ENSG00000269707', 'ENSG00000274552', 'ENSG00000250544', 'ENSG00000257258', 'ENSG00000014138', 'ENSG00000101307', 'ENSG00000145864', 'ENSG00000128422', 'ENSG00000256250', 'ENSG00000145700', 'ENSG00000183793', 'ENSG00000104695', 'ENSG00000138379', 'ENSG00000229163', 'ENSG00000253716', 'ENSG00000113578', 'ENSG00000271626', 'ENSG00000120451', 'ENSG00000170043', 'ENSG00000264735', 'ENSG00000227737', 'ENSG00000248713', 'ENSG00000248837', 'ENSG00000267390']
#random_genes=['ENSG00000249028', 'ENSG00000248713']
#print(random_genes)

all_genes = list(merged_data.index)
# 유전자 리스트의 개수 출력
gene_count = len(all_genes)
print("Number of genes:", gene_count)
print(all_genes)

def plot_results2(data, labels, centers, combo):
    log_data = np.log2(1 + data) #로그2 변환
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(log_data.iloc[:, 0], log_data.iloc[:, 1], c=labels, cmap='viridis')

    # 클러스터 중심 표시
    #plt.scatter(centers[:, 0], centers[:, 1], s=50, c='red', marker='x', label='Centroids')

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

def plot_results(data, labels, centers, combo):
   
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')

    # 클러스터 중심 표시
    #plt.scatter(centers[:, 0], centers[:, 1], s=50, c='red', marker='x', label='Centroids')

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

def calculate_slope_intercept(cluster_data):
    """클러스터 데이터로부터 기울기와 절편을 계산하는 함수입니다."""
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
        selected_genes = merged_data.loc[list(combo)]
        preprocessed_data[combo] = remove_outliers(selected_genes.T, 0.01)
    except KeyError as e:
        print(f"KeyError: {e} - Skipping this gene pair.")
        
        
preprocessed_data


# 클러스터링 및 네트워크 구축을 병렬로 수행하는 함수
def process_combo(combo, data):
    remaining_samples = len(data)
    
    if remaining_samples < 2:
        print(f"Error: Not enough data points ({remaining_samples}) for {combo}")
        return None, None
    
    n_clusters = min(7, remaining_samples)
    bgm = BayesianGaussianMixture(n_components=n_clusters, max_iter=600, n_init=20,
                                  init_params='random', tol=1e-3, reg_covar=1e-4,
                                  covariance_type='diag')
    bgm.fit(data)
    labels = bgm.predict(data)
    
    unique_labels = np.unique(labels)
    results = []
    if len(unique_labels) > 1:
        silhouette_vals = silhouette_samples(data, labels)
        overall_silhouette = silhouette_score(data, labels)
        
        if overall_silhouette >= 0.4:
            for idx, label in enumerate(unique_labels):
                cluster_data = data[labels == label]
                cluster_silhouette = silhouette_vals[labels == label].mean()
                slope, intercept = calculate_slope_intercept(cluster_data)
                
                if len(cluster_data) > 1:
                    correlation_matrix = cluster_data.corr()
                    average_correlation = correlation_matrix.stack().mean()
                else:
                    average_correlation = np.nan
                
                if cluster_silhouette > overall_silhouette and 0.1 < abs(slope) < 45 and abs(average_correlation) > 0.3:
                    print(f"Overall Silhouette Score for {combo}: {overall_silhouette:.2f}")
                    print(f"Cluster {label}: Silhouette Score: {cluster_silhouette:.2f}, Slope: {slope:.2f}, "
                          f"Intercept: {intercept:.2f}, Pearson Correlation: {average_correlation:.2f}")
                    
                    samples_in_cluster = cluster_data.index.tolist()
                    results.append((combo, samples_in_cluster, abs(slope)))
    return combo, results

# 병렬 처리로 클러스터링 수행
results = Parallel(n_jobs=-1)(delayed(process_combo)(combo, data) for combo, data in preprocessed_data.items())

# 네트워크 구축
G = nx.Graph()
for combo, result in results:
    if result is not None:
        for sample in result:
            G.add_edge(combo[0], combo[1], weight=sample[2], sample=sample[1])

# 엣지 데이터 출력

for edge in G.edges(data=True):

    print(edge)

from collections import Counter
# 엣지 이름 카운트
edge_names = [d['edge_name'] for _, _, d in G.edges(data=True)]
edge_count = Counter(edge_names)

# 카운트 출력
print("\nEdge Counts:")
for edge_name, count in sorted(edge_count.items(), key=lambda item: item[1], reverse=True):
    print(f"{edge_name}: {count}")




# 네트워크 그래프 시각화
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # 노드 위치 결정
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]

nx.draw(G, pos, with_labels=True, node_size=700, font_size=15, width=weights) #width : weights 간선의 두께를 가중치에 비례하게 설정
plt.title("Network Graph of Genes with Sample Edges")
plt.show()

# 특정 샘플 'TCGA-B8-4146-01B-11R-1672-07'이 edge_name인 경우에만 필터링
sample_edge_name = 'TCGA-GJ-A3OU-01A-31R-A38B-07'
filtered_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['edge_name'] == sample_edge_name]

# 서브 그래프 생성
sub_graph = nx.Graph()
sub_graph.add_edges_from(filtered_edges)

# 서브 그래프 시각화
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(sub_graph, seed=42)  # 노드 위치 결정
weights = [d['weight'] for _, _, d in sub_graph.edges(data=True)]

# 간선 두께를 가중치에 비례하게 설정
nx.draw(sub_graph, pos, with_labels=True, node_size=500, font_size=10, width=weights, edge_color='blue')

plt.title(f"Network Graph of Genes with Sample Edge Name: {sample_edge_name}")
plt.show()


