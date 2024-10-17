from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
from spectral import *
from spectral2 import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

class SphericalKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-6):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Step 1: 데이터 정규화
        X = normalize(X)
        np.random.seed(100)  # 초기화

        # 무작위로 초기 클러스터 중심 선택
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centers_ = X[random_indices]

        for iteration in range(self.max_iter):
            # Step 3: 각 포인트를 가장 가까운 중심에 할당
            similarity = X.dot(self.centers_.T)
            labels = np.argmax(similarity, axis=1)

            # Step 4: 새로운 클러스터 중심 계산
            new_centers = np.zeros_like(self.centers_)
            for i in range(self.n_clusters):
                points_in_cluster = X[labels == i]
                if points_in_cluster.shape[0] > 0:
                    new_centers[i] = points_in_cluster.mean(axis=0)

            # Step 5: 중심을 다시 정규화
            new_centers = normalize(new_centers)

            # 변화가 tol 이하라면 수렴했다고 간주하고 종료
            if np.linalg.norm(self.centers_ - new_centers) < self.tol:
                break

            self.centers_ = new_centers

        self.labels_ = labels
        return self.labels_

    def predict(self, X):
        similarity = X.dot(self.centers_.T)
        return np.argmax(similarity, axis=1)

    def calculate_inertia(self, X):
        """클러스터 내의 거리를 계산하여 성능 측정"""
        X = normalize(X)  # X도 정규화
        similarity = X.dot(self.centers_.T)
        labels = np.argmax(similarity, axis=1)
        distances = 1 - np.max(similarity, axis=1)  # 거리 계산 (1 - 유사도)
        inertia = np.sum(distances)  # 거리의 합이 inertia
        return inertia
    
def find_best_kmeans(X, n_clusters, n_init):
    best_inertia = np.inf
    best_model = None
    best_labels = None

    for i in range(n_init):
        model = SphericalKMeans(n_clusters=n_clusters)
        labels = model.fit(X)
        
        # inertia 계산
        inertia = model.calculate_inertia(X)
        
        # inertia가 더 작으면 업데이트
        if inertia < best_inertia:
            best_inertia = inertia
            best_model = model
            best_labels = labels
    
    return  best_labels


def find_best_lda(X, num_topics, n_init):
    best_perplexity = np.inf
    best_model = None
    best_labels = None

    for i in range(n_init):
        # LDA 모델 학습
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=(i+50), max_iter=50, learning_method='batch')

        # LDA 모델 피팅
        lda.fit(X)

        # Perplexity 계산
        perplexity = lda.perplexity(X)

        # 가장 낮은 perplexity 값을 가진 모델 선택
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_model = lda
            # 각 문서에 대해 가장 높은 확률을 가진 주제를 클러스터로 할당
            best_labels = np.argmax(lda.transform(X), axis=1)

    return best_labels



def evaluate_metrics(true_labels, predicted_labels):
    # 정확도 계산
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # 정밀도 계산
    precision = precision_score(true_labels, predicted_labels)
    
    # 재현율 계산
    recall = recall_score(true_labels, predicted_labels)
    
    # F1-Score 계산
    f1 = f1_score(true_labels, predicted_labels)
    # 결과 출력
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return accuracy, precision, recall, f1
    
def mapping(true_labels, predicted_labels):
    # 혼동 행렬 계산
    confusion = confusion_matrix(true_labels, predicted_labels)
    
    # 헝가리안 알고리즘(최대 일치 문제 해결)을 사용하여 최적의 매핑 찾기
    row_ind, col_ind = linear_sum_assignment(-confusion)
    
    # 최적의 매핑을 기반으로 라벨을 재정렬
    mapped_labels = np.zeros_like(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        mapped_labels[predicted_labels == j] = i
        
    return mapped_labels
            
def get_dominant_topic(doc_topic_distribution):
    return np.argmax(doc_topic_distribution, axis=1)

def process_binary_list(binary_list):
    # 첫 번째 원소 확인
    if binary_list[0] == 0:
        return binary_list
    else:
        # 리스트의 모든 값을 반전 (0 -> 1, 1 -> 0)
        inverted_list = [1 - x for x in binary_list]
        return inverted_list
    
def clustering(before_tf_idf, math_tf_idf, n_clusters, method, true_label):    
    
    if method == 'cos_dis_kmeans':

        
        labels1 = find_best_kmeans(before_tf_idf, n_clusters, n_init=30)
        labels2 = find_best_kmeans(math_tf_idf, n_clusters, n_init=30)



    elif method == 'basic_spectral':
    
        cos_sim1 = cosine_similarity(before_tf_idf)
        cos_sim2 = cosine_similarity(math_tf_idf)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels1 = spectral.fit_predict(cos_sim1)
        labels2 = spectral.fit_predict(cos_sim2)

    elif method == 'tuned_spectral':
    
        cos_sim1 = spec_cluster2(before_tf_idf)
        cos_sim2 = spec_cluster2(math_tf_idf)
        

        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels1 = spectral.fit_predict(cos_sim1)
        labels2 = spectral.fit_predict(cos_sim2)
    

    
    elif method == 'LDA':
        labels1 = find_best_lda(before_tf_idf, num_topics=n_clusters, n_init =30)
        labels2 = find_best_lda(math_tf_idf, num_topics=n_clusters, n_init = 30)


        


    before_ari = adjusted_rand_score(true_label, labels1)
    after_ari = adjusted_rand_score(true_label, labels2)
    

    before_ami = adjusted_mutual_info_score(true_label, labels1)
    after_ami = adjusted_mutual_info_score(true_label, labels2)




    print(method)
    print(true_label)
    print(labels1)
    print(labels2)
    print(f"before_rand_index : {before_ari:.3f}")
    print(f"after_rand_index : {after_ari:.3f}")
    print('\n')
    print(f"before Adjusted Mutual Information: {before_ami:.3f}")
    print(f"after Adjusted Mutual Information: {after_ami:.3f}")
    print('\n')
    