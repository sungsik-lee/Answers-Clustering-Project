from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def digraph_to_undirected(digraph_matrix):
        n = len(digraph_matrix)
        undirected_matrix = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if digraph_matrix[i][j] == 1:
                    undirected_matrix[i][j] = 1
                    undirected_matrix[j][i] = 1

        return np.array(undirected_matrix)
    
def spec_cluster(tfidf):
    cos_distances_total =cosine_similarity(tfidf)
    num = len(tfidf)
    cos_distances_total= np.round(cos_distances_total, decimals=4)
    cos_distances_total = cos_distances_total- np.eye(num)
    # np.set_printoptions(precision=5, suppress=True)
    # print(cos_distances_total)
    
    row_means = (num /(num-1)) *np.mean(cos_distances_total, axis=1)

    # 0과 1로 변환된 대칭 행렬 생성
    final_similarity_matrix = np.zeros_like(cos_distances_total)

    # 각 원소를 비교하여 0과 1로 변환
    for i in range(cos_distances_total.shape[0]):
        for j in range(cos_distances_total.shape[1]):
            if cos_distances_total[i, j] > row_means[i]:
                final_similarity_matrix[i, j] = 1
            else:
                final_similarity_matrix[i, j] = 0        
    

    
    final_similarity_matrix2 = digraph_to_undirected(final_similarity_matrix)
    final_similarity_matrix3 = final_similarity_matrix2 + np.eye(num)
    
    return final_similarity_matrix3



