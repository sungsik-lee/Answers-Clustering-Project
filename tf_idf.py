from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import re
import numpy as np
import warnings
from sklearn.preprocessing import Normalizer


def replace_nan_with_zero(matrix):
    # NaN 값을 0으로 대체
    matrix = np.nan_to_num(matrix, nan=0)
    return matrix


def minmax_matrix(matrix):
    min_max_matrix  = np.empty((0, len(matrix[0])))
    for i in range(len(matrix)):
        min_val = np.min(matrix[i])
        max_val = np.max(matrix[i])
        normalized_row = (matrix[i] - min_val) / (max_val - min_val)
        min_max_matrix = np.append(min_max_matrix, [normalized_row] , axis =0)
    return min_max_matrix

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def math_tfidf(n, norm, concept, nrange, num_df, text_df, answers, stop_tag, stop_word, kiwi, concept_list, decay):
    # 모든 경고 무시
    warnings.filterwarnings("ignore")
    
    # 텍스트에서 수식 파트와 자연어 파트를 따로따로 분리하는 코드 
    def is_korean_word(word):
        return re.search(r'[가-힣]', word) is not None

    def kiwi_tokenizer_num(text):
        tokens = kiwi.analyze(text)[0][0]
        
        token_doc = [word.form for word in tokens if not word.tag in stop_tag]
        token_doc = [word for word in token_doc if not word in stop_word]
        
        filtered_num_tokens = [token for token in token_doc if not is_korean_word(token)]
        return   filtered_num_tokens

    def kiwi_tokenizer_text(text):
        tokens = kiwi.analyze(text)[0][0]
        
        token_doc = [word.form for word in tokens if not word.tag in stop_tag]
        token_doc = [word for word in token_doc if not word in stop_word]
        
        filtered_text_tokens = [token for token in token_doc if is_korean_word(token)]
        return   filtered_text_tokens

    vectorizer1 = TfidfVectorizer(ngram_range=nrange, min_df = num_df, tokenizer=kiwi_tokenizer_num, lowercase=False, norm = None)  
    vectorizer2 = TfidfVectorizer(min_df = text_df, tokenizer=kiwi_tokenizer_text, lowercase=False, norm = None)
    tfidf_matrix_num = vectorizer1.fit_transform(answers)
    tfidf_matrix_num = tfidf_matrix_num.toarray()
    tfidf_matrix_text = vectorizer2.fit_transform(answers)
    tfidf_matrix_text = tfidf_matrix_text.toarray()


    a= pd.DataFrame(tfidf_matrix_num, columns=list(vectorizer1.get_feature_names_out()))
    a.to_excel(f'./Matrix/{n}번_num.xlsx', index=False)

    b= pd.DataFrame(tfidf_matrix_text, columns=list(vectorizer2.get_feature_names_out()))
    b.to_excel(f'./Matrix/{n}번_text.xlsx', index=False)
    # 각 행을 정규화한 결과
    
    
    
    if concept == True:
        #개념 용어를 강조하기 위한 count 벡터를 만들기 위한 term 
        vectorizer3 = CountVectorizer(min_df = text_df, tokenizer=kiwi_tokenizer_text, lowercase=False)
        bow_matrix_text = vectorizer3.fit_transform(answers)
        bow_matrix_text = bow_matrix_text.toarray() 
        
        for idx, feature in enumerate(list(vectorizer3.get_feature_names_out())):
            if feature in concept_list:
                bow_matrix_text  = bow_matrix_text.astype(float)
                column = bow_matrix_text[:, idx]
            
                # 변환 적용: a를 (2^a + 1) / 2^a로 변환 따라서 일단 개념어면 1보다 큰 값을 할당하지만, 그 개수가 많아질수록 강조포인트는 약간 낮아짐.
                transformed_column = (decay ** column + 2) / (decay ** column)
                
                # 원래 배열에 변환된 열을 다시 할당
                bow_matrix_text[:, idx] = transformed_column
            
            # 개념 단어가 아닌 열은 모두 0으로 처리    
            else:
                bow_matrix_text [:, idx] = 0


        # 개념 열 안에서 기존의 0으로 차있던 칸은 모두 2로 될것이니 이를 0으로 바꾸고 모든 0을 1로 바꿈 (text_matrix와 곱해주기 위함)
        bow_matrix_text[ bow_matrix_text == 2] = 0    
        bow_matrix_text[bow_matrix_text == 0] = 1


        #강조 상수를 곱해주어 최종 text matrix 사용
        tfidf_matrix_text = bow_matrix_text*tfidf_matrix_text
    
    total_feature = list(vectorizer1.get_feature_names_out()) + list(vectorizer2.get_feature_names_out())
    mid_tfidf = np.concatenate((tfidf_matrix_num, tfidf_matrix_text), axis=1)
    c= pd.DataFrame(mid_tfidf, columns=total_feature)
    c.to_excel(f'./Matrix/{n}번_unnormalized.xlsx', index=False) 

    if norm==True:
    # 각 행을 정규화
        tfidf_matrix_num = minmax_matrix(tfidf_matrix_num)
        tfidf_matrix_text = minmax_matrix(tfidf_matrix_text)
        tfidf_matrix_num = replace_nan_with_zero(tfidf_matrix_num)
        tfidf_matrix_text = replace_nan_with_zero(tfidf_matrix_text)
        
        
    total_feature = list(vectorizer1.get_feature_names_out()) + list(vectorizer2.get_feature_names_out())
    final_tfidf = np.concatenate((tfidf_matrix_num, tfidf_matrix_text), axis=1)
    d= pd.DataFrame(final_tfidf, columns=total_feature)
    d.to_excel(f'./Matrix/{n}번_normalized.xlsx', index=False)       
    print(final_tfidf.shape)
    return final_tfidf