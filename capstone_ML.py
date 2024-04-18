# 캡스톤 머신러닝 컨텐츠기반추천알고리즘 - 코사인유사도와 가중치 적용
# v0.1
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns

# 1. 내가 원하는 상대의 기술스택, 내가 원하는 상대의 협업프로젝트 경험 횟수, 내가 원하는 투자가능시간, 내 평점
# 리스트로 저장
# 내가 원하는 상대의 기술스택
my_want_skill = 'C++, MySQL'
want_exp_count = 2
my_investable_time = 40
my_raiting = 4.5

# 2. userid, 상대의 기술스택, 상대 협업프로젝트 경험 횟수, 상대가 실제 투자가능한 시간, 상대의 평점
# (상대방들은 여러명 -> 리스트 내에 튜플로 저장)

other_people = [(2, 'C++, SQLite, Unreal Engine', 2, 50, 5.0),
                (3, 'Java, Springboot', 3, 40, 4.2),
                (16, 'Python, Django, php', 4, 30, 4.5),
                (4, 'Kotlin, Android SDK, Room Persistence Library', 2, 35, 3.8),
                (7, 'Unity, C#', 1, 45, 0),
                (22, 'Python, TensorFlow', 2, 35, 4.0),
                (34, 'JS(JavaScript), MySQL, Seaborn, Python, Node.js', 2, 4, 4.5),
                (25, 'Photon Unity Networking(PUN), MySQL, C++, unreal engine', 3, 50, 4.3)]

cnt_vect = CountVectorizer(token_pattern=r'[^,\s]+') # CountVectorizer객체. 나와 상대들의 기술스택을 각각 벡터화하는 데 이용.

# 나와 다른 사람의 기술스택을 벡터화한 행렬 저장.
skills_matrix = cnt_vect.fit_transform([my_want_skill] + [other_person[1] for other_person in other_people])
# print(skills_matrix) # CSR 매트릭스(0 제거 희소행렬)로 반환되어 단어 최초단어등장 위치를 기준으로 벡터화된 데이터임.
# print(type(skills_matrix))
print('단어장(벡터화된 토큰의 단어 정보. 실제로는 벡터값이 매핑되어 있습니다.)',cnt_vect.vocabulary_)

# 코사인 유사도 계산하기
cosine_similarities = cosine_similarity(skills_matrix)
print('여기까지의 유사도 (기본 코사인 유사도) : \n', cosine_similarities[0],end='\n-------------------------------\n') 
# 우리가 생각하는 코사인 유사도는 반환결과의 첫 번째 행(cosine_similarities[0])입니다. 첫 번째 요소(cosine_similarities[0][0])는 내 스택 vs 내 스택이고 그 다음 요소부터(cosine_similarities[0][1]) 내 스택 vs 상대1, (cosine_similarities[0][2])내 스택 vs 상대 2 ... 이런식이에요

# 상대방 별 가중치 계산하기
weighted_similarities = [] 
for i in range(1, len(other_people)+1) : # 0번째는 내가 원하는 스택과 내가 원하는 스택의 유사도임..
    similarity_score = cosine_similarities[0][i]
    print('나 vs {0}번째 상대방 기본 유사도'.format(i), similarity_score)
    # 사용자와 상대방의 협업프로젝트경험횟수 차
    exp_diff = abs(want_exp_count - other_people[i-1][2])
    
    # 투자가능한 시간의 차
    inv_time_diff = abs(my_investable_time - other_people[i-1][3])

    # 평점 차
    raiting_diff = abs(my_raiting - other_people[i-1][4])

    # 협업프로젝트경험 가중치 계산
    if exp_diff == 0 :
        exp_weight = 1.2
    elif exp_diff >= 10 :
        exp_weight = 0.8
    else :
        exp_weight = 1.2 - (exp_diff / 10) * 0.4
    
    # 투자가능한 시간 가중치 계산
    if inv_time_diff >= 15 :
        inv_time_weight = 0.9
    elif inv_time_diff == 0:
        inv_time_weight = 1.1
    else :
        inv_time_weight = 1.1 - (inv_time_diff / 15) * 0.2
    
    # 평점 가중치 계산
    if raiting_diff == 0 :
        raiting_weight = 1.05
    else :
        raiting_weight = 1.05 - (raiting_diff / 5) * 0.1

    # 총 가중치 계산
    total_weight = exp_weight * inv_time_weight * raiting_weight

    # 가중치 적용한 최종 유사도
    current_weighted_similarity = similarity_score * total_weight
    print('나 vs {0}번째 상대방의 가중치 적용된 최종 유사도'.format(i), current_weighted_similarity)
    # 유사도 저장 후 다음 반복(다음 사람과 비교)
    weighted_similarities.append(current_weighted_similarity)
    print('--------------------------------------')

# print(weighted_similarities)
# 상위 3명의 id 추출
top3_indices = np.argsort(weighted_similarities)[::-1][:3]
print('최종 순위 3명')
# 상위 3명의 정보 출력
for idx in top3_indices :
    print("상대방 id : ", other_people[idx][0], ', 최종 유사도 : ', weighted_similarities[idx])