import pymysql
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

connection = pymysql.connect(host='localhost',
                             user='root',  
                             password='990524', #비번
                             database='capd', #db이름
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

# 사용자 및 대회 ID
user_id = 1
contest_id = 28

try:
    with connection.cursor() as cursor:
        # profile 테이블에서 데이터 가져오기
        profile_sql = f"SELECT id, collaboration_count, my_time, desired_time, rate FROM profile WHERE id = '{user_id}'"
        cursor.execute(profile_sql)
        profile_row = cursor.fetchone()  # 해당 유저의 정보를 가져옴

          
        # 다른 유저들의 프로필 정보 저장할 리스트 초기화
        other_users_profiles = []

        if profile_row:
            # 참여 데이터 가져오기
            participation_sql = f"""
                SELECT psl.stack_list
                FROM participation_stack_list psl
                JOIN participation p ON psl.participation_id = p.id
                JOIN contest c ON p.contest_id = c.id
                WHERE p.user_id = '{user_id}'
                AND c.id = '{contest_id}';
            """
            cursor.execute(participation_sql)
            participation_rows = cursor.fetchall()

            # 스택 리스트 조합
            stack_lists = [participation['stack_list'] for participation in participation_rows]
            combined_stack_list = ', '.join(stack_lists) #원하는 스택 정보

            # 프로필 정보 저장 (id, 원하는 스택, 원하는 경력 횟수, 원하는 투자시간, 내 평점)
            profile_info = {
                'id': profile_row['id'],
                'stack_list': combined_stack_list,
                'collaboration_count': profile_row['collaboration_count'],
                'desired_time': profile_row['desired_time'],
                'rate': profile_row['rate']
            }
            print("Current User Profile:")
            print(profile_info)

            # 다른 유저들의 프로필 정보 가져오기 (id, 본인 스택, 경력횟수, 투자가능시간 ,별점)
            other_users_profile_sql = f"""
                                       SELECT p.id, GROUP_CONCAT(DISTINCT psl.stack_list) AS stack_list, COUNT(DISTINCT c.id) AS collaboration_count, p.my_time, p.rate
                                       FROM profile p
                                       JOIN profile_stack_list psl ON p.id = psl.profile_id
                                       LEFT JOIN career c ON p.id = c.profile_id
                                       WHERE p.id != '{user_id}'
                                       GROUP BY p.id, p.my_time, p.rate;
                                    """
            cursor.execute(other_users_profile_sql)
            other_users_profile_rows = cursor.fetchall()

            for row in other_users_profile_rows:
                other_user_profile_info = {
                    'id': row['id'],
                    'stack_list': row['stack_list'],
                    'collaboration_count': row['collaboration_count'],
                    'my_time': row['my_time'],
                    'rate': row['rate']
                }
                other_users_profiles.append(other_user_profile_info)

            # 결과 출력
            print("\nOther Users Profiles:")
            df = pd.DataFrame(other_users_profiles)
            print(df)

        else:
            print(f"No profile found for user with id {user_id}")
finally:
    # 연결 닫기
    connection.close()

# 1. 내가 원하는 상대의 기술스택, 내가 원하는 상대의 협업프로젝트 경험 횟수, 내가 원하는 투자가능시간, 내 평점
# 리스트로 저장
# 내가 원하는 상대의 기술스택
my_want_skill = combined_stack_list
want_exp_count = profile_row['collaboration_count']
my_investable_time = profile_row['desired_time']
my_raiting = profile_row['rate']

# 2. userid, 상대의 기술스택, 상대 협업프로젝트 경험 횟수, 상대가 실제 투자가능한 시간, 상대의 평점
# (상대방들은 여러명 -> 리스트 내에 튜플로 저장)

other_people = []

for row in other_users_profile_rows:
    other_person = {
        'id': row['id'],
        'stack_list': row['stack_list'],
        'collaboration_count': row['collaboration_count'],
        'my_time': row['my_time'],
        'rate': row['rate']
    }
    other_people.append(other_person)

# other_people을 원하는 형식으로 변경
other_people_formatted = [
    (
        person['id'],
        person['stack_list'],
        person['collaboration_count'],
        person['my_time'],
        person['rate']
    )
    for person in other_people
]
cnt_vect = CountVectorizer(token_pattern=r'[^,\s]+') # CountVectorizer객체. 나와 상대들의 기술스택을 각각 벡터화하는 데 이용.

# 나와 다른 사람의 기술스택을 벡터화한 행렬 저장.
skills_matrix = cnt_vect.fit_transform([my_want_skill] + [other_person[1] for other_person in other_people_formatted])
# print(skills_matrix) # CSR 매트릭스(0 제거 희소행렬)로 반환되어 단어 최초단어등장 위치를 기준으로 벡터화된 데이터임.
# print(type(skills_matrix))
print('단어장(벡터화된 토큰의 단어 정보. 실제로는 벡터값이 매핑되어 있습니다.)',cnt_vect.vocabulary_)

# 코사인 유사도 계산하기
cosine_similarities = cosine_similarity(skills_matrix)
print('여기까지의 유사도 (기본 코사인 유사도) : \n', cosine_similarities[0],end='\n-------------------------------\n') 
# 우리가 생각하는 코사인 유사도는 반환결과의 첫 번째 행(cosine_similarities[0])입니다. 첫 번째 요소(cosine_similarities[0][0])는 내 스택 vs 내 스택이고 그 다음 요소부터(cosine_similarities[0][1]) 내 스택 vs 상대1, (cosine_similarities[0][2])내 스택 vs 상대 2 ... 이런식이에요

# 상대방 별 가중치 계산하기
weighted_similarities = [] 
for i in range(1, len(other_people_formatted)+1) : # 0번째는 내가 원하는 스택과 내가 원하는 스택의 유사도임..
    similarity_score = cosine_similarities[0][i]
    print('나 vs {0}번째 상대방 기본 유사도'.format(i), similarity_score)
    
    exp_diff = abs(want_exp_count - other_people_formatted[i-1][2])  # 수정된 부분
    # 투자가능한 시간의 차
    inv_time_diff = abs(my_investable_time - other_people_formatted[i-1][3])  # 수정된 부분
    # 평점 차
    raiting_diff = abs(my_raiting - other_people_formatted[i-1][4])

    
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

# 코사인 유사도 시각화 - x축이 유사도, y축이 비교대상 스택 문자열인 가로 바그래프
# ----------------- 1. 각각 막대그래프 따로 그리기 (순위변화 보기 용이함) ------------------------
# matplotlib의 barh() 가로막대그래프
# X축 : 코사인 유사도
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
xData = cosine_similarities[0][1:]
# Y축 : 각 userId
yData = [str(other_person[0]) for other_person in other_people_formatted]

plt.barh(yData, xData, label='Cosine Similarity', color='r')

plt.ylabel('userId') # y축 이름
plt.xlabel('Cosine Similarity') # x축 이름
plt.title('Cosine Similarity by userId Horizental bar graph\n Before applying weight') # 그래프 제목
plt.grid()
# 각 유사도 값 그래프 바 옆에 표시하기
for index, value in enumerate(xData) :
    if value != 0.0 : # ha속성은 바 끝으로부터 어느쪽으로 텍스트를 보여줄건지 결정
        plt.text(value, index, str(value), ha='right')
    else : # 값이 0.0일때 왼쪽으로 표시하면 왼쪽에 userId가 겹쳐서 잘 안보임..
        plt.text(value, index, str(value), ha='left')

# 가중치 적용 이후 코사인 유사도 가로 바 그래프
xData = weighted_similarities
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 그래프
plt.barh(yData, xData, color='skyblue', label='Weighted Cosine Similarity')
plt.ylabel('user Id')
plt.xlabel('Weighted Cosine Similarity')
plt.title('Cosine Similarity by userId Horizental bar graph\nAfter applying weight')
plt.grid()
for index, value in enumerate(xData):
    if value != 0.0:
        plt.text(value, index, str(value), ha='right')
    else:
        plt.text(value, index, str(value), ha='left')
plt.show()

# ----------------- 2. 한번에 그리기 (가중치 적용 전후 얼마나 변화했는지 보기 용이함) ------------------------
xData = cosine_similarities[0][1:]  # 적용 전 코사인 유사도
xData_weighted = weighted_similarities  # 적용 후 코사인 유사도
yData = [str(other_person[0]) for other_person in other_people_formatted]  # userId 혹은 다른 식별자

# 그래프 그리기
plt.figure(figsize=(10, 6))  # 그래프 사이즈 설정

# 가로 막대 그래프 그리기 (적용 전)
plt.barh(np.arange(len(yData)), xData, color='skyblue', label='Before Weighted', height=0.4)

# 가로 막대 그래프 그리기 (적용 후)
plt.barh(np.arange(len(yData)) + 0.4, xData_weighted, color='orange', label='After Weighted', height=0.4)

# 그래프에 텍스트 표시
for i, value in enumerate(xData):
    if value != 0.0:
        plt.text(value, i, str(round(value, 2)), ha='right', va='center', fontsize=10)  # 적용 전 막대 오른쪽에 텍스트 표시
    else:
        plt.text(value, i, str(round(value, 2)), ha='left', va='center', fontsize=10)  # 값이 0.0일 때 왼쪽에 텍스트 표시
for i, value in enumerate(xData_weighted):
    if value != 0.0:
        plt.text(value, i + 0.4, str(round(value, 2)), ha='right', va='center', fontsize=10)  # 적용 후 막대 오른쪽에 텍스트 표시
    else:
        plt.text(value, i + 0.4, str(round(value, 2)), ha='left', va='center', fontsize=10)  # 값이 0.0일 때 왼쪽에 텍스트 표시

# 그래프 제목, 축 이름 설정
plt.title('Cosine Similarity Before and After Weighted')
plt.xlabel('Cosine Similarity')
plt.ylabel('User Id')
plt.yticks(np.arange(len(yData)) + 0.2, yData)  # y 축에 userId 표시
plt.grid(axis='x')  # x 축에만 그리드 표시
plt.legend()  # 범례 표시
plt.tight_layout()  # 그래프 간격 조정
plt.show()

# print(weighted_similarities)
# 상위 3명의 id 추출
top3_indices = np.argsort(weighted_similarities)[::-1][:3]
print('최종 순위 3명')
# 상위 3명의 정보 출력
for idx in top3_indices :
    print("상대방 id : ", other_people_formatted[idx][0], ', 최종 유사도 : ', weighted_similarities[idx])