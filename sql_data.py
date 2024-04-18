import pymysql
import pandas as pd
#import json
# JSON 파일에서 contest_id와 user_id 읽기
#with open('data.json', 'r') as json_file:
#    data = json.load(json_file)
#    contest_id = data['contestId'] # 공모전 id
#    user_id = data['userId'] # 유저 id -> 추천 리스트 요청한 user

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='db12',
                             database='capd',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
user_id =1
contest_id=1
try:
    with connection.cursor() as cursor:
        # profile 테이블에서 데이터 가져오기
        profile_sql = f"SELECT id, collaboration_count, my_time, desired_time, rate FROM profile WHERE id = '{user_id}'"
        cursor.execute(profile_sql)
        profile_row = cursor.fetchone()  # 해당 유저의 정보를 가져옴

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
            combined_stack_list = ', '.join(stack_lists)

            # 프로필 정보 저장
            profile_info = {
                'id': profile_row['id'],
                'stack_list': combined_stack_list,
                'collaboration_count': profile_row['collaboration_count'],
                'desired_time': profile_row['desired_time'],
                'rate': profile_row['rate']
            }

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
            combined_stack_list = ', '.join(stack_lists)

            # 프로필 정보 저장
            profile_info = {
                'id': profile_row['id'],
                'stack_list': combined_stack_list,
                'collaboration_count': profile_row['collaboration_count'],
                'desired_time': profile_row['desired_time'],
                'rate': profile_row['rate']
            }
            print("Current User Profile:")
            print(profile_info)

            # 다른 유저들의 프로필 정보 가져오기
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
            other_users_profile_df = pd.DataFrame(other_users_profiles)
            print(other_users_profile_df)

        

        else:
            print(f"No profile found for user with id {user_id}")
finally:
    # 연결 닫기
    connection.close()