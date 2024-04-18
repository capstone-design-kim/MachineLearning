import re

# 입력 텍스트
text = "C++, C#, Python, Java"

# 정규표현식 패턴
pattern = r"\b\w+\b|C\+\+|C#"

# 정규표현식을 사용하여 단어 추출
words = re.findall(pattern, text)

# 추출된 단어 출력
print(words)
