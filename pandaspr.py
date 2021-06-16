import pandas as pd
import numpy as np
import matplotlib as plt

# s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

dates = pd.date_range('20130101', periods=6)
# print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
# print(df)

# df2 = pd.DataFrame({'A': 1, 'B': pd.Timestamp('20130102'), 'C': pd.Series(1, index=list(range(4)), dtype='float32'), 'D': np.array([3]*4, dtype='int32'), 'E': pd.Categorical(['test', 'train', 'text', 'train']), 'F': 'foo'})

# print(df2)

# print(df2.dtypes)

# print(df.tail(3), f'\n', df.tail(), f'\n', df.head())

# print(f'{df.index} \n {df.columns} \n {df.values} \n {df.describe}')

# print(df.T) # 데이터 전치

# print(df.sort_index(axis=1, ascending=False)) # 값 별로 정렬

# print(df.sort_values(by='B'))

# print(df['A'], f'\n', df[0:3], f'\n', df['20130102':'20130104'])  # 행을 분할하는 []을 통해 선택

# print(df.loc[dates[0]], f'\n', df.loc[:, ['A', 'B']], f'\n', df.loc['20130102':'20130104', ['A', 'B']])
# 라벨을 사용하여 여러축(데이터)를 얻음 / 양쪽 종단점을 포함한 라벨 슬라이싱 / 반환되는 객체의 차원축소

# print(df.loc['20130102', ['A', 'B']], f'\n', df.loc[dates[0]], f'\n', 'A', df.at[dates[0], 'A'])
# 스칼라값 얻음 / 스칼라 값을 더 빠르게 구함(앞과 동일) / 넘겨받은 정수의 위치를 기준으로 선택

# print(df.iloc[3], f'\n', df.iloc[3:5, 0:2], f'\n', df.iloc[[1, 2, 4], [0, 2]], f'\n', df.iloc[1:3, :])
# 정수로 표기된 슬라이스들을 통해 numpy/python과 유사하게 작동 / 정수로 표기된 위치값의 리스트를 통해 / 명시적으로 행을 나눌때 / 명시적으로 열을 나눌때

# print(df.iloc[:, 1:3], f'\n', df.iloc[1, 1], f'\n', df.iat[1, 1], f'\n', df[df.A > 0])
# 명시적으로 (특정한)값을 얻을 경우 / 스칼라값을 빠르게 얻는 방법(좌동) / 데이터 선택 위해 단일 열 값 사용 / Boolean 조건 충족하는 DF에서 값 선택

'''
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1  # 라벨에 의해 값을 설정
'''

# print(df[df > 0], f'\n', df2, f'\n', df2[df2['E'].isin(['two', 'four'])], f'\n', s1, f'\n')
# 필더링을 위한 메소드 isin() 사용 / / 새 열을 설정하면 인덱스 별로 데이터 자동 정렬


df.at[dates[0], 'A'] = 0  # 위치에 의해 값을 설정
df.iat[0, 1] = 0  # Numpy 배열을 사용한 할당에 의해 값을 설정
df.loc[:, 'D'] = np.array([5]*len(df))
# print(df)  # 위 설정대로 작동한 결과


'''
df2 = df.copy()
df2[df2 > 0] = -df2
'''

# print(df2)


# 4. Missing Data (결측치)


'''
Pandas는 결측치를 표현하기 위해 주로 np.nan 값을 사용합니다. 이 방법은 기본 설정값이지만 계산에는 포함되지 않습니다.
Missing data section을 참조하세요. Reindexing으로 지정된 축 상의 인덱스를 변경 / 추가 /삭제할 수 있습니다.
Reindexing은 데이터의 복사본을 반환합니다.
'''

df1 = df.reindex(index=dates[0: 4], columns=list(df.columns)+['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
# print(df1)

df1.dropna(how='any')  # 결측치를 가지고 있는 행 삭제
df1.fillna(value=5)  # 결측치를 채워넣음

pd.isna(df1)  # nan인 값에 boolean을 통한 표식 얻음
# 역자 주 : DF의 모든 값을 boolean 형태로 표시, nan인 값에만 True가 표시되게 하는 함수


# 5. Operation (연산)

# Stats(통계)

df.mean()  # 일반적으로 결측치 제외 후 연산, 기술통계 수행

df.mean(1)  # 다른축에서 동일한 연산 수행

'''
정렬이 필요하며, 차원이 다른 객체로 연산, pandas는 지정된 차원을 따라 자동으로 브로드 캐스팅됨
역자 주 : broadcast란 numpy에서 n차원이나 스칼라 값으로 연산 후 도출되는 결과의 규칙을 설명하는 것을 의미
'''

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)

df.sub(s, axis='index')

# Apply(적용)

df.apply(np.cumsum)  # 데이터에 함수를 적용

df.apply(lambda x: x.max() - x.min())

#  Histogrammin (히스토그래밍)

s = pd.Series(np.random.randint(0, 7, size=10))

s.value_counts()

# String Method (문자열 메소드)
'''
Series는 다음의 코드와 같이 문자열 처리 메소드 모음 (set)을 보유
이 모음은 배열의 각 요소를 쉽게 조작할 수 있도록 만들어주는 문자열의 속성에 포함
문자열의 패턴 일치 확인은 기본적으로 정규 표현식을 사용, 몇몇 경우 항상 정규 표현식을 사용함에 유의
'''

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()


# 6. Merge (병합)


'''
Concat (연결)
결합 (join) / 병합 (merge) 형태의 연산에 대한 인덱스, 관계 대수 기능을 위한 다양한 형태의 논리를 포함한 
Series, 데이터프레임, Panel 객체를 손쉽게 결합할 수 있도록 하는 다양한 기능을 pandas 에서 제공
'''

df = pd.DataFrame(np.random.randn(10, 4))

pieces = [df[:3], df[3:7], df[7:]]  # concat() 으로 pandas 객체를 연결
pd.concat((pieces))

#  JOIN (결합) SQL 방식으로 병합

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [4, 5]})


# Append (추가)

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])

s = df.iloc[3]
df.append(s, ignore_index=True)


# 7. Grouping (그룹화)

'''
그룹화는 다음 단계 중 하나 이상을 포함하는 과정. 
1. 몇몇 기준에 따라 여러 그룹으로 데이터를 분할(splitting)
2. 각 그룹에 독립적으로 함수를 적용 (applying)
3. 결과물들을 하나의 데이터 구조로 결합 (combining)
'''

# 여기부터 해야함
