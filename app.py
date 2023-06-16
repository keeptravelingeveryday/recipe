from typing import List, Any

from flask import Flask
from flask import request

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import sys

app = Flask(__name__)

path = 'C:\\Users\\Playdata\\Desktop\\recipe'
recipe_db = pd.read_csv(os.path.join(path, 'recipe.csv'), encoding='utf-8')
#레시피번호를 인덱스로 설정
recipe_db.set_index("레시피번호", inplace=True)
recipe_db = recipe_db.drop_duplicates(subset=['음식이름'])
# recipe_db 데이터프레임에서 ["레시피번호","방법별","상황별","종류별","식재료명"] 컬럼만 리턴해서
# tfidf_df 데이터 프레임에 저장
tfidf_df = recipe_db[["음식이름","방법별", "상황별", "종류별", "식재료명"]]
print('tfidf_df.index=',tfidf_df.index)
print('len(tfidf_df)=',len(tfidf_df))
# tfidf_df 데이터프레임의 data 컬럼의 단어들의 Tfidf를 계산할 객체 vectorizer 생성
# `min_df`는 단어가 포함된 문서의 최소 개수를 설정
# norm 매개 변수를 사용하여 벡터의 정규화 방법을 지정, "l2"는 유클리드 거리 정규화
vectorizer = TfidfVectorizer(min_df=1, norm="l2")


#유통기한 임박한 식재료 추천
@app.route('/survey', methods=['POST'])
def survey():
    category1 = request.form.get('category1', "입력값 없음")
    print('category1=', category1)
    category2 = request.form.get('category2', "입력값 없음")
    print('category2=', category2)
    category3 = request.form.get('category3', "입력값 없음")
    print('category3=', category3)
    my_data1 = request.form.getlist('my_data1')
    print(my_data1)
    my_data1 = " ".join(my_data1)
    favorite_list = request.form.getlist('favorite_list')

    my_data1 = [my_data1 + " " +" ".join(favorite_list) ]
    print("my_data1=",my_data1)

    #print('my_data1=', my_data1)
    #print('favorite_list=', favorite_list)

    #favorite1 = " ".join(favorite_list)

    #my_data1 = [my_data1 + favorite1]
    #print("my_data1=", my_data1)

    #전역변수 불러옴.
    global tfidf_df

    # if favorite_list is None:
    #     # 카테고리 설정
    #     tfidf_df = tfidf_df[
    #         (filtered_df["종류별"] == category1) | (filtered_df["종류별"] == category2)
    #         | (filtered_df["종류별"] == category3)
    #         ]
    # else:
    #     print("recipe_db[recipe_db['음식이름'].str.contains('|'.join(favorite_list))]=", recipe_db[recipe_db["음식이름"].str.contains('|'.join(favorite_list))])
    #     # recipe_db에서 favorite_list에 해당하는 레시피들의 인덱스 추출
    #     recipe_index = recipe_db[recipe_db["음식이름"].str.contains('|'.join(favorite_list))].index
    #
    #     # recipe_index를 인덱스로 하는 새로운 DataFrame 생성
    #     filtered_df = tfidf_df.loc[recipe_index]
    #
    #     # 추출한 인덱스가 filtered_df의 인덱스 범위 내에 없는 경우,
    #     # #filtered_df와 recipe_db에서 해당 인덱스와 일치하는 값을 추출하여 filtered_df에 추가
    #     missing_index = [i for i in recipe_index if i not in filtered_df.index]
    #     if missing_index:
    #         missing_recipes = recipe_db.loc[missing_index]
    #         filtered_df = pd.concat([filtered_df, missing_recipes], axis=0)

        # 카테고리 설정
    tfidf_df = tfidf_df[
        (tfidf_df["종류별"] == category1) | (tfidf_df["종류별"] == category2)
        | (tfidf_df["종류별"] == category3)
        ]

        # # 카테고리 설정
        # tfidf_df = tfidf_df[
        #     (filtered_df["종류별"] == category1) | (filtered_df["종류별"] == category2)
        #     | (filtered_df["종류별"] == category3)
        #     ]
        #
    tfidf_df["식재료명"] = tfidf_df["식재료명"].str.replace(",", " ")

    tfidf_df["data"] = tfidf_df["방법별"] + " " \
                       + tfidf_df["상황별"] + " " \
                       + tfidf_df["종류별"] + " " \
                       + tfidf_df["식재료명"] + " " \
                       + tfidf_df["음식이름"]
        # tfidf_df["data"] 의 Tfidf 를 계산하기 위해서 카운터 전체 단어수 등을 계산해서 데이터의 어휘집 구축
    vectorizer.fit(tfidf_df["data"])


    # tfidf_df["data"] 의 Tfidf 를 계산해서 recipe_vector에 저장
    recipe_vector = vectorizer.transform(tfidf_df["data"])

    recipe_vector_df = pd.DataFrame(recipe_vector.toarray(),
                                     columns=vectorizer.get_feature_names_out())
    # print("len(tfidf_df)=",len(tfidf_df))
    # print("len(recipe_vector_df)=", len(recipe_vector_df))

    # Convert user input to list and transform it to a vector
    my_data1_vectorizer = vectorizer.transform(my_data1)

    # my_data1_vectorizer.toarray() : my_data1_vectorizer 를 numpy 배열로 변환
    # my_data1_vectorizer를 DataFrame으로 변환
    my_data1_vectorizer_df = pd.DataFrame(my_data1_vectorizer.toarray(), columns=vectorizer.get_feature_names_out())
    print("len(my_data1_vectorizer_df)=",len(my_data1_vectorizer_df))

    # receipe_vector_df (전체 레시피)
    # my_data1_vectorizer_df (입력한 레시피 식재료)
    # 의 유사도를 계산해서 cosine_sim에 저장
    cosine_sim = cosine_similarity(recipe_vector_df.values,
                                   my_data1_vectorizer_df.values)
    #cosine_sim을 1차원 배열로 변환 후 유사도를 result_df에 저장
    result_df = pd.DataFrame(cosine_sim.flatten(), columns=["유사도"])
    print("len(tfidf_df.index)=",len(tfidf_df.index))
    print("len(result_df)=", len(result_df))
    # result_df에 레시피 번호를 추가
    result_df["레시피번호"] = tfidf_df.index
    result_df["음식이름"] = recipe_db["음식이름"]
    result_df = result_df.set_index('레시피번호')
    print('result_df=', result_df)
    result_df['레시피명'] = recipe_db.loc[result_df.index, '음식이름'].values
    print('result_df=', result_df)
    result_df = result_df[~result_df['레시피명'].isin(favorite_list)]
    # 유사도가 가장 높은 순으로 정렬하고 가장 높은 5개 조회
    top_5_df = result_df.sort_values(by=["유사도"], ascending=False)[0:5]
    print('top_5_df=', top_5_df)
    recipe_numbers = top_5_df.index.astype(str).tolist()
    recipe_numbers_str = ' '.join(recipe_numbers)

    return recipe_numbers_str

import pandas as pd
from flask import Flask, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(_name_)
column_names = ['레시피번호', '음식이름', '방법별', '상황별', '종류별', '식재료명']
recipe_db = pd.read_csv(r"C:\Users\Park\Downloads\recipe.csv", encoding='cp949', names=column_names)
# 레시피번호를 인덱스로 설정
recipe_db.set_index('레시피번호', inplace=True)
recipe_db = recipe_db.drop_duplicates(subset=['음식이름'])
# tfidf_df 데이터 프레임에 저장
tfidf_df = recipe_db[['음식이름', '방법별', '상황별', '종류별', '식재료명']]
# tfidf_df 데이터프레임의 data 컬럼의 단어들의 Tfidf를 계산할 객체 vectorizer 생성
# `min_df`는 단어가 포함된 문서의 최소 개수를 설정
# norm 매개 변수를 사용하여 벡터의 정규화 방법을 지정, "l2"는 유클리드 거리 정규화
vectorizer = TfidfVectorizer(min_df=1, norm="l2")
# 유통기한 임박한 식재료 추천
@app.route('/recommendationControl', methods=['POST'])
def survey():
    categories = request.form.get('categories')
    category_nm = ['밑반찬', '메인반찬', '김치/젓갈/장류', '국/탕', '찌개', '면/만두', '밥/죽/떡', '퓨전', '양식', '샐러드', '양념/소스/잼', '스프', '빵',
                   '디저트', '과자', '차/음료/술']
    category_lst = []
    ingrdsToUse = request.form.get('ingrdsToUse')
    ingrds = []
    ingrds.append(ingrdsToUse.replace(',', ' '))
    for category in categories.split(',') :
        category_lst.append(category_nm[int(category)])
    category1 = category_lst[0]
    category2 = category_lst[1]
    category3 = category_lst[2]
    # 전역변수 불러옴
    global tfidf_df
    # 카테고리 설정
    tfidf_df = tfidf_df[
        (tfidf_df['종류별'] == category1) | (tfidf_df['종류별'] == category2)
        | (tfidf_df['종류별'] == category3)
    ]
    tfidf_df['식재료명'] = tfidf_df['식재료명'].str.replace(",", " ")
    tfidf_df['data'] = tfidf_df['방법별'] + " " \
                       + tfidf_df['상황별'] + " " \
                       + tfidf_df['종류별'] + " " \
                       + tfidf_df['식재료명'] + " " \
                       + tfidf_df['음식이름']
    # tfidf_df['data'] 의 Tfidf 를 계산하기 위해서 데이터의 어휘집 구축
    vectorizer.fit(tfidf_df['data'])
    # tfidf_df['data'] 의 Tfidf 를 계산해서 recipe_vector에 저장
    recipe_vector = vectorizer.transform(tfidf_df['data'])
    recipe_vector_df = pd.DataFrame(recipe_vector.toarray(), columns=vectorizer.get_feature_names_out())
    # 사용자 입력을 벡터로 변환
    ingrds_vectorizer = vectorizer.transform(ingrds)
    # ingrds_vectorizer를 DataFrame으로 변환
    ingrds_vectorizer_df = pd.DataFrame(ingrds_vectorizer.toarray(), columns=vectorizer.get_feature_names_out())
    # receipe_vector_df(전체 레시피)와 ingrds_vectorizer_df(입력한 레시피 식재료)의 유사도를 계산
    cosine_sim = cosine_similarity(recipe_vector_df.values, ingrds_vectorizer_df.values)
    # 유사도를 result_df에 저장
    result_df = pd.DataFrame(cosine_sim.flatten(), columns=['유사도'])
    # result_df에 레시피 번호와 음식이름 추가
    result_df['레시피번호'] = tfidf_df.index
    result_df['음식이름'] = recipe_db['음식이름']
    result_df = result_df.set_index('레시피번호')
    result_df['레시피명'] = recipe_db.loc[result_df.index, '음식이름'].values
    # 유사도가 0.5이상인 값들을 가장 높은 순으로 정렬후 df 생성
    top_df = result_df[result_df['유사도'] >= 0.5].sort_values(by='유사도', ascending=False)
    top_5 = []
    for i in range(len(top_df)):
        # 앞 리스트 값이 뒤 리스트 안에 하나라도 있으면 true 레시피 번호가 item
        if any(item in ingrdsToUse.split(',') for item in recipe_db.loc[top_df.index[i], '식재료명'].split(',')):
            if len(top_5) == 4 :
                top_5.append(str(top_df.index[i]))
                break
            else :
                top_5.append(str(top_df.index[i]))
    if len(top_5) < 5 :
        # 레시피 DB에 인기 정보(평점, etc..) 추가
        for ingrId, ingrList in recipe_db['식재료명'].items() : # db 구성 변경 필요 => ingredient_db에 레시피 번호열 추가해 비효율 감소
            if any(item in ingrdsToUse.split(',') for item in ingrList.split(',')) :
                if ingrList.index not in top_5 :
                    if len(top_5) == 4 :
                        top_5.append(str(ingrId))
                        break
                    else :
                        top_5.append(str(ingrId))
    print("ingrsToUse :", ingrdsToUse)
    print("top_5 : ", top_5)
    return ' '.join(top_5)
if _name_ == '_main_':
    app.run()

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()