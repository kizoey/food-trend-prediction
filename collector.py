# -*- coding: utf-8 -*-
import urllib.request
import json
import pandas as pd
import config  # 네이버 api 접근코드
from urllib.request import urlopen
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import datetime
import re

# 네이버 api 접근코드
CID = config.CID
SECRET = config.SECRET
# ANCHOR CATG
ANCHOR_CATG = config.ANCHOR_CATG


# send_request : 3개 단위로 request를 보내고, df 형태로 결과값 리턴
def send_request(url, body, curr_catgs):
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", CID)
    request.add_header("X-Naver-Client-Secret", SECRET)
    request.add_header("Content-Type", "application/json")
    response = urllib.request.urlopen(request, data=json.dumps(body).encode("utf-8"))
    rescode = response.getcode()

    if (rescode == 200):
        response_body = response.read()
        result = json.loads(response_body.decode('utf-8'))["results"]

        # rearrange to dataframe
        ret = pd.DataFrame.from_dict(result[0]['data'])
        ret.rename(columns={"ratio": curr_catgs[0]['name']}, inplace=True)

        for i in range(1, len(result)):
            ret_partial = pd.DataFrame.from_dict(result[i]['data'])
            ret_partial.rename(columns={"ratio": curr_catgs[i]['name']}, inplace=True)
            ret = pd.merge(ret, ret_partial, on="period", how='outer').fillna(0)
    else:
        print("Error Code:" + rescode)

    return ret

def scale_n_merge(original_df, new_df, anchor_nm):
    # 기존 dataframe과 새로 생성된 dataframe 결합
    # scaled = partial_anchor / partial_anchor * result_anchor
    comp_ratio = original_df.iloc[0, 1] / new_df.iloc[0, 1]
    new_df.iloc[:, 1:] = new_df.iloc[:, 1:].multiply(comp_ratio)

    # 기준 칼럼 드롭 후 merge
    new_df.drop(columns=anchor_nm, inplace=True)
    merged = pd.merge(original_df, new_df, on="period", how='outer').fillna(-1)
    return merged

def saveResponse(response, fname='DATALAB_카테고리별_트렌드지수'):
    out_dir = "./data/source/"
    fname = fname + '_' + datetime.datetime.now().strftime("%Y%m%d") + ".csv"
    response.to_csv(out_dir + fname, encoding='utf-8-sig', index=False, float_format='%.4f')
    print(fname + " saved")

def collect_related(fname = 'Trend Chart_2020-08-13_STL_20대여성.csv'):
    # 크롬드라이버 옵션 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 해당 옵션 때문에 크롬 창이 뜨지 않음
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    data = pd.read_csv("./data/output/" + fname, encoding='CP949')

    data_top10 = data[:10]
    top10_list = list(data_top10.iloc[:, 0])

    cate_lists = []
    for cate in top10_list:
        cate_list = cate.split('_')
        cate_lists.append(cate_list)


    driver = webdriver.Chrome('chromedriver', options=options)

    # 전체 Top 10 카테고리 연관검색어 저장하기
    cate_words_df = pd.DataFrame(data={'rank': list(range(1, 41))})

    num = 0
    for cate in cate_lists:
        driver.implicitly_wait(3)
        driver.get('https://datalab.naver.com/shoppingInsight/sCategory.naver')
        time.sleep(3)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/span').click()  # 1분류
        time.sleep(1)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/ul/li[7]/a').click()  # 식품 카테고리
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/span').click()  # 2분류
        time.sleep(1)
        driver.find_element_by_link_text(cate[0]).click()
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[3]/span').click()  # 3분류
        time.sleep(1)
        driver.find_element_by_link_text(cate[1]).click()

        if cate[2] != 'NA':
            driver.find_element_by_xpath(
                '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[4]/span').click()  # 4분류
            time.sleep(1)
            driver.find_element_by_link_text(cate[2]).click()

        # 시작 기간 클릭
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[1]/span').click()
        time.sleep(1)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[1]/ul/li[4]/a').click()  # 2020년 클릭
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[2]/span').click()
        time.sleep(1)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[2]/ul/li[8]/a').click()  # 8월 클릭
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[3]/span').click()
        time.sleep(1)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[3]/ul/li[10]/a').click()  # 10일 클릭

        # 종료 기간 클릭(시작 기간 설정에 따라 같은 날짜라도 xpath 조금씩 달라짐)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[1]/span').click()
        time.sleep(1)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[1]/ul/li[1]/a').click()  # 2020년 클릭
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[2]/span').click()
        time.sleep(1)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[2]/ul/li[1]/a').click()  # 8월 클릭
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[3]/span').click()
        time.sleep(1)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[3]/ul/li[7]/a').click()  # 16일 클릭

        # 성별/연령
        driver.find_element_by_xpath('//*[@id="19_gender_1"]').click()  # 여성 클릭
        driver.find_element_by_xpath('//*[@id="20_age_2"]').click()  # 20대 클릭

        # 조회하기 버튼 클릭
        driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/a').click()
        time.sleep(3)

        # 해당 카테고리 (ex. 초콜릿) 인기검색어 추출(이후에 순위도 추출?)
        words = []
        for j in range(1, 3):  # 페이지 수
            for i in range(1, 21):  # 페이지 당 검색어 수
                word = driver.find_element_by_xpath(
                    '//*[@id="content"]/div[2]/div/div[2]/div[2]/div/div/div[1]/ul/li[' + str(i) + ']/a')
                a = word.text
                words.append(a)
            driver.find_element_by_xpath(
                '//*[@id="content"]/div[2]/div/div[2]/div[2]/div/div/div[2]/div/a[2]').click()  # 다음 페이지 클릭
            time.sleep(2)

        cate_words_df['cate_words'] = words
        cate_words_df.rename(columns={'cate_words': top10_list[num] + ' 연관검색어'}, inplace=True)

        num += 1
        time.sleep(3)

    for i in range(1, 11):
        cate_words_df.iloc[:, i] = cate_words_df.iloc[:, i].str.replace("\n", "")

    # 숫자만 제거하거나, 숫자도 보존 고민 (kg 수)
    cate_words_df.to_csv('/content/drive/MyDrive/DTB 공모전/연관검색어 분석/Trend_연관검색어_2020-08-13_20대 여성.csv', index=None)

    driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/span').click()  # 1분류
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/ul/li[7]/a').click()  # 식품 카테고리

    driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/span').click()  # 2분류

    driver.find_element_by_link_text("김치").text

    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[1]/span').click()
    driver.find_element_by_link_text("2021").text

    # 해당 카테고리(ex. 과자 > 초콜릿) 및 기간(ex. 2021-02-08 ~ 2021-02-14)에 해당하는 버튼(xpath) 클릭하는 코드
    # 나중에 반복문 혹은 함수화하기
    # NA가 있는 카테고리(3분류까지)와 아닌 카테고리(4분류까지) 조건 나누기

    # 카테고리 클릭
    driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/span').click()  # 1분류
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[1]/ul/li[7]/a').click()  # 식품 카테고리

    driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/span').click()  # 2분류
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/ul/li[7]/a').click()  # 과자
    # driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[2]/ul/li[5]/a').click()   # 김치

    driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[3]/span').click()  # 3분류
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[1]/div/div[3]/ul/li[2]/a').click()  # 초콜릿
    # 다른 카테고리는 4분류도 만들어야겠군...(조건문 달기)

    # 기간 클릭(연도 부분도 추가)
    # 시작 기간 클릭
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[1]/span').click()
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[1]/ul/li[5]/a').click()  # 2021년 클릭
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[2]/span').click()
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[2]/ul/li[2]/a').click()  # 2월 클릭
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[3]/span').click()
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[1]/div[3]/ul/li[8]/a').click()  # 8일 클릭

    # 종료 기간 클릭(시작 기간 설정에 따라 같은 날짜라도 xpath 조금씩 달라짐)
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[1]/span').click()
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[1]/ul/li/a').click()  # 2021년 클릭
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[2]/span').click()
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[2]/ul/li[1]/a').click()  # 2월 클릭
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[3]/span').click()
    driver.find_element_by_xpath(
        '//*[@id="content"]/div[2]/div/div[1]/div/div/div[2]/div[2]/span[3]/div[3]/ul/li[7]/a').click()  # 14일 클릭

    # 성별/연령

    # 조회하기 버튼 클릭
    driver.find_element_by_xpath('//*[@id="content"]/div[2]/div/div[1]/div/a').click()

    # 해당 카테고리 (ex. 초콜릿) 인기검색어 추출(이후에 순위도 추출?)
    words = []

    for j in range(1, 3):  # 페이지 수
        for i in range(1, 21):  # 페이지 당 검색어 수
            word = driver.find_element_by_xpath(
                '//*[@id="content"]/div[2]/div/div[2]/div[2]/div/div/div[1]/ul/li[' + str(i) + ']/a')
            a = word.text
            words.append(a)
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[2]/div/div[2]/div[2]/div/div/div[2]/div/a[2]').click()  # 다음 페이지 클릭
        time.sleep(2)

    words_df = pd.DataFrame(words, columns=['search_words'])
    words_df['search_words'] = words_df['search_words'].str.replace('\n', '')

    # 정규표현식 써서 순위 컬럼 만들기
    word_rank = []
    for word in words_df['search_words']:
        rank = re.findall("\d+", word)
        word_rank.append(rank[0])

    words_df['rank'] = word_rank
    words_df['search_words'] = words_df['search_words'].str.replace("[0-9]", "")

    words_df.to_csv("./data/output/" + "연관검색어" + datetime.datetime.now().strftime("%Y%m%d"))

    return words_df

def collect_shopping_index(fname='NAVER_카테고리목록_식품_210708.csv',
                           start_dt='2017-08-01', end_dt=datetime.datetime.now().strftime("%Y-%m-%d"),
                           timeunit='date', ages=["20", "30", "40", "50"], gender="f"):
    # 카테고리 목록 리딩
    catg_list = pd.read_csv('./data/source/' + fname, encoding='cp949', header=0,
                            usecols=['param', 'catg_nm_full'])
    # catg_list파일에서 anchor catg 정보는 제외 (response에서 수치 오류 발생)
    catg_list = catg_list[catg_list['param'] != ANCHOR_CATG['param']]
    catg_list = catg_list.to_dict('records')
    print("네이버 식품 카테고리 parse 완료")

    # api용 url
    url = "https://openapi.naver.com/v1/datalab/shopping/categories"

    # 카테고리별 트렌드 지수 크롤링
    print("\n카테고리별 트렌드 지수 수집중...\n")
    catg_num = len(catg_list)

    for i in range(0, catg_num, 2):
        # 비교 대상 카테고리가 1개인 경우 예외처리
        try:
            curr_catgs = [
                {"name": ANCHOR_CATG['catg_nm_full'], "param": [ANCHOR_CATG['param']]},
                {"name": catg_list[i]['catg_nm_full'], "param": [str(catg_list[i]["param"])]},
                {"name": catg_list[i + 1]['catg_nm_full'], "param": [str(catg_list[i + 1]["param"])]}
            ]
        except:
            curr_catgs = [
                {"name": ANCHOR_CATG['catg_nm_full'], "param": [ANCHOR_CATG['param']]},
                {"name": catg_list[i]['catg_nm_full'], "param": [str(catg_list[i]["param"])]}
            ]

        body = {
            "startDate": start_dt,
            "endDate": end_dt,
            "timeUnit": timeunit,
            "category": curr_catgs,
            "ages": ages,
            "gender": gender
        }
        partial_result = send_request(url, body, curr_catgs)

        try:
            result = scale_n_merge(result, partial_result, ANCHOR_CATG["catg_nm_full"])
        except NameError:
            result = partial_result

        print("크롤링 진행 중... %s / %s\n" % (i, catg_num))
        for c in curr_catgs: print("진행중인 카테고리 : ", c['name'])

        print("카테고리별 트렌드지수 수집 완료\n")
        response = result
    
    if gender=="f":
        saveResponse(response, fname=f'DATALAB_트렌드지수_{age}대_여성')
    elif gender=="m":
        saveResponse(response, fname=f'DATALAB_트렌드지수_{age}대_남성')
    else:
        saveResponse(response, fname=f'DATALAB_트렌드지수_{age}대_성별무관')


def collect_search_index():
    pass


if __name__ == '__main__':
    pass




