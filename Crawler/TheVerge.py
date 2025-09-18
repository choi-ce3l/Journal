import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import os
from tqdm import tqdm
import time  # 요청 사이 간격을 두고 싶을 경우 사용

def collect_theverge_links(start_date: str, end_date: str) -> list:
    """
    주어진 기간 동안 The Verge AI 아카이브에서 기사 링크 수집
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    valid_links = []

    current_date = start
    while current_date <= end:
        year = current_date.year
        month = current_date.month

        first_url = f'https://www.theverge.com/archives/ai-artificial-intelligence/{year}/{month}/1'
        print(f"페이지 수 확인 중: {first_url}")

        try:
            first_response = requests.get(first_url)
            first_response.raise_for_status()
            first_soup = BeautifulSoup(first_response.text, 'html.parser')
            pages = first_soup.select_one("span.i0ukxu3.i0ukxu1")
            match = re.search(r'of\s*(\d+)', pages.text) if pages else None
            total_pages = int(match.group(1)) if match else 1
        except Exception as e:
            print(f"⚠️ 페이지 수 추출 실패: {e}")
            total_pages = 1

        for page in range(1, total_pages + 1):
            url = f'https://www.theverge.com/archives/ai-artificial-intelligence/{year}/{month}/{page}'
            print(f"크롤링 중: {url}")
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                a_tag = soup.select('div.hp1qhq3 div div a')

                for a in a_tag:
                    link = a.get('href')
                    # print(link)
                    if link and not link.endswith("#comments"):
                        valid_links.append(link)
                        print(link)
            except requests.exceptions.RequestException as e:
                print(f"⚠️ 요청 실패: {e}")

        current_date += relativedelta(months=1)

    print(f"총 {len(valid_links)}개의 링크를 수집했습니다.")
    return valid_links

def scrape_theverge_article(href: str, file_path: str = "data.csv"):
    """
    The Verge 기사 하나를 크롤링하여 CSV 파일에 저장
    """
    url = f'https://www.theverge.com{href}'

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        head = soup.select_one('#content h1')
        contents = soup.select('#zephr-anchor > div > p')
        keywords = soup.select('#zephr-anchor > div > ul > li')
        date = soup.select_one('time')

        title_text = head.get_text(strip=True) if head else "N/A"
        content_text = "\n".join(p.get_text(strip=True) for p in contents) if contents else "N/A"
        keyword_text = ", ".join(k.get_text(strip=True) for k in keywords) if keywords else "N/A"
        date_text = date.get_text(strip=True) if date else "N/A"

        new_data = pd.DataFrame([{
            "date": date_text,
            "title": title_text,
            "content": content_text,
            "keywords": keyword_text,
            "url": url
        }])

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data

        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {url}")
    except Exception as e:
        print(f"❌ 크롤링 실패: {url} / 오류: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The Verge AI 기사 크롤러")
    parser.add_argument('--start', type=str, required=True, help="시작 날짜 (예: 2023-01-01)")
    parser.add_argument('--end', type=str, required=True, help="끝 날짜 (예: 2023-01-31)")
    parser.add_argument('--output', type=str, default="data.csv", help="저장할 파일 경로")

    args = parser.parse_args()

    # 링크 수집
    links = collect_theverge_links(args.start, args.end)

    # 기사 크롤링 및 저장 with tqdm
    for href in tqdm(links, desc="📰 기사 크롤링 진행 중"):
        relative_href = href.replace("https://www.theverge.com", "") if href.startswith("https") else href
        scrape_theverge_article(relative_href, file_path=args.output)
        time.sleep(1)

'''
실행 방법

'''