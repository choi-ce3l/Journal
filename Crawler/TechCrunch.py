import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scrape_techcrunch_ai_articles(start_page=1, end_page=20):
    """TechCrunch AI 카테고리에서 기사를 수집하는 함수"""

    # 저장용 리스트
    data = []
    failed_urls = []

    # 세션 사용으로 연결 효율성 향상
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    # 페이지 진행률 바
    page_progress = tqdm(range(start_page, end_page + 1), desc="페이지 진행", unit="페이지")

    for page in page_progress:
        try:
            url = f'https://techcrunch.com/category/artificial-intelligence/page/{page}/'
            page_progress.set_description(f"페이지 {page} 처리 중")

            response = session.get(url, timeout=10)
            response.raise_for_status()  # HTTP 에러 발생시 예외 발생

            soup = BeautifulSoup(response.text, 'html.parser')

            # 기사 목록 선택
            articles = soup.select('.wp-block-post-template.is-layout-flow.wp-block-post-template-is-layout-flow > li')

            if not articles:
                logger.warning(f"페이지 {page}에서 기사를 찾을 수 없습니다.")
                continue

            # 각 페이지의 기사 진행률 바
            article_progress = tqdm(articles, desc=f"페이지 {page} 기사", leave=False, unit="기사")

            for article in article_progress:
                try:
                    # 'AI' 태그가 있는 기사만
                    label_link = article.select_one('div > div > div > div > a')
                    if label_link and label_link.text.strip() == 'AI':
                        # 기사 링크 추출
                        title_link = article.select_one('div > div > div > h3 > a')
                        if title_link and title_link.get('href'):
                            article_url = title_link.get('href')

                            try:
                                response2 = session.get(article_url, timeout=10)
                                response2.raise_for_status()

                                soup2 = BeautifulSoup(response2.text, 'html.parser')

                                # 제목
                                head = soup2.select_one('.article-hero__title.wp-block-post-title')
                                title = head.get_text(strip=True) if head else "제목 없음"

                                # 날짜
                                date_tag = soup2.select_one('.wp-block-post-date > time')
                                date = date_tag.get('datetime') if date_tag else None

                                # 본문
                                content_tags = soup2.select(
                                    '.entry-content.wp-block-post-content.is-layout-constrained.wp-block-post-content-is-layout-constrained > p'
                                )
                                content = "\n".join(
                                    p.get_text(strip=True) for p in content_tags) if content_tags else "본문 없음"

                                # 키워드
                                keywords_tag = soup2.select_one('.wp-block-tc23-post-relevant-terms > div')
                                keywords = keywords_tag.get_text(strip=True) if keywords_tag else None

                                # 누적 저장
                                data.append({
                                    'title': title,
                                    'date': date,
                                    'content': content,
                                    'keywords': keywords,
                                    'url': article_url
                                })

                                article_progress.set_description(f"수집된 기사: {len(data)}개")

                            except requests.RequestException as e:
                                logger.error(f"기사 상세 페이지 요청 실패 ({article_url}): {e}")
                                failed_urls.append(article_url)
                            except Exception as e:
                                logger.error(f"기사 파싱 중 오류 ({article_url}): {e}")
                                failed_urls.append(article_url)

                            # 예의상 딜레이
                            time.sleep(0.5)

                except Exception as e:
                    logger.error(f"기사 처리 중 오류: {e}")
                    continue

        except requests.RequestException as e:
            logger.error(f"페이지 {page} 요청 실패: {e}")
            continue
        except Exception as e:
            logger.error(f"페이지 {page} 처리 중 오류: {e}")
            continue

    return data, failed_urls


def main():
    """메인 실행 함수"""
    print("TechCrunch AI 기사 수집을 시작합니다...")

    # 기사 수집 실행
    data, failed_urls = scrape_techcrunch_ai_articles(start_page=1, end_page=20)

    # 결과 출력
    print(f"\n수집 완료!")
    print(f"총 {len(data)}개의 기사를 수집했습니다.")

    if failed_urls:
        print(f"실패한 URL {len(failed_urls)}개:")
        for url in failed_urls[:5]:  # 처음 5개만 출력
            print(f"  - {url}")
        if len(failed_urls) > 5:
            print(f"  ... 그 외 {len(failed_urls) - 5}개")

    if data:
        # 데이터프레임으로 변환
        df = pd.DataFrame(data)

        # 기본 통계 정보
        print(f"\n데이터 정보:")
        print(f"  - 제목이 있는 기사: {df['title'].notna().sum()}개")
        print(f"  - 날짜가 있는 기사: {df['date'].notna().sum()}개")
        print(f"  - 본문이 있는 기사: {df['content'].notna().sum()}개")
        print(f"  - 키워드가 있는 기사: {df['keywords'].notna().sum()}개")

        # CSV 파일로 저장
        filename = 'techcrunch_ai_articles.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n결과가 '{filename}' 파일로 저장되었습니다.")

        # 샘플 데이터 출력
        if len(df) > 0:
            print(f"\n첫 번째 기사 샘플:")
            first_article = df.iloc[0]
            print(f"  제목: {first_article['title'][:100]}...")
            print(f"  날짜: {first_article['date']}")
            print(f"  키워드: {first_article['keywords']}")
            print(f"  URL: {first_article['url']}")
    else:
        print("수집된 데이터가 없습니다.")


if __name__ == "__main__":
    main()