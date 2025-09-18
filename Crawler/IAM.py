import os
import time
import random
import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, WebDriverException

# ===== 사용자 환경 =====
# 60 - 62
# 1- 8
vol_from=62
vol_to=63
issue_from=8
issue_to=9
SAVE_DIR = r"/Crawler/IAM"
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(SAVE_DIR, f"iam_vol{vol_from}to{vol_to}_issue{issue_from}.csv")  # 파일명 정정
HEADLESS = False  # 필요 시 True

# ===== 유틸 =====
def random_wait(a=1.0, b=3.0):
    time.sleep(random.uniform(a, b))

def get_driver():
    opts = uc.ChromeOptions()
    # macOS 공통 안전 옵션
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    # User-Agent
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    # 별도 프로필 사용을 권하지 않음(탐지 위험↑). 꼭 필요하면 아래 주석 해제 후 경로 수정.
    # opts.add_argument("--user-data-dir=/Users/choihj/Library/Application Support/Google/Chrome/Default")

    drv = uc.Chrome(options=opts)
    try:
        drv.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
        )
    except WebDriverException:
        pass
    return drv

def extract_text(el):
    return el.get_text(strip=True) if el else ""

def parse_article_page(html):
    soup = BeautifulSoup(html, "html.parser")

    # 제목
    title = extract_text(soup.select_one("span.title-text"))

    # 저자
    authors = [extract_text(a) for a in soup.select("div.author-group span.react-xocs-alternative-link")]
    authors = ", ".join([a for a in authors if a])

    # 초록
    abs_el = soup.select_one("div.abstract.author") or soup.select_one("div[id^='sp']") \
             or soup.select_one("div.Abstracts div.abstract")
    abstract = extract_text(abs_el)
    if abstract.lower().startswith("abstract"):
        abstract = abstract[len("abstract"):].strip()

    # 날짜: 메타 우선 -> 페이지 내 텍스트 보조
    pub_date = ""
    meta_date = soup.select_one("meta[name='citation_publication_date']")
    if meta_date and meta_date.get("content"):
        pub_date = meta_date["content"].strip()
    if not pub_date:
        date_candidate = soup.select_one("div.text-xs, dl.article-header-details")
        pub_date = extract_text(date_candidate)

    # 키워드
    # ScienceDirect는 'Author keywords' 섹션이 있거나 없을 수 있음
    keywords = [extract_text(k) for k in soup.select("div.keywords-section div.keyword > span")]
    if not keywords:
        keywords = [extract_text(k) for k in soup.select("div.Keywords div.keyword")]
    keywords = ", ".join([k for k in keywords if k])

    return title, authors, abstract, pub_date, keywords

# ===== 크롤링 =====
driver = get_driver()
wait = WebDriverWait(driver, 25)
all_rows = []

try:
    for vol in range(vol_from, vol_to):
        for issue in range(issue_from, issue_to):
            toc_url = f"https://www.sciencedirect.com/journal/information-and-management/vol/{vol}/issue/{issue}"
            print(f"\n[DSS] Volume {vol} 접속 중...")
            driver.get(toc_url)
            random_wait(2, 4)

            # 쿠키 수락 시도
            try:
                cookie_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                )
                cookie_btn.click()
                print("쿠키 수락 완료")
                random_wait()
            except Exception:
                print("쿠키 수락 스킵")

            # TOC 로딩
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".text-l")))
            except TimeoutException:
                print(f"❌ 논문 목록 로딩 실패: Vol {vol}")
                continue

            # 동적 페이지에서 참조 무효화 방지: href 수집 후 순회
            title_spans = driver.find_elements(By.CSS_SELECTOR, ".text-l")
            if not title_spans:
                print(f"⚠️ 논문 링크 없음: Vol {vol}")
                continue

            hrefs = []
            for el in title_spans:
                try:
                    parent = el.find_element(By.XPATH, '//*[@id="S0167923625001083"]')
                    hrefs.append(parent.get_attribute("href"))
                except Exception:
                    # 일부 구조는 span 자체에 클릭 이벤트만 있음 → span 클릭 처리용 마커
                    hrefs.append(None)

            for idx, href in enumerate(hrefs):
                try:
                    if href:
                        driver.get(href)
                    else:
                        # href가 없으면 다시 TOC로 가서 해당 index 클릭
                        driver.get(toc_url)
                        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.js-article-title.text-l")))
                        spans = driver.find_elements(By.CSS_SELECTOR, "span.js-article-title.text-l")
                        if idx >= len(spans):
                            print(f"인덱스 초과 스킵: Vol {vol}, idx {idx}, issue {issue}")
                            continue
                        link = spans[idx]
                        driver.execute_script("arguments[0].scrollIntoView({behavior:'instant',block:'center'});", link)
                        random_wait(0.6, 1.2)
                        driver.execute_script("arguments[0].click();", link)

                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.title-text")))
                    # 하단까지 스크롤하여 동적 섹션 로딩
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    random_wait(1.2, 2.2)

                    title, authors, abstract, pub_date, keywords = parse_article_page(driver.page_source)

                    all_rows.append({
                        "volume": vol,
                        "issue": "suppl/C",
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "date": pub_date,
                        "keywords": keywords,
                        "url": driver.current_url
                    })

                    # 중간 저장 및 드라이버 재시작
                    if len(all_rows) % 30 == 0:
                        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
                        print(f"중간 저장됨 ({len(all_rows)}개)")
                        driver.quit()
                        driver = get_driver()
                        wait = WebDriverWait(driver, 25)
                        driver.get(toc_url)
                        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.js-article-title.text-l")))
                        random_wait(2, 4)

                except (StaleElementReferenceException, TimeoutException, WebDriverException) as e:
                    print(f"🚧 실패 (Vol {vol}, idx {idx}): {e}")
                    # TOC 복구
                    try:
                        driver.get(toc_url)
                        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.js-article-title.text-l")))
                    except Exception:
                        # 드라이버 리셋
                        driver.quit()
                        driver = get_driver()
                        wait = WebDriverWait(driver, 25)
                        driver.get(toc_url)
                        try:
                            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.js-article-title.text-l")))
                        except Exception:
                            print("TOC 복구 실패. 다음 볼륨으로 진행.")
                            break
                    random_wait()
                    continue

finally:
    try:
        driver.quit()
    except Exception:
        pass

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 저장 완료! 총 {len(df)}개 논문 수집됨")
print(f"📁 저장 위치: {OUTPUT_CSV}")