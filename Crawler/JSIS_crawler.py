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

# ===== 사용자 설정 =====
vol_start = 34
vol_end = 35  # 34까지 포함되도록 +1
iss_list = ["1", "2"]
HEADLESS = False

SAVE_DIR = r"/Users/choihj/PycharmProjects/Journal/Data/JSIS"
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(SAVE_DIR, f"JSIS_vol{vol_start}to{vol_end-1}_iss1to4.csv")

# ===== 유틸 =====
def random_wait(a=1.0, b=3.0):
    time.sleep(random.uniform(a, b))

def get_driver():
    opts = uc.ChromeOptions()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
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
    title = extract_text(soup.select_one("span.title-text"))
    authors = [extract_text(a) for a in soup.select("div.author-group span.react-xocs-alternative-link")]
    authors = ", ".join([a for a in authors if a])
    abs_el = soup.select_one("div.abstract.author") or soup.select_one("div[id^='sp']") or soup.select_one("div.Abstracts div.abstract")
    abstract = extract_text(abs_el)
    if abstract.lower().startswith("abstract"):
        abstract = abstract[len("abstract"):].strip()
    pub_date = ""
    meta_date = soup.select_one("meta[name='citation_publication_date']")
    if meta_date and meta_date.get("content"):
        pub_date = meta_date["content"].strip()
    if not pub_date:
        date_candidate = soup.select_one("div.text-xs, dl.article-header-details")
        pub_date = extract_text(date_candidate)
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
    for vol in range(vol_start, vol_end):
        for iss in iss_list:
            toc_url = f"https://www.sciencedirect.com/journal/the-journal-of-strategic-information-systems/vol/{vol}/issue/{iss}"
            print(f"\n[jsis] Volume {vol} Issue {iss} 접속 중...")
            driver.get(toc_url)
            random_wait(2, 4)

            try:
                cookie_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                )
                cookie_btn.click()
                print("쿠키 수락 완료")
                random_wait()
            except Exception:
                print("쿠키 수락 스킵")

            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".text-l")))
            except TimeoutException:
                print(f"❌ 논문 목록 로딩 실패: Vol {vol} Issue {iss}")
                continue

            title_spans = driver.find_elements(By.CSS_SELECTOR, ".text-l")
            if not title_spans:
                print(f"⚠️ 논문 링크 없음: Vol {vol} Issue {iss}")
                continue

            hrefs = []
            for el in title_spans:
                try:
                    parent = el.find_element(By.XPATH, "..")
                    hrefs.append(parent.get_attribute("href"))
                except Exception:
                    hrefs.append(None)

            for idx, href in enumerate(hrefs):
                try:
                    if href:
                        driver.get(href)
                    else:
                        driver.get(toc_url)
                        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.js-article-title.text-l")))
                        spans = driver.find_elements(By.CSS_SELECTOR, "span.js-article-title.text-l")
                        if idx >= len(spans):
                            print(f"인덱스 초과 스킵: Vol {vol} Issue {iss}, idx {idx}")
                            continue
                        link = spans[idx]
                        driver.execute_script("arguments[0].scrollIntoView({behavior:'instant',block:'center'});", link)
                        random_wait(0.6, 1.2)
                        driver.execute_script("arguments[0].click();", link)

                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.title-text")))
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    random_wait(1.2, 2.2)

                    title, authors, abstract, pub_date, keywords = parse_article_page(driver.page_source)

                    all_rows.append({
                        "volume": vol,
                        "issue": iss,
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "date": pub_date,
                        "keywords": keywords,
                        "url": driver.current_url
                    })

                    if len(all_rows) % 30 == 0:
                        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
                        print(f"중간 저장됨 ({len(all_rows)}개)")
                        driver.quit()
                        driver = get_driver()
                        wait = WebDriverWait(driver, 25)

                except (StaleElementReferenceException, TimeoutException, WebDriverException) as e:
                    print(f"🚧 실패 (Vol {vol} Issue {iss}, idx {idx}): {e}")
                    try:
                        driver.get(toc_url)
                        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.js-article-title.text-l")))
                    except Exception:
                        driver.quit()
                        driver = get_driver()
                        wait = WebDriverWait(driver, 25)
                        driver.get(toc_url)
                        try:
                            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.js-article-title.text-l")))
                        except Exception:
                            print("TOC 복구 실패. 다음 이슈로 진행.")
                            break
                    random_wait()
                    continue

finally:
    try:
        driver.quit()
    except Exception:
        pass

pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 저장 완료! 총 {len(all_rows)}개 논문 수집됨")
print(f"📁 저장 위치: {OUTPUT_CSV}")
