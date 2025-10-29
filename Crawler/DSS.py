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
vol_from = 199 # 164 ~ 199
vol_to = 199 # 176부터 해야함

SAVE_DIR = r"/Users/choihj/PycharmProjects/Journal/Data/Academia/DSS"
os.makedirs(SAVE_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(SAVE_DIR, f"dss_vol_{vol_from}_to_{vol_to}.csv")
HEADLESS = False
MAX_RETRIES = 1  # 재시도 횟수


# ===== Volume별 날짜 매핑 =====
def get_date_by_volume(volume):
    """Volume 번호를 기준으로 날짜를 반환합니다."""
    start_year = 2023
    start_month = 1
    start_volume = 164

    months_diff = volume - start_volume
    total_months = (start_year - 2000) * 12 + start_month + months_diff
    year = 2000 + (total_months - 1) // 12
    month = ((total_months - 1) % 12) + 1

    return f"{year}-{month:02d}"


# ===== 유틸 =====
def random_wait(a=1.0, b=3.0):
    time.sleep(random.uniform(a, b))


def get_driver():
    """드라이버 생성 및 설정"""
    opts = uc.ChromeOptions()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")  # 더 큰 창 크기
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-popup-blocking")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")  # GPU 가속 비활성화
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    try:
        drv = uc.Chrome(options=opts)
        drv.set_page_load_timeout(60)  # 페이지 로드 타임아웃 설정
        try:
            drv.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
            )
        except WebDriverException:
            pass
        return drv
    except Exception as e:
        print(f"❌ 드라이버 생성 실패: {e}")
        raise


def safe_get_url(driver, url, max_retries=MAX_RETRIES):
    """URL 접속 재시도 로직"""
    for attempt in range(max_retries):
        try:
            driver.get(url)
            random_wait(2, 3)
            return True
        except TimeoutException:
            print(f"⏱️ 타임아웃 (시도 {attempt + 1}/{max_retries}): {url}")
            if attempt < max_retries - 1:
                random_wait(3, 5)
            else:
                return False
        except WebDriverException as e:
            print(f"🚧 WebDriver 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                random_wait(5, 8)
            else:
                return False
    return False


def accept_cookies(driver):
    """쿠키 수락"""
    try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        cookie_btn.click()
        print("✅ 쿠키 수락 완료")
        random_wait(1, 2)
        return True
    except Exception:
        print("ℹ️ 쿠키 수락 버튼 없음")
        return False


def extract_text(el):
    return el.get_text(strip=True) if el else ""


def parse_article_page(html):
    """논문 페이지 파싱"""
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

    # 키워드
    keywords = [extract_text(k) for k in soup.select("div.keywords-section div.keyword > span")]
    if not keywords:
        keywords = [extract_text(k) for k in soup.select("div.Keywords div.keyword")]
    keywords = ", ".join([k for k in keywords if k])

    return title, authors, abstract, keywords


def restart_driver(driver, wait):
    """드라이버 재시작"""
    try:
        driver.quit()
        print("🔄 드라이버 재시작 중...")
        random_wait(3, 5)
    except Exception:
        pass

    new_driver = get_driver()
    new_wait = WebDriverWait(new_driver, 30)
    return new_driver, new_wait


def save_progress(rows, filepath):
    """진행 상황 저장"""
    try:
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"💾 진행 상황 저장됨 ({len(rows)}개 논문)")
        return True
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return False


# ===== 크롤링 메인 로직 =====
def main():
    driver = get_driver()
    wait = WebDriverWait(driver, 30)
    all_rows = []

    # 기존 진행 상황 불러오기
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.DataFrame(pd.read_csv(OUTPUT_CSV))
            all_rows = existing_df.to_dict('records')
            print(f"📂 기존 데이터 로드: {len(all_rows)}개 논문")
        except Exception as e:
            print(f"⚠️ 기존 데이터 로드 실패: {e}")

    try:
        for vol in range(vol_from, vol_to + 1):
            toc_url = f"https://www.sciencedirect.com/journal/decision-support-systems/vol/{vol}/suppl/C"
            print(f"\n{'=' * 60}")
            print(f"📚 [DSS] Volume {vol} 크롤링 시작")
            print(f"{'=' * 60}")

            # TOC 페이지 접속
            if not safe_get_url(driver, toc_url):
                print(f"❌ Volume {vol} TOC 접속 실패. 건너뛰기.")
                continue

            # 쿠키 수락 (첫 번째 볼륨에서만)
            if vol == vol_from:
                accept_cookies(driver)

            # 논문 목록 로딩 대기
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".text-l")))
                random_wait(2, 3)
            except TimeoutException:
                print(f"❌ 논문 목록 로딩 실패: Volume {vol}")
                # 드라이버 재시작 시도
                driver, wait = restart_driver(driver, wait)
                continue

            # 논문 링크 수집
            title_spans = driver.find_elements(By.CSS_SELECTOR, ".text-l")
            if not title_spans:
                print(f"⚠️ 논문 링크 없음: Volume {vol}")
                continue

            hrefs = []
            for el in title_spans:
                try:
                    parent = el.find_element(By.XPATH, "./ancestor::a")
                    hrefs.append(parent.get_attribute("href"))
                except Exception:
                    hrefs.append(None)

            pub_date = get_date_by_volume(vol)
            print(f"📅 Volume {vol} → Date: {pub_date}")
            print(f"📄 총 {len(hrefs)}개 논문 발견")

            # 각 논문 크롤링
            for idx, href in enumerate(hrefs):
                article_num = idx + 1
                print(f"\n[{article_num}/{len(hrefs)}] 처리 중...", end=" ")

                retry_count = 0
                success = False

                while retry_count < MAX_RETRIES and not success:
                    try:
                        # 논문 페이지 접속
                        if href:
                            if not safe_get_url(driver, href):
                                raise TimeoutException("URL 접속 실패")
                        else:
                            # href 없을 경우 다시 TOC 가서 클릭
                            if not safe_get_url(driver, toc_url):
                                raise TimeoutException("TOC 복귀 실패")

                            wait.until(EC.presence_of_all_elements_located(
                                (By.CSS_SELECTOR, "span.js-article-title.text-l")))
                            spans = driver.find_elements(By.CSS_SELECTOR, "span.js-article-title.text-l")

                            if idx >= len(spans):
                                print(f"⚠️ 인덱스 초과")
                                break

                            link = spans[idx]
                            driver.execute_script(
                                "arguments[0].scrollIntoView({behavior:'instant',block:'center'});", link)
                            random_wait(0.8, 1.5)
                            driver.execute_script("arguments[0].click();", link)

                        # 논문 페이지 로딩 대기
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.title-text")))
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        random_wait(1.5, 2.5)

                        # 논문 정보 파싱
                        title, authors, abstract, keywords = parse_article_page(driver.page_source)

                        # 데이터 저장
                        all_rows.append({
                            "volume": vol,
                            "issue": "suppl/C",
                            "date": pub_date,
                            "title": title,
                            "authors": authors,
                            "abstract": abstract,
                            "keywords": keywords,
                            "url": driver.current_url
                        })

                        print(f"✅ 수집 완료")
                        success = True

                        # 주기적 저장 및 드라이버 재시작
                        if len(all_rows) % 20 == 0:
                            save_progress(all_rows, OUTPUT_CSV)
                            driver, wait = restart_driver(driver, wait)
                            # TOC로 복귀
                            safe_get_url(driver, toc_url)
                            wait.until(EC.presence_of_all_elements_located(
                                (By.CSS_SELECTOR, "span.js-article-title.text-l")))
                            random_wait(2, 3)

                    except (StaleElementReferenceException, TimeoutException, WebDriverException) as e:
                        retry_count += 1
                        print(f"🚧 오류 발생 (재시도 {retry_count}/{MAX_RETRIES}): {type(e).__name__}")

                        if retry_count < MAX_RETRIES:
                            # TOC 복구 시도
                            try:
                                safe_get_url(driver, toc_url)
                                wait.until(EC.presence_of_all_elements_located(
                                    (By.CSS_SELECTOR, "span.js-article-title.text-l")))
                                random_wait(2, 3)
                            except Exception:
                                # 드라이버 재시작
                                driver, wait = restart_driver(driver, wait)
                                safe_get_url(driver, toc_url)
                                try:
                                    wait.until(EC.presence_of_all_elements_located(
                                        (By.CSS_SELECTOR, "span.js-article-title.text-l")))
                                except Exception:
                                    print("❌ 복구 실패. 다음 볼륨으로 진행.")
                                    break

                        random_wait(3, 5)

                if not success:
                    print(f"❌ 최종 실패: Volume {vol}, Article {article_num}")

            # Volume 완료 후 저장
            save_progress(all_rows, OUTPUT_CSV)
            print(f"\n✅ Volume {vol} 완료!")

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류: {e}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

        # 최종 저장
        save_progress(all_rows, OUTPUT_CSV)

        print(f"\n{'=' * 60}")
        print(f"✅ 크롤링 완료!")
        print(f"📊 총 {len(all_rows)}개 논문 수집됨")
        print(f"📁 저장 위치: {OUTPUT_CSV}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()