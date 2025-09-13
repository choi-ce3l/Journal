import os
import time
import random
import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# 저장 경로 설정
save_path='Academia/'
os.makedirs(save_path, exist_ok=True)
output_file = os.path.join(save_path, "informs_isre_vol36.csv")

# 대기 함수
def random_wait(a=1, b=3):
    time.sleep(random.uniform(a, b))

# 드라이버 생성 함수
def get_driver():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1280,800")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument("referer=https://pubsonline.informs.org/")

    driver = uc.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
              get: () => undefined
            })
        """
    })
    return driver

# 실행 시작
driver = get_driver()
wait = WebDriverWait(driver, 20)
all_results = []

# ✅ Volume별 Issue 범위 지정
issue_map = {
    # 34: range(4,5),
    # 35: range(1, 5),
    36: range(3, 5)
}

for vol, issue_range in issue_map.items():
    for iss in issue_range:
        toc_url = f"https://pubsonline.informs.org/toc/isre/{vol}/{iss}"
        print(f"\n📄 Volume {vol}, Issue {iss} 접속 중...")
        driver.get(toc_url)
        random_wait(2, 4)

        if vol == 36 and iss == 3:
            try:
                cookie_btn = wait.until(EC.element_to_be_clickable((By.ID, "hs-eu-confirmation-button")))
                cookie_btn.click()
                print("🍪 쿠키 수락 완료")
                random_wait()
            except:
                print("⚠️ 쿠키 수락 스킵")

        try:
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h5.issue-item__title > a")))
        except TimeoutException:
            print(f"❌ 논문 목록 로딩 실패: Vol {vol}, Iss {iss}")
            continue

        links = driver.find_elements(By.CSS_SELECTOR, "h5.issue-item__title > a")

        for i in range(len(links)):
            try:
                links = driver.find_elements(By.CSS_SELECTOR, "h5.issue-item__title > a")
                link = links[i]
                paper_url = link.get_attribute("href")

                driver.execute_script("window.scrollBy({top: 400, behavior: 'smooth'});")
                random_wait(0.5, 1.5)
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", link)
                random_wait(1, 1.8)
                driver.execute_script("arguments[0].click();", link)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.citation__title")))
                random_wait(1.5, 2.5)

                soup = BeautifulSoup(driver.page_source, "html.parser")
                title = soup.select_one("h1.citation__title")
                authors = soup.select("a.entryAuthor")
                date = soup.select_one("span.epub-section__date")
                abstract = soup.select_one("div.abstractSection.abstractInFull > p")
                keywords = soup.select("section.article__keyword ul.rlist li a")

                all_results.append({
                    "volume": vol,
                    "issue": iss,
                    "title": title.text.strip() if title else "",
                    "authors": ", ".join([a.text.strip() for a in authors]) if authors else "",
                    "date": date.text.strip() if date else "",
                    "abstract": abstract.text.strip() if abstract else "",
                    "keywords": ", ".join([k.text.strip() for k in keywords]) if keywords else "",
                    "url": paper_url
                })

                if len(all_results) % 30 == 0:
                    pd.DataFrame(all_results).to_csv(output_file, index=False, encoding="utf-8-sig")
                    print(f"💾 중간 저장됨 ({len(all_results)}개)")
                    driver.quit()
                    driver = get_driver()
                    wait = WebDriverWait(driver, 20)
                    driver.get(toc_url)
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h5.issue-item__title > a")))
                    random_wait(3, 6)

                try:
                    driver.back()
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h5.issue-item__title > a")))
                    random_wait()
                except:
                    driver.get(toc_url)
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h5.issue-item__title > a")))
                    random_wait()

            except Exception as e:
                print(f"🚧 실패 (Vol {vol}, Iss {iss}, idx {i}): {e}")
                driver.get(toc_url)
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h5.issue-item__title > a")))
                random_wait()
                continue

# 종료 및 저장
driver.quit()
df = pd.DataFrame(all_results)
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✅ 전체 크롤링 완료! 총 {len(df)}개 논문 수집됨")
print(f"📁 저장 위치: {output_file}")
