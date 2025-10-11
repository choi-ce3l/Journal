import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import pandas as pd
import time

def scrape_issue(vol:int, iss:int, out_csv:str):
    driver = uc.Chrome(headless=False)
    toc_url=f"https://www.tandfonline.com/toc/mmis20/{vol}/{iss}?nav=tocList"
    driver.get(toc_url)
    time.sleep(5)

    # 논문 링크 추출
    links = [a.get_attribute("href") for a in driver.find_elements(By.CSS_SELECTOR, "div.art_title.linkable > a")]
    print(f"Found {len(links)} articles.")

    data = []
    for link in links:
        driver.get(link)
        time.sleep(2)
        try:
            title = driver.find_element(By.CSS_SELECTOR, ".hlFld-title").text.strip()
        except:
            title = ""
        try:
            abstract = driver.find_element(By.CSS_SELECTOR, ".last").text.strip()
        except:
            abstract = ""
        try:
            keywords = [k.text.strip() for k in driver.find_elements(By.CSS_SELECTOR, ".keyword-click")]
            keywords = ", ".join(keywords)
        except:
            keywords = ""

        data.append({
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "url": link
        })
        print("→", title)

    driver.quit()
    pd.DataFrame(data).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("완료:", out_csv)


# 실행 예시
if __name__ == "__main__":
    # vol: 40-42
    # iss: 1-4
    vol=40
    for iss in range(1,5):
        scrape_issue(vol, iss, f"/Users/choihj/PycharmProjects/Journal/Data/JMIS/JMIS_vol{vol}_iss{iss}.csv")