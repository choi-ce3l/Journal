import re, time, requests, pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HDRS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

def clean(s):
    if not s: return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def get_soup(url):
    r = requests.get(url, headers=HDRS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def collect_article_urls(vol:int, iss:int):
    toc_url = f"https://aisel.aisnet.org/misq/vol{vol}/iss{iss}/"
    soup = get_soup(toc_url)
    urls = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        full = urljoin(toc_url, href)
        if f"/misq/vol{vol}/iss{iss}/" in full and re.search(rf"/misq/vol{vol}/iss{iss}/\d+/?$", full):
            urls.add(full)
    return sorted(urls)

def extract_title(soup):
    el = soup.select_one("#title a")
    return clean(el.get_text()) if el else ""

def extract_abstract(soup):
    el = soup.select_one("#abstract p")
    if el and clean(el.get_text()):
        return clean(el.get_text())
    lbl = soup.find(lambda tag: tag.name in ["h2","h3"] and "abstract" in tag.get_text(strip=True).lower())
    if lbl:
        p = lbl.find_next("p")
        if p: return clean(p.get_text())
    return ""

def extract_keywords(soup):
    block = soup.select_one("div.keywords, section.keywords, #keywords")
    items = []
    if block:
        items = [clean(x.get_text()) for x in block.select("li, span, a") if clean(x.get_text())]
    if not items:
        meta = soup.select_one("meta[name='keywords']")
        if meta and meta.get("content"):
            items = [clean(x) for x in re.split(r",|;", meta["content"]) if clean(x)]
    seen, out = set(), []
    for k in items:
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

def scrape_article(url):
    soup = get_soup(url)
    return {
        "title": extract_title(soup),
        "abstract": extract_abstract(soup),
        "keywords": ", ".join(extract_keywords(soup)),
        "url": url
    }

def scrape_issue(vol:int, iss:int, out_csv:str):
    urls = collect_article_urls(vol, iss)
    if not urls:
        print(f"vol{vol} iss{iss}: 논문 URL을 찾지 못했습니다.")
        return
    rows = []
    for i,u in enumerate(urls,1):
        try:
            row = scrape_article(u)
            rows.append(row)
            print(f"[{vol}-{iss} {i}/{len(urls)}] {row['title'][:80]}")
            time.sleep(0.5)
        except Exception as e:
            print("실패:", u, "->", e)
    pd.DataFrame(rows, columns=["title","abstract","keywords","url"]).to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    print("완료:", out_csv)

# 사용 예시
if __name__ == "__main__":
    # vol=47
    # issue=2
    # # Vol45 Issue2
    # scrape_issue(vol, issue, f"misq_vol{vol}_iss{issue}.csv")

    # 여러 권호 반복 처리 가능
    for vol, iss in [(47,3),(47,4),(48,1),(48,2),(48,3),(48,4),(49,1),(49,2),(49,3)]:
        scrape_issue(vol, iss, f"misq_vol{vol}_iss{iss}.csv")