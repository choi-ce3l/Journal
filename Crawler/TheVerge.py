# -*- coding: utf-8 -*-
"""
The Verge AI 아카이브 크롤러 (링크 수집 + 기사 파싱 + CSV 저장)

사용 예:
python theverge_ai_scraper.py --start 2025-09-01 --end 2025-09-01 --output ./TheVerge.csv
python theverge_ai_scraper.py --start 2025-01-01 --end 2025-09-01 --section ai-artificial-intelligence --output ~/Downloads/theverge_ai.csv
"""

import os, re, sys, json, time, argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# 설정
# =========================
BASE = "https://www.theverge.com"
DEFAULT_SECTION = "ai-artificial-intelligence"  # 아카이브 섹션
PAUSE = 1.2  # 서버 예의상 대기

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

ARTICLE_HREF_PAT = re.compile(
    r"^(/|https?://www\.theverge\.com/)(\d{4}/\d{1,2}/\d{1,2}/|news/|tech/|ai-artificial-intelligence/|column/|podcast/|video/)"
)

# =========================
# 유틸
# =========================
def get_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def to_abs(href: str) -> str:
    return urljoin(BASE + "/", href)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def coerce_date_iso(s: str) -> str:
    if not s or s == "N/A":
        return "N/A"
    # ISO in datetime attribute or JSON-LD datePublished 우선
    try:
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        return dt.astimezone().isoformat()
    except Exception:
        pass
    # 공지 문자열에서 날짜 추정
    try:
        dt = datetime.strptime(s[:25], "%B %d, %Y at %I:%M %p")
        return dt.isoformat()
    except Exception:
        pass
    return s  # 원문 보존

# =========================
# 링크 수집
# =========================
def guess_total_pages(html: str) -> int:
    """
    'Page 1 of N' 패턴 우선 탐지. 실패 시 1 페이지로 가정.
    """
    soup = BeautifulSoup(html, "html.parser")
    # 자주 보이는 'Page X of Y' 텍스트 탐색
    text = soup.get_text(" ", strip=True)
    m = re.search(r"[Pp]age\s+\d+\s+of\s+(\d+)", text)
    if m:
        return max(1, int(m.group(1)))
    # 페이지네이션 a[aria-label="Page N"] 최댓값 탐색
    nums = []
    for a in soup.find_all("a"):
        label = (a.get("aria-label") or "").strip()
        m2 = re.match(r"Page\s+(\d+)$", label)
        if m2:
            nums.append(int(m2.group(1)))
    if nums:
        return max(nums)
    return 1

def extract_archive_links(html: str) -> list:
    """
    아카이브 페이지에서 기사 링크 추출. 패턴 기반 필터로 안정화.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "#comments" in href:
            continue
        if ARTICLE_HREF_PAT.search(href):
            links.add(to_abs(href))
    return list(links)

def collect_theverge_links(start_date: str, end_date: str, section: str = DEFAULT_SECTION) -> list:
    """
    start_date~end_date 사이 각 월의 아카이브에서 기사 URL 수집.
    YYYY-MM-DD 형식.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    sess = get_session()

    collected = set()
    current = start

    while current <= end:
        y, m = current.year, current.month
        first_url = f"{BASE}/archives/{section}/{y}/{m}/1"
        print(f"\n🔍 페이지 수 확인: {first_url}")
        try:
            r = sess.get(first_url, timeout=30)
            r.raise_for_status()
            total_pages = guess_total_pages(r.text)
        except Exception as e:
            print(f"⚠️ 페이지 수 추출 실패: {e}")
            total_pages = 1

        # 페이지 순회
        seen_any = False
        for page in range(1, total_pages + 1):
            url = f"{BASE}/archives/{section}/{y}/{m}/{page}"
            print(f"📄 크롤링: {url}")
            try:
                resp = sess.get(url, timeout=30)
                if resp.status_code >= 400:
                    print(f"  ↳ 건너뜀 HTTP {resp.status_code}")
                    break
                page_links = extract_archive_links(resp.text)
                if not page_links:
                    print("  ↳ 기사 링크 없음. 다음 달로.")
                    break
                for lk in page_links:
                    collected.add(lk)
                seen_any = True
                time.sleep(PAUSE)
            except requests.RequestException as e:
                print(f"  ↳ 요청 실패: {e}")
                break

        if not seen_any:
            print("  ↳ 이 달은 수집 링크 없음.")
        current += relativedelta(months=1)

    print(f"\n✅ 수집 링크: {len(collected)}개")
    return sorted(collected)

# =========================
# 기사 파싱
# =========================
def parse_json_ld(soup: BeautifulSoup) -> dict:
    """
    JSON-LD에서 headline, description, datePublished, keywords를 우선 추출.
    """
    data = {"title": None, "abstract": None, "date": None, "keywords": None}
    scripts = soup.find_all("script", type="application/ld+json")
    for sc in scripts:
        try:
            payload = json.loads(sc.string or "")
        except Exception:
            continue
        cand = []
        if isinstance(payload, dict):
            cand = [payload]
        elif isinstance(payload, list):
            cand = payload
        else:
            continue

        for item in cand:
            if not isinstance(item, dict):
                continue
            typ = item.get("@type") or item.get("type")
            if isinstance(typ, list):
                types = [t.lower() for t in typ if isinstance(t, str)]
            elif isinstance(typ, str):
                types = [typ.lower()]
            else:
                types = []
            if any(t in ("newsarticle", "article", "reportageNewsArticle".lower()) for t in types):
                data["title"] = item.get("headline") or data["title"]
                data["abstract"] = item.get("description") or data["abstract"]
                data["date"] = item.get("datePublished") or data["date"]
                kw = item.get("keywords")
                if isinstance(kw, list):
                    data["keywords"] = ", ".join([str(k) for k in kw if k])
                elif isinstance(kw, str):
                    data["keywords"] = kw
                return data
    return data

def parse_meta_fallback(soup: BeautifulSoup, current: dict) -> dict:
    """
    메타 태그와 본문에서 폴백 추출.
    """
    out = dict(current)

    # 제목
    if not out.get("title"):
        h1 = soup.find("h1")
        out["title"] = clean_text(h1.get_text()) if h1 else None

    # 요약
    if not out.get("abstract"):
        md = soup.find("meta", attrs={"name": "description"})
        if not md:
            md = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "twitter:description"})
        if md and md.get("content"):
            out["abstract"] = clean_text(md["content"])
        else:
            # 본문 일부
            paras = soup.select("article p")
            if not paras:
                # 다른 컨테이너 폴백
                paras = soup.find_all("p")
            text = " ".join([clean_text(p.get_text()) for p in paras[:5]])
            out["abstract"] = text if text else None

    # 키워드
    if not out.get("keywords"):
        mk = soup.find("meta", attrs={"name": "news_keywords"}) or soup.find("meta", attrs={"name": "keywords"})
        if mk and mk.get("content"):
            out["keywords"] = clean_text(mk["content"])
        else:
            # 목록 폴백
            kws = [li.get_text(strip=True) for li in soup.select("#zephr-anchor ul li")]
            out["keywords"] = ", ".join(kws) if kws else None

    # 날짜
    if not out.get("date"):
        t = soup.find("time")
        dt = t.get("datetime") if t and t.has_attr("datetime") else (t.get_text(strip=True) if t else None)
        out["date"] = dt or None

    return out

def scrape_article(url: str, sess: requests.Session) -> dict:
    """
    단일 기사 파싱 → dict 반환
    """
    try:
        r = sess.get(url, timeout=40)
        r.raise_for_status()
    except requests.RequestException as e:
        return {"url": url, "error": f"request_failed: {e}"}

    soup = BeautifulSoup(r.text, "html.parser")

    data = parse_json_ld(soup)
    data = parse_meta_fallback(soup, data)

    return {
        "date": coerce_date_iso(data.get("date") or "N/A"),
        "title": clean_text(data.get("title") or "N/A"),
        "abstract": clean_text(data.get("abstract") or "N/A"),
        "keywords": clean_text(data.get("keywords") or "N/A"),
        "url": url,
    }

# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser(description="The Verge AI 아카이브 크롤러")
    ap.add_argument("--start", required=True, help="시작 날짜 YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="종료 날짜 YYYY-MM-DD")
    ap.add_argument("--section", default=DEFAULT_SECTION, help="아카이브 섹션 경로 (기본: ai-artificial-intelligence)")
    ap.add_argument("--output", default="/Users/choihj/PycharmProjects/Journal/Data/TheVerge.csv", help="출력 CSV 경로")
    ap.add_argument("--resume", action="store_true", help="이미 저장된 URL은 건너뛰기")
    ap.add_argument("--limit", type=int, default=0, help="최대 기사 수 (0=무제한)")
    args = ap.parse_args()

    ensure_parent_dir(args.output)
    sess = get_session()

    # 기존 CSV 로드 및 스킵 목록
    existing_urls = set()
    if args.resume and os.path.exists(args.output):
        try:
            prev = pd.read_csv(args.output)
            if "url" in prev.columns:
                existing_urls = set(prev["url"].dropna().tolist())
                print(f"↺ 재개 모드: 기존 {len(existing_urls)}개 URL 스킵")
        except Exception as e:
            print(f"⚠️ 기존 CSV 로드 실패: {e}")

    # 링크 수집
    links = collect_theverge_links(args.start, args.end, section=args.section)
    if args.resume and existing_urls:
        links = [u for u in links if u not in existing_urls]
        print(f"↺ 스킵 후 잔여 링크: {len(links)}")

    # 기사 파싱
    rows = []
    count = 0
    for url in tqdm(links, desc="📰 기사 파싱"):
        row = scrape_article(url, sess)
        rows.append(row)
        count += 1
        if args.limit and count >= args.limit:
            break
        time.sleep(PAUSE)

    # CSV 저장
    df_new = pd.DataFrame(rows, columns=["date", "title", "abstract", "keywords", "url"])
    if os.path.exists(args.output) and not df_new.empty:
        try:
            df_old = pd.read_csv(args.output)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    # 중복 제거
    if not df_all.empty:
        df_all.drop_duplicates(subset=["url"], inplace=True)

    try:
        df_all.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n✅ 저장 완료: {os.path.abspath(args.output)} (총 {len(df_all)}건)")
    except PermissionError:
        print("❌ 저장 실패: Permission denied. 쓰기 가능한 경로를 지정하세요. 예: --output ~/Downloads/theverge.csv")
        sys.exit(1)

if __name__ == "__main__":
    main()