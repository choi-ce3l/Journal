import pandas as pd

# 1. 데이터 로드
academia_path = "/home/dslab/choi/Journal/LDA_ALL/Data/corpus_with_topics_academia_lda_bow.csv"
industry_path = "/home/dslab/choi/Journal/LDA_ALL/Data/corpus_with_topics_industry_lda_bow.csv"

df_academia = pd.read_csv(academia_path)
df_industry = pd.read_csv(industry_path)

print(f"Academia: {len(df_academia)}건 / Industry: {len(df_industry)}건")
print(f"Academia 토픽 범위: {df_academia['topic'].min()} ~ {df_academia['topic'].max()}")
print(f"Industry 토픽 범위: {df_industry['topic'].min()} ~ {df_industry['topic'].max()}")

# ============================================================
# 2. 함수 정의 (한 번만 실행)
# ============================================================

def show_topic_docs(df, topic_num, corpus_name="", n=5):
    """특정 토픽의 대표 문서 출력"""

    # 해당 토픽 문서만 필터링
    topic_df = df[df['topic'] == topic_num].copy()

    if len(topic_df) == 0:
        print(f"⚠️ Topic {topic_num}에 해당하는 문서가 없습니다.")
        return

    # topic_probability 기준 상위 n개
    top_docs = topic_df.nlargest(n, 'topic_probability')

    for i, (_, row) in enumerate(top_docs.iterrows(), 1):
        prob = row['topic_probability']

        # 제목 찾기
        title = row.get('title')

        # 연도 찾기
        year = row.get('date')
        if year and pd.notna(year):
            year = f" ({int(year)})"
        else:
            year = ""

        # 한국어 요약 찾기
        kor_sum = None
        for col in ['korean_summary']:
            if col in row.index and pd.notna(row.get(col)):
                kor_sum = str(row[col])[:500]
                break

        # Abstract 찾기
        abstract = None
        for col in ['abstract']:
            if col in row.index and pd.notna(row.get(col)):
                abstract = str(row[col])[:500]
                break

        print(f"\n{'─'*80}")
        print(f"📄 [{i}] 토픽확률: {prob:.4f}{year}")
        print(f"   📌 {title}")

        if kor_sum:
            print(f"\n   🇰🇷 한국어 요약:")
            print(f"      {kor_sum}{'...' if len(str(kor_sum)) >= 500 else ''}")

        if abstract:
            print(f"\n   📝 Abstract:")
            print(f"      {abstract}{'...' if len(str(abstract)) >= 500 else ''}")


def compare_topics(acad_topic, ind_topic, n=5):
    """Academia와 Industry 토픽 비교"""
    print("\n" + "🔬")
    print(f"  Academia Topic {acad_topic}  vs  Industry Topic {ind_topic}")
    print("🔬")

    show_topic_docs(df_academia, acad_topic, "ACADEMIA", n)
    print("\n" + "⬇️  ")
    show_topic_docs(df_industry, ind_topic, "INDUSTRY", n)


def topic_summary(df, corpus_name=""):
    """각 토픽별 문서 수 요약"""
    summary = df.groupby('topic').agg({
        'topic_probability': ['count', 'mean', 'max']
    }).round(4)
    summary.columns = ['문서수', '평균확률', '최대확률']
    print(f"\n📊 {corpus_name} 토픽별 요약:")
    print(summary)
    return summary


# ============================================================
# 3. 사용법 (아래 셀에서 실행)
# ============================================================
"""
# 토픽별 문서 수 확인
topic_summary(df_academia, "Academia")
topic_summary(df_industry, "Industry")

# 단일 코퍼스 특정 토픽 확인
show_topic_docs(df_academia, 3, "Academia", n=5)   # topic 3
show_topic_docs(df_industry, 5, "Industry", n=5)   # topic 5

# 두 토픽 비교 (Shared Topics 비교용)
compare_topics(3, 5, n=5)  # Academia topic 3 vs Industry topic 5
"""

# ============================================================
# 4. 여기서 토픽 번호 수정 후 실행!
# ============================================================

# # 👇 확인하고 싶은 토픽 번호를 여기서 수정하세요
# ACADEMIA_TOPIC = 0   # Academia 토픽 번호
# INDUSTRY_TOPIC = 11   # Industry 토픽 번호
# NUM_DOCS = 5         # 확인할 문서 수

# 실행 (주석 해제 후 사용)
# compare_topics(ACADEMIA_TOPIC, INDUSTRY_TOPIC, NUM_DOCS)