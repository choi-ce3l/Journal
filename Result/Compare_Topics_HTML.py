# ============================================================
# 🎯 토픽 대표 문서 뷰어 (HTML 버전) - Jupyter Notebook용
# ============================================================

import pandas as pd
from IPython.display import display, HTML

# 1. 데이터 로드
academia_path = "/home/dslab/choi/Journal/LDA_ALL/Data/corpus_with_topics_academia_lda_bow.csv"
industry_path = "/home/dslab/choi/Journal/LDA_ALL/Data/corpus_with_topics_industry_lda_bow.csv"

df_academia = pd.read_csv(academia_path)
df_industry = pd.read_csv(industry_path)

print(f"Academia: {len(df_academia)}건 / Industry: {len(df_industry)}건")

# ============================================================
# 2. HTML 스타일 정의
# ============================================================

STYLE = """
<style>
.topic-container {
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 100%;
    margin: 20px 0;
}
.topic-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 10px 10px 0 0;
    font-size: 18px;
    font-weight: bold;
}
.doc-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin: 10px 0;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.doc-header {
    background: #f8f9fa;
    padding: 12px 15px;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.doc-title {
    font-weight: bold;
    color: #2c3e50;
    font-size: 14px;
}
.doc-meta {
    color: #7f8c8d;
    font-size: 12px;
}
.prob-badge {
    background: #3498db;
    color: white;
    padding: 4px 10px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
}
.doc-body {
    padding: 15px;
}
.section-label {
    font-weight: bold;
    color: #8e44ad;
    font-size: 12px;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 5px;
}
.section-content {
    color: #34495e;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 12px;
    padding: 10px;
    background: #fafafa;
    border-radius: 5px;
    border-left: 3px solid #9b59b6;
}
.abstract-content {
    border-left-color: #3498db;
}
.divider {
    text-align: center;
    padding: 20px;
    font-size: 24px;
    color: #95a5a6;
}
.comparison-wrapper {
    display: flex;
    gap: 20px;
}
.corpus-column {
    flex: 1;
    min-width: 0;
}
</style>
"""

# ============================================================
# 3. HTML 생성 함수
# ============================================================

def get_topic_docs_html(df, topic_num, corpus_name, n=5, color="#667eea"):
    """특정 토픽의 대표 문서를 HTML로 반환"""

    topic_df = df[df['topic'] == topic_num].copy()

    if len(topic_df) == 0:
        return f"<p>⚠️ Topic {topic_num}에 해당하는 문서가 없습니다.</p>"

    top_docs = topic_df.nlargest(n, 'topic_probability')

    html = f"""
    <div class="topic-container">
        <div class="topic-header" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
            📚 {corpus_name} - Topic {topic_num} (총 {len(topic_df)}건 중 Top {n})
        </div>
    """

    for i, (_, row) in enumerate(top_docs.iterrows(), 1):
        prob = row['topic_probability']
        title = row.get('title', '제목 없음')
        year = row.get('date', '')
        year_str = f" ({int(year)})" if pd.notna(year) else ""

        # 한국어 요약
        kor_sum = row.get('korean_summary', '')
        kor_html = ""
        if pd.notna(kor_sum) and str(kor_sum).strip():
            kor_text = str(kor_sum)[:400] + ('...' if len(str(kor_sum)) > 400 else '')
            kor_html = f"""
            <div class="section-label">🇰🇷 한국어 요약</div>
            <div class="section-content">{kor_text}</div>
            """

        # Abstract
        abstract = row.get('abstract', '')
        abs_html = ""
        if pd.notna(abstract) and str(abstract).strip():
            abs_text = str(abstract)[:400] + ('...' if len(str(abstract)) > 400 else '')
            abs_html = f"""
            <div class="section-label">📝 Abstract</div>
            <div class="section-content abstract-content">{abs_text}</div>
            """

        html += f"""
        <div class="doc-card">
            <div class="doc-header">
                <div>
                    <span class="doc-meta">#{i}{year_str}</span>
                    <div class="doc-title">{title}</div>
                </div>
                <span class="prob-badge">확률: {prob:.4f}</span>
            </div>
            <div class="doc-body">
                {kor_html}
                {abs_html}
            </div>
        </div>
        """

    html += "</div>"
    return html


def show_topic_docs(df, topic_num, corpus_name="", n=5):
    """특정 토픽의 대표 문서를 HTML로 표시"""
    color = "#667eea" if "academia" in corpus_name.lower() else "#e74c3c"
    html = STYLE + get_topic_docs_html(df, topic_num, corpus_name, n, color)
    display(HTML(html))


def compare_topics(acad_topic, ind_topic, n=5):
    """Academia와 Industry 토픽을 나란히 비교"""

    acad_html = get_topic_docs_html(df_academia, acad_topic, "ACADEMIA", n, "#667eea")
    ind_html = get_topic_docs_html(df_industry, ind_topic, "INDUSTRY", n, "#e74c3c")

    html = STYLE + f"""
    <h2 style="text-align:center; color:#2c3e50; margin:20px 0;">
        🔬 Topic 비교: Academia [{acad_topic}] vs Industry [{ind_topic}]
    </h2>
    <div class="comparison-wrapper">
        <div class="corpus-column">{acad_html}</div>
        <div class="corpus-column">{ind_html}</div>
    </div>
    """
    display(HTML(html))


def compare_topics_vertical(acad_topic, ind_topic, n=5):
    """Academia와 Industry 토픽을 세로로 비교 (좁은 화면용)"""

    acad_html = get_topic_docs_html(df_academia, acad_topic, "ACADEMIA", n, "#667eea")
    ind_html = get_topic_docs_html(df_industry, ind_topic, "INDUSTRY", n, "#e74c3c")

    html = STYLE + f"""
    <h2 style="text-align:center; color:#2c3e50; margin:20px 0;">
        🔬 Topic 비교: Academia [{acad_topic}] vs Industry [{ind_topic}]
    </h2>
    {acad_html}
    <div class="divider">⬇️ ⬇️ ⬇️</div>
    {ind_html}
    """
    display(HTML(html))


# ============================================================
# 4. 사용법
# ============================================================
"""
# 단일 토픽 확인
show_topic_docs(df_academia, 0, "Academia", n=5)
show_topic_docs(df_industry, 11, "Industry", n=5)

# 두 토픽 나란히 비교 (넓은 화면)
compare_topics(0, 11, n=5)

# 두 토픽 세로로 비교 (좁은 화면)
compare_topics_vertical(0, 11, n=5)
"""

# ============================================================
# 5. 바로 실행
# ============================================================
# ACADEMIA_TOPIC = 0
# INDUSTRY_TOPIC = 11
# NUM_DOCS = 5

# 아래 중 하나 선택해서 실행
# compare_topics(ACADEMIA_TOPIC, INDUSTRY_TOPIC, NUM_DOCS)  # 나란히
# compare_topics_vertical(ACADEMIA_TOPIC, INDUSTRY_TOPIC, NUM_DOCS)  # 세로로