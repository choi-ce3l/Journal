import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')

# ========================================
# 설정
# ========================================
# 입력 파일
ACADEMIA_CORPUS = "/home/dslab/choi/Journal/BERTopic/Data/corpus_with_topics_academia.csv"
ACADEMIA_TOPICS = "/home/dslab/choi/Journal/BERTopic/Data/topics_academia.csv"
INDUSTRY_CORPUS = "/home/dslab/choi/Journal/BERTopic/Data/corpus_with_topics_industry.csv"
INDUSTRY_TOPICS = "/home/dslab/choi/Journal/BERTopic/Data/topics_industry.csv"

# 출력 파일
OUTPUT_FILE = "/home/dslab/choi/Journal/BERTopic/Data/topic_pairs.csv"

# 매칭 설정
TOP_N_MATCHES = 3  # 각 토픽당 상위 3개 매칭 (5 → 3으로 감소)
MIN_SIMILARITY = 0.4  # 최소 유사도 임계값 (0.3 → 0.4로 증가)
# ========================================

print(f"{'=' * 70}")
print(f"산업-학계 토픽 매핑 분석")
print(f"{'=' * 70}\n")

# ========================================
# 1단계: 데이터 로드
# ========================================
print("[1단계] 데이터 로드")

# 학계 데이터
academia_corpus = pd.read_csv(ACADEMIA_CORPUS)
academia_topics = pd.read_csv(ACADEMIA_TOPICS)
print(f"  ✓ 학계 문서: {len(academia_corpus)}개")
print(f"  ✓ 학계 토픽: {len(academia_topics)}개")

# 산업계 데이터
industry_corpus = pd.read_csv(INDUSTRY_CORPUS)
industry_topics = pd.read_csv(INDUSTRY_TOPICS)
print(f"  ✓ 산업계 문서: {len(industry_corpus)}개")
print(f"  ✓ 산업계 토픽: {len(industry_topics)}개")

# outlier(-1) 제외
academia_topics = academia_topics[academia_topics['topic_id'] != -1].reset_index(drop=True)
industry_topics = industry_topics[industry_topics['topic_id'] != -1].reset_index(drop=True)
print(f"  ✓ outlier 제외 후 - 학계: {len(academia_topics)}개, 산업계: {len(industry_topics)}개")

# ========================================
# 2단계: 토픽 벡터화 (Sentence-BERT)
# ========================================
print(f"\n[2단계] 토픽 벡터화 (Sentence-BERT)")

# 임베딩 모델 로드
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"  ✓ 모델 로드: all-MiniLM-L6-v2")

# 학계 토픽 임베딩
print(f"  ► 학계 토픽 임베딩 생성 중...")
academia_topic_texts = academia_topics['keywords'].tolist()
academia_embeddings = embedding_model.encode(
    academia_topic_texts,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"    ✓ 학계 임베딩: {academia_embeddings.shape}")

# 산업계 토픽 임베딩
print(f"  ► 산업계 토픽 임베딩 생성 중...")
industry_topic_texts = industry_topics['keywords'].tolist()
industry_embeddings = embedding_model.encode(
    industry_topic_texts,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"    ✓ 산업계 임베딩: {industry_embeddings.shape}")

# ========================================
# 3단계: 유사도 계산
# ========================================
print(f"\n[3단계] 유사도 계산")

# 3-1. Cosine Similarity
print(f"  ► Cosine Similarity 계산 중...")
cosine_sim_matrix = cosine_similarity(academia_embeddings, industry_embeddings)
print(f"    ✓ Cosine Similarity Matrix: {cosine_sim_matrix.shape}")
print(f"    - 평균: {cosine_sim_matrix.mean():.4f}")
print(f"    - 최대: {cosine_sim_matrix.max():.4f}")
print(f"    - 최소: {cosine_sim_matrix.min():.4f}")

# 3-2. Jensen-Shannon Divergence
print(f"  ► Jensen-Shannon Divergence 계산 중...")


def calculate_jsd_matrix(topics1, topics2):
    """키워드 분포 기반 JSD 계산"""
    n1, n2 = len(topics1), len(topics2)
    jsd_matrix = np.zeros((n1, n2))

    for i, kw1 in enumerate(topics1['keywords']):
        # 키워드를 단어로 분리하고 빈도 계산
        words1 = kw1.lower().split(', ')
        word_counts1 = {w: words1.count(w) for w in set(words1)}

        for j, kw2 in enumerate(topics2['keywords']):
            words2 = kw2.lower().split(', ')
            word_counts2 = {w: words2.count(w) for w in set(words2)}

            # 전체 단어 집합
            all_words = set(word_counts1.keys()) | set(word_counts2.keys())

            # 확률 분포 생성
            p = np.array([word_counts1.get(w, 0) for w in all_words])
            q = np.array([word_counts2.get(w, 0) for w in all_words])

            # 정규화
            p = p / (p.sum() + 1e-10)
            q = q / (q.sum() + 1e-10)

            # JSD 계산
            jsd_matrix[i, j] = jensenshannon(p, q)

    return jsd_matrix


jsd_matrix = calculate_jsd_matrix(academia_topics, industry_topics)
print(f"    ✓ JSD Matrix: {jsd_matrix.shape}")
print(f"    - 평균: {jsd_matrix.mean():.4f}")
print(f"    - 최대: {jsd_matrix.max():.4f}")
print(f"    - 최소: {jsd_matrix.min():.4f}")

# 3-3. 종합 점수 계산
print(f"  ► 종합 점수 계산 중...")
combined_score_matrix = 0.7 * cosine_sim_matrix + 0.3 * (1 - jsd_matrix)
print(f"    ✓ Combined Score Matrix: {combined_score_matrix.shape}")
print(f"    - 평균: {combined_score_matrix.mean():.4f}")

# ========================================
# 4단계: 관심 강도 분석
# ========================================
print(f"\n[4단계] 관심 강도 분석")

# 학계 토픽별 통계
academia_stats = {}
total_academia = len(academia_corpus[academia_corpus['topic'] != -1])

for _, topic_row in academia_topics.iterrows():
    topic_id = topic_row['topic_id']
    topic_docs = academia_corpus[academia_corpus['topic'] == topic_id]

    academia_stats[topic_id] = {
        'count': len(topic_docs),
        'ratio': len(topic_docs) / total_academia * 100,
        'avg_prob': topic_docs['topic_probability'].mean() if len(topic_docs) > 0 else 0
    }

# 산업계 토픽별 통계
industry_stats = {}
total_industry = len(industry_corpus[industry_corpus['topic'] != -1])

for _, topic_row in industry_topics.iterrows():
    topic_id = topic_row['topic_id']
    topic_docs = industry_corpus[industry_corpus['topic'] == topic_id]

    industry_stats[topic_id] = {
        'count': len(topic_docs),
        'ratio': len(topic_docs) / total_industry * 100,
        'avg_prob': topic_docs['topic_probability'].mean() if len(topic_docs) > 0 else 0
    }

print(f"  ✓ 학계 토픽 통계 계산 완료")
print(f"  ✓ 산업계 토픽 통계 계산 완료")

# ========================================
# 5단계: 시차 분석
# ========================================
print(f"\n[5단계] 시차 분석 (Time Lag)")


def calculate_peak_year(corpus, topic_id):
    """토픽의 피크 연도 계산"""
    topic_docs = corpus[corpus['topic'] == topic_id]

    if len(topic_docs) == 0 or 'year' not in topic_docs.columns:
        return None

    # 연도별 문서 수
    year_counts = topic_docs['year'].value_counts()

    if len(year_counts) == 0:
        return None

    # 최대 문서 수를 가진 연도
    peak_year = year_counts.idxmax()
    return peak_year


# 학계 피크 연도
academia_peak_years = {}
for topic_id in academia_topics['topic_id']:
    peak = calculate_peak_year(academia_corpus, topic_id)
    academia_peak_years[topic_id] = peak

# 산업계 피크 연도
industry_peak_years = {}
for topic_id in industry_topics['topic_id']:
    peak = calculate_peak_year(industry_corpus, topic_id)
    industry_peak_years[topic_id] = peak

print(f"  ✓ 피크 연도 계산 완료")

# ========================================
# 6단계: 토픽 페어 매칭
# ========================================
print(f"\n[6단계] 토픽 페어 매칭")

topic_pairs = []
seen_pairs = set()  # 중복 방지

# 각 학계 토픽에 대해 상위 N개 산업계 토픽 매칭
for i, academia_row in academia_topics.iterrows():
    academia_id = academia_row['topic_id']

    # 유사도 점수 추출
    scores = combined_score_matrix[i, :]

    # 상위 N개 인덱스 (유사도 높은 순)
    top_indices = np.argsort(scores)[::-1][:TOP_N_MATCHES]

    for rank, j in enumerate(top_indices, 1):
        industry_row = industry_topics.iloc[j]
        industry_id = industry_row['topic_id']

        # 중복 체크 (이미 매칭된 페어는 건너뛰기)
        pair_key = (academia_id, industry_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # 유사도 점수
        cosine_sim = cosine_sim_matrix[i, j]
        jsd = jsd_matrix[i, j]
        combined_score = combined_score_matrix[i, j]

        # 최소 유사도 체크
        if combined_score < MIN_SIMILARITY:
            continue

        # 시차 계산
        academia_peak = academia_peak_years.get(academia_id)
        industry_peak = industry_peak_years.get(industry_id)

        if academia_peak and industry_peak:
            time_lag = industry_peak - academia_peak

            if time_lag > 0.5:
                lag_type = "학계선도"
            elif time_lag < -0.5:
                lag_type = "산업선도"
            else:
                lag_type = "동시발전"
        else:
            time_lag = None
            lag_type = "정보부족"

        # 관심도 격차
        academia_ratio = academia_stats[academia_id]['ratio']
        industry_ratio = industry_stats[industry_id]['ratio']
        interest_gap = abs(academia_ratio - industry_ratio)

        # 페어 정보 저장
        topic_pairs.append({
            # 학계 정보
            'academia_topic_id': academia_id,
            'academia_keywords': academia_row['keywords'],
            'academia_count': academia_stats[academia_id]['count'],
            'academia_ratio': academia_ratio,
            'academia_avg_prob': academia_stats[academia_id]['avg_prob'],
            'academia_peak_year': academia_peak,

            # 산업계 정보
            'industry_topic_id': industry_id,
            'industry_keywords': industry_row['keywords'],
            'industry_count': industry_stats[industry_id]['count'],
            'industry_ratio': industry_ratio,
            'industry_avg_prob': industry_stats[industry_id]['avg_prob'],
            'industry_peak_year': industry_peak,

            # 유사도
            'cosine_similarity': cosine_sim,
            'jsd': jsd,
            'combined_score': combined_score,

            # 시차 분석
            'time_lag': time_lag,
            'lag_type': lag_type,

            # 기타
            'interest_gap': interest_gap,
            'match_rank_from_academia': rank
        })

print(f"  ✓ 총 {len(topic_pairs)}개 토픽 페어 생성 (중복 제거 후)")
print(f"  ✓ 학계 토픽당 평균 {len(topic_pairs) / len(academia_topics):.1f}개 매칭")

# ========================================
# 7단계: 결과 저장
# ========================================
print(f"\n[7단계] 결과 저장")

# DataFrame 생성
pairs_df = pd.DataFrame(topic_pairs)

# 종합 점수로 정렬
pairs_df = pairs_df.sort_values('combined_score', ascending=False).reset_index(drop=True)

# 저장
pairs_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"  ✓ 저장 완료: {OUTPUT_FILE}")
print(f"  ✓ 총 페어 수: {len(pairs_df)}")

# ========================================
# 8단계: 결과 요약
# ========================================
print(f"\n{'=' * 70}")
print(f"[결과 요약]")
print(f"{'=' * 70}\n")

# 상위 10개 매칭
print("■ 상위 10개 토픽 매칭:")
for idx, row in pairs_df.head(10).iterrows():
    print(f"\n  [{idx + 1}] 유사도: {row['combined_score']:.3f}")
    print(f"      학계 #{row['academia_topic_id']:2d} ({row['academia_count']:3d}개): {row['academia_keywords'][:60]}...")
    print(f"      산업 #{row['industry_topic_id']:2d} ({row['industry_count']:3d}개): {row['industry_keywords'][:60]}...")
    if row['time_lag'] is not None:
        print(f"      시차: {row['time_lag']:.1f}년 ({row['lag_type']})")

# 시차 유형별 통계
print(f"\n■ 시차 유형별 통계:")
lag_type_counts = pairs_df[pairs_df['lag_type'] != '정보부족']['lag_type'].value_counts()
for lag_type, count in lag_type_counts.items():
    print(f"  - {lag_type}: {count}개 ({count / len(pairs_df) * 100:.1f}%)")

# 평균 시차
valid_lags = pairs_df[pairs_df['time_lag'].notna()]['time_lag']
if len(valid_lags) > 0:
    print(f"\n■ 평균 시차: {valid_lags.mean():.2f}년")
    print(f"  - 표준편차: {valid_lags.std():.2f}년")

# 관심도 격차
print(f"\n■ 관심도 격차:")
print(f"  - 평균: {pairs_df['interest_gap'].mean():.2f}%")
print(f"  - 최대: {pairs_df['interest_gap'].max():.2f}%")

print(f"\n{'=' * 70}")
print(f"✓ 산업-학계 토픽 매핑 완료!")
print(f"{'=' * 70}")