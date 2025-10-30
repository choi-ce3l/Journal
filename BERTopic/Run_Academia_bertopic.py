import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import warnings

warnings.filterwarnings('ignore')

# ========================================
# 설정
# ========================================
INPUT_FILE = "/home/dslab/choi/Journal/BERTopic/Data/academia_corpus.csv"
OUTPUT_TOPICS_FILE = "/home/dslab/choi/Journal/BERTopic/Data/topics_academia.csv"
OUTPUT_MODEL_PATH = "/home/dslab/choi/Journal/BERTopic/Model/bertopic_model_academia"

# BERTopic 하이퍼파라미터
N_TOPICS_RANGE = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 시도할 토픽 개수
MIN_TOPIC_SIZE = 5  # 5 → 3: 아주 작은 클러스터도 허용 (아웃라이어 최소화)
N_WORDS = 10  # 토픽당 키워드 개수
# ========================================

print(f"{'=' * 60}")
print(f"BERTopic 토픽 모델링 시작")
print(f"{'=' * 60}\n")

# 1. 데이터 로드
print("[1단계] 데이터 로드")
df = pd.read_csv(INPUT_FILE)
print(f"  ✓ 파일: {INPUT_FILE}")
print(f"  ✓ 문서 수: {len(df)}")
print(f"  ✓ 컬럼: {list(df.columns)}")

# 전처리된 텍스트 추출
docs = df['processed_text'].tolist()
print(f"  ✓ 전처리된 문서 {len(docs)}개 로드 완료")

# 2. 임베딩 모델 로드
print(f"\n[2단계] SBERT 임베딩 모델 로드")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"  ✓ 모델: all-MiniLM-L6-v2")

# 3. 문서 임베딩 생성
print(f"\n[3단계] 문서 임베딩 생성 (시간이 소요될 수 있습니다...)")
print(f"  총 문서 수: {len(docs)}개")
embeddings = embedding_model.encode(
    docs,
    show_progress_bar=True,
    batch_size=32,  # 배치 크기 명시
    convert_to_numpy=True
)
print(f"  ✓ 임베딩 생성 완료: {embeddings.shape}")

# 4. BERTopic 모델 구성
print(f"\n[4단계] BERTopic 모델 구성")

# UMAP 차원 축소 (아웃라이어 대폭 감소)
umap_model = UMAP(
    n_neighbors=100,  # 50 → 100: 훨씬 더 많은 이웃 고려
    n_components=15,  # 10 → 15: 더 많은 차원 보존
    min_dist=0.0,  # 유지: 밀집된 클러스터
    metric='cosine',
    random_state=42
)
print(f"  ✓ UMAP 설정 완료 (아웃라이어 최소화 모드)")

# HDBSCAN 클러스터링 (아웃라이어 대폭 감소)
hdbscan_model = HDBSCAN(
    min_cluster_size=3,  # 5 → 3: 더 작은 클러스터 허용
    min_samples=1,  # 3 → 1: 훨씬 더 유연하게
    cluster_selection_epsilon=0.2,  # 0.1 → 0.2: 더 많은 클러스터 병합
    metric='euclidean',
    cluster_selection_method='leaf',  # eom → leaf: 더 많은 클러스터 생성
    prediction_data=True
)
print(f"  ✓ HDBSCAN 설정 완료 (아웃라이어 최소화 모드)")

# CountVectorizer (n-gram 설정)
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)
print(f"  ✓ CountVectorizer 설정 완료")

# 5. 최적 모델 찾기
print(f"\n[5단계] 최적 BERTopic 모델 찾기")
print(f"  토픽 개수 범위: {N_TOPICS_RANGE}")

best_model = None
best_score = -1
best_n_topics = None
model_results = []

for nr_topics in N_TOPICS_RANGE:
    print(f"\n  [시도] nr_topics={nr_topics}")

    try:
        # BERTopic 모델 생성
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            top_n_words=N_WORDS,
            verbose=False
        )

        # 토픽 모델링 실행
        topics, probs = topic_model.fit_transform(docs, embeddings)

        # 토픽 정보 가져오기
        topic_info = topic_model.get_topic_info()
        n_topics_found = len(topic_info) - 1  # -1 제외

        # outliers 계산 (topics가 numpy 배열이 아닐 수 있음)
        topics_array = np.array(topics) if not isinstance(topics, np.ndarray) else topics
        n_outliers = int((topics_array == -1).sum())

        print(f"    - 발견된 토픽 수: {n_topics_found}")
        print(f"    - Outlier 문서 수: {n_outliers}")

        # Coherence Score 계산 (간단한 방법)
        # 토픽 크기의 분산을 역수로 사용 (균형잡힌 토픽 선호)
        topic_sizes = topic_info['Count'].values[1:]  # -1 제외
        if len(topic_sizes) > 0:
            size_variance = np.var(topic_sizes)
            avg_size = np.mean(topic_sizes)
            # 평균 크기가 크고 분산이 작을수록 좋은 점수
            score = avg_size / (1 + size_variance / 100)
        else:
            score = 0

        print(f"    - Score: {score:.4f}")

        model_results.append({
            'nr_topics': nr_topics,
            'n_topics_found': n_topics_found,
            'outliers': n_outliers,
            'score': score
        })

        # 최고 점수 모델 저장
        if score > best_score:
            best_score = score
            best_model = topic_model
            best_n_topics = nr_topics
            print(f"    ✓ 최고 점수 갱신!")

    except Exception as e:
        print(f"    ❌ 오류: {e}")
        continue

# 결과 출력
print(f"\n{'=' * 60}")
print(f"[모델 선택 결과]")
print(f"{'=' * 60}")
for result in model_results:
    marker = "★" if result['nr_topics'] == best_n_topics else " "
    print(f"{marker} nr_topics={result['nr_topics']:2d} | "
          f"발견={result['n_topics_found']:2d} | "
          f"outliers={result['outliers']:4d} | "
          f"score={result['score']:.4f}")

print(f"\n✓ 최적 모델: nr_topics={best_n_topics}, score={best_score:.4f}")

# 6. 최종 모델로 토픽 추출
print(f"\n[6단계] 최종 토픽 추출")

topics, probs = best_model.fit_transform(docs, embeddings)
topic_info = best_model.get_topic_info()

print(f"  ✓ 총 토픽 수: {len(topic_info) - 1} (outlier 제외)")
print(f"  ✓ 문서당 토픽 할당 완료")

# 7. 토픽별 대표 정보 추출
print(f"\n[7단계] 토픽별 대표 정보 추출")

topic_details = []

for topic_id in topic_info['Topic'].values:
    if topic_id == -1:  # outlier 제외
        continue

    # 토픽 키워드
    topic_words = best_model.get_topic(topic_id)
    keywords = ', '.join([word for word, _ in topic_words])

    # 해당 토픽의 문서들
    topic_docs_idx = [i for i, t in enumerate(topics) if t == topic_id]
    topic_docs_count = len(topic_docs_idx)

    # 대표 문서 (확률이 가장 높은 문서 3개)
    if topic_docs_idx:
        # probs가 2D 배열인 경우와 1D 배열인 경우 모두 처리
        topic_probs_subset = []
        for i in topic_docs_idx:
            try:
                if isinstance(probs, np.ndarray) and probs.ndim == 2:
                    # 2D 배열: probs[i, topic_id]
                    if topic_id < probs.shape[1]:
                        topic_probs_subset.append(probs[i, topic_id])
                    else:
                        topic_probs_subset.append(0)
                elif hasattr(probs[i], '__len__'):
                    # 리스트나 1D 배열: probs[i][topic_id]
                    if topic_id < len(probs[i]):
                        topic_probs_subset.append(probs[i][topic_id])
                    else:
                        topic_probs_subset.append(0)
                else:
                    # 스칼라값
                    topic_probs_subset.append(0)
            except:
                topic_probs_subset.append(0)

        if topic_probs_subset:
            top_doc_indices = sorted(range(len(topic_probs_subset)),
                                     key=lambda i: topic_probs_subset[i],
                                     reverse=True)[:3]

            representative_docs = []
            for idx in top_doc_indices:
                doc_idx = topic_docs_idx[idx]
                if doc_idx < len(df):
                    title = df.iloc[doc_idx]['title']
                    representative_docs.append(title)

            rep_docs_str = ' | '.join(representative_docs)
        else:
            rep_docs_str = ""
    else:
        rep_docs_str = ""

    topic_details.append({
        'topic_id': topic_id,
        'count': topic_docs_count,
        'keywords': keywords,
        'representative_documents': rep_docs_str
    })

topics_df = pd.DataFrame(topic_details)
print(f"  ✓ 토픽 정보 추출 완료: {len(topics_df)}개 토픽")

# 8. 토픽별 상세 정보 출력
print(f"\n{'=' * 60}")
print(f"[토픽 상세 정보]")
print(f"{'=' * 60}\n")

for _, row in topics_df.head(10).iterrows():  # 상위 10개만 출력
    print(f"■ Topic {row['topic_id']} (문서 수: {row['count']})")
    print(f"  키워드: {row['keywords']}")
    if row['representative_documents']:
        print(f"  대표 문서:")
        for i, doc in enumerate(row['representative_documents'].split(' | ')[:3], 1):
            print(f"    {i}. {doc[:80]}...")
    print()

# 9. 결과 저장
print(f"{'=' * 60}")
print(f"[8단계] 결과 저장")
print(f"{'=' * 60}\n")

# 토픽 정보 저장
topics_df.to_csv(OUTPUT_TOPICS_FILE, index=False, encoding='utf-8-sig')
print(f"✓ 토픽 정보 저장: {OUTPUT_TOPICS_FILE}")

# 모델 저장
best_model.save(OUTPUT_MODEL_PATH, serialization="pytorch")
print(f"✓ 모델 저장: {OUTPUT_MODEL_PATH}")

# 문서별 토픽 할당 추가
df['topic'] = topics

# topic_probability 계산 (probs 구조에 따라 다르게 처리)
topic_probs = []
for i, prob in enumerate(probs):
    try:
        if isinstance(prob, np.ndarray):
            if prob.ndim == 1 and len(prob) > 0:
                topic_probs.append(prob.max())
            elif prob.ndim == 0:
                topic_probs.append(float(prob))
            else:
                topic_probs.append(0.0)
        elif hasattr(prob, '__len__') and len(prob) > 0:
            topic_probs.append(max(prob))
        else:
            topic_probs.append(0.0)
    except:
        topic_probs.append(0.0)

df['topic_probability'] = topic_probs

output_with_topics = OUTPUT_TOPICS_FILE.replace('topics_', 'corpus_with_topics_')
df.to_csv(output_with_topics, index=False, encoding='utf-8-sig')
print(f"✓ 토픽이 할당된 코퍼스 저장: {output_with_topics}")

print(f"\n{'=' * 60}")
print(f"✓ BERTopic 모델링 완료!")
print(f"✓ 총 토픽 수: {len(topics_df)}")
print(f"✓ 최적 nr_topics: {best_n_topics}")
print(f"✓ 최고 점수: {best_score:.4f}")
print(f"{'=' * 60}")