import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ===========================
# CONFIGURATION SECTION
# ===========================
class Config:
    """
    모든 설정을 한 곳에서 관리하는 클래스
    """
    # 입력 파일 경로 (Academia만 사용)
    ACADEMIA_FILE = '/home/dslab/choi/Journal/LDA_ALL/Data/corpus_with_topics_academia_lda_bow.csv'

    # 데이터 컬럼명
    TEXT_COLUMN = 'abstract'  # 텍스트가 들어있는 컬럼
    TITLE_COLUMN = 'title'  # 제목 컬럼 (선택사항)
    ID_COLUMN = None  # 문서 ID 컬럼 (없으면 None, 자동으로 인덱스 사용)
    TOPIC_COLUMN = 'topic'  # 토픽 번호가 들어있는 컬럼

    # 텍스트 결합 방식
    USE_TITLE = False  # True면 title + abstract 결합, False면 abstract만
    TITLE_WEIGHT = 1  # 제목에 더 많은 가중치를 주고 싶으면 증가

    # 모델 설정
    MODEL_NAME = 'all-MiniLM-L6-v2'  # 'all-MiniLM-L6-v2' 또는 'all-mpnet-base-v2'

    # 출력 디렉토리 및 파일명
    OUTPUT_DIR = '/home/dslab/choi/Journal/LDA_ALL/Result'
    CLASSIFICATION_OUTPUT = 'academia_type_classification.csv'
    TOPIC_STATS_OUTPUT = 'topic_type_distribution.csv'
    TYPE_STATS_OUTPUT = 'type_distribution_summary.csv'

    # 시각화 파일
    TOPIC_HEATMAP = 'topic_type_heatmap.png'
    TYPE_DISTRIBUTION = 'type_distribution_chart.png'
    TOPIC_BARS = 'topic_dominant_types.png'

    # 시각화 설정
    SAVE_PLOTS = True
    FIGURE_SIZE = (15, 10)
    DPI = 300

    @classmethod
    def get_output_path(cls, filename):
        """출력 파일 경로 생성"""
        output_dir = Path(cls.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        return output_dir / filename

    @classmethod
    def print_config(cls):
        """현재 설정 출력"""
        print("\n" + "=" * 80)
        print("CURRENT CONFIGURATION")
        print("=" * 80)
        print(f"\n[Input File]")
        print(f"  Academia: {cls.ACADEMIA_FILE}")
        print(f"\n[Data Columns]")
        print(f"  Text Column: {cls.TEXT_COLUMN}")
        print(f"  Title Column: {cls.TITLE_COLUMN}")
        print(f"  ID Column: {cls.ID_COLUMN}")
        print(f"  Topic Column: {cls.TOPIC_COLUMN}")
        print(f"\n[Text Processing]")
        print(f"  Use Title: {cls.USE_TITLE}")
        if cls.USE_TITLE:
            print(f"  Title Weight: {cls.TITLE_WEIGHT}x")
        print(f"\n[Model Settings]")
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"\n[Classification Mode]")
        print(f"  Assumption: All documents are PURE (assign to most similar type)")
        print(f"\n[Output Settings]")
        print(f"  Output Directory: {cls.OUTPUT_DIR}")
        print("=" * 80 + "\n")


# ===========================
# Type 정의 (5개)
# ===========================
type_definitions = {
    "Type 1": """
Type 1 — ML Methods Development (Padmanabhan et al., 2022).
Research that develops new or significantly improved ML algorithms, models, or computational frameworks 
to solve business, technical, or social problems. The central contribution is methodological or artifact-level 
novelty. Domain insights, empirical findings, or social/organizational implications are secondary.

Core characteristics:
- Introduces a novel ML method, computational design, or algorithmic enhancement.
- Evaluates performance using benchmarks, robustness tests, ablation studies, or simulations.
- Emphasizes generalizable computational contribution rather than IS-specific theorization.

Typical cues: develop, design, propose, novel model, framework, performance improvement, robustness.
""",

    "Type 2": """
Type 2 — Understanding Phenomena Using ML (Padmanabhan et al., 2022).
Research that applies existing ML techniques as analytical tools to extract variables, predict outcomes, 
or measure constructs for understanding IS-related phenomena. ML is instrumental—not the contribution.
The main contribution is empirical or theoretical insight about users, organizations, markets, platforms, 
or societal mechanisms.

Core characteristics:
- ML used to construct features, extract constructs (psychometric or behavioral), or enable causal inference.
- Theoretical contribution concerns human/organizational/market dynamics, not ML algorithms.
- May combine ML-derived variables with econometric, behavioral, or organizational theory.

Typical cues: apply, use, predict, analyze, causal effect, behavioral pattern, empirical insights.
""",

    "Type 3": """
Type 3 — ML in Complex Systems (Padmanabhan et al., 2022).
Research examining ML as a component within complex adaptive or sociotechnical systems.  
The contribution focuses on *interaction, feedback, co-evolution, or emergent outcomes* resulting from 
ML embedded within human, organizational, or multi-agent environments. This type investigates how ML 
affects system dynamics and how the system adapts to ML.

Core characteristics:
- ML interacts with humans, organizations, or other algorithms as part of a system.
- Studies feedback loops, performative effects, systemic risks, or multi-agent dynamics.
- Often uses simulations, agent-based models, or experiments to analyze dynamic co-adaptation.

Typical cues: feedback, emergent, adaptive, agent-based, simulation, socio-technical interaction.
""",

    "Type 4": """
Type 4 — Predominantly Social (Sarker et al., 2019 — Sociotechnical Type I/II).
Research where the primary focus is on social, organizational, institutional, or behavioral phenomena. 
Technology (including ML) appears only as background, context, or one of many environmental factors.  
The theoretical contribution concerns human actors, routines, norms, power structures, identity, or 
organizational processes—not the technology itself.

Core characteristics:
- Social component dominates; technology is peripheral or indirect.
- Explains outcomes using social or organizational theory (e.g., identity, trust, culture, governance).
- Reflects Sarker's “Predominantly Social” or “Social Imperative on the Technical.”

Typical cues: organizational change, trust, identity, institutional pressure, power, collaboration, culture.
""",

    "Type 5": """
Type 5 — Predominantly Technical (Sarker et al., 2019 — Sociotechnical Type VI).
Research where the technical artifact dominates and social or organizational aspects are minimal or absent.  
Different from Type 1: it does *not* introduce novel ML methods, but focuses on system architecture, 
infrastructure, performance engineering, or automation. The purpose is technical optimization, not 
methodological innovation or sociotechnical insight.

Core characteristics:
- Technical system design, deployment, infrastructure, or optimization is the main focus.
- Lacks theorization of human, organizational, or institutional dynamics.
- Mirrors Sarker’s “Predominantly Technical” category on the sociotechnical continuum.

Typical cues: system architecture, automation, throughput, latency, deployment, performance optimization.
"""
}


# ===========================
# 임베딩 모델 로드
# ===========================
def load_embedding_model(model_name=None):
    """임베딩 모델 로드"""
    if model_name is None:
        model_name = Config.MODEL_NAME

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    return model


# ===========================
# Type 정의 임베딩
# ===========================
def embed_type_definitions(model, type_definitions):
    """Type 정의를 임베딩"""
    print("Embedding type definitions...")
    type_embeddings = {}
    for type_name, definition in type_definitions.items():
        type_embeddings[type_name] = model.encode(definition, convert_to_tensor=False)
    return type_embeddings


# ===========================
# 문서 분류 함수 (5번 반복 + 다수결)
# ===========================
def classify_documents(documents, doc_ids, topics, model, type_embeddings, n_iterations=5):
    """
    문서를 5개 Type으로 분류 (5번 반복 실행 후 다수결로 최종 결정)

    Parameters:
    -----------
    n_iterations : int
        분류를 반복할 횟수 (기본값: 5)
    """
    print(f"Embedding {len(documents)} documents (will run {n_iterations} times for voting)...")

    # 최종 결과를 저장할 리스트
    results = []

    # 각 문서별로 반복 실행 결과를 저장
    all_iterations_results = {i: [] for i in range(len(documents))}

    # n_iterations 번 반복
    for iteration in range(n_iterations):
        print(f"\n  → Iteration {iteration + 1}/{n_iterations}...")

        # 문서 임베딩 (매번 새로 계산 - 약간의 변동성 있음)
        doc_embeddings = model.encode(documents, convert_to_tensor=False, show_progress_bar=True)

        # 각 문서에 대해 유사도 계산 및 Type 할당
        for i, (doc_id, topic, doc_emb) in enumerate(zip(doc_ids, topics, doc_embeddings)):
            similarities = {}

            # 각 Type과의 유사도 계산
            for type_name, type_emb in type_embeddings.items():
                sim = cosine_similarity(
                    doc_emb.reshape(1, -1),
                    type_emb.reshape(1, -1)
                )[0][0]
                similarities[type_name] = sim

            # 최대 유사도를 가진 Type으로 할당
            assigned_type = max(similarities, key=similarities.get)

            # 이번 iteration 결과 저장
            all_iterations_results[i].append({
                'iteration': iteration,
                'assigned_type': assigned_type,
                'similarities': similarities.copy()
            })

    print("\n  → Calculating final types by majority voting...")

    # 다수결로 최종 Type 결정
    for i, (doc_id, topic) in enumerate(zip(doc_ids, topics)):
        # 각 iteration에서 할당된 Type 수집
        assigned_types = [result['assigned_type'] for result in all_iterations_results[i]]

        # 가장 많이 나온 Type 찾기 (다수결)
        from collections import Counter
        type_counter = Counter(assigned_types)
        final_type = type_counter.most_common(1)[0][0]
        vote_count = type_counter.most_common(1)[0][1]

        # 평균 유사도 계산 (모든 iteration의 평균)
        avg_similarities = {}
        for type_name in type_embeddings.keys():
            sims = [result['similarities'][type_name] for result in all_iterations_results[i]]
            avg_similarities[type_name] = np.mean(sims)

        final_max_sim = avg_similarities[final_type]

        # 결과 저장
        result_dict = {
            "Document_ID": doc_id,
            "Topic": topic,
            "Assigned_Type": final_type,
            "Vote_Count": vote_count,  # 다수결에서 몇 번 선택되었는지
            "Total_Votes": n_iterations,
            "Vote_Percentage": (vote_count / n_iterations) * 100,
            "Max_Similarity": final_max_sim
        }

        # 각 Type별 평균 유사도 추가
        for type_name in sorted(type_embeddings.keys()):
            result_dict[f"{type_name}_Similarity"] = avg_similarities[type_name]

        # 각 iteration별 할당 결과 추가 (디버깅용)
        for iter_num in range(n_iterations):
            result_dict[f"Iter_{iter_num + 1}_Type"] = all_iterations_results[i][iter_num]['assigned_type']

        results.append(result_dict)

    # 다수결 통계 출력
    print("\n  → Voting Statistics:")
    vote_percentages = [r['Vote_Percentage'] for r in results]
    print(f"     Average vote agreement: {np.mean(vote_percentages):.1f}%")
    print(f"     100% agreement: {sum(1 for v in vote_percentages if v == 100)} documents")
    print(f"     < 60% agreement: {sum(1 for v in vote_percentages if v < 60)} documents")

    return pd.DataFrame(results)


# ===========================
# 토픽별 Type 분포 분석
# ===========================
def analyze_topic_type_distribution(df_results):
    """
    토픽별로 어떤 Type을 품고 있는지 분석 (Crosstab 방식)
    """
    print("\n" + "=" * 80)
    print("TOPIC-LEVEL TYPE DISTRIBUTION")
    print("=" * 80)

    # Crosstab으로 빈도 계산 (Topic x Type)
    cross_tab = pd.crosstab(df_results['Topic'], df_results['Assigned_Type'])

    # Type 컬럼 순서 정렬 (Type 1, 2, 3, 4, 5)
    type_cols = [f'Type {i}' for i in range(1, 6)]
    available_types = [col for col in type_cols if col in cross_tab.columns]
    cross_tab = cross_tab[available_types]

    print("\n[1] Type별 Topic 빈도 (문서 개수):")
    print("-" * 80)
    print(cross_tab)
    print()

    # 백분율로 변환 (각 Topic별로 100%가 되도록)
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    print("\n[2] Type별 Topic 분포 (%):")
    print("-" * 80)
    print(cross_tab_pct.round(2))
    print()

    # 각 토픽의 주도적인 Type 계산
    dominant_types = cross_tab.idxmax(axis=1)
    dominant_pcts = cross_tab_pct.max(axis=1)

    print("\n[3] 각 Topic의 Dominant Type:")
    print("-" * 80)
    for topic in cross_tab.index:
        total_docs = cross_tab.loc[topic].sum()
        dom_type = dominant_types[topic]
        dom_pct = dominant_pcts[topic]
        print(f"Topic {topic:2d} ({int(total_docs):3d} docs): {dom_type} ({dom_pct:.1f}%)")

    # 전체 요약
    print("\n" + "-" * 80)
    print("OVERALL SUMMARY:")
    print("-" * 80)

    dominant_type_counts = dominant_types.value_counts()
    total_topics = len(cross_tab)

    for type_name in available_types:
        count = dominant_type_counts.get(type_name, 0)
        pct = (count / total_topics * 100) if total_topics > 0 else 0
        topics_list = dominant_types[dominant_types == type_name].index.tolist()
        print(f"{type_name}: {count}/{total_topics} topics ({pct:.1f}%) - {topics_list}")

    # DataFrame으로 변환하여 반환 (기존 형식 유지를 위해)
    topic_stats = []
    for topic in cross_tab.index:
        result = {
            'Topic': topic,
            'Total_Docs': int(cross_tab.loc[topic].sum())
        }

        for type_num in range(1, 6):
            type_name = f'Type {type_num}'
            if type_name in cross_tab.columns:
                result[f'Type_{type_num}_Count'] = int(cross_tab.loc[topic, type_name])
                result[f'Type_{type_num}_Percentage'] = cross_tab_pct.loc[topic, type_name]
            else:
                result[f'Type_{type_num}_Count'] = 0
                result[f'Type_{type_num}_Percentage'] = 0.0

        result['Dominant_Type'] = dominant_types[topic]
        result['Dominant_Type_Percentage'] = dominant_pcts[topic]

        topic_stats.append(result)

    return pd.DataFrame(topic_stats)


# ===========================
# 전체 Type 분포 통계
# ===========================
def analyze_type_distribution(df_results):
    """전체 문서의 Type 분포 분석"""
    print("\n" + "=" * 80)
    print("OVERALL TYPE DISTRIBUTION")
    print("=" * 80)

    total_docs = len(df_results)
    print(f"\nTotal Documents: {total_docs}")

    # Type별 분포
    print("\n" + "-" * 80)
    print("Type Distribution:")
    print("-" * 80)

    type_counts = df_results['Assigned_Type'].value_counts()
    type_stats_data = []

    for type_num in range(1, 6):
        type_name = f'Type {type_num}'
        count = type_counts.get(type_name, 0)
        percentage = (count / total_docs) * 100
        print(f"{type_name:12s}: {count:4d} ({percentage:5.1f}%)")
        type_stats_data.append({
            'Type': type_name,
            'Count': count,
            'Percentage': percentage
        })

    # 평균 유사도
    print("\n" + "-" * 80)
    print("Average Similarities:")
    print("-" * 80)
    for type_num in range(1, 6):
        col_name = f'Type {type_num}_Similarity'
        if col_name in df_results.columns:
            avg_sim = df_results[col_name].mean()
            print(f"Type {type_num}: {avg_sim:.4f}")
    print(f"Max (Assigned Type): {df_results['Max_Similarity'].mean():.4f}")

    return pd.DataFrame(type_stats_data)


# ===========================
# 토픽-Type 히트맵 시각화
# ===========================
def visualize_topic_type_heatmap(df_topic_stats, save_path=None):
    """토픽별 Type 분포를 히트맵으로 시각화"""

    # 컬럼 존재 확인
    type_columns = [f'Type_{i}_Percentage' for i in range(1, 6)]

    # 피벗 테이블 생성
    pivot_data = df_topic_stats.set_index('Topic')[type_columns]
    pivot_data.columns = [f'Type {i}' for i in range(1, 6)]

    # 히트맵 그리기
    plt.figure(figsize=(10, max(8, len(pivot_data) * 0.4)))

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title('Type Distribution by Topic (Academia)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('ML Research Type', fontsize=12)
    plt.ylabel('Topic', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"  ✓ Topic heatmap saved: {save_path}")

    plt.show()


# ===========================
# 전체 Type 분포 시각화
# ===========================
def visualize_type_distribution(df_type_stats, save_path=None):
    """전체 Type 분포를 막대 그래프와 파이 차트로 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 막대 그래프
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    axes[0].bar(df_type_stats['Type'], df_type_stats['Percentage'], color=colors)
    axes[0].set_title('Type Distribution - Bar Chart', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Type', fontsize=12)
    axes[0].set_ylabel('Percentage (%)', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # 막대 위에 퍼센트 표시
    for i, (type_name, pct) in enumerate(zip(df_type_stats['Type'], df_type_stats['Percentage'])):
        axes[0].text(i, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    # 2. 파이 차트
    axes[1].pie(df_type_stats['Percentage'], labels=df_type_stats['Type'],
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Type Distribution - Pie Chart', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"  ✓ Type distribution chart saved: {save_path}")

    plt.show()


# ===========================
# 토픽별 주도 Type 시각화
# ===========================
def visualize_topic_dominant_types(df_topic_stats, save_path=None):
    """각 토픽의 주도적인 Type을 막대 그래프로 시각화"""

    plt.figure(figsize=(12, max(6, len(df_topic_stats) * 0.3)))

    # 토픽별 주도 Type 데이터 준비
    topics = df_topic_stats['Topic'].tolist()
    dominant_types = df_topic_stats['Dominant_Type'].tolist()
    dominant_pcts = df_topic_stats['Dominant_Type_Percentage'].tolist()

    # Type별 색상 매핑
    type_colors = {
        'Type 1': '#1f77b4',
        'Type 2': '#ff7f0e',
        'Type 3': '#2ca02c',
        'Type 4': '#d62728',
        'Type 5': '#9467bd'
    }

    colors = [type_colors.get(t, '#gray') for t in dominant_types]

    # 막대 그래프
    bars = plt.barh(range(len(topics)), dominant_pcts, color=colors)

    # 토픽 레이블 설정
    plt.yticks(range(len(topics)), [f'Topic {t}' for t in topics])
    plt.xlabel('Dominant Type Percentage (%)', fontsize=12)
    plt.ylabel('Topic', fontsize=12)
    plt.title('Dominant Type by Topic', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)

    # 막대 옆에 Type 이름 표시
    for i, (pct, dtype) in enumerate(zip(dominant_pcts, dominant_types)):
        plt.text(pct + 1, i, f'{dtype} ({pct:.1f}%)',
                 va='center', fontsize=9)

    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=type_name)
                       for type_name, color in type_colors.items()]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"  ✓ Topic dominant types chart saved: {save_path}")

    plt.show()


# ===========================
# 데이터 로드 함수
# ===========================
def load_data(file_path, text_column=None, title_column=None, id_column=None,
              topic_column=None, use_title=None):
    """
    데이터 파일 로드 및 title + abstract 결합
    """
    if text_column is None:
        text_column = Config.TEXT_COLUMN
    if title_column is None:
        title_column = Config.TITLE_COLUMN
    if id_column is None:
        id_column = Config.ID_COLUMN
    if topic_column is None:
        topic_column = Config.TOPIC_COLUMN
    if use_title is None:
        use_title = Config.USE_TITLE

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 파일 로드
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # 필요한 컬럼 확인
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available: {df.columns.tolist()}")

    if topic_column not in df.columns:
        raise ValueError(f"Topic column '{topic_column}' not found. Available: {df.columns.tolist()}")

    # Title + Abstract 결합
    if use_title and title_column in df.columns:
        print(f"  → Combining title and abstract (title weight: {Config.TITLE_WEIGHT}x)")

        # Title을 여러 번 반복하여 가중치 부여
        df['combined_text'] = df.apply(
            lambda row: ' '.join([str(row[title_column])] * int(Config.TITLE_WEIGHT) +
                                 [str(row[text_column])])
            if pd.notna(row[title_column]) and pd.notna(row[text_column])
            else str(row[text_column]),
            axis=1
        )
        docs = df['combined_text'].tolist()
    else:
        if use_title:
            print(f"  Warning: Title column '{title_column}' not found. Using abstract only.")
        docs = df[text_column].tolist()

    # Topic 정보 추출
    topics = df[topic_column].tolist()

    # ID 컬럼 처리
    if id_column is None or id_column not in df.columns:
        if id_column is not None:
            print(f"  Warning: ID column '{id_column}' not found. Using row index as ID.")
        else:
            print(f"  Info: No ID column specified. Using row index as ID.")
        ids = list(range(len(docs)))
    else:
        ids = df[id_column].tolist()

    return docs, ids, topics, len(df)


# ===========================
# 메인 실행 함수
# ===========================
def main(config=None):
    """
    메인 실행 함수 (Academia only, Pure 가정)
    """
    # Config 출력
    if config is None:
        config = Config

    config.print_config()

    print("=" * 80)
    print("ML TYPE CLASSIFICATION: ACADEMIA (5 Types, Pure Assumption)")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[Step 1] Loading data...")

    try:
        docs, ids, topics, doc_count = load_data(
            config.ACADEMIA_FILE,
            config.TEXT_COLUMN,
            config.TITLE_COLUMN,
            config.ID_COLUMN,
            config.TOPIC_COLUMN,
            config.USE_TITLE
        )
        print(f"  ✓ Loaded: {doc_count} documents, {len(set(topics))} unique topics")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return None, None, None

    # 2. 모델 로드 및 Type 임베딩
    print("\n[Step 2] Loading model and embedding type definitions...")
    model = load_embedding_model(config.MODEL_NAME)
    type_embeddings = embed_type_definitions(model, type_definitions)

    # 3. 문서 분류 (Pure 가정)
    print("\n[Step 3] Classifying documents (Pure assumption)...")
    df_results = classify_documents(docs, ids, topics, model, type_embeddings)

    # 4. 전체 Type 분포 분석
    print("\n[Step 4] Analyzing overall type distribution...")
    df_type_stats = analyze_type_distribution(df_results)

    # 5. 토픽별 Type 분포 분석
    print("\n[Step 5] Analyzing topic-level type distribution...")
    df_topic_stats = analyze_topic_type_distribution(df_results)

    # 6. 결과 저장
    print("\n[Step 6] Saving results...")
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # 문서별 분류 결과
    df_results.to_csv(config.get_output_path(config.CLASSIFICATION_OUTPUT), index=False)
    print(f"  ✓ {config.get_output_path(config.CLASSIFICATION_OUTPUT)}")

    # 토픽별 통계
    df_topic_stats.to_csv(config.get_output_path(config.TOPIC_STATS_OUTPUT), index=False)
    print(f"  ✓ {config.get_output_path(config.TOPIC_STATS_OUTPUT)}")

    # 전체 Type 통계
    df_type_stats.to_csv(config.get_output_path(config.TYPE_STATS_OUTPUT), index=False)
    print(f"  ✓ {config.get_output_path(config.TYPE_STATS_OUTPUT)}")

    # 7. 시각화
    print("\n[Step 7] Creating visualizations...")

    # 토픽별 히트맵
    visualize_topic_type_heatmap(
        df_topic_stats,
        config.get_output_path(config.TOPIC_HEATMAP)
    )

    # 전체 Type 분포
    visualize_type_distribution(
        df_type_stats,
        config.get_output_path(config.TYPE_DISTRIBUTION)
    )

    # 토픽별 주도 Type
    visualize_topic_dominant_types(
        df_topic_stats,
        config.get_output_path(config.TOPIC_BARS)
    )

    # 8. 최종 요약
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n[Topics by Dominant Type]")
    for type_num in range(1, 6):
        type_name = f'Type {type_num}'
        topic_list = df_topic_stats[df_topic_stats['Dominant_Type'] == type_name]['Topic'].tolist()
        print(f"  {type_name}: {len(topic_list)} topics - {topic_list}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    return df_results, df_topic_stats, df_type_stats


# ===========================
# 사용 예시
# ===========================
if __name__ == "__main__":
    # 기본 Config 사용
    results, topic_stats, type_stats = main()

    # 커스텀 Config 사용 예시
    # class CustomConfig(Config):
    #     ACADEMIA_FILE = 'my_data/academic_papers.csv'
    #     TEXT_COLUMN = 'abstract'
    #     TITLE_COLUMN = 'title'
    #     TOPIC_COLUMN = 'topic_id'
    #     USE_TITLE = True
    #     TITLE_WEIGHT = 2.0
    #     MODEL_NAME = 'all-mpnet-base-v2'
    #
    # results, topic_stats, type_stats = main(CustomConfig)