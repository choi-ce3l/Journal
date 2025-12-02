import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup

# NLP 라이브러리
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# LDA
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt

# 필요한 NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# spaCy 모델 로드
nlp = spacy.load('en_core_web_sm')

# ========================================
# 설정
# ========================================
INPUT_FILE = "/home/dslab/choi/Journal/LLM/Data/Academia_classified.csv"
OUTPUT_TOPICS_FILE = "/home/dslab/choi/Journal/LDA_ALL/Data/topics_academia_lda_bow.csv"
OUTPUT_CORPUS_FILE = "/home/dslab/choi/Journal/LDA_ALL/Data/corpus_with_topics_academia_lda_bow.csv"
OUTPUT_MODEL_PATH = "/home/dslab/choi/Journal/LDA_ALL/Model/lda_model_academia_bow"
OUTPUT_PLOT = "/home/dslab/choi/Journal/LDA_ALL/Visualization/lda_coherence_scores_bow.png"

# LDA 설정
MIN_TOPICS = 8
MAX_TOPICS = 25
N_WORDS = 10
MIN_WORD_LENGTH = 3

# 축소된 도메인 불용어 (핵심만 유지)
DOMAIN_STOPWORDS = {
    # 너무 일반적인 학술 용어만
    'paper', 'study', 'research', 'article', 'journal', 'conference',
    'result', 'finding', 'analysis', 'method', 'approach',

    # 의미없는 동사
    'based', 'using', 'used', 'propose', 'show', 'present',

    # 모호한 형용사
    'different', 'various', 'new', 'several', 'many', 'medium',
    'information', 'news', 'system', 'data'
}
# ========================================

print(f"{'=' * 70}")
print(f"LDA Topic Modeling with BoW - Academia Dataset")
print(f"{'=' * 70}\n")

# ========================================
# 1. Data Loading
# ========================================
print("[Step 1] Data Loading")
df = pd.read_csv(INPUT_FILE)
print(f"  ✓ File: {INPUT_FILE}")
print(f"  ✓ Number of documents: {len(df)}")
print(f"  ✓ Columns: {list(df.columns)}")

# Combine title and abstract
df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
print(f"  ✓ Combined title and abstract")

# ========================================
# 2. Text Preprocessing
# ========================================
print(f"\n[Step 2] Text Preprocessing")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | DOMAIN_STOPWORDS


def preprocess_text(text):
    """보수적 전처리 파이프라인"""
    if pd.isna(text) or text == '':
        return []

    # HTML, 수식 제거
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\\(.*?\\\)', '', text)
    text = re.sub(r'<math>.*?</math>', '', text, flags=re.DOTALL)
    text = re.sub(r'http\S+|www\S+|\S+@\S+|\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    # 불용어 제거
    tokens = [word for word in tokens
              if word not in stop_words and len(word) > MIN_WORD_LENGTH]

    # POS 태깅 - 명사만 (원본 방식 유지)
    pos_tags = nltk.pos_tag(tokens)
    tokens = [word for word, pos in pos_tags if pos.startswith('NN')]

    # Lemmatization - 명사로만
    tokens = [lemmatizer.lemmatize(word, pos='n') for word in tokens]

    # NER - PERSON, ORG 제거
    doc = nlp(' '.join(tokens))
    tokens = [token.text for token in doc
              if token.ent_type_ not in ['PERSON', 'ORG']]

    return tokens


print(f"  Processing documents...")
processed_docs = []
for idx, text in enumerate(df['text']):
    if (idx + 1) % 500 == 0:
        print(f"    Progress: {idx + 1}/{len(df)}")
    processed_docs.append(preprocess_text(text))

df['processed_tokens'] = processed_docs
print(f"  ✓ Preprocessing completed: {len(processed_docs)} documents")

# Remove empty documents
df = df[df['processed_tokens'].apply(len) > 0].reset_index(drop=True)
processed_docs = df['processed_tokens'].tolist()
print(f"  ✓ After removing empty documents: {len(processed_docs)} documents")

# ========================================
# 3. Bigram Detection
# ========================================
print(f"\n[Step 3] Bigram Detection")

# Bigram만 사용 (원본 방식 + threshold만 조정)
bigram = Phrases(processed_docs, min_count=5, threshold=40)  # 중간값
bigram_mod = Phraser(bigram)

processed_docs = [bigram_mod[doc] for doc in processed_docs]
df['processed_tokens'] = processed_docs

print(f"  ✓ Bigram phrases detected")

# Show sample bigrams
bigram_counter = Counter()
for doc in processed_docs:
    bigram_counter.update([token for token in doc if '_' in token])

if bigram_counter:
    print(f"  Top 10 bigrams/trigrams:")
    for bigram, count in bigram_counter.most_common(10):
        print(f"    {bigram}: {count}")

# ========================================
# 4. Dictionary and Corpus Creation
# ========================================
print(f"\n[Step 4] Dictionary and Corpus Creation")

dictionary = corpora.Dictionary(processed_docs)
print(f"  ✓ Initial vocabulary size: {len(dictionary)}")

# 완화된 필터링
dictionary.filter_extremes(
    no_below=3,  # 최소 3문서 이상 등장
    no_above=0.6,  # 70% 이하 문서에서 등장
    keep_n=12000  # 상위 15000 단어 유지
)
print(f"  ✓ Vocabulary size after filtering: {len(dictionary)}")

# BoW corpus 생성
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# 너무 짧은 문서 제거
min_doc_length = 10
valid_indices = [i for i, bow in enumerate(corpus) if len(bow) >= min_doc_length]
corpus = [corpus[i] for i in valid_indices]
df = df.iloc[valid_indices].reset_index(drop=True)
processed_docs = [processed_docs[i] for i in valid_indices]

print(f"  ✓ Final corpus size: {len(corpus)} documents")
print(f"  ✓ Documents with at least {min_doc_length} unique terms")

# ========================================
# 5. Using BoW (No TF-IDF)
# ========================================
print(f"\n[Step 5] Using Bag-of-Words (BoW) Corpus for LDA")
print(f"  ✓ No TF-IDF transformation - using raw BoW")

# ========================================
# 6. Optimal Number of Topics
# ========================================
print(f"\n[Step 6] Finding Optimal Number of Topics with BoW")
print(f"  Topic range: {MIN_TOPICS} to {MAX_TOPICS}")

coherence_scores_cv = []  # c_v 점수 저장
coherence_scores_umass = []  # u_mass 점수 저장
perplexity_scores = []
models = []

for num_topics in range(MIN_TOPICS, MAX_TOPICS + 1):
    print(f"\n  Training model with {num_topics} topics...")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=20,
        iterations=500,
        chunksize=500,
        alpha=0.01,
        eta=0.01,
        per_word_topics=True,
        minimum_probability=0.0,
        update_every=1,
        eval_every=10
    )

    # Coherence score (c_v)
    coherence_model_cv = CoherenceModel(
        model=lda_model,
        texts=processed_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score_cv = coherence_model_cv.get_coherence()

    # Coherence score (u_mass)
    coherence_model_umass = CoherenceModel(
        model=lda_model,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass'
    )
    coherence_score_umass = coherence_model_umass.get_coherence()

    # Perplexity
    perplexity = lda_model.log_perplexity(corpus)

    # 리스트에 추가
    coherence_scores_cv.append(coherence_score_cv)
    coherence_scores_umass.append(coherence_score_umass)
    perplexity_scores.append(perplexity)
    models.append(lda_model)

    print(f"    Coherence Score (c_v): {coherence_score_cv:.4f}")
    print(f"    Coherence Score (u_mass): {coherence_score_umass:.4f}")
    print(f"    Perplexity: {perplexity:.4f}")

# 최적 모델 선택 (c_v Coherence Score 기준)
best_idx = np.argmax(coherence_scores_cv)
best_num_topics = MIN_TOPICS + best_idx
best_model = models[best_idx]
best_coherence_cv = coherence_scores_cv[best_idx]
best_coherence_umass = coherence_scores_umass[best_idx]
best_perplexity = perplexity_scores[best_idx]

print(f"\n{'=' * 70}")
print(f"[Optimal Model Selection]")
print(f"{'=' * 70}")
print(f"✓ Optimal number of topics: {best_num_topics}")
print(f"✓ Coherence Score (c_v): {best_coherence_cv:.4f}")
print(f"✓ Coherence Score (u_mass): {best_coherence_umass:.4f}")
print(f"✓ Perplexity: {best_perplexity:.4f}")

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Coherence (c_v) plot
ax1.plot(range(MIN_TOPICS, MAX_TOPICS + 1), coherence_scores_cv,
         marker='o', linewidth=2, markersize=6, color='blue', label='Coherence (c_v)')
ax1.axvline(x=best_num_topics, color='r', linestyle='--',
            label=f'Optimal: {best_num_topics} topics')
ax1.set_xlabel('Number of Topics', fontsize=12)
ax1.set_ylabel('Coherence Score (c_v)', fontsize=12)
ax1.set_title('Coherence Scores (c_v) for Different Numbers of Topics', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Perplexity plot
ax2.plot(range(MIN_TOPICS, MAX_TOPICS + 1), perplexity_scores,
         marker='s', linewidth=2, markersize=6, color='green')
ax2.axvline(x=best_num_topics, color='r', linestyle='--',
            label=f'Optimal: {best_num_topics} topics')
ax2.set_xlabel('Number of Topics', fontsize=12)
ax2.set_ylabel('Perplexity', fontsize=12)
ax2.set_title('Perplexity for Different Numbers of Topics', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"✓ Coherence plot saved: {OUTPUT_PLOT}")
plt.close()

# ========================================
# 7. Topic Information Extraction
# ========================================
print(f"\n[Step 7] Topic Information Extraction")

topic_details = []

for topic_id in range(best_num_topics):
    # Get top N keywords with probabilities
    topic_words = best_model.show_topic(topic_id, topn=N_WORDS)
    keywords = ', '.join([f"{word}({prob:.3f})" for word, prob in topic_words])

    # Find documents assigned to this topic
    topic_docs_idx = []
    for doc_idx, bow in enumerate(corpus):
        doc_topics = best_model.get_document_topics(bow, minimum_probability=0.01)
        if doc_topics:
            main_topic = max(doc_topics, key=lambda x: x[1])
            if main_topic[0] == topic_id:
                topic_docs_idx.append(doc_idx)

    topic_count = len(topic_docs_idx)

    # Get representative documents
    representative_docs = []
    if topic_docs_idx:
        doc_probs = []
        for doc_idx in topic_docs_idx[:100]:
            bow = corpus[doc_idx]
            doc_topics = best_model.get_document_topics(bow)
            topic_prob = dict(doc_topics).get(topic_id, 0)
            doc_probs.append((doc_idx, topic_prob))

        doc_probs.sort(key=lambda x: x[1], reverse=True)

        for doc_idx, prob in doc_probs[:3]:
            title = df.iloc[doc_idx]['title']
            representative_docs.append(f"{title} (prob: {prob:.3f})")

    rep_docs_str = ' | '.join(representative_docs)

    topic_details.append({
        'topic_id': topic_id,
        'count': topic_count,
        'keywords': keywords,
        'representative_documents': rep_docs_str
    })

topics_df = pd.DataFrame(topic_details)
print(f"  ✓ Topic extraction completed: {len(topics_df)} topics")

# ========================================
# 8. Document-Topic Assignment
# ========================================
print(f"\n[Step 8] Document-Topic Assignment")

doc_topics = []
doc_topic_probs = []

for bow in corpus:
    topics = best_model.get_document_topics(bow, minimum_probability=0.01)
    if topics:
        main_topic = max(topics, key=lambda x: x[1])
        doc_topics.append(main_topic[0])
        doc_topic_probs.append(main_topic[1])
    else:
        doc_topics.append(-1)
        doc_topic_probs.append(0.0)

df['topic'] = doc_topics
df['topic_probability'] = doc_topic_probs
print(f"  ✓ Document-topic assignment completed")

# ========================================
# 9. Display Results
# ========================================
print(f"\n{'=' * 70}")
print(f"[Topic Details]")
print(f"{'=' * 70}\n")

for _, row in topics_df.iterrows():
    print(f"■ Topic {row['topic_id']} (Documents: {row['count']})")
    print(f"  Keywords: {row['keywords']}")
    if row['representative_documents']:
        print(f"  Representative documents:")
        for i, doc in enumerate(row['representative_documents'].split(' | ')[:3], 1):
            print(f"    {i}. {doc[:100]}...")
    print()

# ========================================
# 10. Topic Distribution Analysis
# ========================================
print(f"\n{'=' * 70}")
print(f"[Topic Distribution Analysis]")
print(f"{'=' * 70}\n")

topic_counts = df[df['topic'] != -1]['topic'].value_counts().sort_index()
avg_prob = df[df['topic'] != -1].groupby('topic')['topic_probability'].mean()

print("Documents per topic (with average probability):")
for topic_id in range(best_num_topics):
    count = topic_counts.get(topic_id, 0)
    pct = count / len(df) * 100
    avg_p = avg_prob.get(topic_id, 0)
    print(f"  Topic {topic_id:2d}: {count:5d} documents ({pct:5.1f}%) - Avg prob: {avg_p:.3f}")

# Calculate topic concentration (Gini coefficient)
counts = topic_counts.values
if len(counts) > 0:
    gini = np.sum(np.abs(counts[:, None] - counts[None, :])) / (2 * len(counts) * np.sum(counts))
    print(f"\n✓ Topic distribution Gini coefficient: {gini:.4f}")
    print(f"  (0 = perfectly balanced, 1 = completely concentrated)")
else:
    gini = 0.0
    print(f"\n✓ Warning: Could not calculate Gini coefficient")

# ========================================
# 11. Save Results
# ========================================
print(f"\n{'=' * 70}")
print(f"[Step 11] Save Results")
print(f"{'=' * 70}\n")

# Save topic information
topics_df.to_csv(OUTPUT_TOPICS_FILE, index=False, encoding='utf-8-sig')
print(f"✓ Topic information saved: {OUTPUT_TOPICS_FILE}")

# Save corpus with topic assignments
df.to_csv(OUTPUT_CORPUS_FILE, index=False, encoding='utf-8-sig')
print(f"✓ Corpus saved: {OUTPUT_CORPUS_FILE}")

# Save model
best_model.save(OUTPUT_MODEL_PATH)
dictionary.save(f"{OUTPUT_MODEL_PATH}.dict")
corpora.MmCorpus.serialize(f"{OUTPUT_MODEL_PATH}.mm", corpus)
print(f"✓ Model saved: {OUTPUT_MODEL_PATH}")

# ========================================
# 12. Summary Statistics
# ========================================
print(f"\n{'=' * 70}")
print(f"✓ LDA Topic Modeling with BoW Completed!")
print(f"{'=' * 70}")
print(f"✓ Optimal number of topics: {best_num_topics}")
print(f"✓ Coherence Score (c_v): {best_coherence_cv:.4f}")
print(f"✓ Coherence Score (u_mass): {best_coherence_umass:.4f}")
print(f"✓ Total documents: {len(df)}")
print(f"✓ Topic distribution Gini: {gini:.4f}")
print(f"{'=' * 70}")