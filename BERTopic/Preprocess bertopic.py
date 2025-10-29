import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import spacy
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')

# NLTK 데이터 다운로드 (최초 1회)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

# spaCy 모델 로드
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("spaCy 모델을 다운로드합니다...")
    import subprocess

    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# ========================================
# 설정
# ========================================
INPUT_FILE = "/mnt/user-data/outputs/Academia.csv"
OUTPUT_FILE = "/mnt/user-data/outputs/academia_corpus.csv"

# 사용자 정의 불용어
CUSTOM_STOPWORDS = {
    'information', 'system', 'information system',
    'research', 'study', 'paper', 'article',
    'data', 'result', 'analysis', 'finding'
}

# POS 태그 필터 (명사·동사 계열만 유지)
ALLOWED_POS = {
    'NN', 'NNS', 'NNP', 'NNPS',  # 명사
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'  # 동사
}
# ========================================

print(f"{'=' * 60}")
print(f"BERTopic 전처리 시작")
print(f"{'=' * 60}\n")

# 1. 데이터 로드
print("[1단계] 데이터 로드")
df = pd.read_csv(INPUT_FILE)
print(f"  ✓ 파일: {INPUT_FILE}")
print(f"  ✓ 행 수: {len(df)}")
print(f"  ✓ 컬럼: {list(df.columns)}")

# 2. title과 abstract 결합
print(f"\n[2단계] title과 abstract 결합")
df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
print(f"  ✓ 결합된 텍스트 생성 완료")

# 3. NaN 및 빈 텍스트 처리
print(f"\n[3단계] NaN 및 빈 텍스트 제거")
before_nan = len(df)
df = df[df['text'].notna() & (df['text'].str.strip() != '')]
after_nan = len(df)
print(f"  - 제거 전: {before_nan}개")
print(f"  - 제거 후: {after_nan}개")
print(f"  - 제거됨: {before_nan - after_nan}개")

# 4. 전처리 함수 정의
print(f"\n[4단계] 전처리 함수 정의")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(CUSTOM_STOPWORDS)


def clean_text(text):
    """텍스트 전처리 파이프라인"""

    # 1) HTML 태그 제거
    text = BeautifulSoup(text, 'html.parser').get_text()

    # 2) 수식 제거 (LaTeX, MathML 등)
    text = re.sub(r'\$.*?\$', '', text)  # LaTeX inline
    text = re.sub(r'\\\[.*?\\\]', '', text)  # LaTeX display
    text = re.sub(r'<math>.*?</math>', '', text)  # MathML

    # 3) 알파벳과 공백만 유지, 소문자화
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()

    # 4) 토큰화
    tokens = word_tokenize(text)

    # 5) 불용어 제거 & 길이 > 2 필터
    tokens = [token for token in tokens
              if token not in stop_words and len(token) > 2]

    # 6) POS 태그 필터링 (명사·동사만)
    pos_tagged = pos_tag(tokens)
    tokens = [word for word, pos in pos_tagged if pos in ALLOWED_POS]

    # 7) Lemmatization
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]

    return tokens


def remove_person_entities(tokens):
    """spaCy NER로 PERSON 엔터티 제거"""
    text = ' '.join(tokens)
    doc = nlp(text)

    # PERSON 엔터티 추출
    person_entities = {ent.text.lower() for ent in doc.ents if ent.label_ == 'PERSON'}

    # PERSON 엔터티 제거
    filtered_tokens = [token for token in tokens
                       if token.lower() not in person_entities]

    return filtered_tokens


print(f"  ✓ 전처리 함수 정의 완료")

# 5. 전처리 실행
print(f"\n[5단계] 전처리 실행 (시간이 소요될 수 있습니다...)")

processed_texts = []
valid_indices = []

for idx, text in enumerate(df['text']):
    if idx % 100 == 0:
        print(f"  처리 중... {idx}/{len(df)} ({idx / len(df) * 100:.1f}%)")

    try:
        # 토큰화 및 전처리
        tokens = clean_text(text)

        # PERSON 엔터티 제거
        tokens = remove_person_entities(tokens)

        # 최종 텍스트로 변환
        processed_text = ' '.join(tokens)

        # 빈 텍스트가 아닌 경우만 저장
        if processed_text.strip():
            processed_texts.append(processed_text)
            valid_indices.append(idx)

    except Exception as e:
        print(f"    ⚠ 오류 (index {idx}): {e}")
        continue

print(f"  ✓ 전처리 완료: {len(processed_texts)}개 문서")

# 6. 최종 데이터프레임 생성
print(f"\n[6단계] 최종 데이터프레임 생성")

# 유효한 인덱스의 원본 데이터 선택
final_df = df.iloc[valid_indices].copy()
final_df['processed_text'] = processed_texts

# 필요한 컬럼만 선택
if 'date' in final_df.columns:
    # date에서 연도 추출
    def extract_year(date_str):
        if pd.isna(date_str):
            return None
        match = re.search(r'(\d{4})', str(date_str))
        if match:
            return int(match.group(1))
        return None


    final_df['year'] = final_df['date'].apply(extract_year)

output_columns = ['title', 'abstract', 'processed_text', 'affiliations']
if 'year' in final_df.columns:
    output_columns.append('year')
if 'date' in final_df.columns:
    output_columns.append('date')

corpus_df = final_df[output_columns].copy()

print(f"  ✓ 최종 행 수: {len(corpus_df)}")
print(f"  ✓ 컬럼: {list(corpus_df.columns)}")

# 7. 통계 확인
print(f"\n[7단계] 통계 확인")

# 토큰 수 통계
token_counts = corpus_df['processed_text'].str.split().str.len()
print(f"  ■ 토큰 수 통계:")
print(f"    - 평균: {token_counts.mean():.1f}")
print(f"    - 중앙값: {token_counts.median():.1f}")
print(f"    - 최소: {token_counts.min()}")
print(f"    - 최대: {token_counts.max()}")

# Affiliations별 통계
if 'affiliations' in corpus_df.columns:
    print(f"\n  ■ Affiliations별 문서 수:")
    affil_stats = corpus_df['affiliations'].value_counts()
    for affil, count in affil_stats.items():
        print(f"    - {affil}: {count}개")

# 연도별 통계
if 'year' in corpus_df.columns:
    print(f"\n  ■ 연도별 문서 수:")
    year_stats = corpus_df['year'].value_counts().sort_index()
    for year, count in year_stats.items():
        print(f"    - {year}: {count}개")

# 8. 결과 저장
print(f"\n{'=' * 60}")
print(f"[8단계] 결과 저장")
print(f"{'=' * 60}\n")

corpus_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"✓ 전처리 완료!")
print(f"✓ 파일 경로: {OUTPUT_FILE}")
print(f"✓ 최종 문서 수: {len(corpus_df)}")
print(f"✓ 컬럼: {list(corpus_df.columns)}")

# 샘플 데이터 미리보기
print(f"\n[데이터 미리보기]")
print(corpus_df.head(3)[['title', 'processed_text']].to_string())

print(f"\n{'=' * 60}")
print(f"✓ academia_corpus.csv 생성 완료!")
print(f"✓ 다음 단계: BERTopic 모델링")
print(f"{'=' * 60}")