import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========================================
# 설정
# ========================================
INPUT_FILE = "/home/dslab/choi/Journal/BERTopic/Data/topic_pairs.csv"
OUTPUT_DIR = "/home/dslab/choi/Journal/Cross_Domain_Mapping/Visualization/"

# 시각화 설정
FIGSIZE_LARGE = (16, 12)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_SMALL = (10, 6)
# ========================================

print(f"{'='*70}")
print(f"토픽 매핑 시각화")
print(f"{'='*70}\n")

# 데이터 로드
print("[1단계] 데이터 로드")
df = pd.read_csv(INPUT_FILE)
print(f"  ✓ 토픽 페어: {len(df)}개")

# ========================================
# 1. 유사도 히트맵
# ========================================
print("\n[2단계] 유사도 히트맵 생성")

# 유사도 매트릭스 재구성
n_academia = df['academia_topic_id'].max() + 1
n_industry = df['industry_topic_id'].max() + 1

similarity_matrix = np.zeros((n_academia, n_industry))

for _, row in df.iterrows():
    i = int(row['academia_topic_id'])
    j = int(row['industry_topic_id'])
    similarity_matrix[i, j] = row['combined_score']

# 히트맵 생성
fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
sns.heatmap(
    similarity_matrix,
    cmap='YlOrRd',
    vmin=0, vmax=1,
    cbar_kws={'label': 'Similarity Score'},
    ax=ax
)
ax.set_xlabel('Industry Topics', fontsize=14, weight='bold')
ax.set_ylabel('Academia Topics', fontsize=14, weight='bold')
ax.set_title('Academia-Industry Topic Similarity Heatmap', fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}01_similarity_heatmap.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: 01_similarity_heatmap.png")
plt.close()

# ========================================
# 2. 시차 분포 히스토그램
# ========================================
print("\n[3단계] 시차 분포 히스토그램")

valid_lags = df[df['time_lag'].notna()]['time_lag']

fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
ax.hist(valid_lags, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Lag')
ax.axvline(x=valid_lags.mean(), color='green', linestyle='--', linewidth=2,
           label=f'Mean: {valid_lags.mean():.2f} years')

ax.set_xlabel('Time Lag (years)', fontsize=12, weight='bold')
ax.set_ylabel('Frequency', fontsize=12, weight='bold')
ax.set_title('Time Lag Distribution between Academia and Industry',
             fontsize=14, weight='bold', pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}02_time_lag_histogram.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: 02_time_lag_histogram.png")
plt.close()

# ========================================
# 3. 시차 유형별 파이 차트
# ========================================
print("\n[4단계] 시차 유형별 파이 차트")

lag_type_counts = df[df['lag_type'] != '정보부족']['lag_type'].value_counts()

fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax.pie(lag_type_counts.values, labels=lag_type_counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax.set_title('Topic Lead Type Distribution', fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}03_lag_type_pie.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: 03_lag_type_pie.png")
plt.close()

# ========================================
# 4. 관심도 스캐터 플롯
# ========================================
print("\n[5단계] 관심도 스캐터 플롯")

# 각 토픽 쌍을 최고 유사도만 남기기
best_pairs = df.loc[df.groupby('academia_topic_id')['combined_score'].idxmax()]

fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

# 색상은 유사도로 표시
scatter = ax.scatter(
    best_pairs['academia_ratio'],
    best_pairs['industry_ratio'],
    c=best_pairs['combined_score'],
    cmap='viridis',
    s=100,
    alpha=0.6,
    edgecolors='black'
)

# 대각선 (동일한 관심도)
max_ratio = max(best_pairs['academia_ratio'].max(), best_pairs['industry_ratio'].max())
ax.plot([0, max_ratio], [0, max_ratio], 'r--', linewidth=2, alpha=0.5, label='Equal Interest')

ax.set_xlabel('Academia Interest Ratio (%)', fontsize=12, weight='bold')
ax.set_ylabel('Industry Interest Ratio (%)', fontsize=12, weight='bold')
ax.set_title('Academia vs Industry Topic Interest', fontsize=14, weight='bold', pad=15)

# 컬러바
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Similarity Score', fontsize=11, weight='bold')

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}04_interest_scatter.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: 04_interest_scatter.png")
plt.close()

# ========================================
# 5. 상위 토픽 매칭 바 차트
# ========================================
print("\n[6단계] 상위 토픽 매칭 바 차트")

top_pairs = df.head(15)

fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

# Y축 레이블 생성
labels = [f"A{row['academia_topic_id']}-I{row['industry_topic_id']}"
          for _, row in top_pairs.iterrows()]

y_pos = np.arange(len(labels))
scores = top_pairs['combined_score'].values

# 바 차트
bars = ax.barh(y_pos, scores, color='steelblue', alpha=0.7, edgecolor='black')

# 시차 유형별 색상
colors = []
for _, row in top_pairs.iterrows():
    if row['lag_type'] == '학계선도':
        colors.append('#ff9999')
    elif row['lag_type'] == '산업선도':
        colors.append('#66b3ff')
    elif row['lag_type'] == '동시발전':
        colors.append('#99ff99')
    else:
        colors.append('#cccccc')

for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Similarity Score', fontsize=12, weight='bold')
ax.set_title('Top 15 Topic Pairs by Similarity', fontsize=14, weight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#ff9999', label='Academia-led'),
    Patch(facecolor='#66b3ff', label='Industry-led'),
    Patch(facecolor='#99ff99', label='Simultaneous')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}05_top_pairs_bar.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: 05_top_pairs_bar.png")
plt.close()

# ========================================
# 6. 관심도 격차 상위 토픽
# ========================================
print("\n[7단계] 관심도 격차 상위 토픽")

top_gap = df.nlargest(15, 'interest_gap')

fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

labels = [f"A{row['academia_topic_id']}-I{row['industry_topic_id']}"
          for _, row in top_gap.iterrows()]
y_pos = np.arange(len(labels))
gaps = top_gap['interest_gap'].values

bars = ax.barh(y_pos, gaps, color='coral', alpha=0.7, edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Interest Gap (%)', fontsize=12, weight='bold')
ax.set_title('Top 15 Topic Pairs by Interest Gap', fontsize=14, weight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}06_interest_gap_bar.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: 06_interest_gap_bar.png")
plt.close()

# ========================================
# 7. 종합 대시보드
# ========================================
print("\n[8단계] 종합 대시보드")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# (1) 유사도 분포
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['combined_score'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Similarity Score', fontsize=10, weight='bold')
ax1.set_ylabel('Frequency', fontsize=10, weight='bold')
ax1.set_title('Similarity Distribution', fontsize=11, weight='bold')
ax1.grid(True, alpha=0.3)

# (2) 시차 분포
ax2 = fig.add_subplot(gs[0, 1])
valid_lags = df[df['time_lag'].notna()]['time_lag']
ax2.hist(valid_lags, bins=15, color='coral', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Time Lag (years)', fontsize=10, weight='bold')
ax2.set_ylabel('Frequency', fontsize=10, weight='bold')
ax2.set_title('Time Lag Distribution', fontsize=11, weight='bold')
ax2.grid(True, alpha=0.3)

# (3) 시차 유형
ax3 = fig.add_subplot(gs[0, 2])
lag_counts = df[df['lag_type'] != '정보부족']['lag_type'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99']
ax3.pie(lag_counts.values, labels=lag_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 9})
ax3.set_title('Lead Type', fontsize=11, weight='bold')

# (4) 관심도 스캐터
ax4 = fig.add_subplot(gs[1, :2])
best_pairs = df.loc[df.groupby('academia_topic_id')['combined_score'].idxmax()]
scatter = ax4.scatter(
    best_pairs['academia_ratio'],
    best_pairs['industry_ratio'],
    c=best_pairs['combined_score'],
    cmap='viridis',
    s=80,
    alpha=0.6,
    edgecolors='black'
)
max_ratio = max(best_pairs['academia_ratio'].max(), best_pairs['industry_ratio'].max())
ax4.plot([0, max_ratio], [0, max_ratio], 'r--', linewidth=2, alpha=0.5)
ax4.set_xlabel('Academia Interest (%)', fontsize=10, weight='bold')
ax4.set_ylabel('Industry Interest (%)', fontsize=10, weight='bold')
ax4.set_title('Interest Comparison', fontsize=11, weight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Similarity')

# (5) 관심도 격차
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(df['interest_gap'], bins=20, color='orange', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Interest Gap (%)', fontsize=10, weight='bold')
ax5.set_ylabel('Frequency', fontsize=10, weight='bold')
ax5.set_title('Interest Gap Distribution', fontsize=11, weight='bold')
ax5.grid(True, alpha=0.3)

# (6) 상위 매칭
ax6 = fig.add_subplot(gs[2, :])
top10 = df.head(10)
labels = [f"A{row['academia_topic_id']}-I{row['industry_topic_id']}"
          for _, row in top10.iterrows()]
y_pos = np.arange(len(labels))
colors = ['#ff9999' if row['lag_type'] == '학계선도'
          else '#66b3ff' if row['lag_type'] == '산업선도'
          else '#99ff99' for _, row in top10.iterrows()]
ax6.barh(y_pos, top10['combined_score'].values, color=colors, alpha=0.7, edgecolor='black')
ax6.set_yticks(y_pos)
ax6.set_yticklabels(labels, fontsize=9)
ax6.set_xlabel('Similarity Score', fontsize=10, weight='bold')
ax6.set_title('Top 10 Topic Pairs', fontsize=11, weight='bold')
ax6.grid(True, alpha=0.3, axis='x')

plt.suptitle('Academia-Industry Topic Mapping Dashboard',
             fontsize=16, weight='bold', y=0.995)

plt.savefig(f"{OUTPUT_DIR}07_dashboard.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: 07_dashboard.png")
plt.close()

print(f"\n{'='*70}")
print(f"✓ 시각화 완료!")
print(f"✓ 저장 위치: {OUTPUT_DIR}")
print(f"{'='*70}")