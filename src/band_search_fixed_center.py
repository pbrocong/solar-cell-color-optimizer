import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- 1. 파일 경로 입력 ----------
file_path = input("📁 CSV 또는 XLSX 파일 경로를 붙여넣어주세요: ").strip().strip("'").strip('"')

if not os.path.exists(file_path):
    print("파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    exit()

# ---------- 2. 파일 불러오기 ----------
if file_path.endswith('.csv'):
    df = pd.read_csv(file_path, header=19)
elif file_path.endswith(('.xls', '.xlsx')):
    df = pd.read_excel(file_path, header=19)
else:
    print("지원되지 않는 파일 형식입니다. CSV 또는 XLSX 파일만 가능합니다.")
    exit()

# ---------- 3. 데이터 전처리 ----------
df = df[['λ', 'S(λ)', 'a(λ)', 'b(λ)', 'c(λ)']].dropna()
df.columns = ['wavelength', 'S_lambda', 'a_lambda', 'b_lambda', 'c_lambda']
df = df.astype(float)
df = df[(df['wavelength'] >= 380) & (df['wavelength'] <= 780)]
df.reset_index(drop=True, inplace=True)

λ = df['wavelength'].values
S = df['S_lambda'].values
a = df['a_lambda'].values
b = df['b_lambda'].values
c = df['c_lambda'].values
inv_lambda = 1 / λ
denominator = np.sum(S * inv_lambda * b)

# ---------- 4. 타겟 XYZ 입력 ----------
try:
    x = float(input("원하는 X 값을 입력하세요: "))
    y = float(input("원하는 Y 값을 입력하세요: "))
    z = float(input("원하는 Z 값을 입력하세요: "))
    target_XYZ = np.array([x, y, z])
except:
    print("숫자를 정확히 입력해주세요.")
    exit()

# ---------- 5. 밴드 조건 설정 ----------
λ_step = λ[1] - λ[0]
n = len(λ)
band_widths = list(range(1, 150, 1))  # 1~149nm 폭, 1nm 간격

center1 = 444
center1_idx = np.argmin(np.abs(λ - center1))

# Band2는 505~670nm 범위 내에서만 탐색
λ_min_2nd = 505
λ_max_2nd = 670
valid_j = np.where((λ >= λ_min_2nd) & (λ <= λ_max_2nd))[0]

best_error = float('inf')
best_combo = None
best_XYZ = None
best_R = None

# 전체 조합 수 사전 계산
total_trials = 0
for len1 in band_widths:
    i_start = center1_idx - len1 // 2
    i_end = center1_idx + (len1 + 1) // 2
    if i_start < 0 or i_end > n:
        continue
    for j in valid_j:
        for len2 in band_widths:
            if j + len2 > n:
                continue
            total_trials += 1

# ---------- 6. 탐색 ----------
trial = 0
progress_mark = 0

for len1 in band_widths:
    i_start = center1_idx - len1 // 2
    i_end = center1_idx + (len1 + 1) // 2
    if i_start < 0 or i_end > n:
        continue

    for j in valid_j:
        for len2 in band_widths:
            if j + len2 > n:
                continue

            trial += 1
            progress = int((trial / total_trials) * 100)
            if progress >= progress_mark * 10:
                print(f"진행 단계: {progress_mark}/10 ({progress}%) 완료")
                progress_mark += 1

            R = np.zeros_like(λ)
            R[i_start:i_end] = 1  # Band 1: center 444nm 고정, 대칭 폭
            R[j:j+len2] = 1       # Band 2: 자유 탐색

            X = np.sum(R * S * inv_lambda * a) / denominator
            Y = np.sum(R * S * inv_lambda * b) / denominator
            Z = np.sum(R * S * inv_lambda * c) / denominator
            error = np.linalg.norm(np.array([X, Y, Z]) - target_XYZ)

            if error < best_error:
                best_error = error
                best_combo = ((i_start, i_end), (j, j + len2))
                best_XYZ = (X, Y, Z)
                best_R = R.copy()

# ---------- 7. 시각화 및 결과 출력 ----------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(λ, best_R, drawstyle='steps-mid', color='red')
plt.title("The best of R(λ)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("R(λ)")
plt.grid(True)
plt.tight_layout()
plt.show()

band1_start, band1_end = best_combo[0]
band2_start, band2_end = best_combo[1]

print("\n계산 완료!\n")
print(f" Target XYZ: {np.round(target_XYZ, 3)}")
print(f"Computed XYZ: {np.round(best_XYZ, 3)}")
print(f"Error (Euclidean distance): {round(best_error, 5)}")
print(f"\n최적의 R(λ)=1인 두 밄드 구간 (조건 적용):")
print(f"Band 1 (Center 444nm): {λ[band1_start]:.1f} nm ~ {λ[band1_end - 1]:.1f} nm")
print(f"Band 2 (within 505~670nm): {λ[band2_start]:.1f} nm ~ {λ[band2_end - 1]:.1f} nm")

