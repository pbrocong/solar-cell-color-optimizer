# ==============================================================================
# 0. 라이브러리
# ==============================================================================
import os, math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

# ==============================================================================
# 1) 설정 (CONFIG)
# ==============================================================================
# 물리 상수
H = 6.62607015e-34
C = 299792458

# 데이터 처리
HEADER_SEARCH_ROWS    = 30
HEADER_FALLBACK_INDEX = 19
LAMBDA_MIN, LAMBDA_MAX = 380, 780

# 탐색 파라미터
#  N1과 N2의 폭을 독립적으로 탐색합니다.
FWHM1_MIN, FWHM1_MAX, FWHM1_STEP = 10, 120, 2
FWHM2_MIN, FWHM2_MAX, FWHM2_STEP = 10, 120, 2

MU_COARSE_STEP = 20    # coarse μ 간격
MU_FINE_STEP   = 2     # fine   μ 간격
R_STEPS        = 21    # r grid size

# 최적화 종료 기준
TOL_XYZ_ERROR = 1e-2   # 만족 시 조기 종료

# 파일 출력
OUTPUT_CSV_FILENAME = "best_solution_spectra.csv"

# ==============================================================================
# 2) 데이터 구조
# ==============================================================================
@dataclass
class Target:
    X: float
    Y: float
    Z: float

@dataclass
class Solution:
    fwhm1: float = -1.0
    fwhm2: float = -1.0
    mu1: float = -1.0
    mu2: float = -1.0
    r: float = -1.0
    alpha: float = -1.0
    X: float = -1.0
    Y: float = -1.0
    Z: float = -1.0
    score: float = np.inf
    N_total: np.ndarray = field(default_factory=lambda: np.array([]))

# ==============================================================================
# 3) 로딩/전처리
# ==============================================================================
def find_header_row(df_no_header: pd.DataFrame) -> Optional[int]:
    need = {'λ', 'S(λ)', 'a(λ)', 'b(λ)', 'c(λ)'}
    for i, row in df_no_header.head(HEADER_SEARCH_ROWS).iterrows():
        if need.issubset({str(v).strip() for v in row.values}):
            return i
    return None

def find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    norm_cols = {
        col: ''.join(filter(str.isalnum, str(col).lower().replace('λ', 'lambda').replace('̄', 'bar')))
        for col in df.columns
    }
    norm_aliases = {
        ''.join(filter(str.isalnum, a.lower().replace('λ', 'lambda').replace('̄', 'bar')))
        for a in aliases
    }
    for orig, norm in norm_cols.items():
        if norm in norm_aliases:
            return orig
    return None

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    try:
        if filepath.lower().endswith('.csv'):
            df_raw = pd.read_csv(filepath, header=None, encoding='utf-8-sig')
        elif filepath.lower().endswith(('.xls', '.xlsx')):
            df_raw = pd.read_excel(filepath, header=None, engine='openpyxl')
        else:
            raise ValueError("지원 형식: .csv, .xls, .xlsx")
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    except Exception as e:
        raise IOError(f"파일 읽기 오류: {e}")

    hdr = find_header_row(df_raw)
    if hdr is not None:
        if filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath, header=hdr, encoding='utf-8-sig')
        else:
            df = pd.read_excel(filepath, header=hdr, engine='openpyxl')
    else:
        try:
            if filepath.lower().endswith('.csv'):
                df = pd.read_csv(filepath, header=HEADER_FALLBACK_INDEX, encoding='utf-8-sig')
            else:
                df = pd.read_excel(filepath, header=HEADER_FALLBACK_INDEX, engine='openpyxl')
        except Exception:
            df = df_raw
            df.columns = [f'col_{i}' for i in range(df.shape[1])]
            print(f"경고: 헤더 자동탐지 실패. (fallback {HEADER_FALLBACK_INDEX})")

    aliases = {
        'lambda_nm': ['λ', 'lambda', 'wavelength', '람다'],
        'S':         ['S(λ)', 'S', 's(λ)', '원스펙트럼', 's_lambda'],
        'a':         ['a(λ)', 'x̄(λ)', 'a', 'xbar', 'x̄', 'a_lambda'],
        'b':         ['b(λ)', 'ȳ(λ)', 'b', 'ybar', 'ȳ', 'b_lambda'],
        'c':         ['c(λ)', 'z̄(λ)', 'c', 'zbar', 'z̄', 'c_lambda'],
    }

    mapped = pd.DataFrame()
    for std, al in aliases.items():
        col = find_column(df, al)
        if col is None:
            raise ValueError(f"필수 컬럼 누락: {std} (aliases={al})")
        mapped[std] = df[col]

    for col in mapped.columns:
        mapped[col] = pd.to_numeric(mapped[col], errors='coerce')

    mapped.dropna(inplace=True)
    if mapped.empty:
        raise ValueError("모든 숫자 변환 후 데이터가 비었습니다.")

    out = mapped[(mapped['lambda_nm'] >= LAMBDA_MIN) & (mapped['lambda_nm'] <= LAMBDA_MAX)].copy()
    out.sort_values('lambda_nm', inplace=True)
    out.reset_index(drop=True, inplace=True)
    if out.empty:
        raise ValueError("파장 범위 내 유효 데이터 없음.")
    return out

def calculate_weights_and_denominator(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    lam_m = df['lambda_nm'].values * 1e-9
    hc_over_lambda = (H * C) / lam_m
    df['w_x'] = hc_over_lambda * df['a'].values
    df['w_y'] = hc_over_lambda * df['b'].values
    df['w_z'] = hc_over_lambda * df['c'].values
    D = np.sum(df['S'].values * df['w_y'].values)
    if D <= 0:
        raise ValueError(f"분모(D) ≤ 0: {D:.4e}")
    return df, D

# ==============================================================================
# 4) 스펙트럼/색 계산
# ==============================================================================
def fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2 * math.sqrt(2 * math.log(2)))

def gaussian(lambda_nm: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    g = np.exp(-((lambda_nm - mu)**2) / (2 * sigma**2))
    g[g < 0.05] = 0.0  # 5% 컷
    return g

def calculate_xyz(N: np.ndarray, df_w: pd.DataFrame, D: float) -> Tuple[float, float, float]:
    X = np.sum(N * df_w['w_x'].values) / D
    Y = np.sum(N * df_w['w_y'].values) / D
    Z = np.sum(N * df_w['w_z'].values) / D
    return X, Y, Z

def xyz_error(X: float, Y: float, Z: float, tgt: Target) -> float:
    return math.sqrt((X - tgt.X)**2 + (Y - tgt.Y)**2 + (Z - tgt.Z)**2)

# ==============================================================================
# 5) 탐색: (FWHM1, FWHM2, μ1, μ2, r)
# ==============================================================================
def search_step(lambda_nm, S, df, D, target,
                fwhm1, fwhm2, mu1_grid, mu2_grid, r_grid,
                best_solution_so_far: Solution) -> Solution:
    sigma1 = fwhm_to_sigma(fwhm1)
    sigma2 = fwhm_to_sigma(fwhm2)
    g1_cache = {}
    g2_cache = {}

    current_best = best_solution_so_far

    for mu1 in mu1_grid:
        if mu1 not in g1_cache:
            g1_cache[mu1] = gaussian(lambda_nm, mu1, sigma1)
        g1 = g1_cache[mu1]
        if np.all(g1 == 0): 
            continue

        for mu2 in mu2_grid:
            if mu2 not in g2_cache:
                g2_cache[mu2] = gaussian(lambda_nm, mu2, sigma2)
            g2 = g2_cache[mu2]
            if np.all(g2 == 0):
                continue

            for r in r_grid:
                mix = r * g1 + (1 - r) * g2
                mask = mix > 1e-12
                if not np.any(mask):
                    continue

                alpha_max = np.min(S[mask] / mix[mask])
                if not np.isfinite(alpha_max) or alpha_max <= 0:
                    continue

                N_total = alpha_max * mix
                X, Y, Z = calculate_xyz(N_total, df, D)
                score = xyz_error(X, Y, Z, target)

                if score < current_best.score:
                    current_best = Solution(
                        fwhm1=fwhm1, fwhm2=fwhm2,
                        mu1=mu1, mu2=mu2, r=r, alpha=alpha_max,
                        X=X, Y=Y, Z=Z, score=score, N_total=N_total
                    )
    return current_best

def find_best_solution(df: pd.DataFrame, target: Target, D: float) -> Solution:
    lambda_nm = df['lambda_nm'].values
    S = df['S'].values
    best = Solution()

    mu_coarse = np.arange(LAMBDA_MIN, LAMBDA_MAX + 1, MU_COARSE_STEP)
    EPS = 0.1
    r_grid = np.linspace(EPS, 1 - EPS, R_STEPS)

    fwhm1_range = range(FWHM1_MIN, FWHM1_MAX + 1, FWHM1_STEP)
    fwhm2_range = range(FWHM2_MIN, FWHM2_MAX + 1, FWHM2_STEP)

    for i, f1 in enumerate(fwhm1_range):
        for j, f2 in enumerate(fwhm2_range):
            print(f"[FWHM 스캔] fwhm1={f1:>3} nm, fwhm2={f2:>3} nm  ({i+1}/{len(fwhm1_range)}, {j+1}/{len(fwhm2_range)})")

            # coarse
            best_in_pair = search_step(lambda_nm, S, df, D, target,
                                       f1, f2, mu_coarse, mu_coarse, r_grid, best)

            if best_in_pair.score < best.score:
                best = best_in_pair

                # fine(각 μ 주변 ±MU_COARSE_STEP 범위, MU_FINE_STEP 간격)
                mu1c, mu2c = best.mu1, best.mu2
                mu1_fine = np.arange(mu1c - MU_COARSE_STEP, mu1c + MU_COARSE_STEP + 1, MU_FINE_STEP)
                mu2_fine = np.arange(mu2c - MU_COARSE_STEP, mu2c + MU_COARSE_STEP + 1, MU_FINE_STEP)
                best = search_step(lambda_nm, S, df, D, target,
                                   f1, f2, mu1_fine, mu2_fine, r_grid, best)

            # 조기 종료
            if best.score <= TOL_XYZ_ERROR:
                print(f"성공: XYZ 오차 {best.score:.3e} ≤ {TOL_XYZ_ERROR:.1e}. 탐색 종료.")
                return best
    return best

# ==============================================================================
# 6) 입출력
# ==============================================================================
def get_user_inputs() -> Tuple[str, Target]:
    while True:
        fp = input("CSV 또는 XLSX 파일 경로를 입력하세요: ").strip().strip('\'"')
        if os.path.isfile(fp):
            break
        print("오류: 파일이 존재하지 않습니다. 경로를 다시 확인해주세요.")
    print("목표 XYZ 값을 순서대로 입력합니다.")
    def ask(msg):
        while True:
            try: return float(input(msg).strip().replace(',', ''))
            except: print("오류: 숫자를 입력하세요.")
    tgt = Target(ask("목표 X: "), ask("목표 Y: "), ask("목표 Z: "))
    return fp, tgt

# ==============================================================================
# 7) 시각화/저장
# ==============================================================================
def plot_spectra(df: pd.DataFrame, sol: Solution):
    plt.figure(figsize=(12, 7))
    plt.plot(df['lambda_nm'], df['S'], label='S(λ) - Original Spectrum', color='gray', linestyle='--', linewidth=1.5)

    lam = df['lambda_nm'].values
    s1 = fwhm_to_sigma(sol.fwhm1)
    s2 = fwhm_to_sigma(sol.fwhm2)
    G1 = gaussian(lam, sol.mu1, s1)
    G2 = gaussian(lam, sol.mu2, s2)
    N1 = sol.alpha * sol.r * G1
    N2 = sol.alpha * (1 - sol.r) * G2

    plt.plot(lam, N1, label=f'N1(μ={sol.mu1:.1f}nm, FWHM={sol.fwhm1:.1f}nm)', linewidth=2)
    plt.plot(lam, N2, label=f'N2(μ={sol.mu2:.1f}nm, FWHM={sol.fwhm2:.1f}nm)', linewidth=2)

    plt.title(f"Optimal Spectrum Components (r={sol.r:.2f})", fontsize=16)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(LAMBDA_MIN, LAMBDA_MAX); plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

def report_and_save_results(sol: Solution, df: pd.DataFrame, tgt: Target):
    if sol.score == np.inf:
        print("\n최적 해를 찾지 못했습니다.")
        return

    print("\n========================= 최적 해 =========================")
    print(f" FWHM1 / FWHM2 : {sol.fwhm1:.1f} nm / {sol.fwhm2:.1f} nm")
    print(f" μ1 / μ2       : {sol.mu1:.1f} nm / {sol.mu2:.1f} nm")
    print(f" r, α          : {sol.r:.2f}, {sol.alpha:.4f}")
    print("-----------------------------------------------------------")
    print(f" XYZ (achvd)   : ({sol.X:.4f}, {sol.Y:.4f}, {sol.Z:.4f})")
    print(f" XYZ (target)  : ({tgt.X:.4f}, {tgt.Y:.4f}, {tgt.Z:.4f})")
    print(f" Euclid error  : {sol.score:.6f}")
    print("===========================================================\n")

    lam = df['lambda_nm'].values
    s1 = fwhm_to_sigma(sol.fwhm1)
    s2 = fwhm_to_sigma(sol.fwhm2)
    G1 = gaussian(lam, sol.mu1, s1)
    G2 = gaussian(lam, sol.mu2, s2)
    N1 = sol.alpha * sol.r * G1
    N2 = sol.alpha * (1 - sol.r) * G2
    N_total = N1 + N2

    out = pd.DataFrame({
        'lambda_nm': lam,
        'S': df['S'].values,
        'N1': N1, 'N2': N2, 'N_total': N_total
    })
    try:
        out.to_csv(OUTPUT_CSV_FILENAME, index=False, float_format='%.6f')
        print(f"CSV 저장 완료: {OUTPUT_CSV_FILENAME}")
    except Exception as e:
        print(f"CSV 저장 실패: {e}")

    # 제약 검증
    max_diff = np.max(N_total - df['S'].values)
    ok = max_diff < 1e-12
    print(f"제약 N(λ) ≤ S(λ) 만족: {ok} (max(N-S)={max_diff:.3e})")

    plot_spectra(df, sol)

# ==============================================================================
# 8) 메인
# ==============================================================================
def main():
    try:
        path, tgt = get_user_inputs()
        print("\n데이터 로딩/전처리 중...")
        df0 = load_and_prepare_data(path)
        df, D = calculate_weights_and_denominator(df0)
        print("데이터 준비 완료. 최적화 시작...")

        best = find_best_solution(df, tgt, D)
        report_and_save_results(best, df, tgt)

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"\n치명적 오류: {e}")
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()
