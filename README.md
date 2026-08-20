# Solar Cell Color-Coordinate Spectrum Optimizer

Colored photovoltaic (c-BIPV) 모듈 설계를 위한 **반사 스펙트럼 최적화 도구**입니다.
목표 색좌표(CIE XYZ)를 입력하면, 태양광 스펙트럼에서 반사 밴드 조합을 탐색하여
목표 색을 근사하면서 발전 손실을 최소화하는 스펙트럼 설계를 도와줍니다.

> 가천대학교 Device Lab × 서울과학기술대학교 공동연구 과정에서 개발
> (개발: 박형빈, Gachon University)

## 기능

- **밴드 조합 탐색** (`src/band_search_*.py`)
  - 태양광 스펙트럼(380–780nm)에서 2-밴드 반사 구간의 폭·위치를 전수 탐색
  - 목표 XYZ와의 유클리디안 오차 최소 조합 산출, 진행률 표시·시각화 포함
  - `fixed_center`: Band 1 중심 444nm 고정 버전 / `symmetric`: 중심 이동·좌우대칭 버전
- **가우시안 반사 스펙트럼 최적화** (`src/gaussian_spectrum_optimizer.py`)
  - 두 가우시안 밴드(FWHM·중심·세기 독립 탐색)로 현실적인 반사 스펙트럼 모델링
  - 물리 상수 기반 광자 플럭스 계산, 설정(CONFIG) 분리 구조

## 사용법

```bash
pip install -r requirements.txt
python src/band_search_symmetric.py
# 실행 후 스펙트럼 데이터(CSV/XLSX) 경로 입력 → 타겟 XYZ 입력
```

샘플 입력 데이터: `data/색좌표_계산_샘플.csv` (수업/연구용 배포본)

## 저장소 구조

```
src/    탐색·최적화 코드 (Python)
docs/   최적화 플로우차트, 코드 발표자료
data/   샘플 스펙트럼 데이터
```

## License

MIT
