# Code Modification Log

이 문서는 프로젝트 진행을 위해 원본 코드에서 변경된 사항들을 기록합니다.
향후 원본 상태로 복구하거나 변경 내역을 추적할 때 이 문서를 참조하십시오.

---

## 1. `code_denoising/params.py`

### 목적
- 로컬 환경에 맞게 데이터셋 경로를 수정하고, 각기 다른 학습 계획(Denoising, Deconvolution, End-to-End)에 맞는 데이터셋을 지정하기 위함.

### 주요 변경 사항
- **`default_root` (Line 17):**
  - **원본:** `"/fast_storage/juhyung/dataset"`
  - **변경:** `../dataset` (로컬 상대 경로)
- **`TRAIN_DATASET` (Line 20):**
  - **원본:** `[DATA_ROOT + "/train"]`
  - **변경:** 학습 계획에 따라 `test_1_noise`, `test_1_conv`, `test_1` 등으로 변경됨.
- **`clean_root` (Line 40에 추가):**
  - 오염된 이미지에 해당하는 원본 이미지를 찾기 위한 경로 (`../dataset/train`)를 지정하는 새로운 파라미터.
- **`noise_sigma` (Line 73):**
  - **원본:** `0.05` (실시간 노이즈 생성용)
  - **변경:** `0.0` (미리 생성된 데이터셋을 사용하므로 실시간 노이즈 생성을 비활성화)

---

## 2. `code_denoising/datawrapper/datawrapper.py`

### 목적
- 미리 생성된 대규모 증강 데이터셋을 효율적으로 사용하고, 학습 시 Overfitting을 방지하기 위한 '스마트 샘플링' 기능을 구현하기 위함.

### 주요 변경 사항
- **`__init__` 함수 수정:**
  - `clean_root` 파라미터를 받도록 수정.
  - 실시간 노이즈 시뮬레이터(`NoiseSimulator`)가 `noise_sigma > 0`일 때만 활성화되도록 변경.
  - **스마트 샘플링 로직 추가:**
    - `train_dataset` 폴더의 파일들을 스캔하여 원본 이미지 이름(예: `L1_...`)을 기준으로 파일들을 그룹화 (`self.file_groups`).
- **`__len__` 함수 수정:**
  - 전체 파일 개수가 아닌, 그룹의 개수를 반환하도록 변경 (스마트 샘플링의 기준).
- **`__getitem__` 함수 수정:**
  - **로직 전체 변경:**
    1. `idx`에 해당하는 원본 그룹을 선택.
    2. 그룹 내에서 오염된 이미지 파일 하나를 **랜덤으로 선택**하여 로드 (`noisy` 데이터).
    3. 그룹 키(원본 이름)와 `clean_root`를 조합하여 **깨끗한 원본 이미지**를 로드 (`label` 데이터).
    4. 기존의 실시간 노이즈 생성 로직은 사용하지 않음.

---

## 3. `code_denoising/train.py`

### 목적
- `datawrapper.py`의 `LoaderConfig` 클래스에 `clean_root` 파라미터가 추가됨에 따라, `train.py`에서 `LoaderConfig`를 생성할 때 이 값을 올바르게 전달해주기 위함.

### 주요 변경 사항
- **`_set_data` 함수 (Line 89-114):**
  - `train_loader_cfg`, `valid_loader_cfg`, `test_loader_cfg`를 생성하는 `LoaderConfig(...)` 호출 부분에 `clean_root=config.clean_root` 인자를 추가함.
