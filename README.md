# Denoising and Deconvolution Project README

## 1. 과제 목표 (Goal)

이 프로젝트의 최종 목표는 **`dataset/test_y`** 폴더의 이미지들에 적용된 미지의 열화(degradation)를 복원하여, 원본 이미지인 **`dataset/label`** 과 가장 유사하게 만드는 것입니다.

## 2. 데이터셋 역할 분석 (Dataset Roles)

-   **`train/`**, **`val/`**: 모델 학습 및 검증에 사용되는 깨끗한 원본 이미지들.
-   **`test_y/`**: **[문제지]** 최종적으로 복원해야 할 대상. 알 수 없는 파라미터로 열화가 적용되어 있습니다.
-   **`label/`**: **[정답지]** `test_y` 이미지들의 원본. 최종 성능 평가에 사용됩니다.
-   **`test_y_v2/`**: **[힌트]** 열화 방식을 분석할 수 있도록 친절하게 파라미터 정보(컨볼루션 방향, 노이즈 레벨)를 포함한 샘플입니다. **학습이나 평가에는 직접 사용되지 않습니다.**
-   **`test_1/`**, **`test_1_noise/`**, **`test_1_conv/`**: **[자체 제작 학습 데이터]** `train` 원본 이미지에 `test_y_v2`에서 분석한 열화 방식을 적용하여 우리가 직접 생성한 학습용 데이터셋입니다.

## 3. 핵심 열화 분석 (Degradation Analysis)

`test_y_v2` 샘플을 분석한 결과, 이 프로젝트의 핵심 열화는 다음 두 가지로 구성됩니다.
1.  **컨볼루션 (Convolution):** `dataset/forward_simulator.py`의 `dipole_kernel`을 사용한 흐림 효과.
2.  **노이즈 (Noise):** 가우시안 노이즈. 분석된 노이즈 레벨은 표준편차(σ) 기준으로 **Level 1: σ≈0.070**, **Level 2: σ≈0.132** 입니다.

## 4. 최종 전략: "하이브리드 On-the-fly" 및 마스터 플랜

속도와 공정성을 모두 만족시키기 위해, **"데이터는 미리 만들지 않지만, 학습 시에는 모든 조합을 사용하는 On-the-fly 방식"**을 최종 전략으로 확정합니다.

### 4.1. 핵심 데이터 처리 방식

1.  **빠른 I/O:** Colab 학습 시, **작은 원본 `train`, `val` 폴더만 로컬 런타임으로 복사**합니다.
2.  **가상 데이터셋 확장:** `DataLoader`가 특정 `index`의 데이터를 요청하면,
    -   `원본 이미지 index` = `index // 10`
    -   `적용할 열화 조합 index` = `index % 10`
    -   위 계산에 따라 로컬의 원본 이미지를 읽고, **메모리 상에서 실시간으로 해당 열화를 적용**하여 모델에 전달합니다.
3.  **결과:** 이 방식은 빠른 I/O, 완벽한 실험 공정성, 저장 공간 효율성의 장점을 모두 가집니다. **모든 학습 기반 모델은 이 방식으로 통일하여 학습을 진행합니다.**

> ***(주의) 과거 실수 기록:***
> -   ***초기 On-the-fly Augmentation 구현 시, 사용자님의 '통제된 환경' 요구사항을 무시하고 '완전 무작위' 방식으로 임의 구현하여 실험의 재현성을 해치는 심각한 오류를 범했음.***
> -   ***'Index-based Sampling' 방식은 모델이 전체 열화 공간의 일부만 학습하게 만들어, 모델 간 공정한 비교를 불가능하게 만드는 치명적 결함이 있었음.***
> -   ***'Pre-generated All' 방식은 Google Drive의 I/O 병목 현상을 고려하지 않은 비현실적인 제안이었음.***

## 5. 최종 마스터 플랜

위 "하이브리드 On-the-fly" 방식을 기반으로, 다음의 모든 조합을 테스트하여 최종 우승자를 가립니다.

| 조합 (Combination) | Denoising 모듈 | Deconvolution 모듈 | 상태 |
| :--- | :--- | :--- | :--- |
| **1. 학습 기반: SbS** | `DnCNN` (✅ 학습 필요) | `U-Net` (✅ 학습 필요) | **계획 확정** |
| **2. 학습 기반: E2E** | `DnCNN` (End-to-End) | N/A | **계획 확정** |
| **3. 학습 기반: E2E** | `U-Net` (End-to-End) | N/A | **계획 확정** |
| **4. 알고리즘: Simple** | `Diffusion Denoiser` | `Least Squares` | **계획 확정** |
| **5. 알고리즘: PnP** | `Diffusion Denoiser` | `Least Squares` | **계획 확정** |
| **6. 하이브리드 A** | `DnCNN` (✅ 학습 필요) | `Least Squares` | **계획 확정** |
| **7. 하이브리드 B** | `Diffusion Denoiser` | `U-Net` (✅ 학습 필요) | **계획 확정** |

### 4.3. 실행 전략: 통합 평가 프레임워크 구축

다양한 조합을 체계적으로 실행하고 평가하기 위해, **`run_master_evaluation.ipynb`** 라는 새로운 **통합 평가 노트북**을 구축합니다.

1.  **모듈 선택 및 로딩:** 체크포인트 또는 알고리즘 클래스를 유연하게 로드합니다.
2.  **파이프라인 구성:** 로드된 모듈들을 원하는 순서로 자유롭게 연결합니다.
3.  **일괄 실행 및 결과 저장:** 구성된 파이프라인으로 전체 테스트 데이터셋을 복원하고, 조합별로 결과를 저장합니다.
4.  **자동 채점 및 비교:** 저장된 모든 결과에 대해 `evaluate.ipynb`를 실행하여 최종 점수를 계산하고 비교표를 생성합니다.

## 5. 실험 로드맵 (Experiment Roadmap)

최종 목표는 다양한 모델과 방법론을 점진적으로 테스트하여 최고의 성능을 내는 조합을 찾는 것입니다.

### 5.1. 핵심 전략

1.  **통제된 비교 실험:** 모든 모델(DnCNN, U-Net 등)은 **동일한 '통제된' 데이터 증강 방식** 하에서 학습하고 평가하여 공정한 성능을 비교합니다.
2.  **점진적 최적화:** 초기에는 빠른 가능성 타진을 위해 **'에폭별 순환'** 데이터 증강 방식을 사용하고, 가장 성능이 좋은 모델 아키텍처가 선정되면 최종적으로 **'모든 조합 활용'** 방식을 적용하여 성능을 극한으로 끌어올립니다.

> ***(주의) 과거 실수 기록:***
> -   ***초기 On-the-fly Augmentation 구현 시, 사용자님의 '통제된 환경' 요구사항을 무시하고 '완전 무작위' 방식으로 임의 구현하여 실험의 재현성을 해치는 심각한 오류를 범했음.***
> -   ***초기 계획 수립 시, 제안된 'Diffusion' 모델에 대해 구현 방안을 설계하지 않고 임의로 계획에서 생략하는 오류를 범했음.***

### 5.2. 세부 실행 계획

#### Phase 1: End-to-End 모델 아키텍처 비교 (현재 진행 단계)

-   **목표:** '통제된(에폭별 순환)' 환경에서 DnCNN과 U-Net의 End-to-End 성능을 비교하여 더 유망한 아키텍처를 선별합니다.
-   **공통 적용 사항:** `ReduceLROnPlateau` 학습률 스케줄러를 적용하여 과적합을 방지합니다.
-   **(1-1) End-to-End: DnCNN (통제된 ver.)**
-   **(1-2) End-to-End: U-Net (통제된 ver.)**

#### Phase 2: Step-by-Step 접근법 탐색

-   **목표:** Phase 1에서 더 우수했던 아키텍처를 기반으로, Denoising과 Deconvolution을 분리하는 Step-by-Step 방식의 성능을 확인합니다.
-   **(2-1) Step-by-Step: Denoising(선정 모델) + Deconvolution(선정 모델)**
-   **(2-2) Step-by-Step: Denoising(선정 모델) + Deconvolution(Least Square)**

#### Phase 3: 최종 모델 최적화 및 Diffusion 도입 (보류)

-   **목표:** 가장 성능이 좋았던 모델과 접근법에 대해 '모든 조합 활용' 데이터 증강을 적용하여 최종 성능을 극대화하고, Diffusion 모델 구현 및 실험을 진행합니다.

---

## 6. 실험 관리 전략 (Experiment Management Strategy)

다양한 모델, 데이터셋, 학습 방식을 체계적으로 실험하고 재현성을 확보하기 위해 다음 전략을 따릅니다.

1.  **파일 버전 관리 (File Versioning):**
    *   **원본 파일 유지:** `train.py`, `datawrapper.py`와 같은 핵심 원본 스크립트는 수정하지 않고 보존합니다.
    *   **파생 스크립트 생성:** 특정 실험을 위한 코드는 `train_{모델명}_{방식}.py` (예: `train_dncnn_controlled.py`)와 같이 명확한 이름의 파일을 새로 생성하여 관리합니다. 이를 통해 각 파일의 역할을 명확히 구분합니다.
    *   **파라미터 외부 주입:** 노이즈 레벨, 학습률 등 주요 하이퍼파라미터는 `params.py`를 직접 수정하는 대신, Colab 노트북에서 스크립트 실행 시 커맨드 라인 인자(argument)로 전달하여 유연성을 확보합니다.

2.  **Colab 노트북 관리 (Colab Notebook Management):**
    *   **실험 단위로 노트북 생성:** `'모델 + 데이터 + 방식'` 조합처럼, 특정 실험 하나당 하나의 Colab 노트북을 생성합니다. (예: `colab_train_dncnn_controlled_noise_v1.ipynb`)
    *   **(중요) 데이터 로딩 최적화:** Colab에서 학습 시, 전체 `dataset` 폴더를 복사하는 대신 **`train`과 `val` 폴더와 같이 학습에 필수적인 데이터만** 로컬 런타임으로 복사하여 I/O 병목 현상을 최소화하고 준비 시간을 단축합니다.

3.  **로그 및 결과 관리 (Log & Result Management):**
    *   실행 스크립트가 `logs/{실험명}/{날짜}` 구조로 로그와 체크포인트 파일을 자동으로 생성하므로, 각 실험 결과를 명확하게 추적하고 비교합니다.

---

### Advanced Course (여력 확보 시): 품질 우선 실험

-   **목표:** 강한 노이즈와 링잉(Ringing) 현상을 억제하는 데 초점을 맞춘 고급 하이브리드 파이프라인을 테스트합니다.
-   **Denoise:** Diffusion (5–10 steps)
-   **Deconvolution:** PnP (Plug-and-Play) (3–5 iterations)
-   **Fusion:** 대역 결합 (Frequency-based merging) 및 가중 평균

## 8. 개발 이력 및 교훈 (Development History & Lessons Learned)

본 프로젝트는 다수의 기술적 문제와 디버깅 과정을 거쳤습니다. 핵심적인 문제 해결 과정을 기록하여 향후 유사한 실수를 방지하고자 합니다.

- **Git 저장소 오염 및 초기화 사태**: 프로젝트 초기에 `.gitignore`를 부실하게 설정한 탓에, `dataset`, `logs`, `*.pdf` 등 수만 개의 불필요한 파일이 Git 추적 대상에 포함되는 심각한 실수가 발생했습니다. 이로 인해 `git push`가 불가능해졌고, 저장소 내부 데이터 구조가 손상되어 복구 불능 상태에 빠졌습니다. 최종적으로 **로컬의 `.git` 폴더를 완전히 삭제하고, 원격 저장소를 새로 생성한 뒤, 완벽하게 재작성된 `.gitignore`를 적용하여 필요한 소스 코드만 다시 Push**하는 '완전 초기화' 절차를 통해 문제를 해결했습니다. 이 과정에서 **"프로젝트 시작 시점에 `.gitignore`를 가장 먼저, 그리고 가장 완벽하게 설정하는 것이 얼마나 중요한지"** 뼈저리게 깨달았습니다.
- **초기 환경 설정 오류**: 로컬 Windows 환경과 Colab 환경의 차이(`pathlib` 경로 문제, `num_workers` 설정 등)로 인해 초기 학습에 어려움을 겪었으며, `num_workers=0` 설정 및 `pathlib`을 통한 경로 처리로 해결했습니다.
- **Git-Colab 동기화 문제**: `.gitignore`에 의해 `dataset` 폴더가 제외된 상태에서, Colab의 `git clone` 로직이 데이터셋 경로를 찾지 못하는 문제가 발생했습니다. 최종적으로 Colab 실행 시 Google Drive의 고정된 프로젝트 폴더로 이동 후 `git pull`로 코드만 업데이트하는 안정적인 'Upversioning' 방식으로 전환하여 해결했습니다.
- **산재된 설정 파일 문제**: 프로젝트 구조 리팩토링 과정에서 `code_denoising` 폴더 내에 구버전 `params.py`가 남아있어, 최상위 폴더의 신버전 `params.py`를 가리는(shadowing) 현상이 발생했습니다. 이로 인해 `ImportError`가 지속적으로 발생했으며, 불필요한 설정 파일을 삭제하여 해결했습니다.
- **반복적인 Import 경로 오류**: 코드 리팩토링 과정에서 `core_funcs.py`, `train_controlled.py` 등의 파일에서 다른 모듈을 불러오는 `import` 구문의 경로 설정(절대 경로 vs 상대 경로)을 일관성 있게 처리하지 못해 수많은 `ImportError`와 `ModuleNotFoundError`가 발생했습니다. 최종적으로 모든 스크립트가 프로젝트 최상위 폴더를 기준으로 모듈을 찾도록 `sys.path` 설정 및 절대/상대 경로를 명확히 구분하여 해결했습니다. 이 과정에서 **"하나의 파일을 수정하면, 그 파일을 참조하는 모든 다른 파일에 미치는 영향을 반드시 함께 점검해야 한다"**는 중요한 교훈을 얻었습니다.
- **노트북/스크립트 역할 분리 실패**: 초기에는 `run_evaluation.ipynb` 하나의 노트북에서 End-to-End와 Step-by-Step 평가를 모두 처리하려다 코드가 복잡해지고 새로운 버그가 발생하는 악순환을 겪었습니다. 사용자님의 지적에 따라, **"잘 되던 코드는 그대로 두고, 새로운 기능은 새로운 파일로 분리한다"**는 원칙을 적용하여 `run_evaluation_sbs.ipynb`를 새로 생성함으로써 문제를 해결했습니다.

## 9. 현재 진행 상황 (Current Status)

- **안정적인 학습/평가 파이프라인 구축 완료:**
    - **End-to-End 모델:** `colab_train_dncnn_e2e_controlled.ipynb`, `colab_train_unet_e2e_controlled.ipynb`를 통해 안정적인 학습이 가능합니다.
    - **Step-by-Step 모델:** `colab_train_step_by_step.ipynb`를 통해 Denoising, Deconvolution 모델의 개별 학습이 가능합니다.
    - **평가:** `run_evaluation.ipynb` (E2E용)와 `run_evaluation_sbs.ipynb` (SBS용)를 통해 어떤 모델이든 최종 성능(PSNR/SSIM)을 안정적으로 측정할 수 있습니다.
- **베이스라인 성능 측정 진행 중:**
    - **U-Net E2E (38 epochs): PSNR 20.902, SSIM 0.610**
    - 다른 모델들의 학습이 완료되는 대로 베이스라인 성능 확보 예정입니다.

## 10. 고급 복원 모델 구현 완료 (Completed)

성능 극대화를 위한 고급 복원 방법론 3종 및 최종 PnP 프레임워크의 구현을 완료했습니다.

1.  **과제 맞춤형 Deconvolution (`ClippedInverseFilter`):**
    *   참조 코드(`tkd.py`) 분석을 통해, 일반적인 Wiener Filter가 아닌 **"제한된 역필터(Clipped Inverse Filter)"** 방식이 과제의 핵심임을 파악하고 `torch` 기반으로 재구현했습니다.
    *   `code_denoising/classical_methods/deconvolution.py`에 구현 완료.

2.  **Diffusion 기반 방법론 (Hugging Face `diffusers` 라이브러리 활용):**
    *   **PnP용 Denoiser (`HuggingFace_Denoiser`):** PnP 프레임워크에 "부품"처럼 사용할 수 있도록, 사전 학습된 Diffusion 모델을 래핑(wrapping)한 Denoising 모듈을 구현했습니다.
    *   **고성능 복원 모델 (`DiffPIR_Pipeline`):** Diffusion 생성 과정의 매 스텝마다 Deconvolution을 위한 물리적 제약(guidance)을 추가하여 복원 성능을 극대화하는 SOTA 모델 DiffPIR을 커스텀 파이프라인으로 구현했습니다.
    *   `code_denoising/diffusion_methods/` 디렉터리에 각각 구현 완료.

3.  **최종 PnP 프레임워크 (`PnP_Restoration`):**
    *   위에서 구현한 Deconvolution 로직(Data-Fidelity)과 `HuggingFace_Denoiser`(Prior)를 PnP-ADMM 알고리즘으로 결합하는 최종 복원 프레임워크를 구현했습니다.
    *   `code_denoising/pnp_restoration.py`에 구현 완료.

## 11. 다음 목표: 모든 조합의 성능 비교를 위한 마스터 플랜

최종 목표는 학습 기반 모델과 알고리즘 기반 모델을 자유롭게 조합하는 **'하이브리드 파이프라인'**을 포함한 모든 가능성을 탐색하고, 그중 최고 성능의 조합을 찾는 것입니다. 이를 위해 다음의 마스터 플랜을 수립합니다.

| 조합 (Combination) | Denoising 모듈 (Component) | Deconvolution 모듈 (Component) | 상태 |
| :--- | :--- | :--- | :--- |
| **1. 학습 기반 (Step-by-Step)** | `DnCNN` (✅ 학습 필요) | `U-Net` (✅ 학습 필요) | **준비 완료** |
| **2. 알고리즘 기반 (Simple)** | `Diffusion Denoiser` (❌ 학습 불필요) | `Least Squares` (❌ 학습 불필요) | **평가 프레임워크 필요** |
| **3. 알고리즘 기반 (PnP)** | `Diffusion Denoiser` (❌ 학습 불필요) | `Least Squares` (❌ 학습 불필요) | **튜닝 진행 중** |
| **4. 하이브리드 A** | `DnCNN` (✅ 학습 필요) | `Least Squares` (❌ 학습 불필요) | **평가 프레임워크 필요** |
| **5. 하이브리드 B** | `Diffusion Denoiser` (❌ 학습 불필요) | `U-Net` (✅ 학습 필요) | **평가 프레임워크 필요** |

### 실행 전략: 통합 평가 프레임워크 구축

다양한 조합을 체계적으로 실행하고 평가하기 위해, **`run_master_evaluation.ipynb`** 라는 새로운 **통합 평가 노트북**을 구축합니다. 이 노트북은 다음 기능을 제공합니다:

1.  **모듈 선택 및 로딩:** 체크포인트 또는 알고리즘 클래스를 유연하게 로드합니다.
2.  **파이프라인 구성:** 로드된 모듈들을 원하는 순서로 자유롭게 연결합니다.
3.  **일괄 실행 및 결과 저장:** 구성된 파이프라인으로 전체 테스트 데이터셋을 복원하고, 조합별로 결과를 저장합니다.
4.  **자동 채점 및 비교:** 저장된 모든 결과에 대해 `evaluate.ipynb`를 실행하여 최종 점수를 계산하고 비교표를 생성합니다.

## 12. 현재 진행 상황 및 이슈 (Current Status & Issues)

- **`트랙 1` (학습 기반 모델):**
    - U-Net Deconvolution 모델 학습이 재개되었으며, 초기 Epoch에서 **SSIM이 음수로 출력되는 현상**이 관찰되었습니다. 원인 분석이 필요합니다.
- **`트랙 2` (알고리즘 기반 모델):**
    - PnP 하이퍼파라미터 튜닝이 진행 중입니다. `run_diffusion_tests.ipynb`를 통해 최적의 `rho` 및 `noise_level`을 찾고 있습니다.

## 13. 디버깅 회고 및 주요 수정사항

본 프로젝트는 `step-by-step deconvolution` 학습 과정에서 심각하고 반복적인 `RuntimeError: channel mismatch` 및 `AttributeError`를 겪었습니다. 수많은 시행착오 끝에 다음과 같은 핵심 원인들을 발견하고 해결하였으며, 이 과정을 기록하여 향후 유사한 실수를 방지하고자 합니다.

- **`ForwardSimulator`의 채널 생성 오류 (가장 근본적인 원인)**:
    - **문제점**: `dataset/forward_simulator.py`가 컨볼루션 시뮬레이션 후, 복소수 결과에서 `.real`을 사용하여 실수부만 반환했습니다. 이로 인해 2채널이 필요한 `deconvolution` 모델이 1채널 데이터를 받아 `channel mismatch` 오류가 발생했습니다.
    - **해결책**: `.real`을 제거하고, `torch.stack`을 사용하여 결과의 실수부와 허수부를 새로운 채널 차원으로 쌓아, 항상 `[2, H, W]` 형태의 2채널 텐서를 반환하도록 수정했습니다.

- **`get_model` 함수의 모델별 호환성 부족**:
    - **문제점**: `Unet`은 입력 채널 속성으로 `in_chans`를, `DnCNN`은 `channels`를 사용합니다. `get_model` 함수가 `in_chans`만 사용하도록 고정되어 있어 `DnCNN` 평가 시 `AttributeError`가 발생했습니다.
    - **해결책**: `get_model` 함수 내부에 모델 타입(`Unet`, `DnCNN`)에 따라 각각 올바른 속성(`in_chans`, `channels`)을 참조하도록 분기 처리 로직을 추가하여 호환성을 확보했습니다.

- **평가 스크립트의 잘못된 채널 해석**:
    - **문제점**: `e2e` 모델(1채널) 체크포인트를 평가할 때, 평가 스크립트가 `sbs deconvolution` 기준으로 2채널 모델을 생성하여 `size mismatch` 오류가 발생했습니다.
    - **해결책**: 평가 시에는 데이터 증강이 불필요하므로, `create_evaluation_results.py`에서 모델을 생성하기 전에 `config.augmentation_mode = 'none'`으로 강제 설정하는 안전장치를 추가했습니다. 이로써 평가 시에는 항상 1채널 모델이 생성되어 `e2e` 모델과 호환됩니다.

- **`Trainer` 클래스의 초기화 로직 파괴 및 복구**:
    - **문제점**: 반복적인 리팩토링 과정에서 `Trainer` 클래스의 `__init__` 함수에서 `optimizer`, `SummaryWriter` 등 필수 속성을 초기화하는 로직이 누락되거나, `run` 메소드가 삭제되는 등 클래스의 기본 구조가 파괴되어 수많은 `AttributeError`가 발생했습니다.
    - **해결책**: `git` 기록과 논리적 분석을 통해, `__init__` 함수가 `_init_essential` 헬퍼 함수를 호출하여 `model`, `optimizer`, `loss`, `writer` 등을 모두 순서대로 초기화하는 원래의 안정적인 구조로 완벽하게 복구했습니다.
