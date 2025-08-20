# 🎯 이미지 디노이징 프로젝트 - 혁신적 접근법 구상

## 📋 과제 목표
**가장 효과적인 이미지 복원 방법을 구상하고 구현하여 최고 성능 달성**

## 🚀 접근 방향들

### A. 딥러닝 방법 발전 (DnCNN 기반)
- **DnCNN 튜닝**: 레이어 수, 필터 수, 활성화 함수 최적화
- **새로운 아키텍처**: U-Net, ResNet, Attention 기반 모델
- **Multi-scale 접근**: 다양한 해상도에서 특징 추출

### B. 고전적 방법론 활용
- **Wiener Filter**: 주파수 도메인에서 최적화
- **Bilateral Filter**: 엣지 보존하면서 노이즈 제거
- **Non-local Means**: 유사한 패턴을 찾아 평균화
- **Wavelet 기반**: 다중 해상도 분석

### C. 하이브리드 접근 (Denoising + Deconvolution)
- **End-to-End 네트워크**: 노이즈 제거와 디컨볼루션을 동시에
- **Cascade 구조**: 1단계 노이즈 제거 → 2단계 디컨볼루션
- **Multi-task Learning**: 두 작업을 동시에 학습

## 🎨 제안하는 혁신적 모델: "DenoiseNet"

### 아키텍처 구조
```
Input (Noisy Image)
    ↓
[Feature Extraction] - ResNet 블록들
    ↓
[Multi-scale Processing] - U-Net 스타일 skip connection
    ↓
[Attention Module] - 어디를 집중할지 학습
    ↓
[Dual Branch]
    ├── [Denoising Branch] - 노이즈 제거
    └── [Deconvolution Branch] - 블러 제거
    ↓
[Fusion Module] - 두 결과를 지능적으로 결합
    ↓
Output (Clean Image)
```

## 🔧 구체적 구현 계획

### 1단계: 기존 방법 성능 베이스라인 확립
```bash
# 현재 구현된 방법들로 성능 측정
python test.py --model_type dncnn
python test.py --model_type mean_filter  
python test.py --model_type median_filter
```

### 2단계: DnCNN 개선
- **아키텍처 수정**: 더 깊은 네트워크, Attention 추가
- **손실 함수**: L1 + L2 + SSIM 조합
- **데이터 증강**: 다양한 노이즈 레벨, 회전, 뒤집기

### 3단계: 새로운 모델 구현
- **U-Net 기반**: Skip connection으로 세부 정보 보존
- **ResNet 블록**: 깊은 네트워크에서도 안정적 학습
- **Attention 메커니즘**: 중요한 영역에 집중

### 4단계: 하이브리드 모델 (Denoising + Deconvolution)
- **Cascade 구조**: 순차적 처리
- **Multi-task**: 동시 학습으로 상호 보완
- **Adaptive Fusion**: 입력에 따라 가중치 조정

## 💡 혁신 포인트들

### A. 아키텍처 혁신
- **Multi-scale Feature Pyramid**: 다양한 해상도에서 특징 추출
- **Cross-scale Attention**: 서로 다른 스케일 간 정보 교환
- **Progressive Refinement**: 점진적으로 결과 개선

### B. 학습 전략
- **Curriculum Learning**: 쉬운 것부터 어려운 것까지
- **Adversarial Training**: GAN으로 더 현실적인 결과
- **Self-supervised Learning**: 레이블 없이도 학습 가능

### C. 손실 함수 혁신
- **Perceptual Loss**: VGG 특징 기반 손실
- **Adversarial Loss**: 판별자와의 경쟁
- **Structural Similarity Loss**: 구조적 유사성 보존

## 📅 실행 계획

### 즉시 시작할 수 있는 것들:
1. **기존 코드 실행**해서 베이스라인 성능 확인
2. **DnCNN 하이퍼파라미터 튜닝** (레이어 수, 필터 수)
3. **손실 함수 실험** (L1, L2, SSIM 조합)

### 단계별 발전:
1. **U-Net 기반 모델** 구현
2. **Attention 메커니즘** 추가
3. **Multi-task 학습** (Denoising + Deconvolution)

## 🎯 우선순위

### High Priority (1-2주)
- [ ] 기존 코드 실행 및 성능 측정
- [ ] DnCNN 하이퍼파라미터 튜닝
- [ ] 베이스라인 성능 확립

### Medium Priority (2-3주)
- [ ] U-Net 기반 모델 구현
- [ ] Attention 메커니즘 추가
- [ ] 손실 함수 실험

### Low Priority (3-4주)
- [ ] 하이브리드 모델 설계
- [ ] Multi-task 학습 구현
- [ ] 최종 성능 비교 및 분석

## 🔍 성능 평가 지표
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **시각적 품질**: 노이즈 제거, 엣지 보존, 디테일 복원

## 📚 참고 자료
- DnCNN 논문
- U-Net 아키텍처
- Attention 메커니즘
- Multi-task Learning
- Image Deconvolution 기법들

