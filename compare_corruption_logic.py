import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

# --- 기존 robust_noise_analyzer.py의 핵심 함수 ---
def estimate_robust_noise_std(image, outlier_percentile=95):
    """
    Isotropic 미분과 이상치 제거를 통해 노이즈 표준편차를 강건하게 추정합니다.
    """
    if image.ndim == 3:
        image = image.mean(axis=2)
    
    diff_h = np.diff(image, axis=1, append=0)
    diff_v = np.diff(image, axis=0, append=0)
    
    magnitude = np.sqrt(diff_h**2 + diff_v**2).flatten()
    
    # 이미지 질감의 영향을 줄이기 위해 상위 percentile을 이상치로 간주하고 제거
    outlier_threshold = np.percentile(magnitude, outlier_percentile)
    pure_noise_magnitudes = magnitude[magnitude < outlier_threshold]
    
    # 가우시안 노이즈의 경우, 미분 크기의 평균과 표준편차 사이의 관계 이용
    estimated_std = np.mean(pure_noise_magnitudes) / np.sqrt(np.pi / 2)
    
    return estimated_std

def parse_v2_filename(path):
    """test_y_v2 파일명에서 노이즈 레벨을 파싱합니다."""
    match = re.search(r'_noise_lv(\d+)\.npy$', path.name)
    if match:
        return int(match.group(1))
    return None

def analyze_dataset(folder_path, is_v2=False):
    """지정된 데이터셋 폴더의 모든 이미지에 대해 노이즈를 분석합니다."""
    path = Path(folder_path)
    files = list(path.glob('*.npy'))
    
    results = {}
    
    print(f"Analyzing {folder_path}...")
    for file in tqdm(files, desc=f"Processing {path.name}"):
        image = np.load(file)
        std = estimate_robust_noise_std(image)
        
        if is_v2:
            level = parse_v2_filename(file)
            if level:
                if level not in results:
                    results[level] = []
                results[level].append(std)
        else:
            if 'all' not in results:
                results['all'] = []
            results['all'].append(std)
            
    return results

def main():
    # 1. 각 데이터셋 분석
    test_y_results_raw = analyze_dataset('dataset/test_y')
    test_y_v2_results = analyze_dataset('dataset/test_y_v2', is_v2=True)

    # 2. test_y_v2 결과 정리
    avg_std_v2_l1 = np.mean(test_y_v2_results.get(1, [0]))
    avg_std_v2_l2 = np.mean(test_y_v2_results.get(2, [0]))

    # 3. test_y 결과 K-Means 클러스터링으로 그룹화
    all_stds_y = np.array(test_y_results_raw['all']).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(all_stds_y)
    
    cluster_centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_

    # 클러스터 센터 값에 따라 L1, L2 그룹 할당
    l1_center = min(cluster_centers)
    l2_center = max(cluster_centers)

    avg_std_y_l1 = l1_center
    avg_std_y_l2 = l2_center
    
    # 4. 최종 결과 출력 및 가설 검증
    print("\n" + "="*50)
    print("       최종 가설 검증 결과")
    print("="*50)
    print("가설: test_y = Noise(f),  test_y_v2 = Conv(f) + Noise(f)")
    print("논리: Conv가 이미지 질감을 제거하므로, σ(test_y) > σ(test_y_v2) 여야 함\n")

    print(f"[test_y 분석 결과 (K-Means 클러스터링)]")
    print(f"  - L1 추정 그룹 평균 σ: {avg_std_y_l1:.5f} (n={np.sum(labels == np.argmin(cluster_centers))})")
    print(f"  - L2 추정 그룹 평균 σ: {avg_std_y_l2:.5f} (n={np.sum(labels == np.argmax(cluster_centers))})")
    print("-" * 50)
    print(f"[test_y_v2 분석 결과 (파일명 기반)]")
    print(f"  - L1 실제 그룹 평균 σ: {avg_std_v2_l1:.5f} (n={len(test_y_v2_results.get(1, []))})")
    print(f"  - L2 실제 그룹 평균 σ: {avg_std_v2_l2:.5f} (n={len(test_y_v2_results.get(2, []))})")
    print("="*50)
    
    # --- 판정 ---
    verdict_l1 = "지지됨" if avg_std_y_l1 > avg_std_v2_l1 else "기각됨"
    verdict_l2 = "지지됨" if avg_std_y_l2 > avg_std_v2_l2 else "기각됨"

    print("판정:")
    print(f"  - Level 1 비교: σ(y)={avg_std_y_l1:.5f} > σ(v2)={avg_std_v2_l1:.5f}  =>  가설이 '{verdict_l1}'")
    print(f"  - Level 2 비교: σ(y)={avg_std_y_l2:.5f} > σ(v2)={avg_std_v2_l2:.5f}  =>  가설이 '{verdict_l2}'")
    print("="*50)

    # 최종 결론
    if verdict_l1 == "지지됨" and verdict_l2 == "지지됨":
        print("\n[최종 결론]")
        print("모든 레벨에서 'σ(test_y) > σ(test_y_v2)'가 관찰되었습니다.")
        print("이는 'test_y는 컨볼루션 없이 노이즈만 추가되었고, test_y_v2는 컨볼루션으로 인해 배경이 깨끗해져 노이즈가 더 정확하게 측정되었다'는 우리의 논리와 정확히 일치합니다.")
        print("따라서, 'test_y는 Denoising 문제'라는 가설이 강력하게 지지됩니다.")
    else:
        print("\n[최종 결론]")
        print("가설이 기각되었습니다. 'test_y'와 'test_y_v2'의 열화 방식은 우리가 가정한 것과 다릅니다.")
        print("두 데이터셋의 관계에 대한 근본적인 재검토가 필요합니다.")

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(all_stds_y[labels == np.argmin(cluster_centers)], np.zeros(np.sum(labels == np.argmin(cluster_centers))), 
                c='blue', label=f'test_y L1 추정 (μ={avg_std_y_l1:.3f})', alpha=0.6)
    plt.scatter(all_stds_y[labels == np.argmax(cluster_centers)], np.zeros(np.sum(labels == np.argmax(cluster_centers))), 
                c='red', label=f'test_y L2 추정 (μ={avg_std_y_l2:.3f})', alpha=0.6)

    plt.axvline(x=avg_std_v2_l1, color='cyan', linestyle='--', label=f'test_y_v2 L1 평균 (μ={avg_std_v2_l1:.3f})')
    plt.axvline(x=avg_std_v2_l2, color='magenta', linestyle='--', label=f'test_y_v2 L2 평균 (μ={avg_std_v2_l2:.3f})')

    plt.title('Noise Level (σ) Distribution Comparison')
    plt.xlabel('Estimated Noise Standard Deviation (σ)')
    plt.yticks([])
    plt.legend()
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    output_path = Path("Noise_Analysis") / "final_hypothesis_verification.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"\n분석 결과 시각화가 '{output_path}'에 저장되었습니다.")


if __name__ == '__main__':
    main()
