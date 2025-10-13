import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
import pandas as pd

# ============== 설정 상수 ==============
SAMPLING_RATE = 256
ANALYSIS_DURATION_MINUTES = 5 
BUFFER_LENGTH = ANALYSIS_DURATION_MINUTES * 60 * SAMPLING_RATE # 5분 분량 샘플 수 (76800)
ANALYSIS_CHANNELS = 2 
ANALYSIS_CHANNEL_NAMES = ['AF7 (L)', 'AF8 (R)']

# 분석할 뇌파 밴드 정의 (주파수 범위)
EEG_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45), 
}
FIXED_RHYTHM_ORDER = ['Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'] # 최종 결과 표시 순서
# ======================================

def preprocess_and_reference(raw_data: np.ndarray):
    """
    1. 원시 데이터에서 5분 분량을 추출합니다.
    2. TP9/TP10 평균을 이용해 AF7/AF8을 재참조합니다.
    
    raw_data: 4채널 (AF7, AF8, TP9, TP10)의 Numpy 배열.
    """
    
    if raw_data.ndim != 2 or raw_data.shape[1] < 4:
        raise ValueError(f"데이터 형태가 잘못되었습니다. 4개 이상의 채널이 필요합니다. 현재 채널 수: {raw_data.shape[1]}")

    if raw_data.shape[0] < BUFFER_LENGTH:
        raise ValueError(f"데이터 길이가 분석에 필요한 5분 ({BUFFER_LENGTH} 샘플) 미만입니다. 현재: {raw_data.shape[0]} 샘플")

    # 5분 분량만 추출 (시작부터)
    data_to_process = raw_data[:BUFFER_LENGTH, :] 
        
    # LSL 채널 순서 (0:AF7, 1:AF8, 2:TP9, 3:TP10)를 가정
    af7_raw = data_to_process[:, 0].astype(float)
    af8_raw = data_to_process[:, 1].astype(float)
    tp9_chunk = data_to_process[:, 2].astype(float)
    tp10_chunk = data_to_process[:, 3].astype(float)
    
    # 평균 참조 (Average Reference)
    avg_ref_chunk = (tp9_chunk + tp10_chunk) / 2.0
    
    # 재참조된 AF7, AF8 데이터
    af7_corrected = af7_raw - avg_ref_chunk
    af8_corrected = af8_raw - avg_ref_chunk
    
    return [af7_corrected, af8_corrected] # [AF7_corrected, AF8_corrected]

def analyze_eeg_rhythms(data_5min_chunks: list):
    """
    재참조된 5분 데이터 덩어리(chunk)에 대해 뇌파 리듬 점유율을 계산합니다.
    """
    all_results = {}
    
    for i, data in enumerate(data_5min_chunks): 
        channel_name = ANALYSIS_CHANNEL_NAMES[i]
        
        # 데이터 유효성 검사
        if np.all(data == 0) or len(data) != BUFFER_LENGTH:
            all_results[channel_name] = {"Error": "Invalid or Low Power Data"}
            continue

        n = BUFFER_LENGTH
        # 1. 윈도우 함수 적용 (Hanning)
        # 2. FFT를 통해 주파수 도메인으로 변환
        fft_data = np.fft.rfft(data * np.hanning(n)) 
        
        # 3. PSD 계산 (Power Spectral Density)
        psd = np.abs(fft_data)**2
        freqs = np.fft.rfftfreq(n, 1.0/SAMPLING_RATE)
        
        absolute_band_powers = {}
        
        # 4. 각 밴드의 절대 파워 합산
        for band_name, (low, high) in EEG_BANDS.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            absolute_band_powers[band_name] = np.sum(psd[idx_band])

        total_power_in_bands = sum(absolute_band_powers.values())
        
        if total_power_in_bands < 1e-12: # 총 파워가 너무 작으면 Low Power 처리
             all_results[channel_name] = {"Error": "Total Power too Low"}
             continue
             
        channel_rhythm_map = {}
        
        # 5. 상대적 점유율 (Relative Power) 계산
        for band_name in FIXED_RHYTHM_ORDER:
            absolute_power = absolute_band_powers.get(band_name, 0)
            relative_power = (absolute_power / total_power_in_bands) * 100
            channel_rhythm_map[band_name] = round(relative_power, 2)

        all_results[channel_name] = channel_rhythm_map
        
    return all_results