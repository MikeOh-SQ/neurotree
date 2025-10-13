import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

# ============== 설정 상수 ==============
SAMPLING_RATE = 256
ANALYSIS_DURATION_MINUTES = 1 
BUFFER_LENGTH = ANALYSIS_DURATION_MINUTES * 60 * SAMPLING_RATE
ANALYSIS_CHANNELS = 2 
ANALYSIS_CHANNEL_NAMES = ['AF7 (L)', 'AF8 (R)']
SKIP_SECONDS = 15
START_INDEX = SKIP_SECONDS * SAMPLING_RATE

# **********************************************
# --- 필터링 상수 수정/추가 ---
HPF_CUTOFF = 0.5
HPF_ORDER = 4
# 50Hz에서 45Hz로 LPF를 조정하여 감마파 상단 EMG 잡음 제거 강화
LPF_CUTOFF = 45.0 
LPF_ORDER = 4
# **********************************************

# 분석할 뇌파 밴드 정의 (주파수 범위)
EEG_BANDS = {
    # 델타파 대역은 3Hz부터 시작하도록 유지
    'Delta': (3.0, 4), 
    'Theta': (4.0, 8), 
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45), 
}
FIXED_RHYTHM_ORDER = ['Gamma', 'Beta', 'Alpha', 'Theta', 'Delta']
# ======================================

# HPF 계수 계산
HPF_B, HPF_A = butter(HPF_ORDER, HPF_CUTOFF, btype='highpass', fs=SAMPLING_RATE)
# LPF 계수 계산
LPF_B, LPF_A = butter(LPF_ORDER, LPF_CUTOFF, btype='lowpass', fs=SAMPLING_RATE)


def preprocess_and_reference(data_chunk: np.ndarray):
    """
    app.py에서 슬라이싱된 1분 데이터 청크를 입력받아 재참조 및 HPF/LPF 필터링을 수행합니다.
    """
    
    if data_chunk.ndim != 2 or data_chunk.shape[1] < 4:
        raise ValueError(f"데이터 형태가 잘못되었습니다. 4개 이상의 채널이 필요합니다. 현재 채널 수: {data_chunk.shape[1]}")

    af7_raw = data_chunk[:, 0].astype(float)
    af8_raw = data_chunk[:, 1].astype(float)
    tp9_chunk = data_chunk[:, 2].astype(float)
    tp10_chunk = data_chunk[:, 3].astype(float)
    
    avg_ref_chunk = (tp9_chunk + tp10_chunk) / 2.0
    
    af7_corrected = af7_raw - avg_ref_chunk
    af8_corrected = af8_raw - avg_ref_chunk
    
    # --- 필터링 적용 ---
    # 1. HPF 필터링 적용 (느린 드리프트 제거)
    zi_hpf = lfilter_zi(HPF_B, HPF_A)
    af7_hpf, _ = lfilter(HPF_B, HPF_A, af7_corrected, zi=zi_hpf * af7_corrected[0])
    af8_hpf, _ = lfilter(HPF_B, HPF_A, af8_corrected, zi=zi_hpf * af8_corrected[0])
    
    # 2. LPF 필터링 적용 (고주파수 잡음 제거, 45Hz 이상 차단)
    zi_lpf = lfilter_zi(LPF_B, LPF_A)
    af7_filtered, _ = lfilter(LPF_B, LPF_A, af7_hpf, zi=zi_lpf * af7_hpf[0])
    af8_filtered, _ = lfilter(LPF_B, LPF_A, af8_hpf, zi=zi_lpf * af8_hpf[0])
    
    return [af7_filtered, af8_filtered] 


def analyze_eeg_rhythms(data_chunk: list):
    # (내용은 변경 없음, 상위 상수 및 필터만 변경됨)
    all_results = {}
    
    for i, data in enumerate(data_chunk): 
        channel_name = ANALYSIS_CHANNEL_NAMES[i]
        
        if np.all(data == 0):
            all_results[channel_name] = {"Error": "Invalid or Low Power Data"}
            continue

        n = BUFFER_LENGTH 
        fft_data = np.fft.rfft(data * np.hanning(n)) 
        psd = np.abs(fft_data)**2
        freqs = np.fft.rfftfreq(n, 1.0/SAMPLING_RATE)
        
        absolute_band_powers = {}
        for band_name, (low, high) in EEG_BANDS.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            absolute_band_powers[band_name] = np.sum(psd[idx_band])

        total_power_in_bands = sum(absolute_band_powers.values())
        
        if total_power_in_bands < 1e-12: 
             all_results[channel_name] = {"Error": "Total Power too Low"}
             continue
             
        channel_rhythm_map = {}
        
        for band_name in FIXED_RHYTHM_ORDER:
            absolute_power = absolute_band_powers.get(band_name, 0)
            relative_power = (absolute_power / total_power_in_bands) * 100
            channel_rhythm_map[band_name] = round(relative_power, 2)

        all_results[channel_name] = channel_rhythm_map
        
    return all_results
