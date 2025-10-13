import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

# ============== 설정 상수 ==============
SAMPLING_RATE = 256
ANALYSIS_DURATION_MINUTES = 1 # 1분으로 변경
BUFFER_LENGTH = ANALYSIS_DURATION_MINUTES * 60 * SAMPLING_RATE # 1분 분량 샘플 수 (15360)
ANALYSIS_CHANNELS = 2 
ANALYSIS_CHANNEL_NAMES = ['AF7 (L)', 'AF8 (R)']
SKIP_SECONDS = 15 # app.py에서 사용되지만, 여기에도 정의하여 일관성 유지
START_INDEX = SKIP_SECONDS * SAMPLING_RATE

# **********************************************
# --- 필터링 상수 추가 ---
# 0.5Hz 하이패스 필터를 적용하여 DC 오프셋 및 느린 드리프트 제거
HPF_CUTOFF = 0.5
HPF_ORDER = 4
# **********************************************

# 분석할 뇌파 밴드 정의 (주파수 범위)
EEG_BANDS = {
    'Delta': (2, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45), 
}
FIXED_RHYTHM_ORDER = ['Gamma', 'Beta', 'Alpha', 'Theta', 'Delta']
# ======================================

# **********************************************
# --- 필터링 로직 정의 ---
# 필터 계수를 미리 계산합니다.
def design_hpf(cutoff, order, fs):
    # 하이패스 필터 설계
    b, a = butter(order, cutoff, btype='highpass', fs=fs)
    return b, a
# **********************************************

# HPF 계수 계산
HPF_B, HPF_A = design_hpf(HPF_CUTOFF, HPF_ORDER, SAMPLING_RATE)


def preprocess_and_reference(data_chunk: np.ndarray):
    """
    app.py에서 슬라이싱된 1분 데이터 청크를 입력받아 재참조 및 HPF 필터링을 수행합니다.
    """
    
    if data_chunk.ndim != 2 or data_chunk.shape[1] < 4:
        raise ValueError(f"데이터 형태가 잘못되었습니다. 4개 이상의 채널이 필요합니다. 현재 채널 수: {data_chunk.shape[1]}")

    # LSL 채널 순서 (0:AF7, 1:AF8, 2:TP9, 3:TP10)를 가정
    af7_raw = data_chunk[:, 0].astype(float)
    af8_raw = data_chunk[:, 1].astype(float)
    tp9_chunk = data_chunk[:, 2].astype(float)
    tp10_chunk = data_chunk[:, 3].astype(float)
    
    # 평균 참조 (Average Reference)
    avg_ref_chunk = (tp9_chunk + tp10_chunk) / 2.0
    
    # 재참조된 AF7, AF8 데이터
    af7_corrected = af7_raw - avg_ref_chunk
    af8_corrected = af8_raw - avg_ref_chunk
    
    # **********************************************
    # --- HPF 필터링 적용 ---
    # 신호 시작점에서의 튀는 현상을 막기 위해 lfilter_zi를 사용하여 초기 상태 설정
    zi = lfilter_zi(HPF_B, HPF_A)
    
    # AF7 필터링
    af7_filtered, _ = lfilter(HPF_B, HPF_A, af7_corrected, zi=zi * af7_corrected[0])
    
    # AF8 필터링
    af8_filtered, _ = lfilter(HPF_B, HPF_A, af8_corrected, zi=zi * af8_corrected[0])
    
    return [af7_filtered, af8_filtered] # [AF7_corrected, AF8_corrected]
    # **********************************************


def analyze_eeg_rhythms(data_5min_chunks: list):
    """
    재참조 및 HPF 필터링된 뇌파 데이터 덩어리(chunk)에 대해 뇌파 리듬 점유율을 계산합니다.
    """
    all_results = {}
    
    for i, data in enumerate(data_5min_chunks): 
        channel_name = ANALYSIS_CHANNEL_NAMES[i]
        
        # 데이터 유효성 검사 (길이 대신 0인지만 확인)
        if np.all(data == 0):
            all_results[channel_name] = {"Error": "Invalid or Low Power Data"}
            continue

        n = BUFFER_LENGTH # 15360 샘플
        # 1. 윈도우 함수 적용 (Hanning)
        fft_data = np.fft.rfft(data * np.hanning(n)) 
        
        # 2. PSD 계산 (Power Spectral Density)
        psd = np.abs(fft_data)**2
        freqs = np.fft.rfftfreq(n, 1.0/SAMPLING_RATE)
        
        absolute_band_powers = {}
        
        # 3. 각 밴드의 절대 파워 합산
        for band_name, (low, high) in EEG_BANDS.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            absolute_band_powers[band_name] = np.sum(psd[idx_band])

        total_power_in_bands = sum(absolute_band_powers.values())
        
        if total_power_in_bands < 1e-12: # 총 파워가 너무 작으면 Low Power 처리
             all_results[channel_name] = {"Error": "Total Power too Low"}
             continue
             
        channel_rhythm_map = {}
        
        # 4. 상대적 점유율 (Relative Power) 계산
        for band_name in FIXED_RHYTHM_ORDER:
            absolute_power = absolute_band_powers.get(band_name, 0)
            relative_power = (absolute_power / total_power_in_bands) * 100
            channel_rhythm_map[band_name] = round(relative_power, 2)

        all_results[channel_name] = channel_rhythm_map
        
    return all_results

