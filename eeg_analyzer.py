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

# 분석할 뇌파 밴드 정의 (주파수 범위)
EEG_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45), 
}
FIXED_RHYTHM_ORDER = ['Gamma', 'Beta', 'Alpha', 'Theta', 'Delta']
# ======================================

def preprocess_and_reference(data_chunk: np.ndarray):
    """
    app.py에서 슬라이싱된 1분 데이터 청크를 입력받아 재참조를 수행합니다.
    (길이 체크는 app.py가 담당하므로 여기서 제거합니다.)
    
    data_chunk: 4채널 (AF7, AF8, TP9, TP10)의 1분 길이 Numpy 배열.
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
    
    return [af7_corrected, af8_corrected] # [AF7_corrected, AF8_corrected]


def analyze_eeg_rhythms(data_5min_chunks: list):
    """
    재참조된 뇌파 데이터 덩어리(chunk)에 대해 뇌파 리듬 점유율을 계산합니다.
    (여전히 BUFFER_LENGTH 상수를 내부적으로 사용하므로, 이 파일에서 BUFFER_LENGTH가 1분 길이로 수정되어야 합니다.)
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
