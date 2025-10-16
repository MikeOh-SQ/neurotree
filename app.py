import streamlit as st
import numpy as np
import pandas as pd
import io
import requests # GitHub 파일 다운로드 및 요청을 위해 requests 모듈 추가
from scipy.signal import butter, lfilter

# eeg_analyzer.py 파일에서 핵심 분석 함수와 상수를 가져옵니다.
try:
    from eeg_analyzer import analyze_eeg_rhythms, ANALYSIS_CHANNEL_NAMES, SAMPLING_RATE, BUFFER_LENGTH
    
    # 분석 로직 변경: 15초 스킵, 60초 분석 상수는 app.py에서 정의
    ANALYSIS_DURATION = 60 
    SKIP_DURATION = 15 
    TOTAL_PROCESSED_TIME = ANALYSIS_DURATION + SKIP_DURATION
    
except ImportError:
    st.error("Error: eeg_analyzer.py 파일을 찾을 수 없습니다. analyze_eeg_rhythms 함수가 정의되어 있는지 확인해주세요.")
    st.stop()


# ============== 필터 정의 (DSP 로직을 app.py에 통합) ==============
# LPF 및 HPF 필터 계수를 미리 계산 (Filter Order = 4 가정)

def design_filters():
    # HPF (0.5Hz)
    b_hpf, a_hpf = butter(4, 0.5, btype='highpass', fs=SAMPLING_RATE)
    # LPF (45.0Hz)
    b_lpf, a_lpf = butter(4, 45.0, btype='lowpass', fs=SAMPLING_RATE)
    return b_hpf, a_hpf, b_lpf, a_lpf

B_HPF, A_HPF, B_LPF, A_LPF = design_filters()
# =========================================================

# 재참조 및 필터링 수행 함수
def simple_preprocess_and_reference(data_chunk: np.ndarray):
    """데이터 슬라이싱 후 재참조 및 필터링 수행 (app.py에서 B_HPF 등을 사용)"""
    
    # 1. TP9/TP10을 이용한 평균 참조 계산
    tp9_chunk = data_chunk[:, 2].astype(float)
    tp10_chunk = data_chunk[:, 3].astype(float)
    avg_ref_chunk = (tp9_chunk + tp10_chunk) / 2.0
    
    af7_raw = data_chunk[:, 0].astype(float)
    af8_raw = data_chunk[:, 1].astype(float)
    
    af7_corrected = af7_raw - avg_ref_chunk
    af8_corrected = af8_raw - avg_ref_chunk

    # 2. 필터링 적용 (HPF -> LPF 순서)
    
    # AF7 필터링
    af7_filtered = lfilter(B_HPF, A_HPF, af7_corrected)
    af7_filtered = lfilter(B_LPF, A_LPF, af7_filtered)

    # AF8 필터링
    af8_filtered = lfilter(B_HPF, A_HPF, af8_corrected)
    af8_filtered = lfilter(B_LPF, A_LPF, af8_filtered)
    
    return [af7_filtered, af8_filtered] 


st.set_page_config(layout="wide", page_title="뇌파 트리 분석기")

st.title("🎄 뇌파 리듬 기반 크리스마스 트리 생성기")
st.markdown(f"EEG 원시 데이터를 업로드하여 **앞 15초를 제외한 1분 동안**의 뇌파 점유율을 분석하고 트리를 생성합니다.")

# =========================================================
# UI 섹션 1: 샘플 파일 다운로드 및 적용 버튼
# =========================================================
st.subheader("샘플 파일 사용")

# 제공받은 GitHub 링크의 원본(RAW) URL
SAMPLE_CSV_URL = "https://raw.githubusercontent.com/MikeOh-SQ/neurotree/main/sample.csv"

col_download, col_apply = st.columns([1, 1])

with col_download:
    # --- 샘플 CSV 파일 다운로드 버튼 ---
    try:
        sample_response = requests.get(SAMPLE_CSV_URL)
        if sample_response.status_code == 200:
            st.download_button(
                label="샘플 CSV 파일 다운로드 (GitHub)",
                data=sample_response.content, # 응답 내용을 바로 파일 데이터로 사용
                file_name="sample_eeg_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("샘플 파일 다운로드 링크를 가져올 수 없습니다.")
    except Exception:
        st.warning("샘플 파일 다운로드 기능을 초기화할 수 없습니다.")


with col_apply:
    # 샘플 파일 적용 버튼 (state를 사용하여 플로우 제어)
    if st.button("샘플 파일 적용 및 분석 시작"):
        st.session_state['use_sample'] = True
        st.rerun() # 버튼 클릭 시 페이지를 재실행하여 분석 시작

# =========================================================
# UI 섹션 2: 파일 업로드 및 CSV 형식 설명
# =========================================================

st.subheader("1. EEG Raw 데이터 업로드")
st.markdown("""
<div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
    <p><strong>🚨 분석 가능 파일 형식:</strong> CSV 파일만 가능합니다.</p>
    <p><strong>(CSV의 경우 MuseLab에서 csv로 출력한 파일을 입력 가능합니다.)</strong></p>
    <p>파일은 'timestamp', 'osc_address', 'osc_type', 'osc_data'와 같은 헤더를 포함하며, EEG 데이터는 'osc_data' 열에 쉼표로 구분되어야 합니다.</p>
</div>
""", unsafe_allow_html=True)


# 파일 업로드 위젯 (CSV만 허용하도록 수정)
uploaded_file = st.file_uploader(
    "EEG Raw 데이터 파일 (.csv) 업로드", 
    type=['csv'],
    disabled=st.session_state.get('use_sample', False) # 샘플 사용 중이면 비활성화
)

# 파일 처리 로직 통합
file_to_analyze = None

if st.session_state.get('use_sample', False):
    # 샘플 파일 적용 모드
    try:
        response = requests.get(SAMPLE_CSV_URL)
        if response.status_code == 200:
            # StringIO로 변환하여 파일 객체처럼 사용
            file_to_analyze = io.StringIO(response.content.decode('utf-8'))
            st.info("GitHub 샘플 파일이 성공적으로 로드되었습니다.")
            st.session_state['ready_to_analyze'] = True 
        else:
            st.error(f"GitHub에서 샘플 파일을 다운로드하는 데 실패했습니다. 상태 코드: {response.status_code}")
            st.session_state['use_sample'] = False
            st.session_state['ready_to_analyze'] = False
    except Exception as e:
        st.error(f"샘플 파일 로딩 중 오류 발생: {e}")
        st.session_state['use_sample'] = False
        st.session_state['ready_to_analyze'] = False
        
elif uploaded_file is not None:
    # 사용자 파일 업로드 모드
    file_to_analyze = uploaded_file
    st.session_state['ready_to_analyze'] = True 
else:
    # 파일이 없거나 업로드 대기 중
    st.session_state['ready_to_analyze'] = False


# =========================================================
# 분석 실행 로직 (샘플이든 업로드 파일이든)
# =========================================================

if st.session_state.get('ready_to_analyze', False) and file_to_analyze is not None:
    try:
        # 1. 데이터 로드 및 파싱 (MuseLab CSV 파일 파싱)
        with st.spinner('데이터 파싱 및 전처리 중입니다...'):
            
            df = pd.read_csv(file_to_analyze)
            
            # 'osc_address'가 'eeg'인 행만 필터링
            eeg_df = df[df['osc_address'].str.contains('/eeg', na=False)].copy()
            
            # 파싱 함수 정의 
            def parse_eeg_string(eeg_str):
                # 큰따옴표 제거 후 쉼표로 분리
                values = eeg_str.strip('"').split(',')
                # 처음 4개 값 (AF7, AF8, TP9, TP10)만 가져와서 float으로 변환
                try:
                    # 'NaN' 문자열 처리 포함
                    return [float(v) if v.lower() not in ('nan', '') else np.nan for v in values[:4]]
                except Exception:
                    return [np.nan] * 4 # 파싱 오류 시 NaN 반환
            
            parsed_eeg_data = eeg_df['osc_data'].apply(parse_eeg_string)
            raw_data = np.array(parsed_eeg_data.tolist())
            raw_data = np.nan_to_num(raw_data, nan=0.0) # NaN 값은 0으로 채웁니다.

        
        # 2. 전처리 (15초 스킵 후 1분 추출 및 재참조)
        
        start_sample = int(SKIP_DURATION * SAMPLING_RATE)
        end_sample = start_sample + int(ANALYSIS_DURATION * SAMPLING_RATE)
        
        if raw_data.shape[0] < end_sample:
            required_samples = end_sample
            st.error(f"❌ 데이터 길이 부족: 총 {TOTAL_PROCESSED_TIME}초 ({required_samples} 샘플)의 데이터가 필요합니다. 현재 샘플 수: {raw_data.shape[0]}")
            st.stop()
            
        data_to_analyze = raw_data[start_sample:end_sample, :]
        
        # 재참조 및 필터링 수행 (app.py에 통합된 함수 사용)
        eeg_chunks_corrected = simple_preprocess_and_reference(data_to_analyze)


        # 3. 뇌파 리듬 계산 (eeg_analyzer.py에 정의된 함수 사용)
        with st.spinner(f'뇌파 점유율을 분석 중입니다 ({SKIP_DURATION}초 스킵 후 1분 데이터)...'):
            rhythm_results = analyze_eeg_rhythms(eeg_chunks_corrected)
        
        st.success(f"분석 완료! (분석 범위: {SKIP_DURATION}초 ~ {TOTAL_PROCESSED_TIME}초 지점)")
        
        # 4. 뇌파 값 기반 트리 시각화 (개선된 Dominant Rhythm 로직 적용)
        
        # AF7 (L) 채널 결과 사용
        rhythm_data = rhythm_results.get(ANALYSIS_CHANNEL_NAMES[0], {})
        
        # Alpha/Theta 비율 계산 (트리의 '집중도' 지표)
        alpha_power = rhythm_data.get('Alpha', 0.01) # 0 방지
        theta_power = rhythm_data.get('Theta', 0.01) # 0 방지
        alpha_theta_ratio = alpha_power / theta_power
        
        # 가장 지배적인 리듬 찾기 (Gamma, Beta, Alpha, Theta, Delta 중에서)
        dominant_rhythm_name = None
        dominant_power = 0
        
        # Gamma, Beta, Alpha, Theta, Delta 순서로 확인하여 가장 높은 파워 찾기
        # (이 순서는 eeg_analyzer.py의 FIXED_RHYTHM_ORDER와 일치한다고 가정)
        for name, power in rhythm_data.items():
            if power > dominant_power:
                dominant_power = power
                dominant_rhythm_name = name

        
        # --- 트리 메시지 및 상태 결정 (최고 리듬에 따른 정확한 출력) ---
        if dominant_rhythm_name == 'Beta':
            tree_emoji = "💥🎄💥"
            tree_message = f"**최고 리듬: Beta ({dominant_power:.1f}%)**: 트리가 활발하게 깜빡입니다. 각성 및 활동 상태입니다!"
        elif dominant_rhythm_name == 'Alpha':
            tree_emoji = "🌟🎄🌟"
            tree_message = f"**최고 리듬: Alpha ({dominant_power:.1f}%)**: 트리가 가장 밝게 빛납니다. 안정적 집중 상태입니다!"
        elif dominant_rhythm_name == 'Theta':
            tree_emoji = "🌙🎄🌙"
            tree_message = f"**최고 리듬: Theta ({dominant_power:.1f}%)**: 트리가 차분한 푸른빛을 띱니다. 깊은 이완/몰입 상태입니다."
        elif dominant_rhythm_name == 'Gamma':
            tree_emoji = "💡🎄💡"
            tree_message = f"**최고 리듬: Gamma ({dominant_power:.1f}%)**: 트리의 조명이 복잡하게 빛납니다. 고차원적 인지 처리 상태입니다!"
        elif dominant_rhythm_name == 'Delta':
            tree_emoji = "💧🎄💧"
            tree_message = f"**최고 리듬: Delta ({dominant_power:.1f}%)**: 트리가 매우 어둡습니다. 깊은 수면 또는 잔여 잡음 상태입니다."
        else:
            tree_emoji = "✨🎄✨"
            tree_message = f"**최고 리듬: {dominant_rhythm_name} ({dominant_power:.1f}%)**: 트리가 균형 있게 빛납니다. (A/T 비율: {alpha_theta_ratio:.2f})"
        
        
        st.header("트리 생성 결과")
        st.markdown(f"## {tree_emoji}")
        st.markdown(tree_message)
        
        # 5. 상세 분석 결과 (표)
        st.subheader("상세 뇌파 점유율 ($\text{AF7, AF8}$)")
        
        df_results = pd.DataFrame(rhythm_results).T 
        st.dataframe(df_results)
        
        # 분석 완료 후 샘플 사용 상태 초기화 (사용자 재시도를 위해)
        if st.session_state.get('use_sample', False):
             st.session_state['use_sample'] = False
             
    except ValueError as e:
        st.error(f"❌ 데이터 오류: {e}")
    except Exception as e:
        st.error(f"🔥 예기치 않은 오류 발생: {e}")
        
    # 에러가 났을 때도 세션 상태는 초기화 해줍니다.
    if st.session_state.get('use_sample', False):
        st.session_state['use_sample'] = False

# 세션 상태 초기화 버튼 (오류 시 재시도)
if st.session_state.get('ready_to_analyze', False) or st.session_state.get('use_sample', False):
    if st.button("초기 상태로 돌아가기"):
        st.session_state['use_sample'] = False
        st.session_state['ready_to_analyze'] = False
        st.rerun()


