import streamlit as st
import numpy as np
import pandas as pd
import io 
import requests # GitHub 파일 다운로드를 위해 requests 모듈 추가

# eeg_analyzer.py 파일에서 핵심 분석 함수와 상수를 가져옵니다.
# (분석 로직 파일에서 SKIP_SECONDS, BUFFER_LENGTH 등을 새로운 값으로 정의했다고 가정합니다.)
try:
    from eeg_analyzer import preprocess_and_reference, analyze_eeg_rhythms, ANALYSIS_CHANNEL_NAMES, SAMPLING_RATE
    # 분석 로직 변경: 15초 스킵, 60초 분석 (eeg_analyzer에서 계산된 상수 사용)
    # Streamlit에 표시하기 위해 분석 시간 정의
    ANALYSIS_DURATION = 60
    SKIP_DURATION = 15
    TOTAL_PROCESSED_TIME = ANALYSIS_DURATION + SKIP_DURATION
    
except ImportError:
    st.error("Error: eeg_analyzer.py 파일을 찾을 수 없습니다. 분석 로직 파일을 확인해주세요.")
    st.stop()


st.set_page_config(layout="wide", page_title="뇌파 트리 분석기")

st.title("🎄 뇌파 리듬 기반 크리스마스 트리 생성기")
st.markdown(f"EEG 원시 데이터를 업로드하여 **앞 15초를 제외한 1분 동안**의 뇌파 점유율을 분석하고 트리를 생성합니다.")

# =========================================================
# UI 섹션 1: 샘플 파일 다운로드 및 적용 버튼
# =========================================================
st.subheader("샘플 파일 사용")

SAMPLE_CSV_URL = "https://raw.githubusercontent.com/MikeOh-SQ/neurotree/blob/main/sample.csv" 

col_download, col_apply = st.columns([1, 1])

with col_download:
    st.markdown(f"""
        <a href="{SAMPLE_CSV_URL}" download="sample.csv">
            <button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer;">
                샘플 CSV 파일 다운로드 (GitHub)
            </button>
        </a>
        """, unsafe_allow_html=True)

with col_apply:
    # 샘플 파일 적용 버튼 (state를 사용하여 플로우 제어)
    if st.button("샘플 파일 적용 및 분석 시작"):
        st.session_state['use_sample'] = True
        st.experimental_rerun() # 버튼 클릭 시 페이지를 재실행하여 분석 시작

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
            file_to_analyze = io.StringIO(response.content.decode('utf-8'))
            st.info("GitHub 샘플 파일이 성공적으로 로드되었습니다.")
            # 샘플 사용 상태를 유지하면서 분석 로직으로 진입
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
        # 1. 데이터 로드 및 전처리 (MuseLab CSV 파일 파싱)
        with st.spinner('데이터 파싱 및 전처리 중입니다...'):
            
            # Pandas로 파일 로드 (헤더 포함)
            df = pd.read_csv(file_to_analyze)
            
            # 'osc_address'가 'eeg'인 행만 필터링하고, ACC 데이터와 NaN 값 제거
            eeg_df = df[df['osc_address'].str.contains('/eeg', na=False)].copy()
            
            # 파싱 함수 정의 (내부 함수로 유지)
            def parse_eeg_string(eeg_str):
                # 큰따옴표 제거 후 쉼표로 분리
                values = eeg_str.strip('"').split(',')
                # 처음 4개 값 (AF7, AF8, TP9, TP10)만 가져와서 float으로 변환
                try:
                    return [float(v) if v.lower() != 'nan' and v else np.nan for v in values[:4]]
                except Exception:
                    return [np.nan] * 4 # 파싱 오류 시 NaN 반환
            
            # 파싱 함수 적용
            parsed_eeg_data = eeg_df['osc_data'].apply(parse_eeg_string)
            
            # 파싱된 리스트를 Numpy 배열로 변환
            raw_data = np.array(parsed_eeg_data.tolist())
            
            # NaN 값은 0으로 채웁니다.
            raw_data = np.nan_to_num(raw_data, nan=0.0)

        
        # 2. 전처리 (15초 스킵 후 1분 추출 및 재참조)
        # raw_data는 이제 4채널 EEG 데이터만 포함
        
        # --- 분석 범위 변경 로직 반영 ---
        start_sample = int(SKIP_DURATION * SAMPLING_RATE)
        end_sample = start_sample + int(ANALYSIS_DURATION * SAMPLING_RATE)
        
        if raw_data.shape[0] < end_sample:
            required_samples = end_sample
            st.error(f"❌ 데이터 길이 부족: 총 {TOTAL_PROCESSED_TIME}초 ({required_samples} 샘플)의 데이터가 필요합니다. 현재 샘플 수: {raw_data.shape[0]}")
            st.stop()
            
        # 필요한 데이터 범위 슬라이싱
        data_to_analyze = raw_data[start_sample:end_sample, :]
        
        # 재참조 수행
        eeg_chunks_corrected = preprocess_and_reference(data_to_analyze)


        # 3. 뇌파 리듬 계산
        with st.spinner(f'뇌파 점유율을 분석 중입니다 ({SKIP_DURATION}초 스킵 후 1분 데이터)...'):
            rhythm_results = analyze_eeg_rhythms(eeg_chunks_corrected)
        
        st.success(f"분석 완료! (분석 범위: {SKIP_DURATION}초 ~ {TOTAL_PROCESSED_TIME}초 지점)")
        
        # 4. 뇌파 값 기반 트리 시각화 (기존 로직 유지)
        
        # 💡 예시: AF7 Alpha를 트리의 색상 밝기에 매핑
        af7_alpha = rhythm_results.get(ANALYSIS_CHANNEL_NAMES[0], {}).get('Alpha', 0)
        
        # 뇌파 값에 따른 설명 및 시각화 매핑
        st.header("트리 생성 결과")
        
        if af7_alpha > 15:
            tree_emoji = "🌟🎄🌟"
            tree_message = f"**높은 알파($\text{{Alpha}}$ {af7_alpha:.1f}%)**: 집중 상태가 양호합니다! 트리가 밝게 빛납니다."
        elif af7_alpha < 8:
            tree_emoji = "🕯️🎄🕯️"
            tree_message = f"**낮은 알파($\text{{Alpha}}$ {af7_alpha:.1f}%)**: 트리가 약간 흐릿하고 어둡습니다."
        else:
            tree_emoji = "✨🎄✨"
            tree_message = f"**평균 알파($\text{{Alpha}}$ {af7_alpha:.1f}%)**: 트리가 적당히 빛나고 있습니다."

        # 💡 Streamlit의 Markdown과 이모티콘으로 시각화
        st.markdown(f"## {tree_emoji}")
        st.markdown(tree_message)
        
        # 5. 상세 분석 결과 (표)
        st.subheader("상세 뇌파 점유율 ($\text{AF7, AF8}$)")
        
        # 분석 결과를 DataFrame으로 변환하여 표로 표시
        df_results = pd.DataFrame(rhythm_results).T 
        st.dataframe(df_results)
        
        # 분석 완료 후 샘플 사용 상태 초기화 (사용자 재시도를 위해)
        if st.session_state.get('use_sample', False):
             st.session_state['use_sample'] = False
             
    except ValueError as e:
        st.error(f"❌ 데이터 오류: {e}")
    except Exception as e:
        st.error(f"🔥 예기치 않은 오류 발생: {e}")

# 세션 상태 초기화 버튼 (오류 시 재시도)
if st.session_state.get('ready_to_analyze', False) or st.session_state.get('use_sample', False):
    if st.button("초기 상태로 돌아가기"):
        st.session_state['use_sample'] = False
        st.session_state['ready_to_analyze'] = False
        st.experimental_rerun()

