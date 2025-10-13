import streamlit as st
import numpy as np
import pandas as pd
import io # io 모듈 추가

# eeg_analyzer.py 파일에서 핵심 분석 함수를 가져옵니다.
from eeg_analyzer import preprocess_and_reference, analyze_eeg_rhythms, BUFFER_LENGTH, ANALYSIS_CHANNEL_NAMES 

st.set_page_config(layout="wide", page_title="뇌파 트리 분석기")

st.title("🎄 뇌파 리듬 기반 크리스마스 트리 생성기")
st.markdown("EEG 원시 데이터를 업로드하여 5분 평균 뇌파 점유율을 분석하고 트리를 생성합니다.")

# 파일 업로드 위젯
uploaded_file = st.file_uploader(
    "EEG Raw 데이터 파일 (.csv 또는 .npy) 업로드)", 
    type=['npy', 'csv']
)

if uploaded_file is not None:
    try:
        # 1. 데이터 로드 및 전처리 (수정된 핵심 부분)
        with st.spinner('데이터를 로드 및 전처리 중입니다...'):
            
            if uploaded_file.name.endswith('.npy'):
                # Numpy 파일 로딩 (기존 방식 유지)
                raw_data = np.load(uploaded_file)
            else:
                # MuseLab CSV 파일 로딩 및 파싱 (수정 로직)
                
                # 1-1. Pandas로 파일 로드 (헤더 포함)
                # 'timestamp', 'osc_address', 'osc_type', 'osc_data' 열이 있는 것으로 가정
                df = pd.read_csv(uploaded_file)
                
                # 1-2. 'osc_address'가 'eeg'인 행만 필터링하고, ACC 데이터와 NaN 값 제거
                eeg_df = df[df['osc_address'].str.contains('/eeg', na=False)].copy()
                
                # 1-3. osc_data 열의 문자열을 파싱하여 4채널 EEG 데이터 추출
                # (예: "790.54944,794.9817,788.1319,790.54944,NaN,NaN" -> [790.5, 794.9, 788.1, 790.5])
                
                def parse_eeg_string(eeg_str):
                    # 큰따옴표 제거 후 쉼표로 분리
                    values = eeg_str.strip('"').split(',')
                    # 처음 4개 값 (AF7, AF8, TP9, TP10)만 가져와서 float으로 변환
                    try:
                        # NaN 문자열은 np.nan으로 대체하여 float 변환 시 오류 방지
                        return [float(v) if v.lower() != 'nan' and v else np.nan for v in values[:4]]
                    except Exception:
                        return [np.nan] * 4 # 파싱 오류 시 NaN 반환
                
                # 파싱 함수 적용
                parsed_eeg_data = eeg_df['osc_data'].apply(parse_eeg_string)
                
                # 파싱된 리스트를 Numpy 배열로 변환
                raw_data = np.array(parsed_eeg_data.tolist())
                
                # NaN 값은 0으로 채우거나 (간단한 처리) 이후 분석에서 제외하는 방식 필요.
                # 여기서는 간단하게 0으로 채웁니다. (더 정교한 처리는 제외)
                raw_data = np.nan_to_num(raw_data, nan=0.0)


            # 2. 전처리 (5분 추출 및 재참조)
            # -- 이 부분부터는 기존 eeg_analyzer.py의 로직이 사용됩니다. --
            
            # 데이터 채널 수가 4개 이상인지 최종 확인
            if raw_data.ndim != 2 or raw_data.shape[1] < 4:
                 st.error(f"데이터 형태가 잘못되었습니다. 추출된 유효 채널 수: {raw_data.shape[1]}")
                 st.stop()
                 
            eeg_chunks_corrected = preprocess_and_reference(raw_data)
        
        # 3. 뇌파 리듬 계산
        with st.spinner('뇌파 점유율을 분석 중입니다...'):
            rhythm_results = analyze_eeg_rhythms(eeg_chunks_corrected)
        
        st.success("분석 완료! 뇌파 점유율을 확인하세요.")
        
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
        
    except ValueError as e:
        st.error(f"❌ 데이터 오류: {e}")
    except Exception as e:
        st.error(f"🔥 예기치 않은 오류 발생: {e}")
