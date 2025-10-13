import streamlit as st
import numpy as np
import pandas as pd
# eeg_analyzer.py 파일에서 핵심 분석 함수를 가져옵니다.
from eeg_analyzer import preprocess_and_reference, analyze_eeg_rhythms, BUFFER_LENGTH, ANALYSIS_CHANNEL_NAMES 

st.set_page_config(layout="wide", page_title="뇌파 트리 분석기")

st.title("🎄 뇌파 리듬 기반 크리스마스 트리 생성기")
st.markdown("EEG 원시 데이터를 업로드하여 5분 평균 뇌파 점유율을 분석하고 트리를 생성합니다.")

# 파일 업로드 위젯
uploaded_file = st.file_uploader(
    "EEG Raw 데이터 파일 (.csv 또는 .npy) 업로드", 
    type=['npy', 'csv']
)

if uploaded_file is not None:
    try:
        # 1. 데이터 로드
        with st.spinner('데이터를 로드 및 전처리 중입니다...'):
            if uploaded_file.name.endswith('.npy'):
                raw_data = np.load(uploaded_file)
            else:
                raw_data = pd.read_csv(uploaded_file, header=None).values
        
            # 2. 전처리 (5분 추출 및 재참조)
            eeg_chunks_corrected = preprocess_and_reference(raw_data)
        
        # 3. 뇌파 리듬 계산
        with st.spinner('뇌파 점유율을 분석 중입니다...'):
            rhythm_results = analyze_eeg_rhythms(eeg_chunks_corrected)
        
        st.success("분석 완료! 뇌파 점유율을 확인하세요.")
        
        # 4. 뇌파 값 기반 트리 시각화 (핵심 로직)
        
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

        # 💡 HTML/CSS 없이 Streamlit의 Markdown과 이모티콘으로 시각화
        st.markdown(f"## {tree_emoji}")
        st.markdown(tree_message)
        
        # 5. 상세 분석 결과 (표)
        st.subheader("상세 뇌파 점유율 ($\text{AF7, AF8}$)")
        
        # 분석 결과를 DataFrame으로 변환하여 표로 표시
        df_results = pd.DataFrame(rhythm_results).T # Transpose (전치)하여 채널을 열로
        st.dataframe(df_results)
        
    except ValueError as e:
        st.error(f"❌ 데이터 오류: {e}")
    except Exception as e:
        st.error(f"🔥 예기치 않은 오류 발생: {e}")