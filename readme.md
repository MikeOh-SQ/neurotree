# 🎄 EEG 크리스마스 트리 분석기 (Neurotree)

이 프로젝트는 Muse 2 또는 유사 기기에서 수집된 EEG 데이터를 분석하여, 뇌파 리듬의 평균 점유율에 따라 **동적으로 변화하는 크리스마스 트리 이미지**를 생성하는 웹 애플리케이션입니다.

파일 업로드부터 FFT 기반 분석, 그리고 결과 시각화까지 모든 과정이 Python과 Streamlit으로 구현되었습니다.

## 🌟 주요 기능

- **5분 데이터 배치 분석:** 사용자가 EEG Raw 데이터(CSV 또는 Numpy)를 업로드하면, 시작부터 **5분 분량**의 데이터를 추출합니다.

- **TP9/TP10 재참조:** Muse 데이터의 AF7, AF8 채널에 TP9/TP10 평균 참조를 적용하여 노이즈를 제거합니다.

- **뇌파 점유율 분석:** **Delta, Theta, Alpha, Beta, Gamma** 각 밴드의 **상대적 점유율**을 계산합니다.

- **뇌파 기반 시각화:** 분석된 뇌파 점유율 값(예: Alpha 레벨)에 따라 트리의 밝기나 장식 상태가 동적으로 변화하는 크리스마스 트리 이미지를 표시합니다.

## 🚀 앱 실행 방법

이 앱은 Streamlit Community Cloud에 배포될 예정입니다. 로컬에서 먼저 테스트할 수 있습니다.

### 1. 필수 라이브러리 설치

프로젝트를 실행하려면 다음 라이브러리가 필요합니다. (`requirements.txt`에 기록된 내용)

```
pip install -r requirements.txt
```

### 2. 로컬 파일 구조

프로젝트는 다음 세 파일로 구성되어야 합니다.

```
neurotree/├── app.py              # Streamlit UI 및 분석 실행 파일
                  ├── eeg_analyzer.py     # 분석 핵심 로직 (재참조, FFT, PSD 계산)
                  └── requirements.txt    # 의존성 목록
```

### 3. 로컬에서 앱 실행

`app.py` 파일이 있는 디렉토리에서 다음 명령을 실행합니다.

```
streamlit run app.py
```

## 💻 분석 로직 (DSP) 상세

핵심 분석 로직은 `eeg_analyzer.py`에 정의되어 있으며, 다음 단계를 따릅니다.

1. **데이터 정렬:** 업로드된 데이터에서 처음 **76,800 샘플**(5분 ×256Hz)을 추출합니다.

2. **재참조 (Referencing):**
   
   AF7 및 AF8 채널은 TP9/TP10의 평균을 기준으로 재참조됩니다.
   
   AF7corr​=AF7raw​−2TP9+TP10​AF8corr​=AF8raw​−2TP9+TP10​
   
   

3. **PSD 계산:** Hanning Window를 적용한 후 FFT를 수행하여 **파워 스펙트럼 밀도 (PSD)**를 얻습니다.

4. **점유율 계산:** **Delta (1−4Hz)**부터 **Gamma (30−45Hz)**까지 각 밴드의 파워를 합산하고, 총 파워 대비 **상대적 점유율** (Relative Power)을 계산합니다.

## ☁️ Streamlit Community Cloud 배포 가이드

이 앱은 **Streamlit Community Cloud**를 통해 무료로 배포할 수 있습니다.

1. 이 저장소의 모든 파일이 **GitHub**에 커밋되어 있는지 확인합니다.

2. [Streamlit Community Cloud](https://share.streamlit.io/ "null")에 접속하여 로그인합니다.

3. **"New app"**을 클릭하고, 이 **GitHub 저장소**를 연결합니다.

4. **Main file path**에 `app.py`를 지정합니다.

5. **"Deploy!"** 버튼을 누르면 자동으로 앱이 빌드되어 실행됩니다.

**라이선스:** 이 프로젝트는 [MIT License](https://www.google.com/search?q=LICENSE "null")를 따릅니다

