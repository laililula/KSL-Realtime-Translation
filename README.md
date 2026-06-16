\# KSL Realtime Keypoint Inference



한국수어(KSL) WORD keypoint 기반 실시간 단어 인식 추론 코드입니다.



\## Files



\- `realtime\_ksl\_keypoint\_infer.py`: 실시간 웹캠 추론 코드

\- `models/ksl\_keypoint\_tcn\_best.pt`: validation 기준 best 모델

\- `models/ksl\_keypoint\_tcn\_final.pt`: final fit 모델



모델 파일은 GitHub Release에서 다운로드해야 합니다.



\## Install



```powershell

py -3 -m venv .venv

.\\.venv\\Scripts\\activate

pip install -r requirements.txt

