## 실행 방법

프로젝트 폴더에서 다음 명령을 실행하면 GUI가 열립니다.

```powershell
.venv\Scripts\python.exe main.py
```

파일 작업 도구를 실행할 때는 GUI 승인 창이 표시됩니다. 모델이 로드된 뒤 입력창에 질문을 입력하고 `전송`을 누르세요.

기존 콘솔 모드가 필요하면 다음처럼 실행합니다.

```powershell
.venv\Scripts\python.exe main.py --cli
```

**qwen2.5-coder:3b** 모델의 한국어 능력을 향상시키려 **nlpai-lab/kullm-v2** 데이터셋으로 파인튜닝을 시도해보았다.
coder 모델 특성상 대화 용도로 파인튜닝을 하니 성능이 저하되는 것을 느껴 **llama3.2:3b** 모델로 변경

파싱을 엉터리로 하고 있었다!!
애초에 태그 자체를 생성 못하고 있었고 <tool>에 한정해서만 보정을 하여서 정상 작동한 것