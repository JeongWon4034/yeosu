# 해수유동(15분) · 조위관측 최신 API 파이프라인 설명

`eda_cch.ipynb` 하단에 추가된 셀에서 수행하는 작업과 설계 선택을 정리합니다.

## 1. 환경 변수 (`.env` / `.env.example`)

- 프로젝트 **루트**에 `.env`를 두고, `python-dotenv`로 로드합니다.
- 변수:
  - `DATA_GO_KR_SERVICE_KEY_CONT_OC15`: [해수유동 15분](https://www.data.go.kr/data/15112620/openapi.do) 인증키 (**Decoding** 키 권장)
  - `DATA_GO_KR_SERVICE_KEY_DT_RECENT`: [조위관측소 최신 관측데이터](https://www.data.go.kr/data/15155508/openapi.do) 인증키
  - `CONT_OC15_QUERY_PARAMS` (**필수**): Swagger「요청파라미터」표와 **동일한 이름·값**의 JSON 객체. `serviceKey`는 코드에서 붙이므로 JSON에 생략 가능. **API는 이 JSON으로 1회만 호출**합니다 (일일 한도 보호).
- `.env`는 `.gitignore`에 포함되어 있어 커밋되지 않습니다. 팀원은 `.env.example`을 복사해 키만 채우면 됩니다.
- **주의**: 인증키가 채팅·PR 등에 노출되면 포털에서 재발급하는 것이 안전합니다.

## 2. API 호출 방식

### 2.1 조위관측소 최신 관측데이터

- 엔드포인트: `https://apis.data.go.kr/1192136/dtRecent/GetDTRecentApiService`  
  (활용가이드 `조위관측소 관측데이터.pdf` 및 [데이터 상세](https://www.data.go.kr/data/15155508/openapi.do)와 동일)
- 요청 예: `obsCode=DT_0016`(여수), `reqDate`, `min`(출력 시간 간격), `type=xml` 등
- 응답 XML의 `<item>`을 파싱해 관측소 위치(`lat`/`lon`), 조위·수온 등 핵심 열만 추립니다.

### 2.2 해수유동 15분

- 엔드포인트: `https://apis.data.go.kr/1192000/apVhdService_ContOc15/getOpnContOc15`
- **요청은 1회만** 보냅니다. 쿼리 문자열은 `serviceKey` + `.env`의 `CONT_OC15_QUERY_PARAMS`(JSON) 병합입니다. Swagger 표에 있는 `type`, `pageNo`, `numOfRows` 등 필요한 항목은 **전부 JSON에 포함**하세요.
- **응답(Swagger `getOpnContOc15_response`)**: `resultCode` / `resultMsg`, 페이지 정보, `body` 안의 `item`(단건은 객체·복수는 배열) 또는 공공포털 관례인 `response.body.items.item` 형태를 모두 처리합니다. JSON이 아니면 기존처럼 XML `<item>` 파싱을 시도합니다.
- API가 실패하면(파라미터 오류 등) `RuntimeError`로 중단합니다. **HTTP 429** 시 더미 없이 안내 후 예외로 종료합니다.

## 3. Geopandas로 좌표 사용

- **조위 API**: 응답의 위도·경도로 관측소 점을 만들 수 있습니다(필요 시 후속 셀에서 `GeoDataFrame`화 가능).
- **섬 데이터**: `yeosu_islands_wgs84.csv`의 `lon`/`lat`로 `Point` 지오메트리 생성 (EPSG:4326).
- **격자 참고 CSV** (`geo/격자2단계_격자번호.csv`): `gid`, `격자번호(grid_no)`만 있고 **좌표 열은 없음**.  
  따라서 노트북에서는 **임시**로 남해안 접두(`GR2_G1`, `GR2_G2`, `GR2_F2`)로 후보 격자를 줄인 뒤, 섬 분포의 bbox 안에 격자번호를 **균등 격자 형태로 배치**해 대표 좌표를 둡니다.  
  **MOHID 등 물리모델과 연동할 때는 반드시 공식 격자 폴리곤/중심좌표로 교체**해야 합니다.

## 4. 핵심 변수 추출 (해수유동)

- XML 태그명이 바뀔 수 있어, 다음 후보들을 매핑해 `격자코드`, `분석일자`, `분석시간`, `유향`, `유속` 정규화:
  - 격자: `cntmGridNo`, `gridCd`, `gridNo`, …
  - 일시: `anlsYmd`, `anlsHm`, …
  - 유향·유속: `ocCrntDrc`/`ocCrntSpd`, `crdir`/`crsp`, …

## 5. 시계열 동기화 (15분 → 1시간)

- 해수유동: `분석일자`+`분석시간`으로 `datetime` 생성 후, 격자코드별 `resample('1h')`.
- **유속**: 시간 구간 내 **산술 평균**(`유속_1h_arith`)과, u/v 벡터 성분 평균으로 복원한 **벡터 평균 속도**(`유속_1h_mean`)를 함께 둡니다.
- **유향**: 단순 각도 산술평균은 부적절하므로, \(\sin\theta,\cos\theta\)로 변환해 평균 후 `atan2`로 `유향_1h_vec` 계산.
- 조위 쪽: `관측일시` 인덱스로 `1h` 평균 후, `hour` 단위로 유동 1h 결과와 `merge`(outer).

## 6. 격자–섬 매핑 (`yeosu_island_grid_mapping.csv`)

- 각 섬 점(5179 투영)과 임시 격자 중심점 간 거리로 **k=3 최근접 이웃**을 구합니다.
- 컬럼: `도서명`, `grid_no`, `gid`, `rank_nn`, `dist_m_est`, `weight_inv_dist2`(거리 제곱 역수 가중, 분모 안정화 +1).
- **한계**: 격자 좌표가 근사이므로, 파일은 “프로토타입 매핑”이며 **공식 격자 geometry 반영 후 재생성**이 필요합니다.

## 7. 결측·이상치

- **결측**: 격자코드별로 시간 인덱스 정렬 후 `interpolate(method='time', limit_direction='both')` (선형·시간 기준).
- **이상치**: `유향`, `유속`에 대해 절대값 Z-score > 5 를 **플래그만** 남기고, 행은 삭제하지 않습니다. 후속에 도메인 규칙으로 검토하면 됩니다.

## 8. 참고 문서

- 공공데이터포털 Gateway Swagger 가이드 (`gateway_swagger_guide.pdf`): `serviceKey`는 **Decoding** 키 입력, URL 인코딩은 클라이언트가 처리.
- 조위 API: `조위관측소 관측데이터.pdf` (요청/응답 필드명).
