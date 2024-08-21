## Dashboard Stream

* 실행위치: `./dashboard_stream`

  ```
  docker compose -p part8-stream -f stream-docker-compose.yaml up -d --build --force-recreate
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/59821c6a-cf21-48d9-82ee-d02a4d4cae19)
  ```
  psql -h localhost -p 5433 -U targetuser -d targetdatabase
  ```
  - password: `targetpassword`
  ```
  SELECT * FROM cargo_prediction LIMIT 100;
  ```
  ```
  docker compose -p part8-dashboard -f grafana-docker-compose.yaml up -d
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/fa1e069e-ff34-43ed-af54-cefe35e051ad)




  [localhost:3000](http://localhost:3000/) 접속 (id: `dashboarduser`, pw: `dashboardpassword`)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0c7b7eca-32db-4746-9821-5fb098ab22ee)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/1aff2a2c-7e6d-4614-9eda-a2d8c28b67c6)
  좌측상단 三자 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f81b1bd1-dd05-4ede-9b83-e688b19d49ba)
  Dashboard
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/9ad328f2-e0d0-4be7-b5d1-8b3a6f7e2155)
  Create Dashboard
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/4667f7fd-4355-43a1-b015-85f08d2034fb)
  우측 상단 톱니바퀴(⚙️) 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f4de5851-4c85-4803-a355-3e6e2f42036a)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/656e9133-f633-46fc-9e31-ae2a61fd77c4)
  - Title: `Cargo classification`
  - Auto refresh: `1s,` 추가
  - `Refresh live dashboards` 토글 클릭하여 활성
  - 우측상단 `Save dashboard`




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/00ccb2a2-5e86-44ac-928d-3d49e3eb2266)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0f388425-c34a-4d30-ad65-2c02d6cf008f)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0541b598-6807-4a2f-b022-6620001a042b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5498e5d5-baa5-4784-a910-d088df7e178b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0fe8b0eb-8507-4d89-a323-40c7693eea9a)
  `Default` 토글 해제
  - Name : `Inference-database`
  - Host : `target-postgres-server:5432`
  - Database : `targetdatabase`
  - User : `targetuser`
  - Password : `targetpassword`
  - TLS/SSL Mode : `disable`
  - Version : `14.0`
  `Save & test` 버튼 클릭 → Database Connection OK 문구가 뜸




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/8f762f9f-e961-4e48-bbd9-e344b7637887)
  다시 좌측 상단 三자 → Dashboard → Cargo classification
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/cc4d1910-cb9e-45bb-a9de-bacea4db2b39)
  Add visualization
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/745b292a-7a23-4b71-a955-8fc817acd6c6)
  Inference-database 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/d9c881c5-6a66-4699-89ea-04e1ab2ba21d)
  - 우측 탭 : 패널의 이름, 차트의 종류 등을 설정합니다.
    - 기본 값으로 `Time series` 차트가 설정되어 있습니다.
    - 오른쪽 탭의 `Title` 에 패널의 이름을 붙여줍니다.
    - 이번 챕터에서는 `Cargo inference result` 로 설정하겠습니다.
  - 하단 탭 : 데이터 베이스에서 시각화할 테이블 및 열 정보를 설정합니다.
      - Data source : `Inference-database`
      - Table : `cargo_prediction`
      - Column : ➕ 버튼을 눌러 시각화 대상의 column 을 추가합니다.
      - `timestamp`
      - `target`
      - `Run query` 버튼 오른쪽의 `Code` 버튼을 클릭하고 `Limit` 부분을 지워줍니다.
  - `Run query` 버튼을 클릭합니다.
  우측상단의 Apply 하면 됨




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c13fc87f-dc26-4c38-ad85-76b534bdca42)
  우측상단 Last 6 hours라고 써있는 부분 클릭해서 From 부분을 now-30s로 수정, 우측상단 새로고침 버튼 눌러서 1s로 설정
  
  ![dashboard - Clipchamp로 제작](https://github.com/duneag2/capstone-mlops/assets/137387521/6502f6d4-b815-4c13-bb4c-abef076dd52d)


  잘 실행되는 것을 확인할 수 있다.
