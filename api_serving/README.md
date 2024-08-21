## API Serving

* 실행위치: `./api_serving`
  [localhost:5001](http://localhost:5001/)에 있는 RUN ID 복붙해서 넣고 아래 코드 실행
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0e93c086-7137-4c4f-ace6-06ea0daff99d)
  ```
  python3 download_model.py --model-name sk_model --run-id 70b965be026d4e5fb33cf6eccaa43b90
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/a12632a4-fdb9-45cc-bfa0-f5e6d214afa3)




  ```
  uvicorn app:app --reload
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/22838778-f479-4602-8751-6b3389a25b9c)
  [http://localhost:8000/docs](http://localhost:8000/docs) 접속
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/291397d8-4f5f-4f69-b1d9-9d7cdb04031e)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/30831132-ce9c-4087-afef-d36b868780f6)
  `Try it out` 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/b3fdbecd-02ba-4447-9e16-42709cdc77c2)
  `"string"` 대신 `"background/background_13.jpg"`를 입력 후 Execute
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/96fae359-ac59-4247-8ee4-d254b3a50470)
  `target body`에 `0`(background를 의미)이 잘 나옴




  한번 더 해보자… `"concrete/concrete29.jpg"`를 입력하고 Execute
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/d68c7f01-62d9-4342-8be9-2a2fbbe5da48)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ea44eb51-528b-4145-b58d-fc40d2e58005)
  `target 2`로 잘 나옴
  (참고로 background: 0 / brick: 1 / concrete: 2 / ground: 3 / wood: 4)




  ```
  docker compose up -d --build --force-recreate
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c5fbdee4-3914-42d5-b15a-db5d627bd4d0)
  ```
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"image_path": "background/background_13.jpg"}'
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/900969e6-513b-4c4b-92da-9ab93cef46f9)
  마찬가지로 target이 0으로 잘 나옴
