## API Serving

* Execution Location: `./api_serving`
  Copy the RUN ID from [localhost:5001](http://localhost:5001/) and use it in the following code:
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0e93c086-7137-4c4f-ace6-06ea0daff99d)
  ```
  python3 download_model.py --model-name sk_model --run-id 70b965be026d4e5fb33cf6eccaa43b90
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/a12632a4-fdb9-45cc-bfa0-f5e6d214afa3)




  ```
  uvicorn app:app --reload
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/22838778-f479-4602-8751-6b3389a25b9c)
  Visit [http://localhost:8000/docs](http://localhost:8000/docs).
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/291397d8-4f5f-4f69-b1d9-9d7cdb04031e)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/30831132-ce9c-4087-afef-d36b868780f6)
  Click `Try it out`
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/b3fdbecd-02ba-4447-9e16-42709cdc77c2)
  Replace `"string"` with `"background/background_13.jpg"` and click Execute. (If you are using a different dataset than the Cargo dataset, adjust the input accordingly.)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/96fae359-ac59-4247-8ee4-d254b3a50470)
  You should see `0` (indicating background) in the target body.




  Let’s try again… Enter `"concrete/concrete29.jpg"` and click Execute.
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/d68c7f01-62d9-4342-8be9-2a2fbbe5da48)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ea44eb51-528b-4145-b58d-fc40d2e58005)
  You should see  `target 2` correctly identified.
  (For reference: background: 0 / brick: 1 / concrete: 2 / ground: 3 / wood: 4)




  ```
  docker compose up -d --build --force-recreate
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c5fbdee4-3914-42d5-b15a-db5d627bd4d0)
  ```
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"image_path": "background/background_13.jpg"}'
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/900969e6-513b-4c4b-92da-9ab93cef46f9)
  Again, you should see the target identified as 0 correctly.
