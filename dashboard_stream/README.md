## Dashboard Stream

* Execution Location: `./dashboard_stream`

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




  Access [localhost:3000](http://localhost:3000/) (id: `dashboarduser`, pw: `dashboardpassword`)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0c7b7eca-32db-4746-9821-5fb098ab22ee)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/1aff2a2c-7e6d-4614-9eda-a2d8c28b67c6)
  Click on the **three lines icon** at the top left.
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f81b1bd1-dd05-4ede-9b83-e688b19d49ba)
  **Dashboard**
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/9ad328f2-e0d0-4be7-b5d1-8b3a6f7e2155)
  **Create Dashboard**
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/4667f7fd-4355-43a1-b015-85f08d2034fb)
  Click the gear icon (⚙️) at the top right.
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f4de5851-4c85-4803-a355-3e6e2f42036a)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/656e9133-f633-46fc-9e31-ae2a61fd77c4)
  - Title: `Cargo classification`
  - Auto refresh: Add `1s,`
  - Click the toggle `Refresh live dashboards` to enable
  - Click `Save dashboard` at the top right.




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/00ccb2a2-5e86-44ac-928d-3d49e3eb2266)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0f388425-c34a-4d30-ad65-2c02d6cf008f)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0541b598-6807-4a2f-b022-6620001a042b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5498e5d5-baa5-4784-a910-d088df7e178b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0fe8b0eb-8507-4d89-a323-40c7693eea9a)
  Disable the `Default` toggle.
  - Name : `Inference-database`
  - Host : `target-postgres-server:5432`
  - Database : `targetdatabase`
  - User : `targetuser`
  - Password : `targetpassword`
  - TLS/SSL Mode : `disable`
  - Version : `14.0`
  Click the `Save & test` button → "Database Connection OK" message should appear.




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/8f762f9f-e961-4e48-bbd9-e344b7637887)
  Click the **three lines icon** at the top left again → **Dashboard** → **Cargo classification**
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/cc4d1910-cb9e-45bb-a9de-bacea4db2b39)
  **Add visualization**
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/745b292a-7a23-4b71-a955-8fc817acd6c6)
  **Inference-database**
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/d9c881c5-6a66-4699-89ea-04e1ab2ba21d)
  - Right tab: Set the panel's name, chart type, etc.
    - By default, the chart is set to `Time series`.
    - Set the panel name in the `Title` section on the right tab.
    - Name it `Cargo inference result`.
  - Bottom tab: Set the table and column information to be visualized from the database.
      - Data source : `Inference-database`
      - Table : `cargo_prediction`
      - Column : Click the ➕ button to add columns for visualization.
      - `timestamp`
      - `target`
      - Click the `Code` button next to the `Run query` button, and remove the `Limit` part.
  - Click the `Run query` button.
  Then click `Apply` at the top right.




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c13fc87f-dc26-4c38-ad85-76b534bdca42)
  Click on the part that says "Last 6 hours" at the top right, change the "From" section to `now-30s`, and set the refresh button at the top right to `1s`.
  
  ![dashboard - Clipchamp로 제작](https://github.com/duneag2/capstone-mlops/assets/137387521/6502f6d4-b815-4c13-bb4c-abef076dd52d)

