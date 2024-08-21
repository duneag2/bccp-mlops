## Kafka

* Execution Location: `./api_serving`
  ```
  docker compose -p part7-kafka -f kafka-docker-compose.yaml up -d --build --force-recreate
  ```
  If it doesn’t work, exit the `bash`, run the above command, and then re-enter `bash` and run `docker compose -p part7-kafka -f kafka-docker-compose.yaml up -d`. This usually resolves the issue.
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5493c236-3892-4fdb-9b24-104dbd6591f6)
  



  ```
  curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @source_connector.json
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5410612a-f58d-43d8-9e88-d0f951f3b647)
  If the `curl` command doesn’t work, it might be because the Docker container was just started. Try again after a short wait.
  If it still doesn’t work, check if the `connect` container in the `part7-kafka` Docker setup has exited.
  If it has exited, try restarting the containers in this order: `zookeeper → broker → schema → connect`.
  If the issue persists, delete the `part7-kafka` Docker setup, clear the cache with `sudo docker system prune -a`, and try starting the Docker setup again.




  ```
  curl -X GET http://localhost:8083/connectors
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/60ca7a33-e3a1-45c5-b318-cfbab5739da8)
  ```
  kafkacat -L -b localhost:9092
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/abac259e-6f66-4577-afe4-402791585a66)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5ddeb69c-c542-4bca-95d5-d5beda477ded)
  Verify if the `topic "postgres-source-cargo"` is present.




  ```
  kafkacat -b localhost:9092 -t postgres-source-cargo
  ```
  ![root@CHPCJ4_ _mnt_c_Users_USERSPC_capstone-mlops_kafka 2023-12-18 15-24-22](https://github.com/duneag2/capstone-mlops/assets/137387521/d8de0041-b5d1-4c3b-b977-3fa7596f8704)
  You should be able to confirm that real-time updates are being reflected.




  ```
  docker compose -p part7-target -f target-docker-compose.yaml up -d
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ab2eceb3-402c-4f7a-823c-17b41a586d9b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f84fc07e-d4e3-473f-a506-0d5e2ab1b522)
  It is normal for `table-creator` to exit after it runs.




  ```
  curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @sink_connector.json
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/793a09b6-cd1f-4f41-9fd0-473bd4812f97)
  ```
  curl -X GET http://localhost:8083/connectors
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/204c92c9-c200-4fdb-b9e8-d229fdac6e45)
  ```
  curl -X GET http://localhost:8083/connectors/postgres-sink-connector
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c9d9d7a9-a2a3-40f6-807e-8f5a3b34c017)
  ```
  psql -h localhost -p 5433 -U targetuser -d targetdatabase
  ```
  - password: `targetpassword`
  ```
  SELECT * FROM cargo LIMIT 100;
  ```
