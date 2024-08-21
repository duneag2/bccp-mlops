## Kafka

* 실행위치: `./api_serving`
  ```
  docker compose -p part7-kafka -f kafka-docker-compose.yaml up -d --build --force-recreate
  ```
  안되면 `bash`에서 exit한 다음 위 명령문 실행하고 다시 `bash`로 들어가서 `docker compose -p part7-kafka -f kafka-docker-compose.yaml up -d`하면 보통 된다.
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5493c236-3892-4fdb-9b24-104dbd6591f6)
  



  ```
  curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @source_connector.json
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5410612a-f58d-43d8-9e88-d0f951f3b647)
  curl문 안되는 경우, 도커 올린 직후라서 그럴수도 있음 좀 이따가 다시하면 될 수도 있음
  그래도 안되는 경우는 part7-kafka 도커 중 connect 도커가 exited 된것은 아닌지 확인해본다.
  만약 꺼졌다면 zookeeper → broker → schema → connect 순으로 켜보면 될 수 도 있음.
  `part7-kafka` 도커 삭제, `sudo docker system prune -a`로 캐시 삭제 하고 도커 다시 올려보면 될 수도 있다.




  ```
  curl -X GET http://localhost:8083/connectors
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/60ca7a33-e3a1-45c5-b318-cfbab5739da8)
  ```
  kafkacat -L -b localhost:9092
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/abac259e-6f66-4577-afe4-402791585a66)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5ddeb69c-c542-4bca-95d5-d5beda477ded)
  중간에 `topic "postgres-source-cargo"`에 해당하는 부분이 나오는지 확인해준다.




  ```
  kafkacat -b localhost:9092 -t postgres-source-cargo
  ```
  ![root@CHPCJ4_ _mnt_c_Users_USERSPC_capstone-mlops_kafka 2023-12-18 15-24-22](https://github.com/duneag2/capstone-mlops/assets/137387521/d8de0041-b5d1-4c3b-b977-3fa7596f8704)
  실시간 업데이트가 반영되고 있는 것을 확인할 수 있다.




  ```
  docker compose -p part7-target -f target-docker-compose.yaml up -d
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ab2eceb3-402c-4f7a-823c-17b41a586d9b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f84fc07e-d4e3-473f-a506-0d5e2ab1b522)
  table-creator는 동작 후 exited 되는 것이 정상




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
