# BCCP: An MLOps Framework for Self-cleansing Real-Time Data Noise via Bayesian Cut-off-based Closest Pair Sampling
### Official PyTorch Implementation

Early Korean Version available at: https://github.com/duneag2/capstone-mlops

This is the implementation of the approach described in the paper:

> 논문 저자, 논문제목. 학회이름, 연도.

논문에 쓴 이미지 하나 넣는다든가..

### Results on BCCP
표 입력하면 좋을 거 같음

## Quick start
To get started as quickly as possible, follow the instructions in this section. This will allow you to prepare the image classification dataset and train the model from scratch.

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3+ distribution
- PyTorch >= 2.1.2

### Dataset setup
Our classification experiments utilize three distinct datasets. To emphasize the practicality of MLOps, we select datasets related to factory management, waste management, and agricultural business. The datasets we used can be downloaded from the link below. You can also test our model using any image classification dataset.
- Cargo Dataset (https://www.kaggle.com/datasets/morph1max/definition-of-cargo-transportation)
- Bag Dataset (https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images)
- Sugarcane Leaf Disease Dataset (https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset)

Once the dataset is prepared, place the image file folders into the `bccp_mlops/api_serving` folder. Make sure the structure looks like the one below.
```
api_serving
├── dataset
│   ├── class1  // contains many image files
│   ├── class2  // contains many image files
│   └── class3  // contains many image files
├── Makefile
├── app.py
├── docker-compose.yaml
├── download_model.py
└── schemas.py
```

To prepare the Monday and Tuesday Datasets and split the images into train and test datasets, run the following command from the `dataset_prepare/` directory. (For a description of the Monday and Tuesday datasets, please refer to our paper.
```
python3 prepare_dataset.py -d dataset_name
```

Once the execution is complete, a JSON file will be generated in the `data_generate` folder.

### Data Generation using Docker

This step requires Docker Desktop and PostgreSQL to be installed.

Run the following command in the `data_generate/` directory to create the Data Generator container.

```
DATASET=dataset_name TARGET_DAY=target_day docker compose up -d --build --force-recreate
```

Monday Dataset만 사용하고 싶은 경우 target_day에 monday를 입력하고, Monday와 Tuesday dataset을 모두 사용하고 싶은 경우 target_day에 tuesday를 입력하세요.

### Training

model_registry/ 폴더에서 모델 학습을 위한 컨테이너를 생성합니다.

```
docker compose up -d --build --force-recreate
```

 ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f51d472d-3748-406c-b65c-664c7a8cf310)




  [localhost:5001](http://localhost:5001/) 접속 (딱히 아무것도 안 뜬다면 정상)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ac82e811-0ed8-4b86-b287-537e045b9e0f)


  [localhost:9001](http://localhost:9001/) 접속 (username: `minio`, password: `miniostorage`)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/e4d6ad20-c912-4b6c-a9d9-b6b70dc8e0e7)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/b6bdf68a-5243-48de-a331-336661b4e4c1)
  처음 들어가면 bucket이 없을 수 있다. -> Create a bucket -> 이름 `mlflow`로 설정 후 생성 (아래 토글들은 클릭하지 않으시면 됨)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/43c2f4c9-9cce-4087-891a-bcbb483a1106)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/7cac725f-50f1-49cb-9946-1ef7ce19b486)





우리의 모델을 학습시키길 원한다면 아래의 명령어를 실행하십시오.

```
python3 save_model_to_registry.py -d dataset_name -t target_day -l label --user_accuracy user_accuracy -b Y/N -s sampling_type --monday_num number_of_images --tuesday_num number_of_images -r ratio --model_name model_name
```

- `-d` or `--dataset`: specifies the dataset to use , e.g. `cargo`.
- `-t` or `--target`: specifies the target day(`monday` or `tuesday`). Monday Dataset만 사용하고 싶은 경우 target_day에 monday를 입력하고, Monday와 Tuesday dataset을 모두 사용하고 싶은 경우 target_day에 tuesday를 입력하세요. Default: `monday`
- `-l` or `--label`: specifies the 학습에 사용할 라벨. ground_truth 옵션은 학습시에 이미지에 대한 정답 라벨을 사용한다. user_feedback 옵션은 사용자가 이미지를 보고 feedback하여 라벨을 생성하기에 라벨의 정확도가 낮은 상황을 가정한 옵션이다. user_feedback 라벨의 정확도는 user_accuracy 옵션을 이용해 설정할 수 있다.  Default: `ground_truth`.
- `--user_accuracy`: user_feedback 라벨을 사용할 경우의 정확도를 설정한다. Default: `0.7`.
- `-b` or `--bayasian_cut_off`: 모델 학습 전 학습할 데이터에 대해 bayesian cut-off를 사용할 지 여부를 결정한다. Y 를 입력 시 사용할 수 있고, N을 입력시 사용하지 않을 수 있다. Default: N.
- `-s` or `--sampling_type`: Reuse Buffer에서 재학습할 이미지를 샘플링하는 방법을 결정한다. none 사용시 Reuse Buffer를 사용하지 않는다. random 사용시 randon sampling이 적용된다. l1-norm 사용시 L1-norm을 적용한 CP sampling 기법이 사용된다.  l2-norm 사용시 L2-norm을 적용한 CP sampling 기법이 사용된다.  cosine_similarity 사용시 Cosine-Similarity을 적용한 CP sampling 기법이 사용된다. Default: `none`.
- `--monday_num`: monday dataset의 이미지 개수를 입력한다.
- `--tuesday_num`: tuesday dataset의 이미지 개수를 입력한다.
- `-r` or `--ratio`: Sampling type이 none이 아닌 경우, reuse Buffer에서 추출할 샘플의 비율을 결정한다.
- `--model-name`: specifies the model name. Default: `cls_model`.
