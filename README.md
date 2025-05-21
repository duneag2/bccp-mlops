# BCCP: An MLOps Framework for Self-cleansing Real-Time Data Noise via Bayesian Cut-off-based Closest Pair Sampling
### Official PyTorch Implementation

Early Korean Version available at: https://github.com/duneag2/capstone-mlops

This is the implementation of the approach described in the paper:

> S. Lee, N. Jeong, J. Je, and S.-Y. Lee, "BCCP: An MLOps Framework for Self-cleansing Real-Time Data Noise via Bayesian Cut-off-based Closest Pair Sampling", 2024 IEEE International Conference on AI x Data & Knowledge Engineering [Link](https://ieeexplore.ieee.org/document/10990103)

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

If you want to use only the Monday dataset, input `monday` for `target_day`. If you want to use both the Monday and Tuesday datasets, input `tuesday` for `target_day`.

### Training

Create a container for model training in the `model_registry/` folder.

```
docker compose up -d --build --force-recreate
```

 ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f51d472d-3748-406c-b65c-664c7a8cf310)




  Access [localhost:5001](http://localhost:5001/) 
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ac82e811-0ed8-4b86-b287-537e045b9e0f)


  Access [localhost:9001](http://localhost:9001/) (username: `minio`, password: `miniostorage`).
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/e4d6ad20-c912-4b6c-a9d9-b6b70dc8e0e7)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/b6bdf68a-5243-48de-a331-336661b4e4c1)
  When you first access it, there will be no buckets. Go to the **Create a bucket** section, set the **Bucket Name** to **mlflow**, and create the bucket. (There is no need to click on the toggles below.)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/43c2f4c9-9cce-4087-891a-bcbb483a1106)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/7cac725f-50f1-49cb-9946-1ef7ce19b486)





If you want to train our model, please execute the following command:

```
python3 save_model_to_registry.py -d dataset_name -t target_day -l label --user_accuracy user_accuracy -b Y/N -s sampling_type --monday_num number_of_images --tuesday_num number_of_images -r ratio --model_name model_name
```
- `-d` or `--dataset`: Specifies the dataset to use, e.g., `cargo`.
- `-t` or `--target`: Specifies the target day (`monday` or `tuesday`). If you want to use only the Monday dataset, input `monday` for the `target_day`. If you want to use both the Monday and Tuesday datasets, input `tuesday` for the `target_day`. Default: `monday`.
- `-l` or `--label`: Specifies the label to use for training. The `ground_truth` option uses the correct labels for the images during training. The `user_feedback` option assumes a scenario where the labels are generated based on user feedback on the images, thus assuming lower accuracy of the labels. The accuracy of the `user_feedback` labels can be set using the `user_accuracy` option. Default: `ground_truth`.
- `--user_accuracy`: Sets the accuracy when using the `user_feedback` labels. Default: `0.7`.
- `-b` or `--bayesian_cut_off`: Determines whether to use Bayesian cut-off for the dataset before model training. Enter `Y` to use it or `N` to not use it. Default: `N`.
- `-s` or `--sampling_type`: Determines the method for sampling images to be re-trained from the Reuse Buffer. If set to `none`, the Reuse Buffer will not be used. If set to `random`, random sampling will be applied. If set to `l1_norm`, the L1-norm-based CP sampling method will be used. If set to `l2_norm`, the L2-norm-based CP sampling method will be used. If set to `cosine_similarity`, the Cosine Similarity-based CP sampling method will be used. Default: `none`.
- `--monday_num`: Specifies the number of images in the Monday dataset.
- `--tuesday_num`: Specifies the number of images in the Tuesday dataset.
- `-r` or `--ratio`: When the sampling type is not `none`, this sets the proportion of samples to extract from the Reuse Buffer.
- `--model-name`: Specifies the model name. Default: `cls_model`.

### Comparative Experiments
Additionally, we conducted comparative experiments with two notable papers in data cleansing: Cleanlab and Ye et al. You can run the two comparative experiments using the following commands.

**Cleanlab** (C. G. Northcutt, et al., ”Confident Learning: : Estimating Uncertainty in Dataset Labels,” Journal of Artificial Intelligence Research, pp. 1373-1411, 2021.)
```
python3 save_model_to_registry_cleanlab.py -d dataset_name -l label --user_accuracy user_accuracy --monday_num number_of_images --tuesday_num number_of_images --model_name model_name
```
**ANL_CE** (X. Ye, et al., ”Active Negative Loss Functions for Learning with Noisy Labels,” 37th International Conference on Neural Information Processing Systems, pp. 6917-6940, 2023.)
```
python3 save_model_to_registry_ANL_CE.py -d dataset_name -l label --user_accuracy user_accuracy --monday_num number_of_images --tuesday_num number_of_images --model_name model_name
```

### Real-Time Data Pipeline and Visualization Setup
We have documented the process of implementing a real-time visualization dashboard using Grafana to display original data and predicted values. This system involves several key steps, including the creation of a data subscriber to retrieve data from a Kafka topic, transmitting this data to an API server, receiving the predicted values, and sending them to the target database. The entire setup allows for real-time monitoring and visualization of both original and predicted data using Grafana.
The implementation process is outlined in detail across the following three documents, presented in sequential order:

1. **Document 1: [API Serving](api_serving/README.md)**
 - This document describes how to implement a REST API using FastAPI. The API receives input data and returns the predicted values generated by the model.

2. **Document 2: [Kafka](kafka/README.md)**
 - In this document, you'll learn how to build a real-time data pipeline using Kafka. This setup is essential for stream serving, enabling the real-time transmission of data.
   
3. **Document 3: [Grafana Dashboard Configuration](dashboard_stream/README.md)**
 - This final document covers how to configure a Grafana dashboard to monitor data in real-time. It includes instructions on visualizing both the original data and the model's predictions.
