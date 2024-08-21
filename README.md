# BCCP: An MLOps Framework for Self-cleansing Real-Time Data Noise via Bayesian Cut-off-based Closest Pair Sampling
### Official PyTorch Implementation

Early Korean Version available at: https://github.com/duneag2/capstone-mlops

This is the implementation of the approach described in the paper:

> 논문 저자, 논문제목. 학회이름, 연도.

### Quick start

--

To get started as quickly as possible, follow the instructions in this section. This will allow you to prepare the image classification dataset and train the model from scratch.

Dependencies

Make sure you have the following dependencies installed before proceeding:

- Python 3+ distribution
- PyTorch >= 2.1.2

Dataset setup

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

```bash
python3 prepare_dataset.py -d dataset_name
```

Once the execution is complete, a JSON file will be generated in the `data_generate` folder.
