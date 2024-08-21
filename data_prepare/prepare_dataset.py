import argparse
import os

from splite_dataset import split_cls_dataset
from image_corruption import corruption
from make_json_file import make_json_list

def prepare_cls_dataset(dataset_name):
    source_folder = f'../api_serving/{dataset_name}'
    monday_folder = f'../api_serving/{dataset_name}/monday'
    tuesday_original_folder = f'../api_serving/{dataset_name}/tuesday_original'
    tuesday_folder = f'../api_serving/{dataset_name}/tuesday'
    split_cls_dataset(source_folder, monday_folder, tuesday_original_folder, ratio = 0.5)

    class_names = [d for d in os.listdir(tuesday_original_folder) if os.path.isdir(os.path.join(tuesday_original_folder, d))]
    print(class_names)
    for class_name in class_names:
        input_folder = os.path.join(tuesday_original_folder, class_name)
        output_folder = os.path.join(tuesday_folder, class_name)
        corruption(input_folder, output_folder, type = 'mixed')
    
    make_json_list(monday_folder, f'../data_generate/{dataset_name}_monday')
    make_json_list(tuesday_folder, f'../data_generate/{dataset_name}_tuesday')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", required=True)
    args = parser.parse_args()

    prepare_cls_dataset(args.dataset_name)

if __name__ == "__main__":
    main()