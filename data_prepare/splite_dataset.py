import argparse
import os
import random
import shutil

def split_images(source_folder, folder1, folder2, ratio=0.5):
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.gif')]  # 확장자에 따라 조정

    random.shuffle(image_files)

    split_index = int(len(image_files) * ratio)

    for i, image in enumerate(image_files):
        img_source_path = os.path.join(source_folder, image)
        print(img_source_path, end = '  ')

        if i < split_index:
            img_destination = os.path.join(folder1, image)
        else:
            img_destination = os.path.join(folder2, image)
        print(img_destination)
        shutil.copyfile(img_source_path, img_destination)

def split_cls_dataset(cls_source_folder, cls_folder1, cls_folder2, ratio=0.5):
    class_names = [d for d in os.listdir(cls_source_folder) if os.path.isdir(os.path.join(cls_source_folder, d))]

    for class_name in class_names:
        source_folder = os.path.join(cls_source_folder, class_name)
        folder1 = os.path.join(cls_folder1, class_name)
        folder2 = os.path.join(cls_folder2, class_name)

        split_images(source_folder, folder1, folder2, ratio)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_folder", "-s", dest="source_folder", type=str, required=True)
    parser.add_argument("--folder1", "-f1", dest="folder1", type=str, required=True)
    parser.add_argument("--folder2", "-f2", dest="folder2", type=str, required=True)
    args = parser.parse_args()

    source_path = args.source_folder
    folder1 = args.folder1
    folder2 = args.folder2

    split_cls_dataset(source_path, folder1, folder2)

if __name__ == "__main__":
    main()