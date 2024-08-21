import argparse
import os
import pandas as pd

def make_json_list(input_folder, output_file):
    columns = ['image_path', 'target']
    df = pd.DataFrame(columns=columns)

    class_names = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

    for i, class_name in enumerate(class_names):
        image_path = os.path.join(input_folder, class_name)
        image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        for image_file in image_files:
            file_path = class_name + '/' + image_file
            new_row = pd.DataFrame([{'image_path': file_path, 'target': i}])
            df = pd.concat([df, new_row], ignore_index = True)

    print(df)
    print("\n")
    output_file = output_file + '.json'
    print("Saving... "+output_file+"\n\n")
    df.to_json(output_file, orient='records', lines=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the input folder containing images.")
    parser.add_argument("-o", "--output_file", required=True, help="Path to the output file to save images list(json file).")
    args = parser.parse_args()

    make_json_list(args.input_folder, args.output_file)

if __name__ == "__main__":
    main()