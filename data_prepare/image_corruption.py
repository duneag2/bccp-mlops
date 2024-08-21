import argparse
from PIL import Image, ImageDraw
import os
import random

def random_boxes(image_path, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    num_boxes = random.randint(10,20)
    
    for _ in range(num_boxes):
        box_width = random.randint(int(img_width/10), int(img_width/5))
        box_height = random.randint(int(img_height/10), int(img_height/5))
        box_left = random.randint(0, img_width-box_width)
        box_top = random.randint(0, img_height-box_height)

        box_right, box_bottom = box_left + box_width, box_top + box_height
        draw.rectangle([box_left, box_top, box_right, box_bottom], fill="white")
    img = img.convert('RGB')
    img.save(output_path)

def mosaic(image_path, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    num_boxes = random.randint(20,30)

    for _ in range(num_boxes):
        mosaic_size = (10, 10)
        box_width = random.randint(int(img_width/10), int(img_width/5))
        box_height = random.randint(int(img_height/10), int(img_height/5))
        box_left = random.randint(0, img_width-box_width)
        box_top = random.randint(0, img_height-box_height)
        box_right, box_bottom = box_left + box_width, box_top + box_height
        box = (box_left, box_top, box_right, box_bottom)

        mosaic_box = img.crop(box)
        mosaic_box = mosaic_box.resize(mosaic_size, resample=Image.NEAREST)
        mosaic_box = mosaic_box.resize((box_width, box_height), resample=Image.NEAREST)
        img.paste(mosaic_box, box)
    img = img.convert('RGB')
    img.save(output_path)

def random_line(image_path, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    type = random.randint(0,1)

    if type == 0:
        r = random.randint(0, img_width)
        start_point = (r, 0)
        end_point = (img_width-r, img_height)
    else:
        r = random.randint(0, img_height)
        start_point = (0, r)
        end_point = (img_width, img_height-r)
    A = ((img_width ** 2) + (img_height ** 2))**(1/2)
    line_width = random.randint(int(A/10), int(A/5))
    draw.line([start_point, end_point], fill="black", width=line_width)
    img = img.convert('RGB')
    img.save(output_path)

def corruption(input_folder, output_folder, type):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        if type == "random_boxes":
            random_boxes(input_path, output_path)
        elif type == "mosaic":
            mosaic(input_path, output_path)
        elif type == "random_line":
            random_line(input_path, output_path)
        elif type == "mixed":
            t = random.randint(0,2)
            if t == 0:
                random_boxes(input_path, output_path)
            elif t == 1:
                mosaic(input_path, output_path)
            else:
                random_line(input_path, output_path)
        else:
            print("type = random_boxes OR mosaic OR random_line OR mixed")
            exit(0)


def main():
    parser = argparse.ArgumentParser(description="Corrupt images by adding random boxes OR mosaic OR random line OR mixed.")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the input folder containing images.")
    parser.add_argument("-o", "--output_folder", required=True, help="Path to the output folder to save corrupted images.")
    parser.add_argument("--type", required=False, help="random_boxes OR mosaic OR random_line OR mixed", default="mixed")
    args = parser.parse_args()

    corruption(args.input_folder, args.output_folder, args.type)    

if __name__ == "__main__":
    main()