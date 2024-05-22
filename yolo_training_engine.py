# Necessary imports
import os
import glob
import errno
import re
import zipfile
import urllib.request
import cv2
from PIL import Image
from ultralytics import YOLO
import random
import math
import sys
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets

random.seed(1)

# Updated 11/4/2024

# Utility functions
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# Global variables
DRIVE_ROOT_DIR = os.path.abspath(os.curdir)
YOLO_ROOT_DIR = os.path.join(DRIVE_ROOT_DIR, "engine")
PROJECTS_FOLDER_NAME = "projects"
BACKUP_DIR = os.path.join(DRIVE_ROOT_DIR, PROJECTS_FOLDER_NAME)
DNN_MODEL_DIR = os.path.join(YOLO_ROOT_DIR, "cfg/training")
DATA_DIR = os.path.join(YOLO_ROOT_DIR, "data")

# Add YOLO dir to load modules
sys.path.insert(0, YOLO_ROOT_DIR)

# Widget configurations
yolomodels = {
    "YOLOv8n": {"image_size": 640, "filename": "yolov8n.pt"},
    "YOLOv8s": {"image_size": 640, "filename": "yolov8s.pt"},
    "YOLOv8m": {"image_size": 640, "filename": "yolov8m.pt"},
    "YOLOv8l": {"image_size": 640, "filename": "yolov8l.pt"}
}

checkbox = widgets.Checkbox(description='Check to invert')

# Initialize project widgets
yolomodel = widgets.Dropdown(options=yolomodels.keys(), value="YOLOv8m", description='Model:')
projectname = widgets.Text(value='demo1', description='Project Name')
initialize_project = widgets.VBox([projectname, yolomodel])

# Get projects list
folder_list = glob.glob(os.path.join(BACKUP_DIR, "*"))
dir_names = [os.path.basename(path) for path in folder_list] or [""]
projectname_test = widgets.Dropdown(options=dir_names, value=dir_names[0], description="Project:")

# Autoconfigure for training
class ConfigModel:
    def __init__(self, project_name, image_size):
        if not re.match(r'^[A-Za-z0-9_]+$', project_name):
            raise TypeError("Project Name should only contain letters, numbers, and underscores. Spaces and special characters are not allowed.")

        self.project_name = project_name
        self.project_path = os.path.join(BACKUP_DIR, self.project_name)
        self.obj_data_path = os.path.join(self.project_path, "data/custom.yaml")
        self.dataset_path = os.path.join(self.project_path, "dataset")
        self.datasetpath = widgets.Text(value=self.dataset_path, description='Dataset Path', layout=widgets.Layout(width='600px'))
        self.epochs = widgets.Text(value='50', description='Epochs')
        self.imagesize = widgets.FloatSlider(value=image_size, min=320, max=1280, step=32, description='Image Size')
        self.configmodel = widgets.VBox([self.datasetpath, self.imagesize, self.epochs])

    def create_project_path(self):
        os.makedirs(self.project_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        print(f"The project can be found here: {self.project_path}")

    @staticmethod
    def yolo_weights_exist(file_path):
        if os.path.isfile(file_path):
            print(f"Weights file found: {file_path}")
        else:
            raise FileNotFoundError(f"Weights file NOT found: {file_path}")

    @staticmethod
    def download_weights(url, file_name):
        urllib.request.urlretrieve(url, file_name)

# CustomDataset class
class CustomDataset:
    def __init__(self, projectpath, projectname, model_name, dataset_path, image_size):
        self.project_path = projectpath
        self.model_name = model_name
        self.project_name = projectname
        self.obj_data_path = os.path.join(projectpath, "data/custom.yaml")
        self.images_folder_path = dataset_path
        self.backup_folder_path = projectpath
        self.image_size = str(int(image_size))
        self.test_dataset_paths = []
        self.test_percentage = 10
        self.validation_percentage = 10
        self.weights_paths = None
        self.weights_list = None

        os.makedirs(os.path.join(projectpath, "data"), exist_ok=True)
        os.makedirs(self.backup_folder_path, exist_ok=True)

        self.n_classes = 0
        self.n_labels = 0

    def get_starting_weights(self):
        pretrained_weights_name = yolomodels[self.model_name]["train_name"]
        pretrained_path = os.path.join(YOLO_ROOT_DIR, pretrained_weights_name)
        return pretrained_path

    def get_available_models(self, chosen_model):
        self.weights_paths = glob.glob(os.path.join(self.project_path, "**/*.pt"), recursive=True)
        return self.weights_paths

    def show_weights_list(self):
        self.weights_list = widgets.Dropdown(options=self.weights_paths, description='Weight path:', layout=widgets.Layout(width='850px'))

    def count_classes_number(self):
        print("Detecting classes number ...")
        txt_file_paths = glob.glob(os.path.join(self.images_folder_path, "**/*.txt"), recursive=True)
        self.n_labels = len(txt_file_paths)
        print(f"{self.n_labels} label files found")
        class_indexes = set()
        for file_path in txt_file_paths:
            with open(file_path, "r") as f_o:
                lines = f_o.readlines()
                for line in lines:
                    numbers = re.findall("[0-9.]+", line)
                    if numbers:
                        class_idx = int(numbers[0])
                        class_indexes.add(class_idx)

        self.n_classes = len(class_indexes)
        print(f"{self.n_classes} classes found")

        if max(class_indexes) > len(class_indexes) - 1:
            print("Normalizing and rewriting classes indexes to have consecutive numbers")
            new_indexes = {cls: idx for idx, cls in enumerate(sorted(class_indexes))}
            for file_path in txt_file_paths:
                with open(file_path, "r") as f_o:
                    lines = f_o.readlines()
                    text_converted = []
                    for line in lines:
                        numbers = re.findall("[0-9.]+", line)
                        if numbers:
                            class_idx = new_indexes.get(int(numbers[0]))
                            text = f"{class_idx} {numbers[1]} {numbers[2]} {numbers[3]} {numbers[4]}"
                            text_converted.append(text)
                with open(file_path, 'w') as fp:
                    for item in text_converted:
                        fp.write(f"{item}\n")

    def generate_custom_yaml(self):
        custom_yaml = f"train: {os.path.join(self.backup_folder_path, 'data/train.txt')}\n" \
                      f"val: {os.path.join(self.backup_folder_path, 'data/validation.txt')}\n" \
                      f"test: {os.path.join(self.backup_folder_path, 'data/test.txt')}\n" \
                      f"nc: {self.n_classes}\n" \
                      f"names: {[str(i) for i in range(self.n_classes)]}\n"
        with open(self.obj_data_path, "w") as f_o:
            f_o.write(custom_yaml)

    def generate_train_val_files(self):
        print("Generating Train/Validation/Test list")
        images_list = glob.glob(os.path.join(self.images_folder_path, "**/*.jpg"), recursive=True)
        random.shuffle(images_list)
        print(f"{len(images_list)} Images found")

        if not images_list:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"Images list not found in {self.images_folder_path}")

        test_number = len(images_list) // self.test_percentage
        validation_number = len(images_list) // self.validation_percentage

        with open(f"{self.backup_folder_path}/data/validation.txt", "w") as f_o:
            for i, path in enumerate(images_list):
                f_o.write(f"{path}\n")
                self.test_dataset_paths.append(path)
                if i == test_number:
                    break

        with open(f"{self.backup_folder_path}/data/test.txt", "w") as f_o:
            for i, path in enumerate(images_list):
                if i > validation_number:
                    f_o.write(f"{path}\n")
                    self.test_dataset_paths.append(path)
                if i == validation_number + test_number:
                    break

        with open(f"{self.backup_folder_path}/data/train.txt", "w") as f_o:
            for i, path in enumerate(images_list):
                if i > validation_number + test_number:
                    f_o.write(f"{path}\n")

    def extract_zip_file(self, path_to_zip_file):
        print("Extracting Images")
        with zipfile.ZipFile(os.path.join(self.images_folder_path, "images.zip"), 'r') as zip_ref:
            zip_ref.extractall(self.images_folder_path)

# CustomTraining class
class CustomTraining:
    def __init__(self, data_path, project_path, yolo_model, epochs, image_size):
        self.project_name = project_path
        self.obj_data_path = data_path
        self.weights_name = yolo_model.get_starting_weights()
        self.project_path = os.path.join(BACKUP_DIR, project_path)
        self.yolo_model = yolo_model
        self.epochs = epochs
        self.image_size = image_size

    def start_training(self):
        print("Preparing Training ...")
        yolo_model = YOLO(self.weights_name)
        yolo_model.train(
            data=self.obj_data_path,
            imgsz=self.image_size,
            epochs=self.epochs,
            batch=2,
            name=self.project_name,
            exist_ok=True
        )
        print("Training Completed")

# Display class
class Display:
    def __init__(self, data_path, project_path):
        self.project_path = project_path
        self.obj_data_path = data_path
        self.yolo_model = YOLO(self.obj_data_path)

    def detect_images(self):
        images = glob.glob(os.path.join(self.project_path, "dataset/images/*.jpg"))
        output_dir = os.path.join(self.project_path, "dataset/detected_images")
        os.makedirs(output_dir, exist_ok=True)
        for img in images:
            img_name = os.path.basename(img)
            result = self.yolo_model(img)
            result[0].plot()
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, result[0].plot())
        print("Detection Completed")
