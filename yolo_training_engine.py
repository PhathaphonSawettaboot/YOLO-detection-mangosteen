# some handy functions to use along widgets
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
random.seed(1)

# Updated 11/4/2024

import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

DRIVE_ROOT_DIR = os.path.abspath(os.curdir)
YOLO_ROOT_DIR = os.path.join(DRIVE_ROOT_DIR, "engine")
PROJECTS_FOLDER_NAME = "projects"
BACKUP_DIR = os.path.join(DRIVE_ROOT_DIR, PROJECTS_FOLDER_NAME)
DNN_MODEL_DIR = os.path.join(YOLO_ROOT_DIR, "cfg/training")
DATA_DIR = os.path.join(YOLO_ROOT_DIR, "data")


import sys
# Add YOLO dir to load modules
sys.path.insert(0, os.path.join(YOLO_ROOT_DIR))


from IPython.display import display, Markdown, clear_output
# widget packages
import ipywidgets as widgets# defining some widgets

yolomodels = {"YOLOv8n": {"image_size": 640, "filename": "yolov8n.pt"},
            "YOLOv8s": {"image_size": 640, "filename": "yolov8s.pt"},
            "YOLOv8m": {"image_size": 640, "filename": "yolov8m.pt"},
            "YOLOv8l": {"image_size": 640, "filename": "yolov8l.pt"}
            }

checkbox = widgets.Checkbox(
           description='Check to invert',)

# Initialize project
yolomodel = widgets.Dropdown(
       options=yolomodels.keys(),
       value="YOLOv8m",
       description='Model:')
projectname = widgets.Text(value='demo1',
       description='Project Name', )
initialize_project = widgets.VBox([projectname, yolomodel])

# Get projects list
folder_list = glob.glob(os.path.join(BACKUP_DIR, "*"))
dir_names = [os.path.basename(path) for path in folder_list]
dir_names = [""] if not dir_names else dir_names # avoid empty array
projectname_test = widgets.Dropdown(options=dir_names,
                               value=dir_names[0],
                               description="Project:")

# Autoconfigure for training
class ConfigModel:
    def __init__(self, project_name, image_size):
        #Validate project name
        if not re.match(r'^[A-Za-z0-9_]+$', project_name):
            raise TypeError("Project Name should only contains letters, numbers and underscores. Spaces and special characters are not allowed.")

        self.project_name = project_name
        self.project_path = os.path.join(BACKUP_DIR, self.project_name)
        self.obj_data_path = os.path.join(self.project_path, "data\custom.yaml")
        self.dataset_path = os.path.join(self.project_path, "dataset")
        self.datasetpath = widgets.Text(value='{}'.format(self.dataset_path),
               description='Dataset Path', layout=widgets.Layout(width='600px'))

        # self.batchsize = widgets.Dropdown(
        #     options=['16', '32'],
        #     value='16',
        #     description='Batch Size:')

        self.epochs = widgets.Text(value='50',
                                   description='Epochs')


        self.imagesize = widgets.FloatSlider(
            value=image_size,
            min=320,
            max=1280,
            step=32,
            description='Image Size')

        # Find interrupted training
        #self.continuetraining_check = widgets.Checkbox(description='Find and Continue Interrupted training', value=False,
        #                                          layout=widgets.Layout(width='600px'))

        self.configmodel = widgets.VBox([self.datasetpath, self.imagesize, self.epochs])


    def create_project_path(self):
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
            print("A new project has been created an can be found on this path: {}".format(self.project_path))
        else:
            print("The project already exists and can be found here: {}".format(self.project_path))
        dataset_path = os.path.join(self.project_path, "dataset")
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    def yolo_weights_exist(file_path):
        if os.path.isfile(file_path):
            print("Weights file found: {}".format(file_path))
        else:
            raise FileNotFoundError("Weights file NOT found: {}".format(file_path))

    def download_weights(url, file_name):
        urllib.request.urlretrieve(str(url), str(file_name))


# class TrainingSettings:
#     def __init__(self):
#         self.list_gpus = self.get_gpu_count()
#
#         self.gpu_settings = widgets.Dropdown(
#             options=self.list_gpus.values(),
#             value=list(self.list_gpus.values())[0],
#             description='Select GPU:')
#
#         self.training_settings = widgets.VBox([self.gpu_settings])
#
#     def get_gpu_count(self):
#         print("Looking for Nvidia CUDA GPUs")
#         num_gpus = torch.cuda.device_count()
#         list_gpus = {}
#         print("{} gpu found".format(num_gpus))
#         for i in range(num_gpus):
#             device_name = torch.cuda.get_device_name(i)
#             gpu_memory = torch.cuda.get_device_properties(i).total_memory
#             print("GPU {}: {} | Memory: {}".format(i, device_name, convert_size(gpu_memory)))
#             list_gpus[i] = "{} {}".format(i, device_name)
#         # Raise warning if no GPUs were found
#         if not list_gpus:
#             list_gpus = {"CPU": "cpu"}
#             print("WARNING: No Nvidia CUDA GPUs were found.")
#         return list_gpus
#
#     def get_gpu_index(self, gpu_name):
#         gpu_idx = [k for k, v in self.list_gpus.items() if v == gpu_name]
#         if gpu_idx:
#             gpu_idx = str(gpu_idx[0])
#         return gpu_idx


class CustomDataset:
    def __init__(self, projectpath, projectname, model_name, dataset_path, image_size):
        # Files location
        #self.new_custom_cfg_path = os.path.join(projectpath, "{}_{}.cfg".format(projectname, model_name))
        # Create data folder
        self.project_path = projectpath
        self.model_name = model_name
        self.project_name = projectname
        self.obj_data_path = os.path.join(projectpath, "data\custom.yaml")
        #self.obj_names_path = os.path.join(projectpath, "data\obj.names")
        self.images_folder_path = "{}".format(dataset_path)
        self.backup_folder_path = projectpath
        self.image_size = str(int(image_size))

        # Dataset paths
        # we use the paths to display dataset images
        self.test_dataset_paths = []

        # Test and Validation images
        self.test_percentage = 10
        self.validation_percentage = 10

        # Model eveluation
        self.weights_paths = None
        self.weights_list = None

        # Settings cfg
        # self.batchsize = batchsize

        data_folder_path = os.path.join(projectpath, "data")
        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)

        if not os.path.exists(self.backup_folder_path):
            os.makedirs(self.backup_folder_path)
            print("Created Backup Folder")

        self.n_classes = 0
        self.n_labels = 0


    def get_starting_weights(self):
        global yolomodels
        pretrained_weights_name = yolomodels[self.model_name]["train_name"]
        pretrained_path = os.path.join(YOLO_ROOT_DIR, pretrained_weights_name)
        return pretrained_path


    def get_available_models(self, chosen_model):
        self.weights_paths = glob.glob(os.path.join(self.project_path, "**/*.pt"),
                                       recursive=True)
        # # Make sure that weights of the chosen model are available
        # model_found = False
        # for path in self.weights_paths:
        #     if chosen_model in path:
        #         model_found = True
        #
        # if model_found is False:
        #     print("WARNING: Weights for the model {} were not found.\n"
        #           "If you trained a different YOLO version, make sure you select that on the step N° 1 of the notebook.".format(chosen_model))

        return self.weights_paths

    def show_weights_list(self):
        self.weights_list = widgets.Dropdown(
            options=self.weights_paths,
            description='Weight path:',
        layout=widgets.Layout(width='850px'))


    def count_classes_number(self):
        print("Detecting classes number ...")
        # Detect number of Classes by reading the labels indexes
        # If there are missing indexes, normalize the number of classes by rewriting the indexes starting from 0
        txt_file_paths = glob.glob(self.images_folder_path + "**/*.txt", recursive=True)
        self.n_labels = len(txt_file_paths)
        print("{} label files found".format(self.n_labels))
        # Count number of classes
        class_indexes = set()
        for i, file_path in enumerate(txt_file_paths):
            # get image size
            with open(file_path, "r") as f_o:
                lines = f_o.readlines()
                for line in lines:
                    numbers = re.findall("[0-9.]+", line)
                    if numbers:
                        # Define coordinates
                        class_idx = int(numbers[0])
                        class_indexes.add(class_idx)

        # Update classes number
        self.n_classes = len(class_indexes)
        print("{} classes found".format(self.n_classes))

        # Verify if there are missing indexes
        if max(class_indexes) > len(class_indexes) - 1:
            print("Class indexes missing")
            print("Normalizing and rewriting classes indexes so that they have consecutive index number")
            # Assign consecutive indexes, if there are missing ones
            # for example if labels are 0, 1, 3, 4 so the index 2 is missing
            # rewrite labels with indexes 0, 1, 2, 3
            new_indexes = {}
            classes = sorted(class_indexes)
            for i in range(len(classes)):
                new_indexes[classes[i]] = i

            for i, file_path in enumerate(txt_file_paths):
                # get image size
                with open(file_path, "r") as f_o:
                    lines = f_o.readlines()
                    text_converted = []
                    for line in lines:
                        numbers = re.findall("[0-9.]+", line)
                        if numbers:
                            # Define coordinates
                            class_idx = new_indexes.get(int(numbers[0]))
                            class_indexes.add(class_idx)
                            text = "{} {} {} {} {}".format(0, numbers[1], numbers[2], numbers[3], numbers[4])
                            text_converted.append(text)
                    # Write file
                    with open(file_path, 'w') as fp:
                        for item in text_converted:
                            fp.writelines("%s\n" % item)

    def generate_custom_yaml(self):
        custom_yaml = "train: {}\nval: {}\ntest: {}\nnc: {}\nnames: {}\n".format(
            os.path.join(self.backup_folder_path, "data/train.txt"),
            os.path.join(self.backup_folder_path, "data/validation.txt"),
            os.path.join(self.backup_folder_path, "data/test.txt"),
            self.n_classes,
            [str(i) for i in range(self.n_classes)]
        )

        # Saving Obj data
        with open(self.obj_data_path, "w") as f_o:
            f_o.writelines(custom_yaml)

        # Saving Obj names
        # with open(self.obj_names_path, "w") as f_o:
        #     for i in range(self.n_classes):
        #         f_o.writelines("CLASS {}\n".format(i))

    def generate_train_val_files(self):
        print("Generating Train/Validation/Test list")
        images_list = glob.glob(self.images_folder_path + "**/*.jpg", recursive=True)
        random.shuffle(images_list)

        print("{} Images found".format(len(images_list)))
        if len(images_list) == 0:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "Images list not found. Make sure that the images are '.jpg' "
                                                         "format and inside the directory {}".format(
                    self.images_folder_path))

        # Read labels
        for img_path in images_list:
            img_name_basename = os.path.basename(img_path)
            img_name = os.path.splitext(img_name_basename)[0]



        # Generate test files, 10% of training
        test_number = len(images_list) // self.test_percentage

        print("Validation images: {}".format(test_number))
        with open("{}/data/validation.txt".format(self.backup_folder_path), "w") as f_o:
            for i, path in enumerate(images_list):
                f_o.writelines("{}\n".format(path))
                self.test_dataset_paths.append(path)
                if i == test_number:
                    break
        print("validation.txt generated")

        validation_number = len(images_list) // self.validation_percentage
        print("Validation images: {}".format(test_number))
        with open("{}/data/test.txt".format(self.backup_folder_path), "w") as f_o:
            for i, path in enumerate(images_list):
                if i > validation_number:
                    f_o.writelines("{}\n".format(path))
                    self.test_dataset_paths.append(path)
                # If we reach the test number, stop
                if i == validation_number + test_number:
                    break
        print("test.txt generated")

        print("Train images: {}".format(len(images_list) - test_number - validation_number))
        with open("{}/data/train.txt".format(self.backup_folder_path), "w") as f_o:
            for i, path in enumerate(images_list):
                if i > validation_number + test_number:
                    f_o.writelines("{}\n".format(path))
        print("train.txt generated")

    def extract_zip_file(self, path_to_zip_file):
        print("Extracting Images")
        with zipfile.ZipFile(self.images_folder_path + "images.zip", 'r') as zip_ref:
            zip_ref.extractall(self.images_folder_path)


class CustomTraining:
    def __init__(self, project_path):
        # Custom values for training
        self.obj_data_path = os.path.join(project_path, "data\custom.yaml")
        self.workers = 1
        self.batch = 16
        self.augment = True

    def train(self, modelname, epochs, device, projectname, imgsz):
        # Train
        model = YOLO(model=yolomodels[modelname]["filename"])
        results = model.train(data=self.obj_data_path, epochs=int(epochs),
                              device=device, workers=self.workers, imgsz=int(imgsz), batch=self.batch,
                              project="{}/{}".format(BACKUP_DIR, projectname),
                              augment=self.augment)



class Display:
    def __init__(self, test_dataset):
        test_dataset = test_dataset[0] if test_dataset else ""
        self.image_path = widgets.Text(value='{}'.format(test_dataset),
                                        description='Image Path', layout=widgets.Layout(width='900px'))

    def show_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
        display(Image.fromarray(img))

