import os
import numpy as np
import torch
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
import re
import cv2
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class Data_Loading():
    def __init__(self, directory):
        self.directory = directory
        self.train_image_generators = {}
        self.test_image_generators = {}
        self.validation_image_generators = {}
        self.Data_Generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
    def plot_images(self, snippets, video_name, label):
        num_snippets = len(snippets)
        num_images_per_snippet = [len(images) for images in snippets.values()]
        max_images_per_row = max(num_images_per_snippet)
        num_rows = num_snippets
        num_cols = max_images_per_row
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        print(f'Video: {video_name}')
        fig.suptitle(f"Video: {video_name}, Label: {label}", fontsize=16)
        sorted_snippets = []
        for key, images in snippets.items():
            sorted_images = sorted(images, key=lambda x: (int(re.findall(r'_frame_(\d+)_', x)[0]), int(re.findall(r'_face_(\d)', x)[0])))
            sorted_snippets.append((key, sorted_images))
        
        for i, (snippet, images) in enumerate(sorted_snippets):
            snippet_num = i + 1
            for j, img_path in enumerate(images):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (244, 244))              
                if num_rows > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]            
                frame_number = int(re.findall(r'_frame_(\d+)_', os.path.basename(img_path))[0])
                face_number = int(re.findall(r'_face_(\d)', os.path.basename(img_path))[0])
                ax.imshow(img_resized)
                ax.set_title(f'Frame: {frame_number}, Face: {face_number}\nClass: {label}')
                ax.axis('off')
            if num_rows > 1:
                axes[i, 0].set_ylabel(f'Snippet: {snippet_num}', fontsize=14)
            else:
                axes[0].set_ylabel(f'Snippet: {snippet_num}', fontsize=14)
        
        plt.tight_layout()
        plt.show()

    def process_image(self, image, target_shape=(244, 244)):
            h, w = target_shape
            image = load_img(image, color_mode='grayscale', target_size=(h, w))
            img_arr = img_to_array(image)
            x = img_arr
            return x

    def display_batch_info(self, image_generators):
        print(f'Number of videos in image_generators: {len(image_generators)}')
        for video_name, video_info in image_generators.items():
            print(f"Video Name: {video_name}, Label: {video_info['label']}")
            for batch_info in video_info['batches']:
                snippet_folder = batch_info['snippet_folder']
                generator = batch_info['batch']
                generator.reset()
                batch = generator[0]
                image_filenames = generator.filenames
                sorted_indices = sorted(
                    range(len(image_filenames)),
                    key=lambda x: (
                        int(re.findall(r'_frame_(\d+)_', image_filenames[x])[0]),
                        int(re.findall(r'_face_(\d)', image_filenames[x])[0])
                    )
                )   
                sorted_images = [batch[0][i] for i in sorted_indices]
                sorted_labels = [batch[1][i] for i in sorted_indices]
                sorted_filenames = [image_filenames[i] for i in sorted_indices]
                fig, axes = plt.subplots(1, len(sorted_images) + 1, figsize=(4 * (len(sorted_images) + 1), 4))
                for j, image in enumerate(sorted_images):
                    ax = axes[j]
                    frame_number = re.findall(r'_frame_(\d+)_', sorted_filenames[j])[0]
                    face_number = re.findall(r'_face_(\d)', sorted_filenames[j])[0]
                    title = f"Frame_{frame_number}_face_{face_number}"
                    ax.imshow(image)
                    ax.set_title(title)
                    ax.axis('off')
                axes[-1].text(0.5, 0.5, f'Snippet: {snippet_folder}', fontsize=12, ha='center', va='center')
                axes[-1].axis('off')
                plt.tight_layout()
                plt.show()

    def load_faces(self, directory):
        train_image_generators = {}
        test_image_generators = {}
        validation_image_generators = {}
        for split in ['test']:
            phase_dir = os.path.join(directory, split)
            for category in os.listdir(phase_dir):
                category_folder = os.path.join(phase_dir, category)
                if os.path.isdir(category_folder):
                    if category == 'PTSD':
                        for category_type in os.listdir(category_folder):
                            category_type_folder = os.path.join(category_folder, category_type)
                            Trauma_type = os.path.basename(category_type_folder)
                            if os.path.isdir(category_type_folder):
                                parent_faces_folder = os.path.join(category_type_folder, 'Frames_Extracted')
                                for video_folder in os.listdir(parent_faces_folder):
                                    print(video_folder)
                                    video_folder_path = os.path.join(parent_faces_folder, video_folder)
                                    if os.path.isdir(video_folder_path):
                                        snippets = {}
                                        for snippet_folder in sorted(os.listdir(video_folder_path)):
                                            print(snippet_folder)
                                            snippet_folder_path = os.path.join(video_folder_path, snippet_folder)
                                            if os.path.isdir(snippet_folder_path):
                                                faces_folder_path = os.path.join(snippet_folder_path, 'Faces')
                                                if os.path.isdir(faces_folder_path):
                                                    images = []
                                                    image_files = sorted([img for img in os.listdir(faces_folder_path) if img.endswith('.jpg')],
                                                                        key=lambda x: int(re.findall(r'_frame_(\d+)_', x)[0]))
                                                    for image in image_files:
                                                        img_path = os.path.join(faces_folder_path, image)
                                                        images.append(img_path)
                                                    snippets[snippet_folder] = images
                                                    map_key = f"PTSD:{video_folder}:{snippet_folder}"
                                                    if len(images) > 0:
                                                        df = pd.DataFrame({
                                                            'filename': images,
                                                            'class': ['PTSD'] * len(images)
                                                        })
                                                        image_batches = self.Data_Generator.flow_from_dataframe(
                                                            dataframe=df,
                                                            x_col='filename',
                                                            y_col='class',
                                                            target_size=(244, 244),
                                                            batch_size=len(images),
                                                            class_mode='raw',
                                                            shuffle=False
                                                        )
                                                        batch_info = {'batch': image_batches, 'snippet_folder': snippet_folder}
                                                        if split == 'train':
                                                            if video_folder not in train_image_generators:
                                                                train_image_generators[video_folder] = {'batches': [], 'label': category}
                                                            train_image_generators[video_folder]['batches'].append(batch_info)
                                                            
                                                        elif split == 'test':
                                                            if video_folder not in test_image_generators:
                                                                test_image_generators[video_folder] = {'batches': [], 'label': category}
                                                            test_image_generators[video_folder]['batches'].append(batch_info)
                                                        elif split == 'validation':
                                                            if video_folder not in validation_image_generators:
                                                                validation_image_generators[video_folder] = {'batches': [], 'label': category}
                                                            validation_image_generators[video_folder]['batches'].append(batch_info)
                                        #plot_images(snippets, video_folder, category)

                    elif category == "NO PTSD":
                        print('Processing NO PTSD')
                        Frames_Extracted_Folder = os.path.join(category_folder, 'Frames_Extracted')
                        for video_folder in os.listdir(Frames_Extracted_Folder):
                            video_folder_path = os.path.join(Frames_Extracted_Folder, video_folder)
                            print(video_folder)
                            if os.path.isdir(video_folder_path):
                                snippets = {}
                                for snippet_folder in sorted(os.listdir(video_folder_path)):
                                    snippet_folder_path = os.path.join(video_folder_path, snippet_folder)
                                    print(snippet_folder)
                                    if os.path.isdir(snippet_folder_path):
                                        faces_folder_path = os.path.join(snippet_folder_path, 'Faces')
                                        if os.path.isdir(faces_folder_path):
                                            images = []
                                            image_files = sorted([img for img in os.listdir(faces_folder_path) if img.endswith('.jpg')],
                                                                key=lambda x: int(re.findall(r'_frame_(\d+)_', x)[0]))
                                            for image in image_files:
                                                img_path = os.path.join(faces_folder_path, image)
                                                images.append(img_path)
                                            snippets[snippet_folder] = images
                                            map_key = f"NO PTSD:{video_folder}:{snippet_folder}"
                                            if len(images) > 0:
                                                df = pd.DataFrame({
                                                    'filename': images,
                                                    'class': ['NO PTSD'] * len(images)
                                                })
                                                image_batches = self.Data_Generator.flow_from_dataframe(
                                                    dataframe=df,
                                                    x_col='filename',
                                                    y_col='class',
                                                    target_size=(244, 244),
                                                    batch_size=len(images),
                                                    class_mode='raw',
                                                    shuffle=False)
                                                batch_info = {'batch': image_batches, 'snippet_folder': snippet_folder}
                                                if split == 'train':
                                                    if video_folder not in train_image_generators:
                                                        train_image_generators[video_folder] = {'batches': [], 'label': category}
                                                    train_image_generators[video_folder]['batches'].append(batch_info)
                                                elif split == 'test':
                                                    if video_folder not in test_image_generators:
                                                        test_image_generators[video_folder] = {'batches': [], 'label': category}
                                                    test_image_generators[video_folder]['batches'].append(batch_info)
                                                elif split == 'validation':
                                                    if video_folder not in validation_image_generators:
                                                        validation_image_generators[video_folder] = {'batches': [], 'label': category}
                                                    validation_image_generators[video_folder]['batches'].append(batch_info)
                                #plot_images(snippets, video_folder, category)
        print(len(test_image_generators))
        for key in test_image_generators.items():
            print(key)
        return self.train_image_generators, self.test_image_generators, self.validation_image_generators
        
 


    
    '''def data_to_model(self,train_data, test_data,validation_data):
        batch_size = len(train_data[0][0])
        print(batch_size)

    
    
    def snippet_wise_data():

    

    def laod_data(self, test_data, train_data, validation_data):
        BATCH_SIZE = len()
        IMG_SIZE = (160, 160)

        train_dataset = tf.keras.utils.image_dataset_from_directory(train_data,
                                                                    shuffle=False,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)
                    
        
    def prepare_dataset(self,face_list):
        images = []
        labels = []
        for face in face_list:
            images.append(face['face'])
            labels.append(face['label'])
        images = np.array(images)
        labels = np.array(labels)
        dataset = list(zip(images, labels))
        return dataset
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.test_dir = os.path.join(root_dir, 'test')
        self.train_dir = os.path.join(root_dir, 'train')
        self.validation_dir = os.path.join(root_dir, 'validation')

    def __len__(self):
        return len(os.listdir(self.test_dir)) + len(os.listdir(self.train_dir)) + len(os.listdir(self.validation_dir))

    def __getitem__(self, idx):
        if idx < len(os.listdir(self.test_dir)):
            img_path = os.path.join(self.test_dir, str(idx) + '.jpg')
            img = Image.open(img_path)
            label = 'test'
        elif idx < len(os.listdir(self.test_dir)) + len(os.listdir(self.train_dir)):
            idx -= len(os.listdir(self.test_dir))
            img_path = os.path.join(self.train_dir, str(idx) + '.jpg')
            img = Image.open(img_path)
            label = 'train'
        else:
            idx -= len(os.listdir(self.test_dir)) + len(os.listdir(self.train_dir))
            img_path = os.path.join(self.validation_dir, str(idx) + '.jpg')
            img = Image.open(img_path)
            label = 'validation'

        if self.transform:
            img = self.transform(img)

        return img, label'''


        
                                            