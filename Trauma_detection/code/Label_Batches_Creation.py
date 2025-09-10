#Creates labels for audio and visuals at a single time. (to be ran first for lables)
#Recent_One

import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

class Create_Labels():
    def __init__(self, config):
        self.config = config
        self.pattern = r'^frame_(\d{6})_face_(\d{2})\.jpg$'

    def load_segment(self, file_path):
        return np.load(file_path)

    def create_continuous_labels(self, total_samples, category):
        label_value = 1 if category == "PTSD" else 0
        return np.full(total_samples, label_value)

    def save_continuous_labels(self, labels, video_folder):
        output_filename = f"{os.path.basename(video_folder)}_Continuous_Label.npy"
        output_path = os.path.join(video_folder, output_filename)
        np.save(output_path, labels)

    def get_category_from_path(self, file_path):
        parts = file_path.split(os.sep)
        return parts[-2]

    def get_sorted_npy_segment_files(self, video_folder):
        segment_files = [f for f in os.listdir(video_folder) if f.startswith("segment_") and f.endswith(".npy")]
        return sorted(segment_files, key=lambda x: int(re.findall(r'segment_(\d+)\.npy', x)[0]))


    def get_sorted_visual_segment_folders(self, video_folder):
        return sorted(
            [f for f in os.listdir(video_folder) if f.startswith("segment_") and os.path.isdir(os.path.join(video_folder, f))],
            key=lambda x: int(re.findall(r'segment_(\d+)', x)[0])
        )

    def process_faces_files(self, directory):
        face_files = []
        Faces_Folder =  os.path.join(directory, 'faces')
        for filename in os.listdir(Faces_Folder):
            match = re.match(self.pattern, filename)
            if match:
                frame_number = int(match.group(1))
                face_number = int(match.group(2))
                face_files.append({
                    'filename': filename,
                    'frame_number': frame_number,
                    'face_number': face_number
                })
        face_files.sort(key=lambda x: (x['frame_number'], x['face_number']))
        return face_files


    def audio_labels(self, video_folder):
        category = self.get_category_from_path(video_folder)
        segment_files = self.get_sorted_npy_segment_files(video_folder)
        
        total_samples = sum(self.load_segment(os.path.join(video_folder, f)).shape[0] for f in segment_files)
        continuous_labels = self.create_continuous_labels(total_samples, category)
        self.save_continuous_labels(continuous_labels, video_folder)

        current_index = 0
        for segment_file in segment_files:
            segment_path = os.path.join(video_folder, segment_file)
            segment_data = self.load_segment(segment_path)
            segment_labels = continuous_labels[current_index:current_index + segment_data.shape[0]]

            print(f"Segment: {segment_file}, Shape: {segment_data.shape}, Labels: {segment_labels}")
            current_index += segment_data.shape[0]
    
    def visual_labels(self, video_folder):
        category = self.get_category_from_path(video_folder)
        segment_folders = self.get_sorted_visual_segment_folders(video_folder)

        total_samples = 0
        all_face_files = []

        for segment_folder in segment_folders:
            segment_path = os.path.join(video_folder, segment_folder)
            all_face_files.extend(self.process_faces_files(segment_path))

        total_samples = len(all_face_files)

        continuous_labels = self.create_continuous_labels(total_samples, category)
        continuous_labels_path = self.save_continuous_labels(continuous_labels, video_folder)
        print(f"Total faces in the video are : {total_samples}")

        current_index = 0
        for segment_folder in segment_folders:
            segment_path = os.path.join(video_folder, segment_folder)
            segment_face_files = self.process_faces_files(segment_path)
            
            #print(f"{segment_folder} has {len(segment_face_files)} faces")
            #print("Frame\tFace\tLabel")
            #print("-" * 20)

            for face_file in segment_face_files:
                face_label = continuous_labels[current_index]
                #print(f"{face_file['frame_number']}\t{face_file['face_number']}\t{face_label}")
                current_index += 1

    def processs_2_modality_labels_at_once(self):
        for category in ["PTSD", "Non-PTSD"]:
            category_path = os.path.join(self.config['data_directory'], category)
            for video_folder in os.listdir(category_path):
                video_path = os.path.join(category_path, video_folder)
                if os.path.isdir(video_path):
                    print(f"\nProcessing video: {video_folder}")
                    if self.config['data_directory'].endswith('Audio'):
                        self.audio_labels(video_path)
                    elif self.config['data_directory'].endswith('Visual'):
                        self.visual_labels(video_path)

#Label loading and creating batches (in a sequential-manner) for Audio and Visual from Final_Train (2nd run)
class Load_Audio_Labels_batches(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
        self.segments = []
        self.labels = []
        self.sample_indices = []
        self.segment_starts = []
        self.video_boundaries = []
    
        for video_folder in self.video_folders:
            folder_path = os.path.join(root_dir, video_folder)
            #print(f"Processing video folder: {folder_path}")
            segment_files = sorted([f for f in os.listdir(folder_path) if f.startswith('segment_') and f.endswith('.npy')],
                key=lambda x: int(re.search(r'segment_(\d+)\.npy', x).group(1)))
            labels_path = os.path.join(folder_path, f"{video_folder}_Continuous_Label.npy")
            #print(f"Loading labels from: {labels_path}")
            labels = np.load(labels_path)

            video_start = len(self.segments)
            current_start = 0
            for segment_file in segment_files:
                segment_path = os.path.join(folder_path, segment_file)
                segment = np.load(segment_path)
                print(f"Loaded {segment_file} from {segment_path} with shape: {segment.shape}")

                num_samples = segment.shape[0]
                self.segments.append(segment_path)
                self.labels.append(labels[current_start:current_start + num_samples])
                self.segment_starts.append(current_start)
                self.sample_indices.extend([(len(self.segments)-1, j) for j in range(num_samples)])
                current_start += num_samples

            video_end = len(self.segments)
            self.video_boundaries.append((video_start, video_end))
        #print(f"Loaded {len(self.sample_indices)} samples from {len(self.segments)} segments across {len(self.video_folders)} videos.")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        segment_idx, sample_idx = self.sample_indices[idx]
        segment = np.load(self.segments[segment_idx])
        sample = segment[sample_idx]
        label = self.labels[segment_idx][sample_idx]
        global_idx = self.segment_starts[segment_idx] + sample_idx
        return torch.from_numpy(sample).float(), torch.tensor(label).int()

class Load_Visual_Labels_batches(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.video_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
        self.segments = []
        self.transform = transform
        self.segment_info = []
        self.label_files = {}
        self.label_counts = {}
        self.pattern = r'^frame_(\d{6})_face_(\d{2})\.jpg$'

        for video_folder in self.video_folders:
            folder_path = os.path.join(root_dir, video_folder)
            print(f"Processing video folder: {folder_path}")
            segment_folders = sorted([f for f in os.listdir(folder_path) if f.startswith('segment_')],
                                     key=lambda x: int(re.search(r'segment_(\d+)', x).group(1)))
            labels_path = os.path.join(folder_path, f"{video_folder}_Continuous_Label.npy")
            #print(f"Loading labels from: {labels_path}")
            self.label_files[video_folder] = np.load(labels_path)
            self.label_counts[video_folder] = len(self.label_files[video_folder])

            current_start = 0
            for segment_folder in segment_folders:
                segment_path = os.path.join(folder_path, segment_folder, 'faces')
                face_files = self.process_files(segment_path)

                num_faces = len(face_files)
                if num_faces == 0:
                    print(f" {segment_folder} skipping as it has no faces")
                    continue
                self.segments.append((segment_path, face_files))
                self.segment_info.append((video_folder, segment_folder, len(self.segments) - 1, num_faces, current_start))
                current_start += num_faces

        #print(f"Loaded {len(self.segment_info)} segments across {len(self.video_folders)} videos.")
        for video_folder, count in self.label_counts.items():
            print(f"Video {video_folder}: {count} labels")

    def __len__(self):
        return len(self.segment_info)

    def process_files(self, directory):
        face_files = []
        for filename in os.listdir(directory):
            match = re.match(self.pattern, filename)
            if match:
                frame_number = int(match.group(1))
                face_number = int(match.group(2))
                face_files.append({
                    'filename': filename,
                    'frame_number': frame_number,
                    'face_number': face_number
                })
        face_files.sort(key=lambda x: (x['frame_number'], x['face_number']))
        return [f['filename'] for f in face_files]
    
    def __getitem__(self, idx):
        video_folder, segment_folder, segment_idx, num_faces, label_start = self.segment_info[idx]
        segment_path, face_files = self.segments[segment_idx]
        
        images = []
        labels = []
        image_paths = []
        
        for i, face_file in enumerate(face_files):
            face_image_path = os.path.join(segment_path, face_file)
            image = Image.open(face_image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            
            label_index = label_start + i
            if label_index < self.label_counts[video_folder]:
                labels.append(self.label_files[video_folder][label_index])
            else:
                print(f"Warning: Label index {label_index} out of bounds for video {video_folder}. Using last available label.")
                labels.append(self.label_files[video_folder][-1])           
            image_paths.append(face_image_path)
        
        return torch.stack(images), torch.tensor(labels).int(), image_paths, segment_folder

def Display_images_with_labels(images, labels, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 3))
    if len(images) == 1:
        axs = [axs]
    for i, (img, label, title) in enumerate(zip(images, labels, titles)):
        axs[i].imshow(img.permute(1, 2, 0))
        axs[i].set_title(f"{title}\nLabel: {label:.2f}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def process_batch_audio_visual_labels(config):
    processed_data = {
        'PTSD': {'Audio': [], 'Visual': []},
        'Non-PTSD': {'Audio': [], 'Visual': []}
    }
    
    for category in ["PTSD", "Non-PTSD"]:
        category_path = os.path.join(config['data_directory'], category)
        
        if config['data_directory'].endswith('Visual'):
            dataset = Load_Visual_Labels_batches(category_path, transform=config['transform'])
            for video_folder in dataset.video_folders:
                video_segments = [seg for seg in dataset.segment_info if seg[0] == video_folder]
                for segment_info in video_segments:
                    video_folder, segment_folder, segment_idx, num_faces, _ = segment_info
                    images, labels, image_paths, segment_name = dataset[segment_idx]
                    dataloader = DataLoader(
                        dataset=[(img, lbl) for img, lbl in zip(images, labels)], 
                        batch_size=len(images),
                        sampler=SequentialSampler(images)
                    )
                    for batch_images, batch_labels in dataloader:
                        processed_data[category]['Visual'].append((video_folder, batch_images, batch_labels))

        elif config['data_directory'].endswith('Audio'):
            dataset = Load_Audio_Labels_batches(category_path)
            video_dataloaders = []
            for video_start, video_end in dataset.video_boundaries:
                video_dataloaders.append([])
                for i in range(video_start, video_end):
                    if len(dataset.labels[i]) == 0:
                        video_name = dataset.video_folders[video_index] 
                        modality_type = "Audio" if "Audio" in config['data_directory'] else "Visual" 
                        print(f"Skipping segment {i} in video '{video_name}' ({modality_type}) due to zero samples.")
                        continue 
                    start_idx = dataset.segment_starts[i]
                    end_idx = start_idx + len(dataset.labels[i])
                    sampler = torch.utils.data.SequentialSampler(range(start_idx, end_idx))
                    dataloader = DataLoader(dataset, batch_size=len(dataset.labels[i]), sampler=sampler)
                    video_dataloaders[-1].append(dataloader)

            for video_index, video_loaders in enumerate(video_dataloaders):
                video_name = dataset.video_folders[video_index]
                for dataloader in video_loaders:
                    for batch_samples, batch_labels in dataloader:
                        processed_data[category]['Audio'].append((video_name, batch_samples, batch_labels))
    return processed_data
