import os
import numpy as np
import torch

from collections import namedtuple, defaultdict
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


class RealEstate10KDataset(Dataset):
    # NamedTuple type for handling parsed camera metadata
    CameraMetadata = namedtuple("CameraMetadata", ["R", "t", "focal_length", "principal_point"])

    def __init__(self, split, dataset_images_path, dataset_metadata_path, min_image_dimension):
        """
        Args:
            split - either train or test
            dataset_images_path - filepath to RE10k dataset images
                (should contain one folder called train and one folder called test)
            dataset_metadata_path - filepath to RE10k metadata (camera extrinsics and intrinsics)
        """
        self.split = split
        self.sequence_metadata_map = defaultdict(list)
        self.sequence_names = []

        self.preprocess_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(min_image_dimension, antialias=True)
        ])
        self.resized_image_min_dim = min_image_dimension

        self.images_split_path = os.path.join(dataset_images_path, split)
        metadata_split_path = os.path.join(dataset_metadata_path, split)

        sequence_names = os.listdir(self.images_split_path)
        for i, sequence_name in tqdm(enumerate(sequence_names)):
            if i % 4 != 0:
                continue

            self.sequence_names.append(sequence_name)

            sequence_metadata_filepath = os.path.join(metadata_split_path, f"{sequence_name}.txt")
            parsed_metadata = self.__parse_metadata_file(sequence_metadata_filepath)

            sequence_directory = os.path.join(self.images_split_path, sequence_name)
            for sequence_image_path in os.listdir(sequence_directory):
                sequence_image_timestamp = int(sequence_image_path.split(".")[0])
                image_metadata = parsed_metadata[sequence_image_timestamp]
                self.sequence_metadata_map[sequence_name].append({
                    "filepath": os.path.join(sequence_directory, sequence_image_path),
                    "R": image_metadata.R,
                    "T": image_metadata.t,
                    "focal_length": image_metadata.focal_length,
                    "principal_point": image_metadata.principal_point
                })

        print(f"Testing on {len(self.sequence_names)} videos")

    def __parse_metadata_file(self, metadata_filepath):
        """
        Provided a RE10K metadata file, returns the parsed camera extrinsics and intrinsics

        Args:
            metadata_filepath - the filepath of the metadata file we want to parse

        Returns:
            A dict mapping timestamp to all metadata for the corresponding image file
            (images files are named based on timestamp).

            Each entry in the list is a NamedTuple whose arguments are as follows:

            R - the 3x3 rotation submatrix of the camera pose matrix (consisting of its first 3 rows and columns)
            t - the final column of the camera pose matrix (translation)
            focal_length - a tuple of the focal length in the x and y axes
            principal_point - a tuple consisting of the x and y coordinates of the principal point
        """

        parsed_metadata = {}
        with open(metadata_filepath) as metadata_file:
            raw_metadata = metadata_file.readlines()
            for i in range(1, len(raw_metadata)):
                frame_data = list(map(float, raw_metadata[i].split()))

                # These indexes are based on the specification at https://google.github.io/realestate10k/download.html
                timestamp = int(frame_data[0])
                focal_length = frame_data[1:3]
                principal_point = frame_data[3:5]
                P = np.reshape(frame_data[7:20], (3, 4))

                R = P[:, :3]
                t = P[:, 3]

                parsed_metadata[timestamp] = self.CameraMetadata(R, t, focal_length, principal_point)

        return parsed_metadata

    def __len__(self):
        return len(self.sequence_names)

    def __getitem__(self, sequence_amount_tuple: Tuple[int, int]):
        """
        Given an input tuple of the form (sequence_index, image_count), returns
        image_count images from the sequence at index sequence_index

        Requires: image_count <= sequence length
        """
        sequence_idx, image_count = sequence_amount_tuple
        sequence = self.sequence_names[sequence_idx]
        sequence_image_count = len(self.sequence_metadata_map[sequence])
        sequence_image_indexes = np.random.choice(sequence_image_count, image_count, replace=False)
        batch, _ = self.get_data(sequence, sequence_image_indexes)
        return batch

    def get_data(self, sequence_name, sequence_image_indexes):
        """
        A method to fetch a batch of images and their corresponding metadata given a
        sequence name and a list of image indexes
        """

        selected_images_metadata: List[Dict] = [self.sequence_metadata_map[sequence_name][i] for i in sequence_image_indexes]

        batch = {}

        images = []
        image_rotations = []
        image_translations = []
        image_focal_lengths = []
        image_principal_points = []
        image_bounding_boxes = []
        image_paths = []

        for image_metadata in selected_images_metadata:
            image_path = image_metadata["filepath"]
            image_paths.append(image_path)

            current_image = Image.open(image_path).convert("RGB")
            images.append(self.preprocess_image_transform(current_image))
            original_image_width, original_image_height = current_image.size

            if original_image_width < original_image_height:
                image_width = self.resized_image_min_dim
                image_height = original_image_height / original_image_width * self.resized_image_min_dim
            else:
                image_width = original_image_width / original_image_height * self.resized_image_min_dim
                image_height = self.resized_image_min_dim

            image_bounding_boxes.append(np.array([0, 0, image_width, image_height]))

            image_rotations.append(torch.Tensor(image_metadata["R"]))
            image_translations.append(torch.Tensor(image_metadata["T"]))

            normalized_focal_length_x, normalized_focal_length_y = image_metadata["focal_length"]
            image_focal_lengths.append(
                torch.Tensor((image_width * normalized_focal_length_x, image_height * normalized_focal_length_y))
            )

            normalized_principal_point_x, normalized_principal_point_y = torch.Tensor(image_metadata["principal_point"])
            image_principal_points.append(
                torch.Tensor((image_width * normalized_principal_point_x, image_height * normalized_principal_point_y))
            )

        batch["images"] = torch.stack(images)
        batch["R"] = torch.stack(image_rotations)
        batch["T"] = torch.stack(image_translations)
        batch["fl"] = torch.stack(image_focal_lengths)
        batch["pp"] = torch.stack(image_principal_points)
        batch["crop_params"] = {
            "size": tuple(images[0].shape),
            "resized_scales": np.ones(len(images)),
            "bboxes_xyxy": np.stack(image_bounding_boxes)
        }

        return batch, image_paths


if __name__ == '__main__':
    dataset = RealEstate10KDataset(
        "test",
        "/share/phoenix/nfs05/S8/rc844/RE10K/RealEstate10K_Downloader/dataset/",
        "/share/phoenix/nfs04/S7/rc844/MegaScenes/RE10K/RealEstate10K/"
    )
    print(dataset.get_data("9339d2793fa7d4b0", (0, 1)))
