# NABirds Dataset Documentation

This document provides an overview of the NABirds dataset used for model training.

## Storage Location

The dataset is stored in the following Google Cloud Storage (GCS) bucket:
`gs://kaggle_nabirds_data/`

## Folder Structure on GCS

The data is organized within the `nabirds_preprocessed` directory in the GCS bucket. The structure is as follows:

```
gs://kaggle_nabirds_data/
└── nabirds_preprocessed/
    ├── annotations_all.csv
    ├── classes.txt
    ├── images/
    │   ├── 001/
    │   ├── ...
    │   └── 1010/
    └── train_test_split.txt
```

-   **`images/`**: Contains the image files, organized into subdirectories by class ID.

## File Descriptions

-   **`annotations_all.csv`**: This CSV file contains the core metadata for each image.
    -   **Columns**: `image_id`, `class_id`, `image_name`, `x`, `y`, `width`, `height`.
    -   **Purpose**: Maps each image to its corresponding class and provides bounding box coordinates for object detection tasks.

-   **`classes.txt`**: A text file that maps numeric `class_id` values to human-readable class names (bird species).
    -   **Format**: Each line contains a `class_id` followed by the species name.
    -   **Purpose**: Used to interpret the model's output and translate class predictions into meaningful labels.

-   **`train_test_split.txt`**: This file specifies how the dataset is partitioned into training and testing sets.
    -   **Format**: Each line contains an `image_id` followed by a boolean flag (`0` for training, `1` for testing).
    -   **Purpose**: Ensures a consistent and reproducible split of data for training and evaluating the model.
