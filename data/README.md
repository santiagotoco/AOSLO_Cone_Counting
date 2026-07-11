# Dataset instructions

The AOSLO images and expert cone centroid annotations are not redistributed in
this repository.

Experiments use the ADAM AOSLO cone photoreceptor dataset used in the CoDE
study. The dataset is available from the original source upon request, according
to the terms described in that paper.

After obtaining the dataset, organize it locally as:

```text
data/
├── images/
├── annotations/
└── splits/
    ├── train_ids.txt
    ├── val_ids.txt
    └── test_ids.txt
