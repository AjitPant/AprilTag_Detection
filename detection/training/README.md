# For training

1) To train, first put the dataset inside the dataset folder, so the final path to the img will be ./dataset/{dataset\_name}/img, and for masks it will be ./dataset/{dataset\_name}/mask

2) Run the train.py file.

# Work of diffent files

1) Unet.py -> It contains the code for the Unet
2) train.py -> For training
3) test.py -> for inference
4) dataset.py -> Code for the dataloader
5) corners\_to\_crop.py -> extracts the april tags from a image given the original image and the masks.
