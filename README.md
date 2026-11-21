FoodBD — Bangladeshi Food Classification with a CNN (PyTorch)

A simple Convolutional Neural Network (CNN) built in PyTorch that classifies 67 Bangladeshi food items from the FoodBD dataset (A Polygon-Annotated Meal Image Dataset of Bangladeshi food).

The notebook has two parts:

CNN Implementation — setup, data loading, a 3-layer CNN, training, and visualization of the learned filters and feature maps.
Result Analysis — accuracy/loss curves, a single-image prediction, misclassified images, a confusion matrix, a per-class report and a written analysis.

Only the basic transform (Resize + ToTensor) is used. There is no data augmentation.

Repository contents
File	Description
FoodBD_CNN_Result_Analysis.ipynb	Main notebook (with saved outputs and plots)
README.md	This file
Dataset

FoodBD has meal photos, already split into train / valid / test, with 67 food classes. Each photo shows a whole meal plate, and every food item on it is marked by a polygon (YOLO segmentation format).

FoodBD/
├── data.yaml        (the 67 class names)
├── train/  images/ (2473 meal photos)   labels/ (polygon .txt files)
├── valid/  images/ (697 meal photos)    labels/
└── test/   images/ (353 meal photos)    labels/

The dataset is not included in this repository. Download it and upload the FoodBD folder to your Google Drive.

From meal photos to single-food images

The CNN predicts one class per image, so the notebook crops every annotated food item (using the box around its polygon, with a 5% margin) and saves it into a folder named after its class. PyTorch's ImageFolder can then read it directly.

Split	Meal photos	Food crops
Train	2473	9,359
Validation	697	2,746
Test	353	1,373

Note: the class egg has no examples in the test split.

Model
Image 3×64×64
  → Conv(16 filters, 3×3) → ReLU → MaxPool   →  16×32×32
  → Conv(32 filters, 3×3) → ReLU → MaxPool   →  32×16×16
  → Conv(64 filters, 3×3) → ReLU → MaxPool   →  64×8×8
  → Flatten                                  →  4096
  → Linear → ReLU                            →  128
  → Linear                                   →  67 classes

Total trainable parameters: 556,643. Loss: CrossEntropyLoss. Optimizer: Adam.

Hyperparameters
Hyperparameter	Value	Why
Image size	64 × 64	128 × 128 gave the same validation accuracy (~61%) but was ~4× slower
Batch size	32	Works well for ~9.4k training images
Learning rate	0.0005	At 0.001, validation loss started rising after ~6 epochs
Epochs	12	Validation loss is lowest around epochs 10–12
Dropout	0.0	Not used
Results

Results from the run saved in the notebook (Google Colab, torch.manual_seed(0)).

Metric	Value
Final train accuracy (epoch 12)	68.6%
Best validation accuracy	60.7% (epoch 12)
Best train loss	1.151 (epoch 12)
Best validation loss	1.588 (epoch 12)
Test accuracy	TEST_ACC
Test loss	TEST_LOSS

For comparison, random guessing over 67 classes would give about 1.5%.

Training log
Epoch	Train loss	Train acc	Val loss	Val acc
1	3.190	18.9%	2.798	29.4%
2	2.612	32.8%	2.323	41.7%
3	2.217	43.7%	2.137	44.1%
4	1.964	49.5%	1.927	49.3%
5	1.798	53.5%	1.774	53.9%
6	1.659	56.6%	1.718	55.6%
7	1.546	59.2%	1.727	56.1%
8	1.460	61.0%	1.656	57.0%
9	1.380	62.8%	1.609	59.2%
10	1.290	65.1%	1.591	58.8%
11	1.226	66.2%	1.608	59.5%
12	1.151	68.6%	1.588	60.7%

Both losses fall steadily, and the final train/validation accuracy gap is about 8 percentage points (68.6% vs 60.7%).

Main sources of error

Per-class precision, recall and F1 are in the notebook's classification report.

Class imbalance: rice has 1,056 training crops, while guava has only 13. Rare classes (guava, mango, apple, badam: 13–16 crops each) are much harder for the model to learn.
Similar-looking curries: brown/yellow cooked dishes such as beef, chicken, fish, achar, vaji and vegetable look alike at 64 × 64 pixels.
Messy crops: some crops include part of a neighbouring food, and a few are black because the original photo has black borders.
Why 12 epochs?

A separate experiment (outside the notebook) with the same settings for 40 epochs showed clear overfitting:

Epoch	Train acc	Val acc	Val loss
12	70.6%	60.9%	1.56 (lowest)
21	87.3%	62.1% (best)	2.01
40	97.1%	59.9%	3.46

Training accuracy climbs to ~97%, but validation accuracy stays at ~60–62% and validation loss more than doubles. Without augmentation or dropout, 10–15 epochs is the right range for this dataset.

How to run (Google Colab)
Upload the FoodBD dataset folder (the one with data.yaml, train, valid and test) to My Drive.
Open FoodBD_CNN_Result_Analysis.ipynb in Google Colab.
Select Runtime → Change runtime type → GPU.
Run all cells from top to bottom. The notebook will:
mount Google Drive and find the FoodBD folder automatically,
copy it to Colab's local disk,
crop the food items into class folders,
train the CNN and show all results.

If the folder isn't found, make sure it's in My Drive and not only in Shared with me. In Drive, right-click it → Organize → Add shortcut → My Drive.
