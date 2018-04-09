# Nuclei-segmentation

Starter PyTorch implementation of U-Net image segmentation for Kaggle Data Science Bowl 2018 competition. Link to competation: https://www.kaggle.com/c/data-science-bowl-2018

## Project structure
```
.
├── data
│   ├── train
│   └── test
├── data.py           
├── model.py
├── test.py
├── train.py
└── README.md
```

## Getting started

1. As per usual, open terminal and clone repository `git clone https://github.com/bvezilic/Nuclei-segmentation.git`
2. Download the **train.zip** and **test.zip** from kaggle and extract the contents to **data/train** and **data/test** directory respectively.
> Directories should look something like:
>```
>train  
>  0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe
>  0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e
>  ...
>test
>  0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732
>  0e132f71c8b4875c3c2dd7a22997468a3e842b46aa9bd47cf7b0e8b7d63f0925
>  ...
>```
3. Change to repository directory `cd Nuclei-segmentation`
4. Run `python train.py`
> Model will be automatically saved after training in **models** directory
5. Run `python test.py`

## Test results

Here are some test samples segmentation:

![test_results](https://user-images.githubusercontent.com/16206648/38512364-5bb56d12-3c2b-11e8-928b-2918080ae4f6.png)



