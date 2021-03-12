# Installation

## Virtual Environment

```bash
conda create -n phone_finder python=3.7  # Python 3.7.9
conda activate phone_finder
```

## PyTorch

```bash
# Install PyTorch 1.7.1 following the instruction on https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

## Pip

```bash
pip install matplotlib  # matplotlib-3.3.4
pip install opencv-python  # opencv-python-4.5.1.48
pip install opencv-contrib-python  # opencv-contrib-python-4.5.1.48
pip install black flake8  # Formatting and linting
```

# Training

```bash
python train_phone_finder.py [path_to_training_folder]
# For example
python train_phone_finder.py ~/find_phone
# More configuration information can be printed by
python train_phone_finder.py -h
# Rendering is disabled by default.
# If you would like to visualize the phone locating process, add --render 1:
python train_phone_finder.py ~/find_phone --render 1
```

## Testing

```bash
python find_phone.py [path_to_test_image]
```

![11](/home/wes/Documents/Chen-Weizhe/find_phone_test_images/11.jpg)

We added some ``homemade'' images for testing. Take this image for example, we find the phone by the following command:

```bash
python find_phone.py ./find_phone_test_images/11.jpg --render 1
```

We will see the following phone locating process and the estimated position 0.3869 0.4617.

![demo](/home/wes/Documents/Chen-Weizhe/demo.gif)