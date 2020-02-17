
pip install --user gdown

mkdir datasets
cd datasets


# MNIST
gdown https://drive.google.com/uc?id=1OeIRIDbqPXcaH6wUiYo684NilBM_WOoW

# MNIST-M
gdown https://drive.google.com/uc?id=1N1HWQlbdxj4Er1oCfrWxLUTiZNfIrk19

# SVHN
gdown https://drive.google.com/uc?id=1dx5bBfA9IGBtpePE1B01VZm89etuuAux

# SYN DIGITS
gdown https://drive.google.com/uc?id=1SlKL1rOfOhGM5GfUi_FxxxKtRNObAPhl

# USPS
gdown https://drive.google.com/uc?id=1Xl3RMbQMLcFx9b-5hJxceV0C6etVi0sH



# GTSRB
gdown https://drive.google.com/uc?id=1FXDqeDPxa7Scj7hs7lFm8VIafnq1oT97

# SYN SIGNS
gdown https://drive.google.com/uc?id=1xTzFWxKctlzQj-2Rsrdmbql-H96VsHVZ



# Office Home
gdown https://drive.google.com/uc?id=1gn6Wm-IUZkv2bzRmWyKNb7P-tc8xB33i

# Office
gdown https://drive.google.com/uc?id=12mu-EK9ImvkgJGL45o2ceTqdGBlt7nOP



# Mensural Symbols
gdown https://drive.google.com/uc?id=1wBdo1dxTUWon7aXgAHqRvKrmBBH0pL95



for z in *.zip; do unzip "$z"; done
rm *.zip


