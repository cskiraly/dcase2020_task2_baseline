#!/bin/sh

URL="https://zenodo.org/record/3678171/files/dev_data_slider.zip?download=1"

curl $URL -o dev_data_slider.zip || wget $URL -O dev_data_slider.zip
mkdir -p dev_data
unzip dev_data_slider.zip -d dev_data
