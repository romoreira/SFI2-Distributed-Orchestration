#!/bin/bash

cnns=("ResNet" "InceptionTime" "OmniScaleCNN" "FCN" "FCNPlus" "ResNetPlus" "ResCNN" "TCN" "InceptionTimePlus" "XCM" "XCMPlus")

for cnn in "${cnns[@]}"; do
   echo "Training CNNs: $cnn"
   python3 results_processing.py "$cnn"
done
