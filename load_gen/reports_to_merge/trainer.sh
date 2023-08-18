#!/bin/bash

cnns=("FCN" "FCNPlus" "ResNet" "ResNetPlus" "ResCNN" "TCN" "InceptionTime" "InceptionTimePlus" "OmniScaleCNN" "XCM" "XCMPlus")

for cnn in "${cnns[@]}"; do
   echo "Training CNNs: $cnn"
   python3 results_processing.py "$cnn"
done