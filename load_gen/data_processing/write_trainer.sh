#!/bin/bash

cnns=("OmniScaleCNN" "ResNet" "InceptionTime" "FCN" "FCNPlus" "ResNetPlus" "ResCNN" "TCN" "InceptionTimePlus" "XCM" "XCMPlus")
#cnns=("TCN" "InceptionTimePlus" "XCM" "XCMPlus")

for cnn in "${cnns[@]}"; do
   echo "Training CNNs: $cnn"
   python3 write_neural_tsai.py "$cnn"
done