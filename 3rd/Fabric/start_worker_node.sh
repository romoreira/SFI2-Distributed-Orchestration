#!/bin/bash

ip=$1
token=$2
cert_hash=$3

{

echo ${ip}

yes | sudo kubeadm reset

echo "Trying to Join"
sudo kubeadm join ${ip}:6443 --token ${token} --discovery-token-ca-cert-hash ${cert_hash}

}  2>&1 | tee -a start_worker_node.log