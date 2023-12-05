#!/bin/bash

subnet=$1
ip=$2

{

yes | sudo kubeadm reset

sudo kubeadm init --pod-network-cidr=192.168.0.0/16 --apiserver-advertise-address=${ip}
#sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=${ip}
#sudo kubeadm init --pod-network-cidr=${subnet} --apiserver-advertise-address=${ip}
#sudo kubeadm init --apiserver-advertise-address=${ip}

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/custom-resources.yaml


kubectl get nodes

}  2>&1 | tee -a start_control_plane.log
