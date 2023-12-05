#!/bin/bash

{

sudo apt update
sudo apt install -y docker.io apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo sh -c "echo deb https://apt.kubernetes.io/ kubernetes-xenial main > /etc/apt/sources.list.d/kubernetes.list"
#cat /etc/apt/sources.list.d/kubernetes.list

sudo apt update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

sudo tee /etc/sysctl.d/kubernetes.conf <<EOF
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1
net.ipv4.conf.all.forwarding = 1
net.ipv6.conf.default.forwarding = 1
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

sudo sysctl --system

sudo sh -c "echo {                                                  >  /etc/docker/daemon.json"
sudo sh -c 'echo \"exec-opts\": [\"native.cgroupdriver=systemd\"]  >>  /etc/docker/daemon.json'
sudo sh -c "echo }                                                 >>  /etc/docker/daemon.json"


sudo systemctl enable docker
sudo systemctl daemon-reload
sudo systemctl restart docker

}  2>&1 | tee -a config_worker_node.log
