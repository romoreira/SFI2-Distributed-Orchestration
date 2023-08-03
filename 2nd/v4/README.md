This directory contains files to deploy a Cassandra Cluster on top of FIBRE-NG cluster v4

## Prerequisites
- A running FIBRE-NG cluster v4
- A running Kubernetes cluster

### About this repository :octocat:
This repository contains the files to deploy a Cassandra Cluster on top of FIBRE-NG cluster v4. 

Files:
* [`README.md`](https://github.com/romoreira/SFI2-Distributed-Orchestration/blob/main/2nd/v4/README.md) : this file
* [`cassandra-statefulset.yaml`](https://github.com/romoreira/SFI2-Distributed-Orchestration/blob/main/2nd/v4/cassandra-statefulset.yaml) : file to deploy the Cassandra Cluster
* [`cassandra-service.yaml`](https://github.com/romoreira/SFI2-Distributed-Orchestration/blob/main/2nd/v4/service.yaml) : file to deploy the Cassandra Cluster Service
* [`clean.sh`](https://github.com/romoreira/SFI2-Distributed-Orchestration/blob/main/2nd/v4/clean.sh) : script to clean the Cassandra Cluster



---

## Deploying the Cassandra Cluster
- Deploy the Cassandra Cluster
1. ```kubectl apply -f cassandra-statefulset.yaml```
   - Check the status of the Cassandra Cluster
   ```kubectl get statefulset```
2. ```kubectl apply -f cassandra-service.yaml```
   - Check the status of the Cassandra Cluster
   ```kubectl get service```
