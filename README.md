# SFI2 Project

This repository contains the code (part) for the SFI2 project. The rationale behind this project is to provide a framework to run experiments (Cassandra) on top of Kubernetes and stress the system with a load generator (Cassandra Stress).

### 0. About this repository :octocat:

This repository contains the following files and directory:

* File
  * [`README.md`](https://github.com/romoreira/SFI2-Distributed-Orchestration/blob/main/README.md): this current file
* Directory
  * [`2nd`](https://github.com/romoreira/SFI2-Distributed-Orchestration/tree/main/2nd) : contains the files to deploy Cassandra on top of Kubernetes and FIBRE-NG and Azure
    * [`v2`](https://github.com/romoreira/SFI2-Distributed-Orchestration/tree/main/2nd/v2): contains the files to deploy Cassandra on top of Kubernetes and Azure.
    * [`v4`](https://github.com/romoreira/SFI2-Distributed-Orchestration/tree/main/2nd/v4): contains the files to deploy Cassandra on top of Kubernetes and FIBRE-NG.
    
---

### 1. Prerequisites :rocket:
1. Deploy Cassandra Kubernetes on top of FIBRE-NG
    * Steps available [here](https://github.com/romoreira/SFI2-Distributed-Orchestration/tree/main/2nd/v4)
2. Deploy Cassandra Stress on top of FIBRE-NG
   * Steps available [here](https://github.com/romoreira/SFI2-Distributed-Orchestration/tree/main/load_gen)
 
