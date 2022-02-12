# SFI2 Project
## Use Case 1: Cassandra Kubernetes on top of RNP Network

https://github.com/IBM/Scalable-Cassandra-deployment-on-Kubernetes


  * kubectl create -f cassandra-service.yaml
  * kubectl get svc cassandra
  * kubectl create -f local-volumes.yaml
  * kubectl create -f cassandra-statefulset.yaml
  * kubectl get statefulsets
  * kubectl get pods -o wide
  * kubectl exec -ti cassandra-0 -- nodetool status

  * ./cassandra-stress write n=1000 cl=ONE -node 10.0.0.4

  * sudo apt-get install apt-transport-https gnupg2 -y
  * wget -q -O - https://www.apache.org/dist/cassandra/KEYS | sudo  apt-key add -
  * sudo sh -c 'echo "deb http://www.apache.org/dist/cassandra/debian 311x main" > /etc/apt/sources.list.d/cassandra.list'
  * sudo apt-get install openjdk-8-jdk -y
  * sudo apt-get install cassandra -y

