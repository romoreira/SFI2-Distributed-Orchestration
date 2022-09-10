# SFI2 Project
## Use Case 1: Cassandra Kubernetes on top of FIBRE-NG


  * kubectl create -f cassandra-service.yaml
  * kubectl get svc cassandra
  * kubectl create -f local-volumes.yaml
  * kubectl create -f cassandra-statefulset.yaml
  * kubectl get statefulsets
  * kubectl get pods -o wide
  * kubectl exec -ti cassandra-0 -- nodetool status

### To run Cassandra Stress follow this [steps](https://github.com/romoreira/k8sfi2/tree/main/load_gen)


