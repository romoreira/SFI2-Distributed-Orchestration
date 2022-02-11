# k8sfi2

https://github.com/IBM/Scalable-Cassandra-deployment-on-Kubernetes


  * kubectl create -f cassandra-service.yaml
  * kubectl get svc cassandra
  * kubectl create -f local-volumes.yaml
  * kubectl create -f cassandra-statefulset.yaml
  * kubectl get statefulsets
  * kubectl get pods -o wide
  * kubectl exec -ti cassandra-0 -- nodetool status

  * ./cassandra-stress write n=1000 cl=ONE -node 10.0.0.4
