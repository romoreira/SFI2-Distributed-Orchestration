## To run Cassandra Stress towards Cassandra Ring
* Deploy cassandra-loadgen on Kubernetes `kubectl create -f stress.yaml`
* Open Pod bash `kubectl exec -it loadgen -- /bin/bash`
* Run cassandra-stress `kubectl exec -it loadgen -- /bin/bash`
