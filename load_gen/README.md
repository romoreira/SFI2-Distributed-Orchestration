## To create a Docker image of Cassandra-Stress
* Edit any feature of Docker image `vim Dockerfile`
* Build the Docker image `docker build -t moreirar/loadgen .`
* To Upload the image to hub.docker.com, is nescessary singin `docker login`
* Upload the image `docker push moreirar/loadgen`
* If you wanna to test the image run: `docker container -d -it --name stress moreirar/loadgen:latest bash`

## To run Cassandra Stress towards Cassandra Ring
* Deploy cassandra-loadgen on Kubernetes `kubectl create -f stress.yaml`
* Open Pod bash `kubectl exec -it loadgen -- /bin/bash`
* Run cassandra-stress `kubectl exec -it loadgen -- /bin/bash`
