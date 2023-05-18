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
* Run cassandra-stress `cassandra-stress write n=1000 cl=ONE -node cassandra -pop dist=UNIFORM\(1..1000000\) -rate threads=10 fixed=500/s -mode native cql3 protocolVersion=3`
* Other option `cassandra-stress mixed ratio\(write=50, read=50\) cl=ONE duration=5m -node cassandra -pop dist=UNIFORM\(1..1000000\) -rate threads=10 fixed=500/s -mode native cql3 protocolVersion=3  user=cassandra password=cassandra`

TIP: You need to Write before Read. (Warmup).

### To run cassandra-stress outside container
  * `./cassandra-stress write n=1000 cl=ONE -node 10.0.0.4`

  * `sudo apt-get install apt-transport-https gnupg2 -y`
  * `wget -q -O - https://www.apache.org/dist/cassandra/KEYS | sudo  apt-key add -`
  * `sudo sh -c 'echo "deb http://www.apache.org/dist/cassandra/debian 311x main" > /etc/apt/sources.list.d/cassandra.list'`
  * `sudo apt-get install openjdk-8-jdk -y`
  * `sudo apt-get install cassandra -y`
