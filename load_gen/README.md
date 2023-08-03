# Cassandra Load Generator (Cassandra-Stress) on Kubernetes (FIBRE-NG)

This project aims to deploy a Cassandra Load Generator on Kubernetes. The Cassandra Load Generator is based on [Cassandra-Stress](https://cassandra.apache.org/doc/latest/cassandra/tools/cassandra_stress.html), and it is deployed on Kubernetes using a personalized Docker [image](https://hub.docker.com/repository/docker/moreirar/loadgen/general).

:rocket: The rationale of this project is to simulate a real workload into a near to real Cassandra cluster spread accross multiple locations (datacenters) in Brazilian Academic Network (FIBRE-NG).


### 1. To create a Docker image of Cassandra-Stress :compass:
* Edit any feature of Docker image `vim Dockerfile`
* Build the Docker image `docker build -t moreirar/loadgen .`
* To Upload the image to hub.docker.com, is nescessary singin `docker login`
* Upload the image `docker push moreirar/loadgen`
* If you want to test the image run: `docker container -d -it --name stress moreirar/loadgen:latest bash`

---

## 2. To run Cassandra Stress towards Cassandra Ring :eye_speech_bubble:
* Deploy cassandra-loadgen on Kubernetes `kubectl create -f stress.yaml`
* Open Pod bash `kubectl exec -it loadgen -- /bin/bash`
* Run cassandra-stress `kubectl exec -it loadgen -- /bin/bash`
* Run cassandra-stress `cassandra-stress write n=1000000 cl=ONE -log file=write_output.txt -node cassandra -pop dist=UNIFORM\(1..1000000\) -rate threads=10 fixed=500/s -mode native cql3 protocolVersion=3`
* Other option `cassandra-stress mixed ratio\(write=50, read=50\) cl=ONE duration=5m -log file=mixed_output.txt -node cassandra -pop dist=UNIFORM\(1..1000000\) -rate threads=10 fixed=500/s -mode native cql3 protocolVersion=3  user=cassandra password=cassandra`
* Changing the replica factor `cassandra-stress write n=500000 no-warmup -log file=write_output.txt -node cassandra -schema "replication(strategy=SimpleStrategy, factor=2)" -mode native cql3 protocolVersion=3`
* Reading operations: `cassandra-stress read duration=60m -node cassandra -mode native cql3 protocolVersion=3`


> TIP: You need to Write before Read. (Warmup). :thermometer:  :fire:

---

### 3. Parameters of cassandra-stress command in Python :hotsprings:
* Warmup: `cassandra-stress write n=500000 -log file=write_output.txt -node cassandra -schema "replication(strategy=SimpleStrategy, factor=2)" -mode native cql3 protocolVersion=3`
* Sinusuidal: `python3 cassandra_loadgen.py --logfile output.txt 50 25 -s 20,10 --poisson`
* Flashcrowd: `python3 load_gen_v2.py --logfile output.txt 50 100`

---

### 4. To run cassandra-stress outside container
  * `./cassandra-stress write n=1000 cl=ONE -node 10.0.0.4`

  * `sudo apt-get install apt-transport-https gnupg2 -y`
  * `wget -q -O - https://www.apache.org/dist/cassandra/KEYS | sudo  apt-key add -`
  * `sudo sh -c 'echo "deb http://www.apache.org/dist/cassandra/debian 311x main" > /etc/apt/sources.list.d/cassandra.list'`
  * `sudo apt-get install openjdk-8-jdk -y`
  * `sudo apt-get install cassandra -y`

---