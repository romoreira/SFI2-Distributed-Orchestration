FROM ubuntu:18.04
RUN \
  apt-get update -y \
  && apt-get install -y && apt-get install default-jdk default-jre apt-transport-https wget gnupg2 -y
RUN sh -c 'echo "deb http://www.apache.org/dist/cassandra/debian 40x main" > /etc/apt/sources.list.d/cassandra.list'

ENV TZ=America/Sao_Paulo
RUN wget -q -O - https://www.apache.org/dist/cassandra/KEYS | apt-key add -
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN \
  apt-get update -y \
  && apt-get install cassandra -y
ADD  cassandra_loadgen.py /home/cassandra_loadgen.py
