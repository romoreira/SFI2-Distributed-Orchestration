FROM ubuntu:20.04
RUN \
  apt-get update -y \
  && apt-get install -y && apt-get install default-jdk default-jre apt-transport-https wget gnupg2 -y

RUN echo "deb https://debian.cassandra.apache.org 41x main" | tee -a /etc/apt/sources.list.d/cassandra.sources.list

ENV TZ=America/Sao_Paulo
RUN wget -q -O - https://www.apache.org/dist/cassandra/KEYS | apt-key add -
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN \
  apt-get update -y \
  && apt-get install cassandra -y
RUN \
  apt-get install iputils-ping -y

RUN \
  apt-get install vim -y

RUN \
  apt-get install iproute2 -y

RUN \
  apt-get install net-tools -y

RUN \
  apt-get install curl -y

ADD  cassandra_loadgen.py /home/cassandra_loadgen.py
