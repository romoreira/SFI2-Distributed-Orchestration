#!/bin/bash
grace=$(kubectl get pod cassandra-0 -o=jsonpath='{.spec.terminationGracePeriodSeconds}') \
  && kubectl delete statefulset cassandra \
  && echo "Sleeping 5 seconds" 1>&2 \
  && sleep 5 \
  && kubectl delete persistentvolumeclaim -l app=cassandra
