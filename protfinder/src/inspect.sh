#!/bin/bash

for i in {1..10} 
do
    sleep 2h && (date +"%FT%T") >> /home/users/agrover/aayush2020-internship/myprocess.txt
    ps auxf | grep agrover | grep python  >> /home/users/agrover/aayush2020-internship/myprocess-training.txt
done
