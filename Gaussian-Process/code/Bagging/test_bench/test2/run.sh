#!/bin/bash
./clear.sh
dir=`pwd`;
echo "workdir $dir" >> conf
cat tmp.conf >> conf
# nohup gpmoog $dir/conf > log 2>&1 &
nohup mobo $dir/conf > log 2>&1 &
