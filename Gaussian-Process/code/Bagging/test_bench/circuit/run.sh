#!/bin/bash
./clear.sh
numproc=$1
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "workdir $dir" > $dir/conf
cat $dir/tmp.conf >> $dir/conf
nohup mpiexec -n 1 -f mach deopt ./conf 2> err > log &
# nohup weibo $dir/conf 2> err >log &
# nohup mpiexec -n $numproc apaweibo $dir/conf 2> err >log &
