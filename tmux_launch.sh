send() {
    tmux send "$1" ENTER;
}

worker () {
    for id in {$1..$2}
    do
       tmux select-pane -t $1 \;
        echo "worker $id is running"\;
        send "cd ~"
        send "git clone https://github.com/xiezhq-hermann/byteps.git"
        send "export DMLC_ROLE=worker"
        send "export NVIDIA_VISIBLE_DEVICES=0,1"
        send "export DMLC_WORKER_ID=$(($id-$1))"
        send "export DMLC_NUM_WORKER=$(($2-$1+1))"
        send "export DMLC_NUM_SERVER=$3"
        send "export DMLC_PS_ROOT_URI=$4"
        send "export DMLC_PS_ROOT_PORT=1234"
        send "export DMLC_INTERFACE=bond0"
#       send "export PS_VERBOSE=2"
        send "bpslaunch python3 byteps/example/pytorch/benchmark_byteps.py 2>&1 | tee event.log"
    done
}

server () {
    for id in {$1..$2}
    do
        tmux select-pane -t $(($id-1)) \;
        echo "server $id is running"\;
        send "export DMLC_NUM_WORKER=$3"
        send "export DMLC_ROLE=server"
        send "export DMLC_NUM_SERVER=$(($2-$1+1))"
        send "export DMLC_PS_ROOT_URI=$4"
        send "export DMLC_PS_ROOT_PORT=1234"
        send "export DMLC_INTERFACE=bond0"
#       send "export PS_VERBOSE=2"
        send "bpslaunch 2>&1 | tee event.log"
    done
}

scheduler () {
    tmux select-pane -t $1 \;
    send "export DMLC_NUM_WORKER=$2"
    send "export DMLC_ROLE=scheduler"
    send "export DMLC_NUM_SERVER=$3"
    send "export DMLC_PS_ROOT_URI=$4"
    send "export DMLC_PS_ROOT_PORT=1234"
#       send "export PS_VERBOSE=2"
    send "bpslaunch 2>&1 | tee event.log"
}



single_machine_exhaust () {
exec='
log_command () {
    echo "$1" | tee -a bench.log
    eval "$1"
    echo "$(tail bench.log | grep "Img/sec per GPU:")"
}

echo "start benchmarking" | tee bench.log
for devices in "0" "0,1"
do
    log_command "export NVIDIA_VISIBLE_DEVICES=$devices"
    declare -A max_batches=( [resnet50]=64 [speech]=64 [transformer]=64 [yolov3]=12 )    
    for model in "resnet50" "speech" "transformer" "yolov3"
    do
        MAX_BATCH=${max_batches[$model]}
        [ "$GPU" == "volta" ] && let MAX_BATCH*=3
        batch=1
        while [[ $batch -le $MAX_BATCH ]] ; do
            log_command "bpslaunch python3 byteps/example/pytorch/benchmark_byteps.py --model $model --num-iters 100 --num-batches-per-iter 10 --batch-size $batch &>> bench.log"
            if [[ $batch -lt 32 ]]; then (( batch += 1 ))
            elif [[ $batch -lt 64 ]]; then (( batch += 2 ))
            else (( batch += 4 )); fi
        done
    done
done
'
    send "echo '$exec' > exec.sh"
    send "GPU=$1 bash exec.sh | tee digest.txt"
}
