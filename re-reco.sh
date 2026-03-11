RUN=$1
mode=$2

source define_envs.sh

LOGS_FOLDER="${RECO_FOLDER}/re-reco-logs/"
DONE_FILE="${RECO_FOLDER}/run_$RUN/done_files.txt"

mkdir -p $LOGS_FOLDER
mkdir -p ${RECO_FOLDER}/run_$RUN
echo > ${RECO_FOLDER}/run_$RUN/done_files.txt

echo "writing list of files re-done in $DONE_FILE"

for spill_str in $(ls -1 "$PROMPT_RECO_FOLDER/run_$RUN/${RUN}_"*reco.root | awk -F "_" '{print $(NF-1)}'); do

    # Convert spill number safely (leading zeros → decimal)
    spill=$((10#$spill_str))

    # Skip spills divisible by 3
    if (( spill % 3 == 0 )); then
        echo "Skipping spill $spill (divisible by 3)"
        continue
    fi

    echo $RECO_FOLDER/run_$RUN/${RUN}_$(printf "%04d" $((10#$spill)))_reco.root >> $DONE_FILE

    echo "Processing spill $spill"

    bash -c "./process_run.sh $RUN $spill $mode noplots nounpack > $LOGS_FOLDER/log_${RUN}_{$spill}.log 2>&1 &"

    while true; do
        running=$(ps aux | grep "bash -c ./process_run.sh" | grep -v grep | wc -l)
        if (( running < 12 )); then
            break
        fi
        sleep 1
    done

done

sleep 5

echo list of files re-recoed in $DONE_FILE

#source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#cat $DONE_FILE | awk '{print $1 " "}' | tr -d "\n" | xargs hadd -f -k ${RECO_FOLDER}/run_$RUN/run_${RUN}_re-reco_merged.root
