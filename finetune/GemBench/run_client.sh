cd  /PATH_TO_BRIDGEVLA/finetune/
sudo apt-get install -y jq
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
cd GemBench

seed=$1  # 200,300,400,500,600
epoch=$2  # model epoch number


declare -A TASKVARS
TASKVARS[train]="open_door+0 open_drawer+0 open_drawer+2 close_jar_peract+15 close_jar_peract+16"
TASKVARS[test_l2]="close_jar_peract+3 close_jar_peract+4"
TASKVARS[test_l3]="open_drawer_long+0 open_drawer_long+1 open_drawer_long+2 open_drawer_long+3 open_door2+0 open_drawer2+0 open_drawer3+0 open_drawer+1 close_drawer+0"
TASKVARS[test_l4]="put_items_in_drawer+0 put_items_in_drawer+2 put_items_in_drawer+4"

for split in train test_l2 test_l3 test_l4; do
    for taskvar in ${TASKVARS[$split]}; do
        xvfb-run -a python3 client.py \
            --port 13003  \
            --output_file /PATH_TO_SAVE_RESULT_JSON/model_${epoch}/seed${seed}/${split}/result.json \
            --microstep_data_dir /PATH_TO_TEST_DATA/test_dataset/microsteps/seed${seed} \
            --taskvar "$taskvar"
    done
done

