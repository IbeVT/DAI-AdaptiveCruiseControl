#!/bin/sh

script_pida=$(python3 carla__tool/carla_dataset_tools/data_record/data_recorder_a.py&)

sleep 10

kill -INT $script_pida

script_pidb=$(python3 carla__tool/carla_dataset_tools/data_record/data_recorder_b.py &)


sleep  10

kill -INT $script_pidb
