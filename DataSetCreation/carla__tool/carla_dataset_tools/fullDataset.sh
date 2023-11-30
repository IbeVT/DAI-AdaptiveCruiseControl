#!/bin/sh

script_pida=$(python3 data_recorder_a.py &)

sleep 10

kill -INT $script_pida

script_pidb=$(python3 data_recorder_b.py &)


sleep  10

kill -INT $script_pidb
