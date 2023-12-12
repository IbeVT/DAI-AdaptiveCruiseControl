#!/bin/sh
echo -n "Running first simulation"
script_pid=$(python3 data_recorder_d.py -w worlds/world2.json &)
sleep 20
echo -n "killing first simulation"
kill -INT $script_pid
sleep 10
echo -n " running second simulation"

script_pid=$(python3 data_recorder_b.py -w worlds/world3.json &)
sleep  20
kill -INT $script_pid
sleep 10
echo -n " running third  simulation"
script_pid=$(python3 data_recorder_e.py -w worlds/world4.json &)
sleep  20
kill -INT $script_pid
sleep 10

echo -n " running 4  simulation"
script_pid=$(python3 data_recorder_d.py -w worlds/world4.json &)
sleep  20
kill -INT $script_pid
sleep 10

echo -n " running 5  simulation"
script_pid=$(python3 data_recorder_b.py -w worlds/world5.json &)
sleep  20
kill -INT $script_pid
sleep 10

echo -n " running 6  simulation"
script_pid=$(python3 data_recorder_e.py -w worlds/world1.json &)
sleep  20
kill -INT $script_pid
sleep 10

echo -n " running 7  simulation"
script_pid=$(python3 data_recorder_d.py -w worlds/world2.json &)
sleep  20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_b.py -w worlds/world3.json &)
sleep  20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_b.py -w worlds/world4.json &)
sleep  20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_a.py -w worlds/world5.json &)
sleep  20
kill -INT $script_pid
sleep 10

