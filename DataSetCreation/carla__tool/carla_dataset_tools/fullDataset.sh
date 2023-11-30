#!/bin/sh

script_pid=$(python3 data_recorder_a.py -w worlds/world1.json &)
sleep 20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_b.py -w worlds/world2.json &)
sleep  20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_c.py -w worlds/world3.json &)
sleep  20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_d.py -w worlds/world4.json &)
sleep  20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_e.py -w worlds/world5.json &)
sleep  20
kill -INT $script_pid
sleep 10

script_pid=$(python3 data_recorder_e.py -w worlds/world1.json &)
sleep  20
kill -INT $script_pid
sleep 10

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

