# ComputerVisionTest
This branch contains a test scenario to visualize the performance of the computer vision module. It allows to let a car drive using autopilot and view the processed data of the computer vision part. It draws every frame of the camera on the screen, draws all detected bounding boxes and the distance to the vehicle that is being followed, the speed of that vehicle and its class. The bounding box of the vehicle that is being followed is drawn in green. The other bounding boxes with a high probability are drawn in blue, while the ones with a low probability are drawn in red.

To run the script, you need to run the command:
`python3 TestComputerVision.py -t x`
With x the number of the testcase (1, 2 or 3). This indicates which predifined route will be followed and where the cars will be spawned. The models of the cars are chosen randomly.

