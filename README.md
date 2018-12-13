[//]: # (Image References)
[simulator_example]: ./figures/simulator_example.png
[tl_detection_example]: ./figures/traffic_lights_sim.png

## System Integration Project

In this project we implement and integrate three systems that will allow the car to drive safely around a track, namely:

* `waypoint_updater`
    * Computes the waypoints (position, heading and velocity) that the car should follow.
    
* `tl_detector`
    * Traffic light detector that signals the position and state (light colour) of the traffic light ahead of the car (if there is any). These signals are gathered by `waypoint_updater` that will stop the car at the limit of a traffic light in red.
    
* `twist_controller`
    * Estimates the DBW (drive-by-wire) signals (throttle/brake and steering) that the car should execute in order to follow the waypoints computed by `waypoint_updater`.

![Performance example][simulator_example]

### Waypoint Updater

The objective of this node is to estimate the waypoints ahead of the vehicle that should be followed. It takes into account the traffic lights ahead in case the car needs to stop due to the red light being switched on.

This node is subscribed to the following topics:

* `/base_waypoints` that contains the waypoints of the entire path (trajectory) that is to be followed.
* `/current_pose` corresponds to the current car position.
* `/traffic_waypoint` waypoint of the detected red traffic light ahead of the car.

The node publishes the waypoints the car should follow (in the near future) in the `/final_waypoints` topic.

In order to compute the final waypoints, the node performs the following steps:

* Get the car current position.
* Compute the closest waypoint **ahead** of the car.
* Generate waypoints to follow by slicing the base waypoints from the closest waypoint (computed in the previous step) to the farthest waypoints to consider (200 waypoints by default).
    * If the a red traffic light is signalled ahead of the car the velocities of the final waypoints are recomputed and gradually decreased so the car stops at the traffic light limit (provided that light is still red). If the light changes to green again the deceleration is canceled and the car starts accelerating in order to achieve the maximum (pre-set) speed.
    
### Traffic Light Detector
 
The goal of this node is to locate the closest traffic light ahead of the car and, if possible, to classify the current light state. For this task, we have opted for a CNN-based approach that performs both the location (detects the traffic light on the input image) and the classification (what state the detected traffic light is in).

This node is subscribed to the following topics:

* `/base_waypoints` provides the complete list of waypoints for the course.
* `/current_pose` determines the vehicle's location.
* `/image_color` provides an image stream from the car's camera. These images are used to determine the color of upcoming traffic lights.
* `/vehicle/traffic_lights` provides the location of all traffic lights (that the base waypoints pass through).
 
The node publishes the index of the waypoint for nearest upcoming red light's stop line to the `/traffic_waypoint` topic.
 
As a starting point, we use a pre-trained [ssd_inception_v2_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) architecture which, according to various reports, exhibits a good balanced performance between speed and accuracy. We have then fine-tuned the architecture using data from the simulator and from real world driving scenario. Both datasets, available [here](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI), contain raw input images as well as annotations with the traffic light state (red, yellow, green or unknown) and their corresponding bounding boxes. For the fine-tuning process, the model has to be reconfigured to consider 4 classes only.
 
Some example of the performance of the fine-tuned model are shown below.

![Traffic lights detection][tl_detection_example]

Note: Since the car simulator uses a rather outdated tensorflow version, there are problems to successfully load a model trained using a more recent version. In order to solve this version incompatibility, we employ pre-trained models that have been frozen using older tensorflow versions and are available [here](https://github.com/mkoehnke/CarND-Capstone/tree/master/data/traffic_light_detection_model).


### Drive-by-Wire (DBW) Module

The objective of this node is to use various controllers to provide appropriate throttle, brake and steering commands. Once messages are being published to `/final_waypoints`, the vehicle's waypoint follower will publish twist commands to the `/twist_cmd topic`. These commands are published to the following topics:

* `/vehicle/throttle_cmd`
* `/vehicle/brake_cmd`
* `/vehicle/steering_cmd`

In addition, if a safety driver takes over, the PID controller will be swithced offf so it does not mistakenly accumulate error. The DBW status (manual or autonomous) is signalled by the `/vehicle/dbw_enabled` topic.



