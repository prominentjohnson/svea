#! /usr/bin/env python3

import ast
import numpy as np

from geometry_msgs.msg import Point
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from visualization_msgs.msg import Marker

from svea_core.interfaces import LocalizationInterface
from svea_core.controllers.pure_pursuit import PurePursuitController
from svea_core.interfaces import ActuationInterface
from svea_core import rosonic as rx
from svea_core.utils import PlaceMarker, ShowPath


class pure_pursuit(rx.Node):  # Inherit from rx.Node

    r"""Pure Pursuit example script for SVEA.

    #**Background**

    This script implements a simple Pure Pursuit controller that follows a
    predefined path. The path is defined by a set of points, and the controller
    computes the steering angle and velocity to follow the path.

    The script also includes visualization of the goal and the path being
    followed.

    #**Preparation**

    TODO: Add instructions for setting up the teleoperation environment.

    #**Simulation**

    To run the Pure Pursuit example in simulation, you can use the following command:
    ```bash
    ros2 launch svea_examples floor2.xml is_sim:=true
    ```
    This launch file includes the following components, with example parameters:

        # Initial state of the robot (x, y, yaw, velocity)
        state:=[-7.4, -15.3, 0.9, 0.0] 
        # Points defining the path to follow. Each point is a string representation of a list.
        points:=['[-2.3,-7.1]','[10.5,11.7]','[5.7,15.0]','[-7.0,-4.0]'] 

    Attributes:
        points: List of points defining the path to follow.
        actuation: Actuation interface for sending control commands.
        localizer: Localization interface for receiving state information.
        mark: PlaceMarker for visualizing the goal.
        path: ShowPath for visualizing the path.
    """

    DELTA_TIME = 0.1
    TRAJ_LEN = 20
    OBSTACLE_MARKER_TOPIC = '/marker/obstacles'

    points = rx.Parameter([
        '[-2.3, -7.1]',
        '[-0.166667, -3.966667]',
        '[1.966667, -0.833333]',
        '[4.1, 2.3]',
        '[6.233333, 5.433333]',
        '[8.366667, 8.566667]',
        '[10.5, 11.7]',
        '[8.9, 12.8]',
        '[7.3, 13.9]',
        '[5.7, 15.0]',
        '[3.583333, 11.833333]',
        '[1.466667, 8.666667]',
        '[-0.65, 5.5]',
        '[-2.766667, 2.333333]',
        '[-4.883333, -0.833333]',
        '[-7.0, -4.0]',
        '[-5.433333, -5.033333]',
        '[-3.866667, -6.066667]'
    ])
    obstacle_points = rx.Parameter([
        # '[-1.2, -5.5]',
        # '[2.8, 1.2]',
        # '[7.8, 10.4]',
    ])
    stop_distance = rx.Parameter(1.2)
    parking_spot = rx.Parameter(['[-3.5, -9.0, 0.9]'])
    parking_approach_distance = rx.Parameter(1.5)
    parking_distance_tolerance = rx.Parameter(0.35)
    parking_angle_tolerance = rx.Parameter(0.35)
    parking_forward_velocity = rx.Parameter(0.35)
    parking_reverse_velocity = rx.Parameter(-0.2)
    state = rx.Parameter([-7.4, -15.3, 0.9, 0.0])  # x, y, yaw, vel
    target_velocity = rx.Parameter(0.6)
    
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    # Goal Visualization
    mark = PlaceMarker()
    # Path Visualization
    #path = ShowPath()

    def on_startup(self):
        """
        Initialize the Pure Pursuit controller and set up the path and goal.
        Controller is initialized with the target velocity and the points
        provided in the parameters. The current state is obtained from the
        localization interface, and the goal is set to the first point in the
        path.
        The trajectory is updated based on the current state and the goal.
        The controller is set to not finished initially, and a timer is created
        to call the loop method at regular intervals.
        """
        # Convert POINTS to numerical lists if loaded as strings
        if isinstance(self.points[0], str):
            self._points = [ast.literal_eval(point) for point in self.points]
        else:
            self._points = self.points

        if len(self.obstacle_points) > 0 and isinstance(self.obstacle_points[0], str):
            self._obstacle_points = [ast.literal_eval(point) for point in self.obstacle_points]
        else:
            self._obstacle_points = self.obstacle_points

        if isinstance(self.parking_spot[0], str):
            self._parking_spot = ast.literal_eval(self.parking_spot[0])
        else:
            self._parking_spot = self.parking_spot

        self.obstacle_marker_pub = self.create_publisher(
            Marker,
            self.OBSTACLE_MARKER_TOPIC,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self._estop_active = False
        self._parking_mode = False
        self._parked = False
        self._parking_logged = False
        self._parking_waypoints = []
        self._parking_index = 0
        self._parking_reverse_start = 0

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.target_velocity

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.curr = 0
        self.goal = self._points[self.curr]
        self.mark.marker('goal','blue',self.goal)
        self.mark.marker('parking_spot', 'green', self._parking_spot[:2] + [0.2])
        self.mark.marker(
            'parking_heading',
            'green',
            [self._parking_spot[0], self._parking_spot[1], 0.35],
            orientation=[0.0, 0.0, self._parking_spot[2]],
            shape=Marker.ARROW,
        )
        self.publish_obstacle_markers()
        self.update_traj(x, y)

        self.create_timer(self.DELTA_TIME, self.loop)

    def loop(self):
        """
        Main loop of the Pure Pursuit controller. It retrieves the current state
        from the localization interface, computes the steering and velocity
        commands using the controller, and sends these commands to the actuation
        interface.
        If the controller has finished following the path, it updates the goal
        and trajectory based on the next point in the path.
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if self._parked:
            self.actuation.send_control(0.0, 0.0)
            return

        if self.controller.is_finished:
            if self._parking_mode:
                self.advance_parking_goal(x, y)
            else:
                if self.curr < len(self._points) - 1:
                    self.update_goal()
                    self.update_traj(x, y)
                else:
                    self.start_parking(x, y)

        min_distance = self.get_min_obstacle_distance(x, y)
        if min_distance <= self.stop_distance:
            if not self._estop_active:
                self.get_logger().warn(
                    f"Emergency stop triggered: nearest obstacle at {min_distance:.2f} m "
                    f"(threshold {self.stop_distance:.2f} m)"
                )
            self._estop_active = True
            self.actuation.send_control(0.0, 0.0)
            return

        if self._estop_active:
            self.get_logger().info(
                f"Emergency stop released: nearest obstacle at {min_distance:.2f} m"
            )
        self._estop_active = False

        if self._parking_mode and self.is_at_parking_spot(x, y, yaw):
            if not self._parking_logged:
                self.get_logger().info("Parking complete: vehicle stopped in designated parking spot")
                self._parking_logged = True
            self._parked = True
            self.controller.is_finished = True
            self.actuation.send_control(0.0, 0.0)
            return

        steering, velocity = self.controller.compute_control(state)
        if self._parking_mode:
            if self._parking_index < self._parking_reverse_start:
                velocity = min(velocity, self.parking_forward_velocity)
            else:
                velocity = self.parking_reverse_velocity
        self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        self.actuation.send_control(steering, velocity)

    def publish_obstacle_markers(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pure_pursuit_obstacles'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.points = [
            Point(x=float(x), y=float(y), z=0.4)
            for x, y in self._obstacle_points
        ]
        self.obstacle_marker_pub.publish(marker)

    def get_min_obstacle_distance(self, x, y):
        if len(self._obstacle_points) == 0:
            return np.inf

        distances = [
            np.hypot(float(obstacle_x) - x, float(obstacle_y) - y)
            for obstacle_x, obstacle_y in self._obstacle_points
        ]
        return min(distances)

    def update_goal(self):
        """
        Update the goal to the next point in the path. If the end of the path
        is reached, it wraps around to the beginning. The current index is
        incremented, and the goal marker is updated.
        """
        self.curr += 1
        self.goal = self._points[self.curr]

        self.controller.is_finished = False
        # Mark the goal
        self.mark.marker('goal','blue',self.goal)

    def update_traj(self, x, y):
        """
        Update the trajectory based on the current state and the goal. It
        generates a linear trajectory from the current position to the goal
        position, and updates the controller's trajectory points.
        The trajectory is visualized using the ShowPath interface.
        """
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.controller.traj_x = xs
        self.controller.traj_y = ys
        #self.path.publish_path(xs,ys)

    def start_parking(self, x, y):
        self._parking_mode = True
        px, py, pyaw = self._parking_spot
        # Use a short sequence of waypoints on the parking heading line:
        # line up in front of the spot, move forward a little more twice,
        # then reverse straight into the final parking point.
        front_far_x = px + 2 * self.parking_approach_distance * np.cos(pyaw)
        front_far_y = py + 2 * self.parking_approach_distance * np.sin(pyaw)
        front_mid_x = px + self.parking_approach_distance * np.cos(pyaw)
        front_mid_y = py + self.parking_approach_distance * np.sin(pyaw)
        front_near_x = px + 0.5 * self.parking_approach_distance * np.cos(pyaw)
        front_near_y = py + 0.5 * self.parking_approach_distance * np.sin(pyaw)

        self._parking_waypoints = [
            [float(front_mid_x), float(front_mid_y)],
            [float(front_far_x), float(front_far_y)],
            [float(front_near_x), float(front_near_y)],
            [float(px), float(py)],
        ]
        self._parking_index = 0
        self._parking_reverse_start = 2
        self.goal = self._parking_waypoints[self._parking_index]
        self.controller.is_finished = False
        self.controller.target_velocity = self.parking_forward_velocity
        self.mark.marker('goal', 'blue', self.goal)
        self.update_traj(x, y)
        self.get_logger().info(
            f"Switching to parking maneuver toward ({px:.2f}, {py:.2f}, yaw={pyaw:.2f})"
        )

    def is_at_parking_spot(self, x, y, yaw):
        px, py, pyaw = self._parking_spot
        position_error = np.hypot(px - x, py - y)
        angle_error = self.wrap_angle(pyaw - yaw)
        return (
            position_error <= self.parking_distance_tolerance
            and abs(angle_error) <= self.parking_angle_tolerance
        )

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def advance_parking_goal(self, x, y):
        if self._parking_index >= len(self._parking_waypoints) - 1:
            self.controller.is_finished = False
            return

        self._parking_index += 1
        self.goal = self._parking_waypoints[self._parking_index]
        self.controller.is_finished = False
        if self._parking_index >= self._parking_reverse_start:
            self.controller.target_velocity = self.parking_reverse_velocity
        else:
            self.controller.target_velocity = self.parking_forward_velocity
        self.mark.marker('goal', 'blue', self.goal)
        self.update_traj(x, y)

if __name__ == '__main__':
    pure_pursuit.main()
