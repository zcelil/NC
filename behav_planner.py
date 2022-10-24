#!/usr/bin/env python3

"""
import debugpy

debugpy.listen(("0.0.0.0", 5555))
print("Waiting for client to attach...")
debugpy.wait_for_client()
"""

import numpy as np
import rospy
import math
import time
from state_estimator.msg import current_state
from visual_perception.msg import RouteInGlobal2D, TrafficSignStates, TrafficSignGlobLoc,WaypointsDone
from motion_planner.msg import CarBehaviour, GoalIndexAndState
from cart_sim.msg import BoldPilotEnabled
import message_filters
from math import sin, cos, pi
from cmath import isnan
# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
DECELERATE_TO_BUS_STOP = 3
TURN_RIGHT = 4
TURN_LEFT = 5
BUS_STOP = 6
STOP_AT_RED = 7
DECELERATE_TO_RED_STOP = 8
STAY_AT_RED = 9
MOVE_FORWARD = 10
GO_TO_PARK_SLOT = 11
DECELERATE_TO_PARK_STOP = 12
STAY_AT_PARK = 13
SWITCH_TO_LEFT_LANE = 14
COLLISION_DETECTED = 15
SWITCH_TO_RIGHT_LANE = 16
AVOID_FROM_COLLISION = 17
LEAVE_BUS_STOP = 18
IDLE_STATE = 19

# Stop speed threshold
STOP_THRESHOLD = 0.002 * 3.6   # 0.0072
# Number of cycles before moving from stop sign.

STOP_COUNTS = 30 * 20


class BehaviouralPlanner:
	def __init__(self, lookahead):
		self._lookahead                     = lookahead
		self._state                         = IDLE_STATE
		self._goal_state                    = [0.0, 0.0, 0.0]
		self._goal_index                    = 0
		self._stop_count                    = 0
		self.stop_sign_found                = False
		self.durak_sign_found               = False
		self.İLERİDEN_SAĞA_MECBURİ_YÖN_sign_found = False
		self.İLERİDEN_SOLA_MECBURİ_YÖN_sign_found = False
		self.ILERI_veya_SAGA_MECBURI_YON_sign_found = False
		self.ILERI_veya_SOLA_MECBURI_YON_sign_found = False
		self.SOLA_DÖNÜLMEZ_sign_found = False
		self.SAĞA_DÖNÜLMEZ_sign_found = False
		self.PARK_YASAK_sign_found = False
		self.PARK_YERI_sign_found = False
		self.GİRİŞ_OLMAYAN_YOL_sign_found = False
		self.TASIT_GIREMEZ_sign_found = False
		self.RED_LIGHT_found = False
		self.GREEN_LIGHT_found = False
		self.bus_stop_location = np.zeros((1,2))
		self.red_stop_location = np.zeros((1,2))
		self.stop_location     = np.zeros((1,2))
		self.park_location     = np.zeros((1,2))
		self.dist_to_bus_stop = 0.0
		self.dist_to_red_stop = 0.0
		self.prev_loc_x = 0
		self.prev_loc_y = 0
		self.prev_last_lane_x = 0
		self.prev_last_lane_y = 0
		self.prev_loc_red_x = 0
		self.prev_loc_red_y = 0
		self.leave_bus_stop_goal_state = []
		self.prev_frame_time = 0   # used to record the time when we processed last frame
		self.new_frame_time = 0    # used to record the time at which we processed current frame
		self._switching_count = 0
		self.flag_spec = 0
		self.waypoints_done = False

		self.waypoints_sub = message_filters.Subscriber("route_in_global_2D", RouteInGlobal2D)
		self.current_state_sub = message_filters.Subscriber("current_state", current_state)
		#self.waypoints_done_sub = message_filters.Subscriber("waypoints_done", WaypointsDone)
		#self.current_state_sub = rospy.Subscriber('/route_in_global_2D',RouteInGlobal2D, self.assign_route_in_global_2D)
		#self.current_state_sub = rospy.Subscriber('/current_state',current_state, self.assign_current_state)
		self.ts = message_filters.ApproximateTimeSynchronizer([self.waypoints_sub, self.current_state_sub], 1, 0.1)
		self.ts.registerCallback( self.transition_state )

		self.waypoints_done_sub = rospy.Subscriber('/waypoints_done', WaypointsDone, self.assign_waypoints_done)
		self.traffic_sign_states_sub = rospy.Subscriber('/traffic_sign_states',TrafficSignStates, self.assign_traffic_sign_states)
		self.traffic_sign_glob_loc_sub = rospy.Subscriber('/traffic_sign_glob_loc',TrafficSignGlobLoc, self.assign_traffic_sign_glob_loc)
		self.car_behaviour_pub = rospy.Publisher("car_behaviour", CarBehaviour, queue_size=1)
		self.goal_index_and_state_pub = rospy.Publisher("goal_index_and_state", GoalIndexAndState, queue_size=1)
		self.boldi_enable_pub = rospy.Publisher('bold_pilot_enabled', BoldPilotEnabled, queue_size=1)



	def set_lookahead(self, lookahead):
		self._lookahead = lookahead

	######################################################
	######################################################
	# MODULE 7: TRANSITION STATE FUNCTION
	#   Read over the function comments to familiarize yourself with the
	#   arguments and necessary internal variables to set. Then follow the TODOs
	#   and use the surrounding comments as a guide.
	######################################################
	######################################################
	# Handles state transitions and computes the goal state.

	def assign_waypoints_done(self, msg):
		self.waypoints_done = msg.waypoints_done
		print("waypoints_done", self.waypoints_done)


	def assign_traffic_sign_states(self, tfsign_states):
		self.DURAK_sign_found = tfsign_states.DURAK_sign_found
		self.ILERIDEN_SAGA_MECBURI_YON_sign_found = tfsign_states.ILERIDEN_SAGA_MECBURI_YON_sign_found
		self.ILERIDEN_SOLA_MECBURI_YON_sign_found = tfsign_states.ILERIDEN_SOLA_MECBURI_YON_sign_found
		self.ILERI_veya_SOLA_MECBURI_YON_sign_found = tfsign_states.ILERI_veya_SOLA_MECBURI_YON_sign_found
		self.ILERI_veya_SAGA_MECBURI_YON_sign_found = tfsign_states.ILERI_veya_SAGA_MECBURI_YON_sign_found
		self.SOLA_DONULMEZ_sign_found = tfsign_states.SOLA_DONULMEZ_sign_found
		self.SAGA_DONULMEZ_sign_found = tfsign_states.SAGA_DONULMEZ_sign_found
		self.GIRIS_OLMAYAN_YOL_sign_found = tfsign_states.GIRIS_OLMAYAN_YOL_sign_found
		self.TASIT_GIREMEZ_sign_found = tfsign_states.TASIT_GIREMEZ_sign_found
		self.RED_LIGHT_found = tfsign_states.RED_LIGHT_found
		self.GREEN_LIGHT_found = tfsign_states.GREEN_LIGHT_found
		self.PARK_YERI_sign_found = tfsign_states.PARK_YERI_sign_found
		self.collision_found = tfsign_states.collision_found
		self.collision_alert = tfsign_states.collision_alert

		print("durak_sign_found:", self.DURAK_sign_found)
		print("park yeri sign found:", self.PARK_YERI_sign_found)

	def assign_traffic_sign_glob_loc(self, tfsign_glob_loc):
		self.red_stop_location_x = tfsign_glob_loc.red_stop_location_x
		self.red_stop_location_y = tfsign_glob_loc.red_stop_location_y
		self.bus_stop_location_x = tfsign_glob_loc.bus_stop_location_x
		self.bus_stop_location_y = tfsign_glob_loc.bus_stop_location_y
		self.park_location_x     = tfsign_glob_loc.park_location_x
		self.park_location_y     = tfsign_glob_loc.park_location_y
		self.park_slot_entry_x   = tfsign_glob_loc.park_slot_entry_x
		self.park_slot_entry_y   = tfsign_glob_loc.park_slot_entry_y
		self.avoidance_goal_x    = tfsign_glob_loc.avoidance_goal_x
		self.avoidance_goal_y    = tfsign_glob_loc.avoidance_goal_y
		self.avoidance_goal_x_switch_right = tfsign_glob_loc.avoidance_goal_x_switch_right
		self.avoidance_goal_y_switch_right = tfsign_glob_loc.avoidance_goal_y_switch_right
		self.last_lane_after_bus_stop_glob_x = tfsign_glob_loc.last_lane_after_bus_stop_glob_x
		self.last_lane_after_bus_stop_glob_y = tfsign_glob_loc.last_lane_after_bus_stop_glob_y


	def transition_state(self, waypoints, ego_state):
		"""Handles state transitions and computes the goal state.  
		
		args:
			waypoints: current waypoints to track (global frame). 
				length and speed in m and m/s.
				(includes speed to track at each x,y location.)
				format: [[x0, y0, v0],
						 [x1, y1, v1],
						 ...
						 [xn, yn, vn]]
				example:
					waypoints[2][1]: 
					returns the 3rd waypoint's y position

					waypoints[5]:
					returns [x5, y5, v5] (6th waypoint)
			ego_state: ego state vector for the vehicle. (global frame)
				format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
					ego_x and ego_y     : position (m)
					ego_yaw             : top-down orientation [-pi to pi]
					ego_open_loop_speed : open loop speed (m/s)
			closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
		variables to set:
			self._goal_index: Goal index for the vehicle to reach
				i.e. waypoints[self._goal_index] gives the goal waypoint
			self._goal_state: Goal state for the vehicle to reach (global frame)
				format: [x_goal, y_goal, v_goal]
			self._state: The current state of the vehicle.
				available states: 
					FOLLOW_LANE         : Follow the global waypoints (lane).
					DECELERATE_TO_STOP  : Decelerate to stop.
					STAY_STOPPED        : Stay stopped.
			self._stop_count: Counter used to count the number of cycles which
				the vehicle was in the STAY_STOPPED state so far.
		useful_constants:
			STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
							  stop when its speed falls within this threshold.
			STOP_COUNTS     : Number of cycles (simulation iterations) 
							  before moving from stop sign.
		"""


		goal_index_and_goal_state  = GoalIndexAndState()
		car_behaviour_pb           = CarBehaviour()
		print("transition state started")


		# In this state, continue tracking the lane by finding the
		# goal index in the waypoint list that is within the lookahead
		# distance. Then, check to see if the waypoint path intersects
		# with any stop lines. If it does, then ensure that the goal
		# state enforces the car to be stopped before the stop line.
		# You should use the get_closest_index(), get_goal_index(), and
		# check_for_stop_signs() helper functions.
		# Make sure that get_closest_index() and get_goal_index() functions are
		# complete, and examine the check_for_stop_signs() function to
		# understand it.
		if self._state == IDLE_STATE:
			goal_index = 0
			x_global = ego_state.current_x + (5.0)*cos(ego_state.current_yaw)
			y_global = ego_state.current_y + (5.0)*sin(ego_state.current_yaw)
			goal_state = [x_global, y_global, 0.5/3.6]
			print("flag spec", self.flag_spec)
			
			if self.waypoints_done:
				self._state = FOLLOW_LANE
				self.flag_spec = 1
			else:
				if self.flag_spec:
					self._state = FOLLOW_LANE
				else:
					if self.collision_found:	
						self._state = COLLISION_DETECTED
					else:
						self._state = IDLE_STATE		
			
			pass
		
		elif self._state == FOLLOW_LANE:
			print("Car state: FOLLOW LANE")

			#car_behaviour_pb.state = FOLLOW_LANE
			# First, find the closest index to the ego vehicle.
			# TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
			# ------------------------------------------------------------------
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			# ------------------------------------------------------------------
			# Next, find the goal index that lies within the lookahead distance
			# along the waypoints.
			# TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
			# ------------------------------------------------------------------
			goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_state = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)
			#print("goal_state:", goal_state)
			# ------------------------------------------------------------------
			# Finally, check the index set between closest_index and goal_index
			# for stop signs, and compute the goal state accordingly.
			# TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
			# ------------------------------------------------------------------
			#goal_index, durak_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)  ##Commented out because no need to check stop signs yet
			#self._goal_index = goal_index
			#self._goal_state = waypoints[self._goal_index]

			#print("waypoints in BehaviouralPlanner:\n", waypoints)
			#print("Goal_state from BehaviouralPlanner:\n", self._goal_state)
			# ------------------------------------------------------------------

			# If stop sign found, set the goal to zero speed, then transition to 
			# the deceleration state.
			# TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
			# ------------------------------------------------------------------
			if self.DURAK_sign_found:
				self._state = BUS_STOP
			
			elif self.ILERIDEN_SAGA_MECBURI_YON_sign_found:
				self._state = TURN_RIGHT

			elif self.ILERIDEN_SOLA_MECBURI_YON_sign_found:
				self._state = SWITCH_TO_LEFT_LANE
				 
				print("switch to left lane:", self.ILERIDEN_SOLA_MECBURI_YON_sign_found)

			elif self.SOLA_DONULMEZ_sign_found:
				self._state = SWITCH_TO_RIGHT_LANE

			elif self.SOLA_DONULMEZ_sign_found and (self.GIRIS_OLMAYAN_YOL_sign_found or self.TASIT_GIREMEZ_sign_found):
				self._state = SWITCH_TO_RIGHT_LANE
			 
			elif self.ILERI_veya_SAGA_MECBURI_YON_sign_found or self.SOLA_DONULMEZ_sign_found :
				self._state = MOVE_FORWARD
							
			elif self.ILERI_veya_SOLA_MECBURI_YON_sign_found or self.SAGA_DONULMEZ_sign_found :
				self._state = MOVE_FORWARD

			elif self.SAGA_DONULMEZ_sign_found and (self.GIRIS_OLMAYAN_YOL_sign_found or self.TASIT_GIREMEZ_sign_found):
				self._state = TURN_LEFT

			elif self.SOLA_DONULMEZ_sign_found and self.SAGA_DONULMEZ_sign_found:
				self._state = MOVE_FORWARD

			elif self.RED_LIGHT_found:
				self._state = STOP_AT_RED

			elif self.GREEN_LIGHT_found:
				self._state = FOLLOW_LANE

			elif self.PARK_YERI_sign_found:
				print("######################################################3gather park yeri sign")
				self._state = GO_TO_PARK_SLOT
				
			elif self.collision_found:
				self._state = COLLISION_DETECTED
			pass

		elif self._state == COLLISION_DETECTED :
			print("avoid from collision")
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			temp = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)
			temp[2] = 0
			goal_state = temp

			"""
			if self.collision_alert:
				self._state = AVOID_FROM_COLLISION
			elif self.collision_found:
				self._state = COLLISION_DETECTED
			else:
				self._state = FOLLOW_LANE
			"""

		elif self._state == AVOID_FROM_COLLISION:

			goal_state = [self.avoidance_goal_x, self.avoidance_goal_y, ego_state.current_speed]
			goal_index = 0
			print("avoidance goal:\n",goal_state)
			self.dist_to_avoidance_goal = np.sqrt((goal_state[0] - ego_state.current_x)**2 + (goal_state[1] - ego_state.current_y)**2)

			print("distance to avoidance goal:",self.dist_to_avoidance_goal)
			if self.dist_to_avoidance_goal < 0.65:
				self._state = SWITCH_TO_RIGHT_LANE
			else:
				self._state = AVOID_FROM_COLLISION

		elif self._state == SWITCH_TO_LEFT_LANE:

			print("switching to left lane")
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_state = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)

			# time when we finish processing for this frame
			self.new_frame_time = time.time()

			print("self.prev_frame_time:", self.prev_frame_time)

			fps = 1/(self.new_frame_time-self.prev_frame_time)
			self.prev_frame_time = self.new_frame_time

			print("self.new_frame_time:", self.new_frame_time)
			# converting the fps into integer
			system_fps = int(fps) + 1
			
			SWITCHING_COUNT_FINAL = 10 * system_fps

			if self._switching_count > SWITCHING_COUNT_FINAL:
				self._state = SWITCH_TO_RIGHT_LANE
				self._switching_count = 0
			else:
				self._state = SWITCH_TO_LEFT_LANE
				self._switching_count += 1

		elif self._state == SWITCH_TO_RIGHT_LANE:


			print("switching to right lane")
			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			#goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			#goal_state = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)
			goal_index = 0
			goal_state = [self.avoidance_goal_x_switch_right, self.avoidance_goal_y_switch_right, ego_state.current_speed]

			# time when we finish processing for this frame
			self.new_frame_time = time.time()

			print("self.prev_frame_time:", self.prev_frame_time)

			fps = 1/(self.new_frame_time-self.prev_frame_time)
			self.prev_frame_time = self.new_frame_time

			print("self.new_frame_time:", self.new_frame_time)
			# converting the fps into integer
			system_fps = int(fps) + 1
			
			SWITCHING_COUNT_FINAL = 20 * system_fps

			print("SWITCHING_COUNT_FINAL:", SWITCHING_COUNT_FINAL)
			print("switching_count: ", self._switching_count)

			if self._switching_count > SWITCHING_COUNT_FINAL:
				self._state = FOLLOW_LANE
				self._switching_count = 0
			else:
				self._state = SWITCH_TO_RIGHT_LANE
				self._switching_count += 1


		elif self._state == GO_TO_PARK_SLOT:
			print("Car state: GO_TO_PARK_SLOT")
			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			#goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

			if isnan(self.park_slot_entry_x) or isnan(self.park_slot_entry_y):
				self.park_slot_entry_x = self.prev_loc_x
				self.park_slot_entry_y = self.prev_loc_y

			goal_speed = 2.5/3.6     #m/s  = 2.5 km/h
			goal_state = [self.park_slot_entry_x, self.park_slot_entry_y, goal_speed]
			self.dist_to_park_slot = np.sqrt((self.park_slot_entry_x - ego_state.current_x)**2 + (self.park_slot_entry_y - ego_state.current_y)**2)
			#print("self._goal_state:", self._goal_state)
			print("self.dist_to_red_stop.", self.dist_to_park_slot)
			print("closed_loop_speed and STOP_THRESHOLD:", ego_state.current_speed, STOP_THRESHOLD)
			if self.dist_to_park_slot > 0.10:
				self._state = GO_TO_PARK_SLOT
				
			else:
				self._state = DECELERATE_TO_PARK_STOP
				
			self.prev_loc_x = self.park_slot_entry_x
			self.prev_loc_y = self.park_slot_entry_y
			pass
			
		elif self._state == DECELERATE_TO_PARK_STOP:
			print("Car state: DECELERATE_TO_PARK_STOP")

			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			#goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

			if isnan(self.park_location_x) or isnan(self.park_location_y):
				self.park_location_x = self.prev_loc_x
				self.park_location_y = self.prev_loc_y
			
			goal_speed = 0     #m/s  
			goal_state = [self.park_location_x, self.park_location_y, goal_speed]
			self.dist_to_park_stop = np.sqrt((self.park_location_x - ego_state.current_x)**2 + (self.park_location_y - ego_state.current_y)**2)
			#print("self._goal_state:", self._goal_state)
			print("self.dist_to_red_stop.", self.dist_to_park_stop)
			print("closed_loop_speed and STOP_THRESHOLD:", ego_state.current_speed, STOP_THRESHOLD)
			if self.dist_to_park_stop > 0.10:
				self._state = DECELERATE_TO_PARK_STOP
				
			else:
				self._state = STAY_AT_PARK
			

			self.park_location_x = self.prev_loc_x
			self.park_location_y = self.prev_loc_y

			pass

		elif self._state == STAY_AT_PARK:
			print("Car state: STAY_AT_PARK")

			goal_state = [self.park_location_x, self.park_location_y, 0]

			# time when we finish processing for this frame
			self.new_frame_time = time.time()

			fps = 1/(self.new_frame_time-self.prev_frame_time)
			self.prev_frame_time = self.new_frame_time
		 
			# converting the fps into integer
			system_fps = int(fps)

			PARK_STOP_COUNTS = 5 * system_fps    

			#The condition below waits for 5 seconds in park, then shuts down the Bold Pilot
			if self._stop_count == PARK_STOP_COUNTS:
				boldi = BoldPilotEnabled()
				boldi.enable = False
				self.boldi_enable_pub.publish(boldi)

				self._stop_count = 0

			# Otherwise, continue counting.
			else:
				
				self._stop_count += 1
				
				pass

		elif self._state == STOP_AT_RED:
			print("Car state: STOP_AT_RED")
			self._state = DECELERATE_TO_RED_STOP
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_state = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)
		
			pass


		elif self._state == DECELERATE_TO_RED_STOP:
			print("Car state: DECELERATE_TO_RED_STOP")

			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			#goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			if isnan(self.red_stop_location_x) or isnan(self.red_stop_location_y) or self.red_stop_location_x == 0 or self.red_stop_location_y == 0:
				goal_state = [self.prev_loc_red_x, self.prev_loc_red_y, 0]
				self.dist_to_bus_stop = np.sqrt((self.prev_loc_red_x - ego_state.current_x)**2 + (self.prev_loc_red_y - ego_state.current_y)**2)
			else:
				goal_state = [self.red_stop_location_x, self.red_stop_location_y, 0]
				self.dist_to_red_stop = np.sqrt((self.red_stop_location_x - ego_state.current_x)**2 + (self.red_stop_location_y - ego_state.current_y)**2)
				self.prev_loc_red_x = self.red_stop_location_x
				self.prev_loc_red_y = self.red_stop_location_y			
			
			goal_index = 0
			#goal_state = [self.red_stop_location_x, self.red_stop_location_y, 0]
			#self.dist_to_red_stop = np.sqrt((self.red_stop_location_x - ego_state.current_x)**2 + (self.red_stop_location_y - ego_state.current_y)**2)
			#print("self._goal_state:", self._goal_state)
			print("self.dist_to_red_stop.", self.dist_to_red_stop)
			print("closed_loop_speed and STOP_THRESHOLD:", ego_state.current_speed, STOP_THRESHOLD)

			if ego_state.current_speed > STOP_THRESHOLD:
				self._state = DECELERATE_TO_RED_STOP
				
			else:
				self._state = STAY_AT_RED
				
			pass

		elif self._state == STAY_AT_RED:
			print("Car state: STAY_AT_RED")
			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = 0
			goal_state = [self.prev_loc_red_x, self.prev_loc_red_y, 0]
			
			if self.GREEN_LIGHT_found:
				self._state = FOLLOW_LANE
				self.RED_LIGHT_found = False    #Behaviour planner has not published to tfsign_states yet, it subscribes to tfsign_states

			pass

		elif self._state == MOVE_FORWARD:
			print("Car state: MOVE_FORWARD")
			print("Determining new trajectory...")
			self._state = FOLLOW_LANE            
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_state = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)
			#print("perceiver.keep_lane_middle:", perceiver.keep_lane_middle)
			
			pass


		elif self._state == TURN_RIGHT:
			
			print("Car state: TURN_RIGHT")
			print("Determining new trajectory...")
			self._state = FOLLOW_LANE
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_state = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)

			pass

		elif self._state == TURN_LEFT:
			print("Car state: TURN_LEFT")                        
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_state = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)
			print("tf_sign state:", self.ILERIDEN_SOLA_MECBURI_YON_sign_found)
			if self.ILERIDEN_SOLA_MECBURI_YON_sign_found == False or (self.SAGA_DONULMEZ_sign_found and self.GIRIS_OLMAYAN_YOL_sign_found)==False:
				self._state = FOLLOW_LANE

			pass

		elif self._state == BUS_STOP:
			print("Car state: BUS_STOP")
			self._state = DECELERATE_TO_BUS_STOP
			closest_len, closest_index = get_closest_index(waypoints, ego_state)
			goal_index = 0
			goal_state = [self.bus_stop_location_x, self.bus_stop_location_y, 1.0/3.6]
			#goal_state = waypoints.route_in_global_2d[goal_index]
			#y = list(waypoints.route_in_global_2d[goal_index].route_in_global_1d)
			#y[2] = 1.0
			#goal_state = y

			pass

		elif self._state == DECELERATE_TO_BUS_STOP:
			print("Car state: DECELERATE_TO_BUS_STOP")

			if isnan(self.bus_stop_location_x) or isnan(self.bus_stop_location_y) or self.bus_stop_location_x == 0 or self.bus_stop_location_y == 0:
				goal_state = [self.prev_loc_x, self.prev_loc_y, 0]
				self.dist_to_bus_stop = np.sqrt((self.prev_loc_x - ego_state.current_x)**2 + (self.prev_loc_y - ego_state.current_y)**2)
			else:
				goal_state = [self.bus_stop_location_x, self.bus_stop_location_y, 0]
				self.dist_to_bus_stop = np.sqrt((self.bus_stop_location_x - ego_state.current_x)**2 + (self.bus_stop_location_y - ego_state.current_y)**2)
				self.prev_loc_x = self.bus_stop_location_x
				self.prev_loc_y = self.bus_stop_location_y
			#print("self.bus_stop_location_x", self.bus_stop_location_x) 
			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			#goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_index = 0
		
			#print("self._goal_state:", self._goal_state)
			print("self.dist_to_bus_stop.", self.dist_to_bus_stop)
			print("closed_loop_speed and STOP_THRESHOLD:", ego_state.current_speed, STOP_THRESHOLD)
			#print("self.prev_loc_x", self.prev_loc_x)


			if self.dist_to_bus_stop > 1.5: # 0.2
				self._state = DECELERATE_TO_BUS_STOP
			else:
				self._state = STAY_STOPPED

			if isnan(self.last_lane_after_bus_stop_glob_x) or isnan(self.last_lane_after_bus_stop_glob_y) or self.last_lane_after_bus_stop_glob_x == 0 or self.last_lane_after_bus_stop_glob_y == 0:
				self.leave_bus_stop_goal_state = [self.prev_last_lane_x, self.prev_last_lane_y, 2.0/3.6]
			
			else:
				self.leave_bus_stop_goal_state = [self.last_lane_after_bus_stop_glob_x, self.last_lane_after_bus_stop_glob_y, 2.0/3.6]
				self.prev_last_lane_x = self.last_lane_after_bus_stop_glob_x
				self.prev_last_lane_y = self.last_lane_after_bus_stop_glob_y


			#print("self_prev_loc_x", self.prev_loc_x)
			pass

		elif self._state == STAY_STOPPED:
			print("Car state: STAY_STOPPED")
			print("Stop_count:", self._stop_count)

			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			#goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
			goal_index = 0
			#if self.DURAK_sign_found:
			goal_state = [self.prev_loc_x, self.prev_loc_y, 0]

			if self._stop_count == STOP_COUNTS:

			#closest_len, closest_index = get_closest_index(waypoints, ego_state)
			#goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

			#stop_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)[1]   #Commented out because no need to check stop signs yet


				print("self.stop_sign_found and self.durak_sign_found:", self.stop_sign_found, self.durak_sign_found)
				if not self.DURAK_sign_found:
					self._state = LEAVE_BUS_STOP
					print("Car is accelerating to follow lane.")
				self._stop_count = 0
				pass

			# Otherwise, continue counting.
			else:

				self._stop_count += 1

			pass

		elif self._state == LEAVE_BUS_STOP:
			print("Car state: LEAVE_BUS_STOP")
			
			goal_index = 0
			#goal_state = self.leave_bus_stop_goal_state
			goal_state = self.leave_bus_stop_goal_state
			last_lane_after_bus_stop_glob = [self.prev_last_lane_x, self.prev_last_lane_y]
			
			if self.SOLA_DONULMEZ_sign_found or self.SAGA_DONULMEZ_sign_found:
				flag = True
			dist_to_last_lane = np.sqrt((self.prev_last_lane_x - ego_state.current_x)**2 + (self.prev_last_lane_y - ego_state.current_y)**2)
			print("distance to last_lane ", dist_to_last_lane)
			print("self.prev_last_lane_x", self.prev_last_lane_x)
			print("self.prev_last_lane_y", self.prev_last_lane_y)

			if dist_to_last_lane < 1.5:
				if flag:
					self._state = TURN_LEFT
				else:
					self._state = FOLLOW_LANE
			else:
				self._state = LEAVE_BUS_STOP	


		else:
			raise ValueError('Invalid state value.')


		goal_index_and_goal_state.goal_index     = goal_index
		goal_index_and_goal_state.state_x        = goal_state[0]
		goal_index_and_goal_state.state_y        = goal_state[1]
		goal_index_and_goal_state.state_velocity = goal_state[2] 

		self.goal_index_and_state_pub.publish(goal_index_and_goal_state)

		car_behaviour_pb.state = self._state
		self.car_behaviour_pub.publish(car_behaviour_pb)  

		######################################################
		######################################################
		# MODULE 7: GET GOAL INDEX FOR VEHICLE
		#   Read over the function comments to familiarize yourself with the
		#   arguments and necessary variables to return. Then follow the TODOs
		#   and use the surrounding comments as a guide.
		######################################################
		######################################################
		# Gets the goal index in the list of waypoints, based on the lookahead and
		# the current ego state. In particular, find the earliest waypoint that has accumulated
		# arc length (including closest_len) that is greater than or equal to self._lookahead.

	def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
		"""Gets the goal index for the vehicle. 
		
		Set to be the earliest waypoint that has accumulated arc length
		accumulated arc length (including closest_len) that is greater than or
		equal to self._lookahead.

		args:
			waypoints: current waypoints to track. (global frame)
				length and speed in m and m/s.
				(includes speed to track at each x,y location.)
				format: [[x0, y0, v0],
						 [x1, y1, v1],
						 ...
						 [xn, yn, vn]]
				example:
					waypoints[2][1]: 
					returns the 3rd waypoint's y position

					waypoints[5]:
					returns [x5, y5, v5] (6th waypoint)
			ego_state: ego state vector for the vehicle. (global frame)
				format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
					ego_x and ego_y     : position (m)
					ego_yaw             : top-down orientation [-pi to pi]
					ego_open_loop_speed : open loop speed (m/s)
			closest_len: length (m) to the closest waypoint from the vehicle.
			closest_index: index of the waypoint which is closest to the vehicle.
				i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
		returns:
			wp_index: Goal index for the vehicle to reach
				i.e. waypoints[wp_index] gives the goal waypoint
		"""
		# Find the farthest point along the path that is within the
		# lookahead distance of the ego vehicle.
		# Take the distance from the ego vehicle to the closest waypoint into
		# consideration.
		arc_length = closest_len
		wp_index = closest_index
		
		# In this case, reaching the closest waypoint is already far enough for
		# the planner.  No need to check additional waypoints.
		#print("arc_length > self._lookahead:", arc_length > self._lookahead)
		if arc_length > self._lookahead:
			return wp_index

		#print("len(waypoints.route_in_global_2d) :", len(waypoints.route_in_global_2d))

		# We are already at the end of the path.
		if wp_index == len(waypoints.route_in_global_2d) - 1:
			return wp_index

		# Otherwise, find our next waypoint.
		# TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
		# ------------------------------------------------------------------
		while wp_index < len(waypoints.route_in_global_2d) - 1:
			# arc_length += math.sqrt(1 + ((waypoints[wp_index+1][1] - waypoints[wp_index][1]) 
			#                     / (waypoints[wp_index+1][0] - waypoints[wp_index][0])) ** 2)
			wp_x_plus_one = waypoints.route_in_global_2d[wp_index + 1].route_in_global_1d[0]
			wp_y_plus_one = waypoints.route_in_global_2d[wp_index + 1].route_in_global_1d[1]
			wp_x          = waypoints.route_in_global_2d[wp_index].route_in_global_1d[0]
			wp_y          = waypoints.route_in_global_2d[wp_index].route_in_global_1d[1]

			arc_length += math.sqrt((wp_x_plus_one - wp_x) ** 2 + (wp_y_plus_one - wp_y) ** 2)
			#print("arc length:", arc_length)
			if self._state == 2 or self._state == 3:
				if arc_length >= self.dist_to_bus_stop:
					wp_index += 1
					break

			
			if self._state == 9 or self._state == 8:
				if arc_length >= self.dist_to_red_stop:
					wp_index += 1
					break

			if arc_length >= self._lookahead:
				wp_index += 1
				break
			else:
				wp_index += 1
			
		# ------------------------------------------------------------------
		#print("wp_index calculated after arc_length:", wp_index, arc_length, self.dist_to_bus_stop)
		#print("wp_index calculated after arc_length:", wp_index, arc_length, self.dist_to_red_stop)
		#print("arc length at the end of while:", arc_length)
		return wp_index
			   
######################################################
######################################################
# MODULE 7: CLOSEST WAYPOINT INDEX TO VEHICLE
#   Read over the function comments to familiarize yourself with the
#   arguments and necessary variables to return. Then follow the TODOs
#   and use the surrounding comments as a guide.
######################################################
######################################################
# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
	"""Gets closest index a given list of waypoints to the vehicle position.

	args:
		waypoints: current waypoints to track. (global frame)
			length and speed in m and m/s.
			(includes speed to track at each x,y location.)
			format: [[x0, y0, v0],
					 [x1, y1, v1],
					 ...
					 [xn, yn, vn]]
			example:
				waypoints[2][1]: 
				returns the 3rd waypoint's y position

				waypoints[5]:
				returns [x5, y5, v5] (6th waypoint)
		ego_state: ego state vector for the vehicle. (global frame)
			format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
				ego_x and ego_y     : position (m)
				ego_yaw             : top-down orientation [-pi to pi]
				ego_open_loop_speed : open loop speed (m/s)

	returns:
		[closest_len, closest_index]:
			closest_len: length (m) to the closest waypoint from the vehicle.
			closest_index: index of the waypoint which is closest to the vehicle.
				i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
	"""
	closest_len = float('Inf')
	closest_index = 0
	# TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
	# ------------------------------------------------------------------
	for i in range(len(waypoints.route_in_global_2d)):
		wp_i_x          = waypoints.route_in_global_2d[i].route_in_global_1d[0]
		wp_i_y          = waypoints.route_in_global_2d[i].route_in_global_1d[1]

		distance = math.sqrt((ego_state.current_x - wp_i_x) ** 2 + (ego_state.current_y - wp_i_y) ** 2)
		if distance <= closest_len:
			closest_len = distance
			closest_index = i
	# ------------------------------------------------------------------
	#print("\n\nClosest waypoint distance from the car: ", closest_len)

	return closest_len, closest_index


if __name__ == '__main__':
	rospy.init_node('behavioural_planner')
	look_ahead = 14.0
	behavioural_planner = BehaviouralPlanner(look_ahead)
	rospy.spin()
