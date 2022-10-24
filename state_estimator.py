class StateEstimator(object): 
	def __init__(self, total_steps):
		self.gravity = np.array([0, 0, -9.81])  # gravity
		self.l_jac = np.zeros([9, 6])
		self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
		self.m_jac = np.eye(3)         
		self.h_jac = np.zeros([3,9])
		self.h_jac[:,:3] = np.eye(3)   # measurement model noise jacobian
		self.array_length = total_steps                      #simulation steps
		self.p_est = np.zeros([self.array_length, 3])  # position estimates
		self.imu_est_loc = np.zeros([self.array_length, 3])
		self.v_est = np.zeros([self.array_length, 3])  # velocity estimates
		self.imu_est_speed = np.zeros([self.array_length, 1])
		self.speed_est = np.zeros([self.array_length, 1])
		#self.ground_truth_speed = np.zeros([self.array_length, 1])
		self.q_est = np.zeros([self.array_length, 4])  # orientation estimates as quaternions
		self.r_est_euler = np.zeros([self.array_length, 3])  # orientation estimates as euler angles
		self.r_cov_euler_std = np.zeros([self.array_length, 3])
		self.p_cov = np.zeros([self.array_length, 9, 9])  # covariance matrices at each timestep
		
		## Time Synchronizer ROS APIs

		self.timestamps = np.zeros([self.array_length, 1]) #retrieved data timestamps  bu satır haricinde kullanılmıyor
		self.imu_sub = message_filters.Subscriber("/zed2/zed_node/imu/data", Imu)
		#self.gnss_sub = message_filters.Subscriber("/gps/fix", NavSatFix)
		#self.base_pose_ground_truth_sub = message_filters.Subscriber("/base_pose_ground_truth", Odometry)
			
		#self.ts = message_filters.ApproximateTimeSynchronizer([self.imu_sub, self.gnss_sub, self.base_pose_ground_truth_sub], 1, 0.01)
		#self.ts.registerCallback( self.es_ekf )
		self.current_state_pub = rospy.Publisher('current_state', current_state, queue_size=1) 
		self.gps_enu_pub = rospy.Publisher('gps_enu', gps_enu, queue_size=1) 
		self.ground_truth_converted_pub = rospy.Publisher('ground_truth_converted', ground_truth_converted, queue_size=1)
		## constant variance and sensor location values, they might be adjusted accordingly
		self.var_imu_f = 12  
		self.var_imu_w = 15
		self.var_gnss  = 2  
		self.imu_location = np.array([-0.2, 0.0, 1.82]) ## urdf den girildi.
		self.depth_location = np.array([0.92,0,0.87])
		self.transl_2car  = np.array([-0.92,0,-0.87]) ## urdf den girildi.
		
		# Set initial values.
		#Initial position, velocity and rotation values were acknowledged. The values in arrays were took from Pro_es_ek.f
		lat = np.array(40.789931, dtype = float) 
		lon = np.array(29.508871, dtype = float) 
		#self.p_est[0] = np.array([P(lat,lon)[0], P(lat,lon)[1], 129.9972]) 
		self.gnss_converter = GPS_utils()
		self.gnss_converter.setENUorigin(40.789931, 29.508871, 129.9972)
		gnss_enu = np.array(self.gnss_converter.geo2enu(lat, lon, 129.9972))
		self.map_ref = gnss_enu.reshape(3)
		self.p_est[0] = np.array([0, 0, 0.0]) 
		self.v_est[0] = np.array([0.0,  0.0,  0.0])
		self.q_est[0] = Quaternion(euler=[0, 0, 0]).to_numpy()  ## simulasyon basladıgı anki orientation degeri ne ise onu girdim.

		self.prev_x = 0
		self.prev_y = 0
		self.prev_z = 0

		self.p_cov[0] = np.zeros(9) 		#estimation covariance
		self.gnss_i  = 0					
		self.k = 1							
		self.ground_truth_p = np.zeros([self.array_length, 3])
		self.ground_truth_vel = np.zeros([self.array_length, 3])
		self.ground_truth_rot_quat = np.zeros([self.array_length, 4])
		self.ground_truth_rot_euler = np.zeros([self.array_length, 3])
		self.ground_truth_speed = np.zeros([self.array_length, 1])
		self.gnss_enu = np.zeros([self.array_length, 3])
		self.rot_z_neg90 = np.array([[cos(-pi/2),         sin(-pi/2),        0],
								  [  -sin(-pi/2),         cos(-pi/2),        0],
								  [            0,         0,                 1]])
		
		self.zed_odom_sub = rospy.Subscriber("/zed2/zed_node/odom", Odometry, self.es_ekf)
		self.hz = rostopic.ROSTopicHz(-1)
		
	def es_ekf(self, odom):
		
		#print("odom hz: \n", self.hz.get_hz("/zed2/zed_node/odom"))
		delta_t = 1 / 15                      # returned "None" so hz value written directy (self.hz.get_hz("/zed2/zed_node/odom")[0])
		
		delta_t = 0.01 # data.delta_t
		
		w = np.array([imu_data.angular_velocity.x, imu_data.angular_velocity.y, imu_data.angular_velocity.z])          ## this line is fine 
		f = np.array([imu_data.linear_acceleration.x, imu_data.linear_acceleration.y, imu_data.linear_acceleration.z]) ## this line is fine 
		
		
		#first converting method
		#gnss_enu = np.array([P(gnss_data.latitude, gnss_data.longitude)[0], P(gnss_data.latitude, gnss_data.longitude)[1], gnss_data.altitude])
		#gnss_enu = gnss_enu - self.map_ref
		
		#second converting method
		
		gnss_enu = np.array(self.gnss_converter.geo2enu(gnss_data.latitude, gnss_data.longitude, gnss_data.altitude))
		gnss_enu = gnss_enu.reshape(3)
		gnss_enu = gnss_enu @ self.rot_z_neg90
		#print("gnss_enu:\n", gnss_enu)

		enu_msg = gps_enu()
		enu_msg.east = gnss_enu[0]
		enu_msg.north = gnss_enu[1]
		enu_msg.up = gnss_enu[2]
		self.gps_enu_pub.publish(enu_msg)

		self.gnss_enu[self.k] = np.array([gnss_enu[0], gnss_enu[1], gnss_enu[2]]) ## up degeri 0 dan 129.9972 e cevrildi.

		
		
		rotation_matrix=Quaternion(*self.q_est[self.k-1]).to_mat()
		a = rotation_matrix @ f + self.gravity
		
		
		self.p_est[self.k] = self.p_est[self.k-1] + self.v_est[self.k-1] * delta_t + (delta_t**2/2) * a
		self.v_est[self.k] = self.v_est[self.k-1] + delta_t * a
		self.q_est[self.k] = Quaternion(axis_angle= w * delta_t).quat_mult_right(self.q_est[self.k-1]) ## yukarıdaki birim cevrimine neden olan satır
		print("Iteration            :", self.k)
		#print('P_est value          :', self.p_est[self.k])
		#print('P_ground_truth value :', self.ground_truth_p[self.k])
		#print('V_ground_truth value :', self.ground_truth_vel[self.k])

		# 1.1 Linearize the motion model and compute Jacobians
		# motion model state jacobian
		f_jac = np.eye(9)
		f_jac[:3, 3:6] = np.eye(3) * delta_t
		f_jac[3:6, 6:] = -(rotation_matrix @ skew_symmetric(f.reshape((3,1))))*delta_t
		

		# 2. Propagate uncertainty
		q_km = delta_t **2 * np.diag([self.var_imu_f, self.var_imu_f, self.var_imu_f, self.var_imu_w, self.var_imu_w, self.var_imu_w])
		self.p_cov[self.k] = f_jac.dot(self.p_cov[self.k-1].dot(f_jac.T)) + self.l_jac.dot(q_km.dot(self.l_jac.T))

				
		self.r_est_euler[self.k] = self.convert_qest_to_euler(self.q_est[self.k])
		self.speed_est[self.k] = np.sqrt(self.v_est[self.k][0]**2 + self.v_est[self.k][1]**2 + self.v_est[self.k][2]**2)

		
		#print('yaw_est value in degree   :', self.r_est_euler[self.k][2] * 180/np.pi)
		#print('yaw_ground in degree      :', self.ground_truth_rot_euler[self.k][2] * 180/np.pi)
		
		
		car_state = current_state()
		car_state.current_x = self.p_est[self.k][0] + self.transl_2car[0]
		car_state.current_y = self.p_est[self.k][1] + self.transl_2car[1]
		car_state.current_z = self.p_est[self.k][2] + self.transl_2car[2]
		car_state.current_vel_x = self.v_est[self.k][0]
		car_state.current_vel_y = self.v_est[self.k][1]
		car_state.current_vel_z = self.v_est[self.k][2]
		car_state.current_speed = self.speed_est[self.k]
		car_state.current_yaw = self.r_est_euler[self.k][2]
		#car_state.estimation_step = self.k
		

		if self.gnss_i < self.array_length: 
			#print("\n############Timestamps of the sensors are EQUAL###########\n")
			
			yk = np.array(gnss_enu)
			
			self.p_est[self.k], self.v_est[self.k], self.q_est[self.k], self.p_cov[self.k] = self.measurement_update(self.var_gnss, self.p_cov[self.k], yk, 
																								 self.p_est[self.k], self.v_est[self.k], self.q_est[self.k])
		
			
			self.r_est_euler[self.k] = self.convert_qest_to_euler(self.q_est[self.k])
			p_est = self.p_est[self.k]
			v_est = self.v_est[self.k]
			r_est_euler = self.r_est_euler[self.k]
			sim_step = self.k 
			return  p_est, v_est, r_est_euler, sim_step
			
			self.gnss_i +=1
			self.k +=1
			
			
		else:

			self.r_est_euler[self.k],  = self.convert_qest_to_euler(self.q_est[self.k])
			p_est = self.p_est[self.k]
			v_est = self.v_est[self.k]
			r_est_euler = self.r_est_euler[self.k]
			sim_step = self.k 
			self.gnss_i +=1
			self.k +=1
			#return p_est, v_est, r_est_euler, sim_step

				 
	#### 4. Measurement Update #####################################################################

		################################################################################################
		# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
		# a function for it.
		################################################################################################
	def measurement_update(self, sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
		# 3.1 Compute Kalman Gain

		R = sensor_var * np.eye(3)
		K_k = p_cov_check @ self.h_jac.T @ inv(self.h_jac @ p_cov_check @ self.h_jac.T + R)
		#print("kalman gain:\n", K_k)

		# 3.2 Compute error state
		
		er_state = K_k.dot(y_k - p_check)  #K_k: 9x3, y_k: 3x1, er_state: 9x1

		# 3.3 Correct predicted state
		p_er_state = er_state[:3]
		v_er_state = er_state[3:6]
		q_er_state = er_state[6:]

		p_hat = p_check + p_er_state
		v_hat = v_check + v_er_state
		"""
		print('v_er_state value : ', v_er_state) 
		print('V_hat value : ', v_hat)
		print('----------------------------------------------------') 
		print('----------------------------------------------------')
		"""
		q_hat = Quaternion(euler=q_er_state).quat_mult_right(q_check)

		# 3.4 Compute corrected covariance
		p_cov_hat = (np.eye(9) - K_k.dot(self.h_jac)).dot(p_cov_check)
		
	
		return p_hat, v_hat, q_hat, p_cov_hat

	def convert_qest_to_euler(self, q_est):

		# Convert estimated quaternions to euler angles
		
		qc = Quaternion(*q_est)
		r_est_euler = qc.to_euler()
		"""
		# First-order approximation of RPY covariance
		J = rpy_jacobian_axis_angle(qc.to_axis_angle())
		r_cov_euler_std = np.sqrt(np.diagonal(J @ p_cov[6:, 6:] @ J.T))
		r_cov_euler_std = np.array(r_cov_euler_std)
		"""
		r_est_euler = np.array(r_est_euler)
		
		return r_est_euler
