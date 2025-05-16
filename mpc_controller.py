# mpc_controller.py
import multiprocessing
import socket
import threading
import time
import traceback
import numpy as np
import math
# from scipy.optimize import minimize # Example using SciPy (Commented out as not fully implemented)

# Placeholder for vehicle dynamics model (e.g., bicycle model)
def vehicle_model(state, control_input, dt):
    """
    Predicts the next state based on current state, control input, and time step.
    This is a simplified kinematic bicycle model.
    :param state: [x, y, yaw_rad, v_mps, current_steer_angle_rad]
    :param control_input: (steer_rate_cmd_rps, accel_cmd_mps2)
    :param dt: Time step (seconds)
    :return: next_state
    """
    x, y, yaw, v, delta = state
    steer_rate_cmd, accel_cmd = control_input
    
    L = 2.5 # Example vehicle wheelbase in meters

    # Update steering angle based on rate command
    new_delta = delta + steer_rate_cmd * dt
    # Max physical steer angle of the vehicle's wheels (e.g., 70 degrees)
    max_phys_steer_angle_rad = np.radians(70.0) # This should ideally match the vehicle
    new_delta = np.clip(new_delta, -max_phys_steer_angle_rad, max_phys_steer_angle_rad)

    # Update state using kinematic bicycle model equations
    new_x = x + v * np.cos(yaw) * dt
    new_y = y + v * np.sin(yaw) * dt
    new_yaw = yaw + (v / L) * np.tan(new_delta) * dt # Avoid division by zero if L is small or v is high
    new_v = v + accel_cmd * dt
    new_v = max(0, new_v) # Velocity cannot be negative

    new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi # Normalize yaw angle to [-pi, pi]

    return np.array([new_x, new_y, new_yaw, new_v, new_delta])

class MPCController:
    """
    Core MPC logic.
    Calculates an optimal steering angle based on predictive control.
    """
    def __init__(self, horizon_N=10, dt=0.1, wheelbase=2.5):
        self.horizon_N = horizon_N
        self.dt = dt
        self.wheelbase = wheelbase # Vehicle wheelbase for the model

        # MPC Tuning Parameters - these require careful adjustment
        self.q_angle_error = 1.5  # Weight for penalizing steering angle deviation from target
        self.q_rate_error = 0.5   # Weight for penalizing steering rate deviation from target
        # self.q_lane_dev = 0.1   # Weight for lateral deviation from a reference path (if applicable)
        self.r_steer_effort = 0.1 # Weight for penalizing large steering rate commands (control effort)
        self.s_steer_change = 0.5 # Weight for penalizing rapid changes in steering rate commands (smoothness)

        # Vehicle constraints (used for optimization bounds and clipping)
        # Max steering angle of the vehicle's wheels (radians)
        self.max_vehicle_steer_angle_rad = np.radians(70.0) # Example: 70 degrees
        # Max rate at which the steering angle can change (radians/sec)
        self.max_steer_rate_rps = np.radians(120.0) # Example: 120 degrees/sec

    def _cost_function(self, control_sequence_flat, current_vehicle_state,
                       target_steer_angle_rad, target_steer_rate_rps):
        """
        Cost function for the MPC optimization.
        Calculates total cost for a sequence of steering rate commands.
        """
        total_cost = 0.0
        predicted_state = np.copy(current_vehicle_state)
        # control_sequence contains only steer_rate_cmds for this simplified version
        steer_rate_cmds = control_sequence_flat #.reshape((self.horizon_N, 1)) # If only optimizing steer_rate

        # Assume a constant (or zero) acceleration for prediction if not optimizing it
        accel_cmd_placeholder = 0.0
        
        last_steer_rate_cmd = 0.0 # For penalizing change in command

        for k in range(self.horizon_N):
            steer_rate_cmd_k = steer_rate_cmds[k]
            
            # Predict next state
            # control_input_k = (steer_rate_cmd_k, accel_cmd_placeholder)
            # For now, assume accel_cmd is part of control_sequence if optimizing it
            # If only optimizing steer_rate, accel_cmd might be fixed or come from phone buttons directly
            # For this example, let's assume accel_cmd is NOT part of the optimization sequence here
            # and is handled separately or is zero for pure steering MPC.
            # If accel_cmd were optimized, control_sequence_flat would be [s0, a0, s1, a1, ...]
            
            predicted_state = vehicle_model(predicted_state, (steer_rate_cmd_k, accel_cmd_placeholder), self.dt)
            predicted_steer_angle_k_rad = predicted_state[4]

            # --- Calculate stage cost for step k ---
            # 1. Angle Tracking Error: Difference between predicted angle and target angle
            angle_error = (predicted_steer_angle_k_rad - target_steer_angle_rad)**2

            # 2. Rate Tracking Error: Difference between commanded rate and target rate
            rate_error = (steer_rate_cmd_k - target_steer_rate_rps)**2
            
            # 3. Control Effort (Steering Rate Magnitude)
            effort_cost = steer_rate_cmd_k**2
            
            # 4. Control Smoothness (Change in Steering Rate Command)
            smoothness_cost = (steer_rate_cmd_k - last_steer_rate_cmd)**2

            stage_cost = (self.q_angle_error * angle_error +
                          self.q_rate_error * rate_error +
                          self.r_steer_effort * effort_cost +
                          self.s_steer_change * smoothness_cost)
            total_cost += stage_cost
            last_steer_rate_cmd = steer_rate_cmd_k
            
        return total_cost

    def compute_steering_command(self, current_vehicle_state, 
                                 target_steer_angle_rad, target_steer_rate_rps):
        """
        Computes the optimal target steering angle for the next step.
        In a full MPC, this would optimize a sequence of control inputs (e.g., steering rates).
        For now, this is a placeholder that directly uses the target angle,
        but a real implementation would use an optimizer like scipy.optimize.minimize.

        :param current_vehicle_state: [x, y, yaw_rad, v_mps, current_steer_angle_rad]
        :param target_steer_angle_rad: Desired steering angle (from phone AY)
        :param target_steer_rate_rps: Desired steering rate (from phone GZ)
        :return: Optimal target steering angle for the vehicle (radians)
        """
        # print(f"MPC Core Input: current_angle={np.degrees(current_vehicle_state[4]):.2f}, target_angle={np.degrees(target_steer_angle_rad):.2f}, target_rate={np.degrees(target_steer_rate_rps):.2f}")

        # --- Placeholder for actual MPC optimization ---
        # initial_control_sequence = np.zeros(self.horizon_N) # If only optimizing steer_rate
        # bounds_steer_rate = [(-self.max_steer_rate_rps, self.max_steer_rate_rps)] * self.horizon_N
        #
        # solver_result = minimize(
        #     self._cost_function,
        #     initial_control_sequence,
        #     args=(current_vehicle_state, target_steer_angle_rad, target_steer_rate_rps),
        #     method='SLSQP', # Example solver
        #     bounds=bounds_steer_rate,
        #     options={'maxiter': 50, 'ftol': 1e-4}
        # )
        #
        # if solver_result.success:
        #     optimal_steer_rate_cmd_for_next_step = solver_result.x[0]
        #     # Integrate the first optimal rate to get the target angle for the next step
        #     current_angle_rad = current_vehicle_state[4]
        #     computed_target_angle_rad = current_angle_rad + optimal_steer_rate_cmd_for_next_step * self.dt
        # else:
        #     print("MPC Optimization failed! Using direct target angle.")
        #     computed_target_angle_rad = target_steer_angle_rad # Fallback
        # --- End Placeholder ---

        # Simplified approach for now: directly use the target angle derived from phone,
        # potentially modified by the target rate if we want to blend them.
        # For this iteration, let's prioritize the target_steer_angle_rad as the primary output,
        # as the MPC's cost function is designed to achieve this over the horizon.
        # A more sophisticated blending or direct rate control would require changes here.
        
        computed_target_angle_rad = target_steer_angle_rad

        # Clip the computed target angle to the vehicle's physical limits
        final_target_angle_rad = np.clip(computed_target_angle_rad,
                                         -self.max_vehicle_steer_angle_rad,
                                         self.max_vehicle_steer_angle_rad)
        
        # print(f"MPC Core Output: final_target_angle={np.degrees(final_target_angle_rad):.2f}deg")
        return final_target_angle_rad


class PhoneSensorListener(threading.Thread):
    """
    Handles TCP communication with the Android phone to receive sensor data.
    Runs in a separate thread.
    """
    def __init__(self, host, port, data_lock, data_update_callback, running_flag_event):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.data_lock = data_lock
        self.data_update_callback = data_update_callback # MPCProcess._update_phone_sensor_data
        self.running_flag = running_flag_event # threading.Event() from MPCProcess
        self.client_conn = None # Socket connection to the phone
        self.client_addr = None
        self.buffer_size = 1024
        print(f"PhoneSensorListener: Initialized for {host}:{port}")

    def run(self):
        server_socket = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"PhoneSensorListener: Socket bound and listening on {self.host}:{self.port}")
        except socket.error as msg:
            print(f"PhoneSensorListener: Error setting up socket: {msg}")
            traceback.print_exc()
            if server_socket: server_socket.close()
            self.running_flag.clear(); return # Signal MPCProcess that listener failed
        except Exception as e:
            print(f"PhoneSensorListener: Unexpected setup error: {e}")
            traceback.print_exc()
            if server_socket: server_socket.close()
            self.running_flag.clear(); return

        while self.running_flag.is_set():
            if self.client_conn: # Close existing connection before accepting new
                try: self.client_conn.close()
                except: pass
                self.client_conn = None
                self.data_update_callback(0.0, 0.0, 0, 0, False, False) # Reset data, mark disconnected

            try:
                server_socket.settimeout(1.0) # Timeout to check running_flag periodically
                try:
                    # print("PhoneSensorListener: Waiting to accept new connection...")
                    self.client_conn, self.client_addr = server_socket.accept()
                    self.client_conn.settimeout(5.0) # Timeout for recv
                    print(f"PhoneSensorListener: Connection from {self.client_addr}")
                    self.data_update_callback(0.0, 0.0, 0, 0, False, True) # Mark connected, reset data
                except socket.timeout:
                    continue # No connection attempt, check running_flag
                except socket.error as e_accept:
                    # print(f"PhoneSensorListener: Error accepting connection: {e_accept}")
                    time.sleep(0.5); continue

                buffer = ""
                while self.running_flag.is_set(): # Inner loop for active connection
                    try:
                        data = self.client_conn.recv(self.buffer_size)
                        if not data:
                            print(f"PhoneSensorListener: Client {self.client_addr} disconnected.")
                            self.data_update_callback(0.0, 0.0, 0, 0, False, False) # Mark disconnected
                            break # Break inner loop to accept new connection

                        buffer += data.decode('utf-8')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line: continue
                            parts = line.split(',')
                            if len(parts) == 5: # ay, gz, accel_btn, brake_btn, reverse_btn
                                try:
                                    ay = float(parts[0])
                                    gz = float(parts[1])
                                    accel_s = int(parts[2])
                                    brake_s = int(parts[3])
                                    reverse_s = int(parts[4])
                                    self.data_update_callback(ay, gz, accel_s, brake_s, reverse_s, True)
                                except ValueError:
                                    print(f"PhoneSensorListener: Invalid values from {self.client_addr}: {line}")
                                except Exception as parse_e:
                                    print(f"PhoneSensorListener: Error parsing values from {self.client_addr}: {parse_e} - Data: {line}")
                            else:
                                print(f"PhoneSensorListener: Malformed data from {self.client_addr}: Got {len(parts)} vals. Expected 5. Data: '{line}'")
                    except socket.timeout:
                        # print(f"PhoneSensorListener: Timeout receiving from {self.client_addr}. Connection might be stale.")
                        # No need to break here, just continue trying to recv or let outer loop handle if running_flag changes
                        continue
                    except (socket.error, ConnectionResetError) as e:
                        print(f"PhoneSensorListener: Socket error/reset with {self.client_addr}: {e}")
                        self.data_update_callback(0.0,0.0,0,0,False,False)
                        break # Break inner to re-accept
                    except UnicodeDecodeError:
                        print(f"PhoneSensorListener: Non-UTF-8 data from {self.client_addr}.")
                        buffer = ""
                    except Exception as e:
                        print(f"PhoneSensorListener: Processing error with {self.client_addr}: {e}")
                        self.data_update_callback(0.0,0.0,0,0,False,False)
                        traceback.print_exc(); break
            except Exception as e: # Catch errors in the outer accept loop
                print(f"PhoneSensorListener: Error in accept/outer loop: {e}")
                self.data_update_callback(0.0,0.0,0,0,False,False)
                traceback.print_exc(); time.sleep(1) # Wait before retrying accept
            finally:
                if self.client_conn:
                    # print(f"PhoneSensorListener: Cleaning up connection to {self.client_addr if self.client_addr else 'previous client'}")
                    try: self.client_conn.close()
                    except: pass
                    self.client_conn = None
                    self.data_update_callback(0.0,0.0,0,0,False,False) # Ensure disconnected state

        print("PhoneSensorListener: Thread stopping.")
        if server_socket:
            try: server_socket.close()
            except: pass
            print("PhoneSensorListener: Server socket closed.")
        if self.client_conn:
            try: self.client_conn.close()
            except: pass
            self.client_conn = None

    def send_haptic_feedback(self, message):
        """Sends a haptic feedback message to the connected phone."""
        if self.client_conn and self._is_connected_unsafe(): # Check connection status (unsafe means not under lock)
            try:
                # print(f"PhoneSensorListener: Sending Haptic: {message.strip()}")
                self.client_conn.sendall(message.encode('utf-8'))
                return True
            except socket.error as e:
                print(f"PhoneSensorListener: Socket error sending haptic data: {e}")
                # Assume connection is lost, update status
                self.data_update_callback(0.0,0.0,0,0,False,False) # Accessing shared state needs care
                try: self.client_conn.close()
                except: pass
                self.client_conn = None
            except Exception as e:
                print(f"PhoneSensorListener: Error sending haptic data: {e}")
        return False

    def _is_connected_unsafe(self):
        # Helper to check connection status without acquiring the lock,
        # for use within methods that might already hold it or where brief stale data is acceptable.
        # For critical checks, use the locked version.
        return self.client_conn is not None


class MPCProcess(multiprocessing.Process):
    def __init__(self, vehicle_data_pipe_recv, control_cmd_pipe_send, mpc_params, phone_tcp_config, control_config):
        super().__init__()
        self.vehicle_data_pipe_recv = vehicle_data_pipe_recv
        self.control_cmd_pipe_send = control_cmd_pipe_send
        self.mpc_controller = MPCController(**mpc_params)
        self.control_config = control_config

        # Phone sensor data attributes (will be updated by PhoneSensorListener)
        self.ay_data_phone = 0.0
        self.gz_data_phone = 0.0
        self.accelerate_pressed_phone = 0
        self.brake_pressed_phone = 0
        self.reverse_enabled_phone = False
        self._is_phone_connected = False
        self._last_phone_update_time = time.time()
        self._phone_data_lock = threading.Lock() # Lock for phone data access

        self._listener_running_flag = threading.Event()
        self._listener_running_flag.set()
        self.phone_listener = PhoneSensorListener(
            host=phone_tcp_config.get("host", "127.0.0.1"),
            port=phone_tcp_config.get("port", 6002),
            data_lock=self._phone_data_lock,
            data_update_callback=self._update_local_phone_sensor_data,
            running_flag_event=self._listener_running_flag
        )
        self.running = True

    def _update_local_phone_sensor_data(self, ay, gz, accel, brake, reverse, connected):
        """Callback for PhoneSensorListener to update MPCProcess's copy of phone data."""
        with self._phone_data_lock:
            self.ay_data_phone = ay
            self.gz_data_phone = gz
            self.accelerate_pressed_phone = accel
            self.brake_pressed_phone = brake
            self.reverse_enabled_phone = reverse
            self._is_phone_connected = connected
            self._last_phone_update_time = time.time()

    def run(self):
        print("MPC_Process: Starting PhoneSensorListener thread...")
        self.phone_listener.start()
        print("MPC_Process: MPC run loop started.")
        try:
            while self.running:
                if self.vehicle_data_pipe_recv.poll(timeout=0.01): # Non-blocking check
                    vehicle_data_from_main = self.vehicle_data_pipe_recv.recv()
                    if vehicle_data_from_main is None: # Sentinel to stop
                        self.running = False; break

                    current_vehicle_state = vehicle_data_from_main['state']
                    max_physical_steer_angle_rad = vehicle_data_from_main.get('max_steer_rad', np.radians(70.0)) # Get from pipe
                    haptic_collision_intensity = vehicle_data_from_main.get('collision_intensity', 0.0)
                    haptic_accel_magnitude = vehicle_data_from_main.get('accel_magnitude', 0.0)

                    # --- Get Local Copy of Phone Sensor Data ---
                    with self._phone_data_lock:
                        ay_raw = self.ay_data_phone
                        gz_raw = self.gz_data_phone
                        accel_pressed = self.accelerate_pressed_phone
                        brake_pressed = self.brake_pressed_phone
                        reverse_active = self.reverse_enabled_phone
                        is_phone_currently_connected = self._is_phone_connected
                        last_phone_update_current = self._last_phone_update_time
                    
                    # --- Haptic Feedback ---
                    COLLISION_THRESHOLD = self.control_config.get('haptic_collision_threshold', 0.5)
                    ACCEL_HAPTIC_THRESHOLD = self.control_config.get('haptic_accel_threshold', 7.0)
                    haptic_msg = None
                    if haptic_collision_intensity > COLLISION_THRESHOLD:
                        norm_intensity = min(haptic_collision_intensity / 5000.0, 1.0)
                        haptic_msg = f"COLLISION,{norm_intensity:.2f}\n"
                    elif haptic_accel_magnitude > ACCEL_HAPTIC_THRESHOLD:
                        norm_accel = np.clip(haptic_accel_magnitude / 20.0, 0, 1)
                        haptic_msg = f"FORCE,{norm_accel:.2f}\n"
                    
                    if haptic_msg:
                        self.phone_listener.send_haptic_feedback(haptic_msg)


                    # --- Phone Data to Control Inputs Conversion ---
                    desired_steer_angle_rad = 0.0
                    desired_steer_rate_rps = 0.0
                    throttle_cmd = 0.0
                    brake_cmd = 0.0

                    if not is_phone_currently_connected or time.time() - last_phone_update_current > 2.0:
                        if is_phone_currently_connected: # Was connected, now stale
                            print("MPC_Process: Phone data stale. Resetting inputs.")
                            # Update local state to reflect disconnection for next iteration if needed
                            self._update_local_phone_sensor_data(0.0,0.0,0,0,reverse_active,False)
                        brake_cmd = 0.5 # Safe default
                    else:
                        # Steering Angle from Accelerometer Y (ay)
                        sa_sens = self.control_config.get('steer_angle_sensitivity', 0.7)
                        sa_dz = self.control_config.get('steering_angle_deadzone', 0.2)
                        max_veh_steer_deg = self.control_config.get('max_vehicle_steer_angle_deg', 70.0)
                        max_ay_scale = self.control_config.get('max_ay_for_scaling', 9.8)
                        
                        if abs(ay_raw) > sa_dz:
                            scale_factor = max(0.001, max_ay_scale - sa_dz)
                            scaled_input = (abs(ay_raw) - sa_dz) / scale_factor
                            target_angle_deg = math.copysign(scaled_input, ay_raw) * sa_sens * max_veh_steer_deg
                            desired_steer_angle_rad = np.radians(np.clip(target_angle_deg, -max_veh_steer_deg, max_veh_steer_deg))

                        # Steering Rate from Gyroscope Z (gz)
                        sr_sens = self.control_config.get('steer_rate_sensitivity', 0.8)
                        sr_dz = self.control_config.get('steering_rate_deadzone', 0.05)
                        max_veh_steer_rate_deg_s = self.control_config.get('max_vehicle_steer_rate_deg_s', 100.0)
                        max_gz_scale_rps = self.control_config.get('max_gz_for_scaling_rps', 5.0)
                        max_veh_steer_rate_rps = np.radians(max_veh_steer_rate_deg_s)

                        if abs(gz_raw) > sr_dz:
                            scale_factor_rate = max(0.001, max_gz_scale_rps - sr_dz)
                            scaled_input_rate = (abs(gz_raw) - sr_dz) / scale_factor_rate
                            desired_steer_rate_rps = math.copysign(scaled_input_rate, -gz_raw) * sr_sens * max_veh_steer_rate_rps
                            desired_steer_rate_rps = np.clip(desired_steer_rate_rps, -max_veh_steer_rate_rps, max_veh_steer_rate_rps)
                        
                        throttle_cmd = 1.0 if accel_pressed == 1 else 0.0
                        brake_cmd = 1.0 if brake_pressed == 1 else 0.0
                        if brake_cmd > 0.0: throttle_cmd = 0.0


                    # --- MPC Calculation ---
                    optimal_target_steer_angle_rad = self.mpc_controller.compute_steering_command(
                        current_vehicle_state,
                        desired_steer_angle_rad, # Target angle from AY
                        desired_steer_rate_rps   # Target rate from GZ
                    )

                    # --- Normalization of Steering Command ---
                    # max_physical_steer_angle_rad is received from Android_control.py
                    normalized_steer_cmd = 0.0
                    if abs(max_physical_steer_angle_rad) > 1e-4:
                        normalized_steer_cmd = np.clip(optimal_target_steer_angle_rad / max_physical_steer_angle_rad, -1.0, 1.0)
                    
                    # --- Prepare Control Command to Send Back ---
                    control_output = {
                        'steer': normalized_steer_cmd, # Normalized steer
                        'throttle': throttle_cmd,
                        'brake': brake_cmd,
                        'reverse': reverse_active # From phone
                    }
                    self.control_cmd_pipe_send.send(control_output)
                else:
                    time.sleep(0.001) # Avoid busy-waiting

        except KeyboardInterrupt:
            print("MPC_Process: KeyboardInterrupt, stopping.")
        except Exception as e:
            print(f"MPC_Process: Error in run loop: {e}")
            traceback.print_exc()
        finally:
            print("MPC_Process: Exiting run loop.")
            self.stop()

    def stop(self):
        print("MPC_Process: stop() called.")
        self.running = False
        self._listener_running_flag.clear() # Signal PhoneSensorListener thread to stop
        if self.phone_listener.is_alive():
            self.phone_listener.join(timeout=1.0)
        # Pipes are closed by the parent process (Android_control.py) that created them
        print("MPC_Process: Fully stopped.")

