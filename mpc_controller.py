# mpc_controller.py
import multiprocessing
import socket
import threading
import time
import traceback
import numpy as np
import math
from scipy.optimize import minimize # Make sure scipy is installed

# Placeholder for vehicle dynamics model (e.g., bicycle model)
# State typically includes [x, y, yaw, velocity, steering_angle]
# Input is [steering_rate_command, acceleration_command]
def vehicle_model(state, control_input, dt):
    """
    Predicts the next state based on current state, control input, and time step.
    """
    x, y, yaw, v, delta = state      # Unpack current state
    steer_rate_cmd, accel_cmd = control_input # Unpack control input

    L = 2.5 # Example wheelbase

    # Update steering angle
    new_delta = delta + steer_rate_cmd * dt
    max_steer_angle_model = np.radians(35.0) # Max physical steer angle of the vehicle model
    new_delta = np.clip(new_delta, -max_steer_angle_model, max_steer_angle_model)

    # Update state using kinematic bicycle model equations
    new_x = x + v * np.cos(yaw) * dt
    new_y = y + v * np.sin(yaw) * dt
    # Ensure L is not zero to avoid division by zero if v is very small
    if abs(v) < 0.01 or abs(L) < 1e-3 : # if speed is very low, yaw change is minimal due to steering
        new_yaw = yaw
    else:
        new_yaw = yaw + (v / L) * np.tan(new_delta) * dt

    new_v = v + accel_cmd * dt
    new_v = max(0, new_v) # Velocity cannot be negative (unless reversing, handle separately if needed)

    new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi # Normalize yaw

    next_state = np.array([new_x, new_y, new_yaw, new_v, new_delta])
    return next_state

class MPCController:
    def __init__(self, horizon_N=10, dt=0.1, wheelbase=2.5,
                 max_steer_angle_deg=35.0, max_steer_rate_rps=np.radians(70.0),
                 max_accel_mps2=3.0, max_decel_mps2=-5.0, # Decel is negative
                 # MPC Tuning Parameters (now arguments with defaults)
                 q_angle_error=1.5, q_rate_error=0.5, q_speed_error=2.0,
                 r_steer_rate_cmd=0.1, r_accel_cmd=0.05,
                 s_delta_steer_rate_cmd=0.5, s_delta_accel_cmd=0.2,
                 **kwargs # To catch any other unexpected arguments if necessary
                 ):
        """
        Initialize MPC parameters.
        """
        self.horizon_N = horizon_N
        self.dt = dt
        self.wheelbase = wheelbase

        # --- MPC Tuning Parameters ---
        self.q_angle_error = q_angle_error
        self.q_rate_error = q_rate_error
        self.q_speed_error = q_speed_error
        # self.q_lane = 0.1 # Weight for lateral deviation (if LKA) - can be added similarly

        self.r_steer_rate_cmd = r_steer_rate_cmd
        self.r_accel_cmd = r_accel_cmd

        self.s_delta_steer_rate_cmd = s_delta_steer_rate_cmd
        self.s_delta_accel_cmd = s_delta_accel_cmd

        # Constraints
        self.max_steer_angle_rad = np.radians(max_steer_angle_deg)
        self.max_steer_rate_rad_s = max_steer_rate_rps # Ensure this is in radians per second
        self.max_acceleration = max_accel_mps2  # m/s^2
        self.max_deceleration = max_decel_mps2 # m/s^2 (should be negative)
        if self.max_deceleration > 0:
            print(f"Warning: max_deceleration should be negative. Forcing to -{abs(self.max_deceleration)}")
            self.max_deceleration = -abs(self.max_deceleration)

        if kwargs:
            print(f"MPCController __init__ received unexpected keyword arguments: {kwargs}")


    def _cost_function(self, control_sequence_flat, current_state,
                       desired_steer_rate_input, desired_steer_angle_input, desired_speed_input,
                       last_applied_steer_rate_cmd, last_applied_accel_cmd):
        """
        Calculates the total cost for a given control sequence.
        """
        total_cost = 0.0
        state = np.copy(current_state)
        # control_sequence has pairs of (steer_rate_cmd, accel_cmd)
        control_inputs = control_sequence_flat.reshape((self.horizon_N, 2))

        current_last_steer_rate_cmd = last_applied_steer_rate_cmd
        current_last_accel_cmd = last_applied_accel_cmd

        for k in range(self.horizon_N):
            steer_rate_cmd = control_inputs[k, 0]
            accel_cmd = control_inputs[k, 1]

            predicted_state = vehicle_model(state, (steer_rate_cmd, accel_cmd), self.dt)
            predicted_steer_angle = predicted_state[4]
            predicted_velocity = predicted_state[3]

            # --- Calculate stage cost for step k ---
            # 1. Angle Tracking Error
            angle_error = (predicted_steer_angle - desired_steer_angle_input) ** 2

            # 2. Rate Tracking Error
            rate_error = (steer_rate_cmd - desired_steer_rate_input) ** 2

            # 3. Speed Tracking Error
            speed_error = (predicted_velocity - desired_speed_input) ** 2

            # 4. Control effort cost (magnitude)
            steer_rate_cmd_cost = steer_rate_cmd ** 2
            accel_cmd_cost = accel_cmd ** 2

            # 5. Control effort cost (rate of change/smoothness)
            delta_steer_rate_cmd_cost = (steer_rate_cmd - current_last_steer_rate_cmd) ** 2
            delta_accel_cmd_cost = (accel_cmd - current_last_accel_cmd) ** 2

            stage_cost = (self.q_angle_error * angle_error +
                          self.q_rate_error * rate_error +
                          self.q_speed_error * speed_error +
                          self.r_steer_rate_cmd * steer_rate_cmd_cost +
                          self.r_accel_cmd * accel_cmd_cost +
                          self.s_delta_steer_rate_cmd * delta_steer_rate_cmd_cost +
                          self.s_delta_accel_cmd * delta_accel_cmd_cost)
            total_cost += stage_cost

            # Update state and last commands for next iteration
            state = predicted_state
            current_last_steer_rate_cmd = steer_rate_cmd
            current_last_accel_cmd = accel_cmd

        return total_cost

    def compute_control_command(self, current_vehicle_state,
                                desired_steer_rate_phone, desired_steer_angle_phone, desired_speed_phone,
                                last_applied_steer_rate_cmd_prev_step, last_applied_accel_cmd_prev_step):
        """
        Computes the optimal steering and acceleration commands using MPC.
        """
        # print(f"MPC Input: State={current_vehicle_state}, DesiredRate={desired_steer_rate_phone:.2f}, "
        #       f"DesiredAngle={np.degrees(desired_steer_angle_phone):.1f}deg, DesiredSpeed={desired_speed_phone:.1f}m/s")

        # Define bounds for control inputs (steer_rate_cmd, accel_cmd)
        steer_rate_bounds = (-self.max_steer_rate_rad_s, self.max_steer_rate_rad_s)
        accel_bounds = (self.max_deceleration, self.max_acceleration) # (min_accel, max_accel)
        
        bounds_list = []
        for _ in range(self.horizon_N):
            bounds_list.extend([steer_rate_bounds, accel_bounds])

        # Initial guess for the control sequence (e.g., repeat last commands or zeros)
        initial_control_sequence = np.tile(
            np.array([last_applied_steer_rate_cmd_prev_step, last_applied_accel_cmd_prev_step]),
            self.horizon_N
        )

        # --- Call Optimization Solver ---
        result = minimize(
            self._cost_function,
            initial_control_sequence,
            args=(current_vehicle_state, desired_steer_rate_phone, desired_steer_angle_phone, desired_speed_phone,
                  last_applied_steer_rate_cmd_prev_step, last_applied_accel_cmd_prev_step),
            method='SLSQP',
            bounds=bounds_list,
            options={'maxiter': 50, 'ftol': 1e-3, 'disp': False} 
        )

        optimal_steer_angle_cmd_rad = current_vehicle_state[4] 
        optimal_throttle_cmd = 0.0
        optimal_brake_cmd = 0.0
        
        current_optimal_steer_rate_cmd = last_applied_steer_rate_cmd_prev_step
        current_optimal_accel_cmd = last_applied_accel_cmd_prev_step

        if result.success:
            optimal_sequence = result.x
            current_optimal_steer_rate_cmd = optimal_sequence[0]
            current_optimal_accel_cmd = optimal_sequence[1]

            current_steer_angle = current_vehicle_state[4]
            optimal_steer_angle_cmd_rad = current_steer_angle + current_optimal_steer_rate_cmd * self.dt
            optimal_steer_angle_cmd_rad = np.clip(optimal_steer_angle_cmd_rad,
                                                  -self.max_steer_angle_rad,
                                                  self.max_steer_angle_rad)

            if current_optimal_accel_cmd > 0:
                optimal_throttle_cmd = np.clip(current_optimal_accel_cmd / self.max_acceleration, 0, 1)
                optimal_brake_cmd = 0.0
            elif current_optimal_accel_cmd < 0:
                optimal_throttle_cmd = 0.0
                optimal_brake_cmd = np.clip(abs(current_optimal_accel_cmd) / abs(self.max_deceleration), 0, 1)
            else:
                optimal_throttle_cmd = 0.0
                optimal_brake_cmd = 0.0
        else:
            print(f"MPC Optimization failed: {result.message}")
            optimal_throttle_cmd = 0.0 
            optimal_brake_cmd = 0.1 

        return optimal_steer_angle_cmd_rad, optimal_throttle_cmd, optimal_brake_cmd, current_optimal_steer_rate_cmd, current_optimal_accel_cmd


class MPCProcess(multiprocessing.Process):
    def __init__(self, vehicle_data_pipe_recv, control_cmd_pipe_send, mpc_init_params, phone_tcp_config, mpc_tuning_params):
        super().__init__()
        self.vehicle_data_pipe_recv = vehicle_data_pipe_recv
        self.control_cmd_pipe_send = control_cmd_pipe_send
        
        controller_params = {**mpc_init_params, **mpc_tuning_params}
        self.mpc_controller = MPCController(**controller_params)

        # Phone Sensor Data
        self.ay_data = 0.0
        self.gz_data = 0.0
        self.accelerate_pressed = 0
        self.brake_pressed = 0
        self.reverse_enabled = False
        self._data_lock = threading.Lock()
        self._last_phone_update_time = time.time()
        self._is_phone_connected = False
        self.client_conn = None 

        # TCP Listener for Phone
        self._tcp_host = phone_tcp_config.get("host", "127.0.0.1")
        self._tcp_port = phone_tcp_config.get("port", 6002)
        self._buffer_size = 1024
        self._running_flag = threading.Event()
        self._running_flag.set()
        self._listener_thread = threading.Thread(target=self._run_phone_sensor_listener, daemon=True)
        self.server_socket = None 

        self.running = True
        self.max_speed_kph_vehicle_limit = 60.0 

        self.last_applied_steer_rate_cmd = 0.0
        self.last_applied_accel_cmd = 0.0

        self.steer_angle_sensitivity_phone = mpc_tuning_params.get("steer_angle_sensitivity_phone", 0.5)
        self.steering_angle_deadzone_phone = mpc_tuning_params.get("steering_angle_deadzone_phone", 0.5)
        self.max_vehicle_steer_angle_deg_phone = mpc_tuning_params.get("max_vehicle_steer_angle_deg_phone", 70)
        self.steer_rate_sensitivity_phone = mpc_tuning_params.get("steer_rate_sensitivity_phone", 0.8)
        self.steering_rate_deadzone_phone = mpc_tuning_params.get("steering_rate_deadzone_phone", 0.1)
        self.max_vehicle_steer_rate_rps_phone = np.radians(mpc_tuning_params.get("max_vehicle_steer_rate_rps_deg_phone", 90.0))


    def _update_phone_sensor_data(self, ay_val, gz_val, accel_state, brake_state, reverse_state, is_connected):
        with self._data_lock:
            self.ay_data = ay_val
            self.gz_data = gz_val
            self.accelerate_pressed = accel_state
            self.brake_pressed = brake_state
            self.reverse_enabled = bool(reverse_state)
            self._last_phone_update_time = time.time()
            self._is_phone_connected = is_connected

    def _run_phone_sensor_listener(self):
        client_addr = None
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self._tcp_host, self._tcp_port))
            self.server_socket.listen(1)
            print(f"MPC_Process Listener: Socket bound and listening on {self._tcp_host}:{self._tcp_port}")
        except socket.error as msg:
            print(f"MPC_Process Listener: Error setting up socket: {msg}")
            traceback.print_exc()
            if self.server_socket: self.server_socket.close(); self.server_socket = None
            self._running_flag.clear(); return
        except Exception as e:
            print(f"MPC_Process Listener: Unexpected setup error: {e}")
            traceback.print_exc()
            if self.server_socket: self.server_socket.close(); self.server_socket = None
            self._running_flag.clear(); return

        while self._running_flag.is_set():
            self.client_conn = None
            try:
                if not self._is_phone_connected:
                    self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                self.server_socket.settimeout(1.0)
                try:
                    self.client_conn, client_addr = self.server_socket.accept()
                    self.client_conn.settimeout(5.0)
                    print(f"MPC_Process Listener: Connection from {client_addr}")
                    self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, True)
                except socket.timeout: continue
                except socket.error as e_accept:
                    print(f"MPC_Process Listener: Error accepting connection: {e_accept}")
                    time.sleep(0.5); continue
                buffer = ""
                while self._running_flag.is_set():
                    try:
                        data = self.client_conn.recv(self._buffer_size)
                        if not data:
                            print(f"MPC_Process Listener: Client {client_addr} disconnected.")
                            self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                            break
                        buffer += data.decode('utf-8')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line: continue
                            parts = line.split(',')
                            if len(parts) == 5:
                                try:
                                    ay, gz, accel_s, brake_s, reverse_s = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                                    self._update_phone_sensor_data(ay, gz, accel_s, brake_s, reverse_s, True)
                                except ValueError: print(f"MPC_Process Listener: Invalid values from {client_addr}: {line}")
                                except Exception as pe: print(f"MPC_Process Listener: Error parsing from {client_addr}: {pe} - Data: {line}")
                            else: print(f"MPC_Process Listener: Malformed data from {client_addr}: Got {len(parts)} vals. Expected 5. Data: '{line}'")
                    except socket.timeout:
                        print(f"MPC_Process Listener: Timeout receiving from {client_addr}.")
                        self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        break
                    except (socket.error, UnicodeDecodeError, Exception) as e_recv:
                        print(f"MPC_Process Listener: Error/Disconnect ({type(e_recv).__name__}) from {client_addr}: {e_recv}")
                        self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        if isinstance(e_recv, UnicodeDecodeError): buffer = ""
                        else: traceback.print_exc()
                        break
            except (socket.error, Exception) as e_outer:
                print(f"MPC_Process Listener: Error in accept loop: {e_outer}")
                self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                time.sleep(1)
            finally:
                if self.client_conn:
                    print(f"MPC_Process Listener: Closing connection to {client_addr if client_addr else 'unknown client'}")
                    try: self.client_conn.shutdown(socket.SHUT_RDWR)
                    except socket.error: pass
                    try: self.client_conn.close()
                    except socket.error: pass
                    self.client_conn = None
                    self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
        print("MPC_Process Listener: Stopping...")
        if self.server_socket:
            try: self.server_socket.close()
            except socket.error: pass
            print("MPC_Process Listener: Server socket closed.")
            self.server_socket = None

    def run(self):
        print("MPC_Process: Starting listener thread for phone sensor data...")
        self._listener_thread.start()
        print("MPC_Process: Started.")
        try:
            while self.running:
                if self.vehicle_data_pipe_recv.poll(timeout=0.01):
                    vehicle_data_msg = self.vehicle_data_pipe_recv.recv()
                    if vehicle_data_msg is None: self.running = False; break

                    current_vehicle_state = vehicle_data_msg['state']
                    haptic_collision_intensity = vehicle_data_msg.get('collision_intensity', 0.0)
                    haptic_accel_magnitude = vehicle_data_msg.get('accel_magnitude', 0.0)
                    self.max_speed_kph_vehicle_limit = vehicle_data_msg.get('max_speed_kph_from_args', 60.0)


                    with self._data_lock:
                        ay_raw, gz_raw = self.ay_data, self.gz_data
                        accel_pressed, brake_pressed = self.accelerate_pressed, self.brake_pressed
                        reverse_active = self.reverse_enabled
                        is_phone_connected, last_phone_update = self._is_phone_connected, self._last_phone_update_time

                    # --- Haptic Feedback ---
                    COLLISION_INTENSITY_THRESHOLD, ACCEL_THRESHOLD = 0.5, 5.0
                    haptic_msg = None
                    if haptic_collision_intensity > COLLISION_INTENSITY_THRESHOLD:
                        haptic_msg = f"COLLISION,{min(haptic_collision_intensity / 5000.0, 1.0):.2f}\n"
                    elif haptic_accel_magnitude > ACCEL_THRESHOLD:
                        haptic_msg = f"FORCE,{np.clip(haptic_accel_magnitude / 20.0, 0, 1):.2f}\n"
                    if haptic_msg and self.client_conn:
                        try: self.client_conn.sendall(haptic_msg.encode('utf-8'))
                        except (socket.error, Exception) as e_haptic:
                            print(f"MPC_Process: Error sending haptic: {e_haptic}")
                            if self.client_conn: 
                                try: self.client_conn.close()
                                except: pass; self.client_conn = None
                            self._update_phone_sensor_data(ay_raw, gz_raw, accel_pressed, brake_pressed, reverse_active, False)


                    # --- Determine Target Inputs for MPC from Phone ---
                    desired_steer_angle_rad_phone, desired_steer_rate_rps_phone = 0.0, 0.0
                    target_speed_for_mpc = current_vehicle_state[3] # Default: maintain current speed

                    if not is_phone_connected or time.time() - last_phone_update > 2.0:
                        if is_phone_connected: # Was connected, now stale
                            self._update_phone_sensor_data(ay_raw, gz_raw, 0, 0, reverse_active, False)
                        # If phone disconnects, aim to stop or maintain (safer to brake gently)
                        target_speed_for_mpc = 0.0 # Aim to stop
                    else:
                        if abs(ay_raw) > self.steering_angle_deadzone_phone:
                            max_ay = 9.8
                            scale_factor = max(0.001, max_ay - self.steering_angle_deadzone_phone)
                            scaled_input = (abs(ay_raw) - self.steering_angle_deadzone_phone) / scale_factor
                            target_angle_deg = math.copysign(scaled_input, ay_raw) * self.steer_angle_sensitivity_phone * self.max_vehicle_steer_angle_deg_phone
                            desired_steer_angle_rad_phone = np.radians(np.clip(target_angle_deg, -self.max_vehicle_steer_angle_deg_phone, self.max_vehicle_steer_angle_deg_phone))

                        if abs(gz_raw) > self.steering_rate_deadzone_phone:
                            max_gz = 5.0
                            scale_factor = max(0.001, max_gz - self.steering_rate_deadzone_phone)
                            scaled_input = (abs(gz_raw) - self.steering_rate_deadzone_phone) / scale_factor
                            desired_steer_rate_rps_phone = math.copysign(scaled_input, -gz_raw) * self.steer_rate_sensitivity_phone * self.max_vehicle_steer_rate_rps_phone
                            desired_steer_rate_rps_phone = np.clip(desired_steer_rate_rps_phone, -self.max_vehicle_steer_rate_rps_phone, self.max_vehicle_steer_rate_rps_phone)

                        if accel_pressed: target_speed_for_mpc = self.max_speed_kph_vehicle_limit / 3.6
                        elif brake_pressed: target_speed_for_mpc = 0.0
                        # If neither, target_speed_for_mpc remains current_vehicle_state[3] (maintain speed)

                    # --- Call MPC ---
                    opt_steer_angle, opt_throttle, opt_brake, \
                    self.last_applied_steer_rate_cmd, \
                    self.last_applied_accel_cmd = self.mpc_controller.compute_control_command(
                        current_vehicle_state,
                        desired_steer_rate_rps_phone,
                        desired_steer_angle_rad_phone,
                        target_speed_for_mpc,
                        self.last_applied_steer_rate_cmd, # Pass previous optimal command
                        self.last_applied_accel_cmd      # Pass previous optimal command
                    )

                    control_command = {
                        'steer_angle_rad': opt_steer_angle,
                        'throttle': opt_throttle,
                        'brake': opt_brake,
                        'reverse': reverse_active # Reverse is still direct from phone
                    }
                    self.control_cmd_pipe_send.send(control_command)
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt: print("MPC_Process: KeyboardInterrupt, stopping.")
        except Exception as e: print(f"MPC_Process: Error in run loop: {e}"); traceback.print_exc()
        finally: self.stop() # Call the refined stop method

    def stop(self):
        print("MPC_Process: stop() called.")
        self.running = False
        if hasattr(self, '_running_flag'): self._running_flag.clear() # Signal listener thread
        if hasattr(self, '_listener_thread') and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)
            if self._listener_thread.is_alive(): print("MPC_Process: Listener thread did not exit cleanly.")
        
        # Close client connection if open
        if self.client_conn:
            try: self.client_conn.shutdown(socket.SHUT_RDWR); self.client_conn.close()
            except (socket.error, Exception): pass # Ignore errors during cleanup
            finally: self.client_conn = None
        
        # Close server socket if open
        if self.server_socket:
            try: self.server_socket.close()
            except (socket.error, Exception): pass # Ignore errors during cleanup
            finally: self.server_socket = None
        print("MPC_Process: Stopped.")


if __name__ == '__main__':
    # This part is for testing mpc_controller.py independently.
    # It will not be run when Android_control.py launches MPCProcess.
    print("mpc_controller.py executed directly (for testing only).")
    # Example test setup (optional)
    # ...
    pass
