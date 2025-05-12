# mpc_controller.py
import multiprocessing
import socket
import threading
import time
import traceback
import numpy as np
import math
# Import your chosen optimization library, e.g., scipy.optimize or cvxpy
# from scipy.optimize import minimize # Example using SciPy

# Placeholder for vehicle dynamics model (e.g., bicycle model)
# State typically includes [x, y, yaw, velocity, steering_angle] or similar
# Input is typically [steering_rate_command, acceleration_command]
def vehicle_model(state, control_input, dt):
    """
    Predicts the next state based on current state, control input, and time step.
    Replace this with your actual vehicle dynamics model (kinematic/dynamic bicycle model).
    """
    # --- Placeholder Model ---
    x, y, yaw, v, delta = state      # Unpack current state
    steer_rate_cmd, accel_cmd = control_input # Unpack control input

    # Simple Kinematic Bicycle Model Example (needs parameters like L - wheelbase)
    L = 2.5 # Example wheelbase

    # Update steering angle (simplistic, assumes direct control over angle change rate)
    # A real model might limit the rate or angle itself.
    # This example assumes steer_rate_cmd is directly the target angle change per second
    new_delta = delta + steer_rate_cmd * dt
    # Clamp steering angle (example limits)
    max_steer_angle = np.radians(35.0)
    new_delta = np.clip(new_delta, -max_steer_angle, max_steer_angle)


    # Update state using kinematic bicycle model equations
    new_x = x + v * np.cos(yaw) * dt
    new_y = y + v * np.sin(yaw) * dt
    new_yaw = yaw + v / L * np.tan(new_delta) * dt
    new_v = v + accel_cmd * dt # Simplistic velocity update

    # Normalize yaw angle (optional but good practice)
    new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi

    next_state = np.array([new_x, new_y, new_yaw, new_v, new_delta])
    # --- End Placeholder ---
    return next_state

class MPCController:
    def __init__(self, horizon_N=10, dt=0.1, wheelbase=2.5):
        """
        Initialize MPC parameters.
        :param horizon_N: Prediction horizon (number of steps)
        :param dt: Time step duration
        :param wheelbase: Vehicle wheelbase for model (example parameter)
        """
        self.horizon_N = horizon_N
        self.dt = dt
        self.wheelbase = wheelbase # Store model parameters if needed

        # --- MPC Tuning Parameters ---
        # --- MPC Tuning Parameters ---
        # Cost function weights (These need careful tuning!)
        # self.q_track = 1.0   # REMOVE or comment out old weight
        self.q_angle = 1.5   # *** ADD: Weight for tracking desired steering angle error *** (TUNE)
        self.q_rate = 0.5    # *** RENAME/ADD: Weight for tracking desired steering rate error *** (TUNE)
        self.q_lane = 0.1    # Weight for lateral deviation from lane center (if using LKA)
        self.r_cmd = 0.1     # Weight for steering command magnitude (penalizes steer_rate_cmd)
        self.s_delta_cmd = 0.5 # Weight for change in steering command (smoothness)

        # Constraints (example)
        self.max_steer_angle = np.radians(35.0) # Max steering wheel angle
        self.max_steer_rate = np.radians(70.0) # Max steering wheel rate (deg/s)

        # You might initialize your solver here if needed

    def _cost_function(self, control_sequence, current_state, desired_steer_rate, desired_steer_angle):
        """
        Calculates the total cost for a given control sequence.
        :param control_sequence: Flattened array of control inputs [steer_rate_cmd_0, accel_cmd_0, steer_rate_cmd_1, ...]
        :param current_state: Initial state [x, y, yaw, v, delta]
        :param desired_steer_rate: Target steering rate from user input (rad/s)
        :param desired_steer_angle: Target steering angle from user input (rad)
        :return: Total cost
        """
        total_cost = 0.0
        state = np.copy(current_state)
        # Reshape control sequence (assuming 2 inputs: steer_rate, accel)
        control_inputs = control_sequence.reshape((self.horizon_N, 2))
        # Use the current actual steer angle as the initial 'last_steer_cmd' basis for smoothness
        last_steer_angle = current_state[4] # Initial steering angle
        # Use 0 as the initial 'last_steer_rate_cmd' for smoothness penalty
        last_steer_rate_cmd = 0.0

        for k in range(self.horizon_N):
            steer_rate_cmd = control_inputs[k, 0]
            accel_cmd = control_inputs[k, 1] # Even if not controlling accel now, model needs it

            # Predict next state using the vehicle model
            # Note: Ensure your vehicle_model correctly uses steer_rate_cmd to update the angle (state[4])
            predicted_state = vehicle_model(state, (steer_rate_cmd, accel_cmd), self.dt)
            predicted_steer_angle = predicted_state[4] # Get predicted steering angle

            # --- Calculate stage cost for step k ---

            # 1. Angle Tracking Error: Difference between predicted angle and desired angle
            angle_error = (predicted_steer_angle - desired_steer_angle) ** 2

            # 2. Rate Tracking Error: Difference between the commanded rate and desired rate
            rate_error = (steer_rate_cmd - desired_steer_rate) ** 2

            # 3. Lateral error cost (Needs lane center info - skip for now if just steering)
            # lateral_error = predicted_state[1] # Assuming y is lateral position relative to lane center
            # lane_cost = lateral_error ** 2

            # 4. Control effort cost (magnitude of steer rate command)
            cmd_cost = steer_rate_cmd ** 2

            # 5. Control effort cost (rate of change/smoothness of steer rate command)
            # Penalize change in the *rate* command itself for smoothness
            delta_cmd_cost = (steer_rate_cmd - last_steer_rate_cmd)**2

            # Combine costs for this stage
            # *** Use the new weights q_angle and q_rate ***
            stage_cost = (self.q_angle * angle_error +
                          self.q_rate * rate_error +
                          # self.q_lane * lane_cost + # Add back when lane info available
                          self.r_cmd * cmd_cost +
                          self.s_delta_cmd * delta_cmd_cost)

            total_cost += stage_cost

            # Update state and last command for next iteration
            state = predicted_state
            last_steer_rate_cmd = steer_rate_cmd # Update last rate command

        return total_cost

    def compute_steering_command(self, current_state, desired_steer_rate, desired_steer_angle):
        """
        Computes the optimal steering command using MPC.
        :param current_state: Current vehicle state [x, y, yaw, v, delta]
        :param desired_steer_rate: Target steering rate from user (rad/s)
        :param desired_steer_angle: Target steering angle from user (rad)
        :return: Optimal steering ANGLE command for the current step (rad)
        """
        # *** Update print statement if desired ***
        print(f"MPC Input: State=[...], DesiredRate={desired_steer_rate:.4f}, DesiredAngle={np.degrees(desired_steer_angle):.2f}deg")

        # Define bounds for control inputs (steer_rate, accel)
        # Steer rate bounds remain the same, as steer_rate is the control variable optimized
        steer_rate_bounds = (-self.max_steer_rate, self.max_steer_rate)
        accel_bounds = (-5.0, 5.0) # Placeholder bounds for acceleration
        bounds = []
        for _ in range(self.horizon_N):
            bounds.extend([steer_rate_bounds, accel_bounds])

        # Initial guess for the control sequence (e.g., all zeros)
        initial_control_sequence = np.zeros(self.horizon_N * 2)

        # --- Call Optimization Solver ---
        # result = minimize(self._cost_function,
        #                   initial_control_sequence,
        #                   # *** Pass desired_steer_angle to args ***
        #                   args=(current_state, desired_steer_rate, desired_steer_angle),
        #                   method='SLSQP',
        #                   bounds=bounds,
        #                   options={'maxiter': 50, 'ftol': 1e-4})

        # --- Placeholder Output (Replace with actual solver result) ---
        # *** Rename output variable ***
        optimal_steer_angle_cmd = 0.0 # Initialize

        if True: # Replace 'True' with 'result.success' if using a solver
            # optimal_sequence = result.x
            # # The solver optimizes for the *rate* sequence.
            # # The first element is the optimal *rate* command for the next step.
            # optimal_steer_rate_cmd_from_solver = optimal_sequence[0]

            # # *** How to get the angle command? ***
            # # Option 1: Integrate the first rate command over dt
            # # This is a common approach in MPC.
            # current_angle = current_state[4]
            # optimal_steer_angle_cmd = current_angle + optimal_steer_rate_cmd_from_solver * self.dt

            # # Option 2: If the solver directly optimized angles (less common for rate-based models)
            # # optimal_steer_angle_cmd = optimal_sequence[INDEX_OF_FIRST_ANGLE_CMD]

            # --- TEMPORARY PLACEHOLDER (Returning Angle) ---
            # This placeholder simply tries to achieve the desired_steer_angle directly,
            # ignoring the rate and the predictive nature of MPC.
            # It's just to make the function return an angle as expected by Android_control.py
            optimal_steer_angle_cmd = desired_steer_angle # Use the desired angle directly

            # *** Apply physical angle limits ***
            optimal_steer_angle_cmd = np.clip(optimal_steer_angle_cmd,
                                              -self.max_steer_angle,
                                              self.max_steer_angle)

            print(f"MPC Placeholder Output: AngleCmd={np.degrees(optimal_steer_angle_cmd):.2f}deg")
            # --- END PLACEHOLDER ---

        else:
            print("MPC Optimization failed!")
            optimal_steer_angle_cmd = current_state[4] # Failsafe: hold current angle
        # --- End Solver Call ---

        # *** Return the optimal steering ANGLE command ***
        return optimal_steer_angle_cmd
    

class MPCProcess(multiprocessing.Process):
    def __init__(self, vehicle_data_pipe_recv, control_cmd_pipe_send, mpc_params, phone_tcp_config):
        super().__init__()
        self.vehicle_data_pipe_recv = vehicle_data_pipe_recv
        self.control_cmd_pipe_send = control_cmd_pipe_send
        self.mpc_controller = MPCController(**mpc_params) # Instantiate your MPC controller

        # --- Phone Sensor Data ---
        self.ay_data = 0.0
        self.gz_data = 0.0
        self.accelerate_pressed = 0
        self.brake_pressed = 0
        self.reverse_enabled = False
        self._data_lock = threading.Lock()
        self._last_phone_update_time = time.time()
        self._is_phone_connected = False
        self.client_conn = None # For haptic feedback

        # --- TCP Listener Setup for Phone Sensors ---
        self._tcp_host = phone_tcp_config.get("host", "127.0.0.1") # Get from config or default
        self._tcp_port = phone_tcp_config.get("port", 6002) # Use a DIFFERENT port than Android_control if it also has a listener
        self._buffer_size = 1024
        self._running_flag = threading.Event()
        self._running_flag.set()
        self._listener_thread = threading.Thread(target=self._run_phone_sensor_listener, daemon=True)
        # --- End Phone Sensor Data ---

        self.running = True

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
        # --- THIS IS THE TCP LISTENER LOGIC MOVED FROM Android_control.py's SensorControl ---
        # --- Adapt it to call self._update_phone_sensor_data ---
        # --- And to potentially use self.client_conn for sending haptic feedback if needed ---
        server_socket = None
        client_addr = None # Define client_addr here
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self._tcp_host, self._tcp_port))
            server_socket.listen(1)
            print(f"MPC_Process Listener: Socket bound and listening on {self._tcp_host}:{self._tcp_port}")
        except socket.error as msg:
            print(f"MPC_Process Listener: Error setting up socket: {msg}")
            traceback.print_exc()
            if server_socket: server_socket.close()
            self._running_flag.clear()
            return
        except Exception as e:
            print(f"MPC_Process Listener: Unexpected setup error: {e}")
            traceback.print_exc()
            if server_socket: server_socket.close()
            self._running_flag.clear()
            return

        while self._running_flag.is_set():
            self.client_conn = None # Reset for each new connection attempt
            try:
                if not self._is_phone_connected: # Update status before waiting
                    self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)

                server_socket.settimeout(1.0)
                try:
                    self.client_conn, client_addr = server_socket.accept() # client_addr is assigned here
                    self.client_conn.settimeout(5.0)
                    print(f"MPC_Process Listener: Connection from {client_addr}")
                    self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, True)
                except socket.timeout:
                    continue
                except socket.error as e_accept:
                    print(f"MPC_Process Listener: Error accepting connection: {e_accept}")
                    time.sleep(0.5)
                    continue

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
                            if len(parts) == 5: # ay, gz, accel_state, brake_state, reverse_state
                                try:
                                    ay = float(parts[0])
                                    gz = float(parts[1])
                                    accel_s = int(parts[2])
                                    brake_s = int(parts[3])
                                    reverse_s = int(parts[4])
                                    self._update_phone_sensor_data(ay, gz, accel_s, brake_s, reverse_s, True)
                                except ValueError:
                                    print(f"MPC_Process Listener: Invalid values from {client_addr}: {line}")
                                except Exception as parse_e:
                                    print(f"MPC_Process Listener: Error parsing values from {client_addr}: {parse_e} - Data: {line}")
                            else:
                                print(f"MPC_Process Listener: Malformed data from {client_addr}: Got {len(parts)} vals. Expected 5. Data: '{line}'")
                    except socket.timeout:
                        print(f"MPC_Process Listener: Timeout receiving from {client_addr}.")
                        self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        break
                    # ... (similar error handling as in Android_control.py) ...
                    except socket.error as e:
                        print(f"MPC_Process Listener: Socket error recv from {client_addr}: {e}")
                        self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        traceback.print_exc()
                        break
                    except UnicodeDecodeError:
                        print(f"MPC_Process Listener: Non-UTF-8 data from {client_addr}.")
                        buffer = "" # Clear buffer on decode error
                    except Exception as e:
                        print(f"MPC_Process Listener: Processing error from {client_addr}: {e}")
                        self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        traceback.print_exc()
                        break # Break inner loop
            # ... (rest of the socket error handling for the accept loop) ...
            except socket.error as e:
                 print(f"MPC_Process Listener: Error accepting connection: {e}")
                 self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                 time.sleep(1)
            except Exception as e:
                 print(f"MPC_Process Listener: Unexpected error in accept loop: {e}")
                 self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                 traceback.print_exc()
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
        if server_socket:
            try: server_socket.close()
            except socket.error: pass
            print("MPC_Process Listener: Server socket closed.")


    def run(self):
        print("MPC_Process: Starting listener thread for phone sensor data...")
        self._listener_thread.start()
        print("MPC_Process: Started.")
        try:
            while self.running:
                if self.vehicle_data_pipe_recv.poll(timeout=0.01): # Non-blocking check
                    vehicle_data = self.vehicle_data_pipe_recv.recv()
                    if vehicle_data is None: # Sentinel to stop
                        self.running = False
                        break

                    current_vehicle_state = vehicle_data['state']
                    # Haptic feedback data also received from Android_control.py now
                    haptic_collision_intensity = vehicle_data.get('collision_intensity', 0.0)
                    haptic_accel_magnitude = vehicle_data.get('accel_magnitude', 0.0)


                    # --- Get Phone Sensor Data (already updated by the listener thread) ---
                    with self._data_lock:
                        ay_raw = self.ay_data
                        gz_raw = self.gz_data
                        accel_pressed = self.accelerate_pressed
                        brake_pressed = self.brake_pressed
                        reverse_active = self.reverse_enabled
                        is_phone_connected = self._is_phone_connected
                        last_phone_update = self._last_phone_update_time

                    # --- Send Haptic Feedback if needed ---
                    # This logic moves from Android_control.py to mpc_controller.py
                    # Define Haptic Thresholds (Tune these!)
                    COLLISION_INTENSITY_THRESHOLD = 0.5
                    ACCEL_THRESHOLD = 5.0
                    haptic_trigger_message = None

                    if haptic_collision_intensity > COLLISION_INTENSITY_THRESHOLD:
                        normalized_intensity = min(haptic_collision_intensity / 5000.0, 1.0) # Adjust 5000
                        haptic_trigger_message = f"COLLISION,{normalized_intensity:.2f}\n"
                    elif haptic_accel_magnitude > ACCEL_THRESHOLD:
                        haptic_trigger_message = f"FORCE,{np.clip(haptic_accel_magnitude / 20.0, 0, 1):.2f}\n"

                    if haptic_trigger_message and self.client_conn: # self.client_conn is the phone socket
                        try:
                            # print(f"MPC_Process: Sending Haptic: {haptic_trigger_message.strip()}")
                            self.client_conn.sendall(haptic_trigger_message.encode('utf-8'))
                        except socket.error as e:
                            print(f"MPC_Process: Socket error sending haptic data: {e}")
                            try: self.client_conn.close()
                            except: pass
                            self.client_conn = None
                            self._update_phone_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        except Exception as e:
                            print(f"MPC_Process: Error sending haptic data: {e}")
                    # --- End Haptic Feedback ---


                    # --- Control Calculation Logic (moved from Android_control.py) ---
                    # This part takes phone sensor data (ay_raw, gz_raw) and vehicle state
                    # to calculate desired_steer_angle_rad and desired_steer_rate_rps.
                    # Then calls MPC.

                    if not is_phone_connected or time.time() - last_phone_update > 2.0:
                        # Handle phone disconnection: zero out inputs for MPC
                        desired_steer_angle_rad_phone = 0.0
                        desired_steer_rate_rps_phone = 0.0
                        # Throttle/brake from phone also zeroed out
                        throttle_cmd_phone = 0.0
                        brake_cmd_phone = 0.0
                        # Keep reverse_active as is, or reset if preferred
                        if is_phone_connected: # If it was connected but now stale
                             self._update_phone_sensor_data(ay_raw, gz_raw, 0,0, reverse_active, False)

                    else:
                        # --- Calculate Desired Angle from AY (Phone) ---
                        # (Use parameters like self.steer_angle_sensitivity, self.steering_angle_deadzone etc.)
                        # These parameters should be part of MPCProcess or passed in mpc_params
                        steer_angle_sensitivity = 0.5 # Example, take from config
                        steering_angle_deadzone = 0.5 # Example
                        max_vehicle_steer_angle_deg = 70 # Example

                        desired_steer_angle_rad_phone = 0.0
                        if abs(ay_raw) > steering_angle_deadzone:
                            max_ay_for_scaling = 9.8
                            scale_factor = max(0.001, max_ay_for_scaling - steering_angle_deadzone)
                            scaled_input = (abs(ay_raw) - steering_angle_deadzone) / scale_factor
                            target_angle_deg = math.copysign(scaled_input, ay_raw) * steer_angle_sensitivity * max_vehicle_steer_angle_deg
                            desired_steer_angle_rad_phone = np.radians(np.clip(target_angle_deg, -max_vehicle_steer_angle_deg, max_vehicle_steer_angle_deg))

                        # --- Calculate Desired Rate from GZ (Phone) ---
                        steer_rate_sensitivity = 0.8 # Example
                        steering_rate_deadzone = 0.1 # Example
                        max_vehicle_steer_rate_rps = np.radians(90.0) # Example

                        desired_steer_rate_rps_phone = 0.0
                        if abs(gz_raw) > steering_rate_deadzone:
                            max_gz_for_scaling = 5.0
                            scale_factor = max(0.001, max_gz_for_scaling - steering_rate_deadzone)
                            scaled_input = (abs(gz_raw) - steering_rate_deadzone) / scale_factor
                            desired_steer_rate_rps_phone = math.copysign(scaled_input, -gz_raw) * steer_rate_sensitivity * max_vehicle_steer_rate_rps # Note the -gz_raw
                            desired_steer_rate_rps_phone = np.clip(desired_steer_rate_rps_phone, -max_vehicle_steer_rate_rps, max_vehicle_steer_rate_rps)

                        throttle_cmd_phone = 1.0 if accel_pressed == 1 else 0.0
                        brake_cmd_phone = 1.0 if brake_pressed == 1 else 0.0
                    # --- End Control Calculation Logic ---

                    # --- Call MPC ---
                    # MPC now takes both desired angle and rate from phone sensors
                    optimal_steer_angle_cmd_rad = self.mpc_controller.compute_steering_command(
                        current_vehicle_state,
                        desired_steer_rate_rps_phone,
                        desired_steer_angle_rad_phone
                    )

                    # Prepare control command to send back
                    control_command = {
                        'steer_angle_rad': optimal_steer_angle_cmd_rad, # MPC now outputs target angle
                        'throttle': throttle_cmd_phone,
                        'brake': brake_cmd_phone,
                        'reverse': reverse_active # from phone
                    }
                    self.control_cmd_pipe_send.send(control_command)
                else:
                    time.sleep(0.001) # Sleep briefly if no data to avoid busy-waiting

        except KeyboardInterrupt:
            print("MPC_Process: KeyboardInterrupt, stopping.")
        except Exception as e:
            print(f"MPC_Process: Error in run loop: {e}")
            traceback.print_exc()
        finally:
            print("MPC_Process: Stopping.")
            self._running_flag.clear() # Signal listener thread to stop
            if self._listener_thread.is_alive():
                self._listener_thread.join(timeout=1.0)
            if self.client_conn:
                try: self.client_conn.close()
                except: pass
            if server_socket: # This was defined inside _run_phone_sensor_listener, handle appropriately
                 try: server_socket.close()
                 except: pass
            print("MPC_Process: Stopped.")

    def stop(self):
        print("MPC_Process: stop() called.")
        self.running = False
        if self._listener_thread.is_alive():
             self._running_flag.clear()
             self._listener_thread.join(timeout=1.0)


if __name__ == '__main__':
    # This part is for testing mpc_controller.py independently if needed.
    # You would typically not run mpc_controller.py directly in the final setup.
    # It will be launched by Android_control.py.

    # Example:
    # parent_conn_vehicle, child_conn_vehicle = multiprocessing.Pipe()
    # parent_conn_control, child_conn_control = multiprocessing.Pipe()

    # mpc_params_example = {'horizon_N': 10, 'dt': 0.05} # from Android_control
    # phone_tcp_config_example = {"host": "127.0.0.1", "port": 6002}


    # mpc_process = MPCProcess(child_conn_vehicle, parent_conn_control, mpc_params_example, phone_tcp_config_example)
    # mpc_process.start()
    # print("Main: MPC Process started for testing.")

    # try:
    #     # Simulate Android_control.py sending data
    #     for i in range(50):
    #         dummy_vehicle_state = np.array([0.0, 0.0, 0.0, 10.0, 0.0]) # x, y, yaw, v, delta
    #         parent_conn_vehicle.send({'state': dummy_vehicle_state, 'collision_intensity':0.0, 'accel_magnitude':0.0 })
    #         print(f"Main: Sent vehicle data {i}")
    #         if parent_conn_control.poll(timeout=0.5):
    #             cmd = child_conn_control.recv()
    #             print(f"Main: Received control command: {cmd}")
    #         else:
    #             print("Main: No control command received in time.")
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     print("Main: Test interrupted.")
    # finally:
    #     print("Main: Stopping MPC process.")
    #     parent_conn_vehicle.send(None) # Signal MPC process to stop
    #     mpc_process.join(timeout=2)
    #     if mpc_process.is_alive():
    #         print("Main: MPC process did not terminate, forcing.")
    #         mpc_process.terminate()
    #     parent_conn_vehicle.close()
    #     child_conn_vehicle.close()
    #     parent_conn_control.close()
    #     child_conn_control.close()
    #     print("Main: Test finished.")
    pass
