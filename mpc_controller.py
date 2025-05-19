# mpc_controller.py
import multiprocessing
import socket
import threading
import time
import traceback
import numpy as np
import math
import collections # For deque

# from scipy.optimize import minimize # Example using SciPy (Commented out)

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

    new_delta = delta + steer_rate_cmd * dt
    max_phys_steer_angle_rad = np.radians(70.0) 
    new_delta = np.clip(new_delta, -max_phys_steer_angle_rad, max_phys_steer_angle_rad)

    new_x = x + v * np.cos(yaw) * dt
    new_y = y + v * np.sin(yaw) * dt
    if abs(L) < 1e-6 : 
        new_yaw = yaw 
    else:
        new_yaw = yaw + (v / L) * np.tan(new_delta) * dt
    
    new_v = v + accel_cmd * dt
    new_v = max(0, new_v) 

    new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi 

    return np.array([new_x, new_y, new_yaw, new_v, new_delta])

class MPCController:
    """
    Core MPC logic.
    Calculates an optimal steering angle based on predictive control.
    """
    def __init__(self, horizon_N=10, dt=0.1, wheelbase=2.5):
        self.horizon_N = horizon_N
        self.dt = dt
        self.wheelbase = wheelbase 

        self.q_angle_error = 1.5  
        self.q_rate_error = 0.5   
        self.r_steer_effort = 0.1 
        self.s_steer_change = 0.5 

        self.max_vehicle_steer_angle_rad = np.radians(70.0) 
        self.max_steer_rate_rps = np.radians(120.0) 

    def _cost_function(self, control_sequence_flat, current_vehicle_state,
                       target_steer_angle_rad, target_steer_rate_rps):
        total_cost = 0.0
        predicted_state = np.copy(current_vehicle_state)
        steer_rate_cmds = control_sequence_flat
        
        accel_cmd_placeholder = 0.0 
        last_steer_rate_cmd = 0.0 

        for k in range(self.horizon_N):
            steer_rate_cmd_k = steer_rate_cmds[k]
            predicted_state = vehicle_model(predicted_state, (steer_rate_cmd_k, accel_cmd_placeholder), self.dt)
            predicted_steer_angle_k_rad = predicted_state[4]

            angle_error = (predicted_steer_angle_k_rad - target_steer_angle_rad)**2
            rate_error = (steer_rate_cmd_k - target_steer_rate_rps)**2
            effort_cost = steer_rate_cmd_k**2
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
        # Placeholder: directly uses the target angle.
        # A real implementation would use an optimizer like scipy.optimize.minimize.
        # from scipy.optimize import minimize
        # initial_steer_rate_sequence = np.zeros(self.horizon_N)
        # bounds_steer_rate = [(-self.max_steer_rate_rps, self.max_steer_rate_rps)] * self.horizon_N
        # solver_result = minimize(self._cost_function, initial_steer_rate_sequence,
        #                          args=(current_vehicle_state, target_steer_angle_rad, target_steer_rate_rps),
        #                          method='SLSQP', bounds=bounds_steer_rate,
        #                          options={'maxiter': 50, 'ftol': 1e-4, 'disp': False})
        # if solver_result.success:
        #     optimal_steer_rate_cmd_for_next_step = solver_result.x[0]
        #     current_angle_rad = current_vehicle_state[4]
        #     computed_target_angle_rad = current_angle_rad + optimal_steer_rate_cmd_for_next_step * self.dt
        # else:
        #     computed_target_angle_rad = target_steer_angle_rad # Fallback
        
        computed_target_angle_rad = target_steer_angle_rad # Current placeholder behavior
        
        final_target_angle_rad = np.clip(computed_target_angle_rad,
                                         -self.max_vehicle_steer_angle_rad,
                                         self.max_vehicle_steer_angle_rad)
        return final_target_angle_rad


class PhoneSensorListener(threading.Thread):
    """Handles TCP communication with the Android phone to receive sensor data."""
    def __init__(self, host, port, data_lock, data_update_callback, running_flag_event):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.data_lock = data_lock
        self.data_update_callback = data_update_callback
        self.running_flag = running_flag_event
        self.client_conn = None
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
            print(f"PhoneSensorListener: Error setting up socket: {msg}"); traceback.print_exc()
            if server_socket: server_socket.close()
            self.running_flag.clear(); return
        except Exception as e:
            print(f"PhoneSensorListener: Unexpected setup error: {e}"); traceback.print_exc()
            if server_socket: server_socket.close()
            self.running_flag.clear(); return

        while self.running_flag.is_set():
            if self.client_conn:
                try: self.client_conn.close()
                except: pass
                self.client_conn = None
                self.data_update_callback(0.0, 0.0, 0, 0, False, False) 

            try:
                server_socket.settimeout(1.0)
                try:
                    self.client_conn, self.client_addr = server_socket.accept()
                    self.client_conn.settimeout(5.0)
                    print(f"PhoneSensorListener: Connection from {self.client_addr}")
                    self.data_update_callback(0.0, 0.0, 0, 0, False, True)
                except socket.timeout: continue
                except socket.error: time.sleep(0.5); continue

                buffer = ""
                while self.running_flag.is_set():
                    try:
                        data = self.client_conn.recv(self.buffer_size)
                        if not data:
                            print(f"PhoneSensorListener: Client {self.client_addr} disconnected.")
                            self.data_update_callback(0.0, 0.0, 0, 0, False, False); break 
                        buffer += data.decode('utf-8', errors='ignore')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line: continue
                            parts = line.split(',')
                            if len(parts) == 5:
                                try:
                                    ay = float(parts[0]); gz = float(parts[1])
                                    accel_s = int(parts[2]); brake_s = int(parts[3]); reverse_s = int(parts[4])
                                    self.data_update_callback(ay, gz, accel_s, brake_s, reverse_s, True)
                                except ValueError:
                                    print(f"PhoneSensorListener: Invalid values from {self.client_addr}: {line}")
                                except Exception as parse_e:
                                    print(f"PhoneSensorListener: Error parsing values from {self.client_addr}: {parse_e} - Data: {line}")
                            else:
                                print(f"PhoneSensorListener: Malformed data from {self.client_addr}: Got {len(parts)} vals. Expected 5. Data: '{line}'")
                    except socket.timeout: continue
                    except (socket.error, ConnectionResetError) as e:
                        print(f"PhoneSensorListener: Socket error/reset with {self.client_addr}: {e}")
                        self.data_update_callback(0.0,0.0,0,0,False,False); break 
                    except UnicodeDecodeError:
                        print(f"PhoneSensorListener: Non-UTF-8 data from {self.client_addr}. Buffer cleared."); buffer = ""
                    except Exception as e:
                        print(f"PhoneSensorListener: Processing error with {self.client_addr}: {e}")
                        self.data_update_callback(0.0,0.0,0,0,False,False); traceback.print_exc(); break
            except Exception as e:
                print(f"PhoneSensorListener: Error in accept/outer loop: {e}")
                self.data_update_callback(0.0,0.0,0,0,False,False); traceback.print_exc(); time.sleep(1)
            finally:
                if self.client_conn:
                    try: self.client_conn.close()
                    except: pass
                    self.client_conn = None
                    self.data_update_callback(0.0,0.0,0,0,False,False)

        print("PhoneSensorListener: Thread stopping.")
        if server_socket:
            try: server_socket.close()
            except: pass
        if self.client_conn:
            try: self.client_conn.close()
            except: pass
            self.client_conn = None

    def send_haptic_feedback(self, message):
        if self.client_conn and self._is_connected_unsafe():
            try:
                self.client_conn.sendall(message.encode('utf-8'))
                return True
            except socket.error as e:
                print(f"PhoneSensorListener: Socket error sending haptic data: {e}")
                self.data_update_callback(0.0,0.0,0,0,False,False) 
                try: self.client_conn.close()
                except: pass
                self.client_conn = None
            except Exception as e:
                print(f"PhoneSensorListener: Error sending haptic data: {e}")
        return False

    def _is_connected_unsafe(self): 
        return self.client_conn is not None


class MPCProcess(multiprocessing.Process):
    def __init__(self, vehicle_data_pipe_recv, control_cmd_pipe_send, mpc_params, phone_tcp_config, control_config):
        super().__init__()
        self.vehicle_data_pipe_recv = vehicle_data_pipe_recv
        self.control_cmd_pipe_send = control_cmd_pipe_send
        self.mpc_controller = MPCController(**mpc_params)
        self.control_config = control_config
        
        self.HISTORY_SIZE = self.control_config.get('data_history_size', 10)

        self.ay_data_phone = 0.0
        self.gz_data_phone = 0.0
        self.ay_history = collections.deque(maxlen=self.HISTORY_SIZE)
        self.gz_history = collections.deque(maxlen=self.HISTORY_SIZE)

        self.vehicle_state_history = collections.deque(maxlen=self.HISTORY_SIZE) 

        self.accelerate_pressed_phone = 0
        self.brake_pressed_phone = 0
        self.reverse_enabled_phone = False
        self._is_phone_connected = False
        self._last_phone_update_time = time.time()
        self._phone_data_lock = threading.Lock()

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
        with self._phone_data_lock:
            self.ay_data_phone = ay
            self.gz_data_phone = gz
            if connected: 
                self.ay_history.append(ay)
                self.gz_history.append(gz)
            self.accelerate_pressed_phone = accel
            self.brake_pressed_phone = brake
            self.reverse_enabled_phone = reverse
            self._is_phone_connected = connected
            self._last_phone_update_time = time.time()

    def _calculate_wma(self, data_deque, default_value=0.0):
        """Calculates the weighted moving average for a deque of scalar values."""
        if not data_deque:
            return default_value
        
        data_list = list(data_deque)
        n = len(data_list)
        if n == 0:
            return default_value
        
        weights = np.arange(1, n + 1) # Linear weights: [1, 2, ..., n]
        weighted_sum = np.sum(np.array(data_list) * weights)
        sum_of_weights = np.sum(weights)
        
        if sum_of_weights == 0: # Should not happen with arange(1, n+1) unless n=0
            return default_value
            
        return weighted_sum / sum_of_weights

    def _calculate_vector_wma(self, data_deque_of_vectors, default_vector=None):
        """Calculates the element-wise weighted moving average for a deque of numpy vectors."""
        if not data_deque_of_vectors:
            return default_vector
        
        data_list_of_vectors = list(data_deque_of_vectors)
        n = len(data_list_of_vectors)
        if n == 0:
            return default_vector
        
        # Ensure all vectors in the deque have the same shape, use the shape of the first
        if default_vector is None and data_list_of_vectors:
             default_vector = np.zeros_like(data_list_of_vectors[0])


        weights = np.arange(1, n + 1) # Linear weights: [1, 2, ..., n]
        
        # Convert list of vectors to a 2D numpy array
        vector_array = np.array(data_list_of_vectors) # Shape (n, num_elements_in_vector)
        
        # Calculate weighted average along axis 0 (across the history of vectors)
        try:
            weighted_avg_vector = np.average(vector_array, axis=0, weights=weights)
        except ZeroDivisionError: # sum_of_weights is zero
             return default_vector if default_vector is not None else np.zeros(vector_array.shape[1])


        return weighted_avg_vector


    def run(self):
        print("MPC_Process: Starting PhoneSensorListener thread...")
        self.phone_listener.start()
        print("MPC_Process: MPC run loop started.")
        # Initialize default_vehicle_state once, e.g., based on expected state vector size
        # Assuming state vector has 5 elements [x, y, yaw, v, steer_angle]
        default_vehicle_state_vec = np.zeros(5) 
        try:
            while self.running:
                if self.vehicle_data_pipe_recv.poll(timeout=0.01):
                    vehicle_data_from_main = self.vehicle_data_pipe_recv.recv()
                    if vehicle_data_from_main is None:
                        self.running = False; break

                    raw_vehicle_state = vehicle_data_from_main['state']
                    if default_vehicle_state_vec is None and raw_vehicle_state is not None:
                        default_vehicle_state_vec = np.zeros_like(raw_vehicle_state)

                    self.vehicle_state_history.append(raw_vehicle_state)
                    
                    processed_vehicle_state = self._calculate_vector_wma(
                        self.vehicle_state_history,
                        default_vector=raw_vehicle_state # Fallback to last raw if history is empty
                    )
                    
                    max_physical_steer_angle_rad = vehicle_data_from_main.get('max_steer_rad', np.radians(70.0))
                    haptic_collision_intensity = vehicle_data_from_main.get('collision_intensity', 0.0)
                    haptic_accel_magnitude = vehicle_data_from_main.get('accel_magnitude', 0.0)

                    with self._phone_data_lock:
                        processed_ay = self._calculate_wma(self.ay_history, default_value=self.ay_data_phone)
                        processed_gz = self._calculate_wma(self.gz_history, default_value=self.gz_data_phone)
                        
                        accel_pressed = self.accelerate_pressed_phone
                        brake_pressed = self.brake_pressed_phone
                        reverse_active = self.reverse_enabled_phone
                        is_phone_currently_connected = self._is_phone_connected
                        last_phone_update_current = self._last_phone_update_time
                    
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

                    desired_steer_angle_rad = 0.0
                    desired_steer_rate_rps = 0.0
                    throttle_cmd = 0.0
                    brake_cmd = 0.0

                    if not is_phone_currently_connected or time.time() - last_phone_update_current > 2.0:
                        if is_phone_currently_connected:
                            print("MPC_Process: Phone data stale. Resetting inputs.")
                            self._update_local_phone_sensor_data(0.0,0.0,0,0,reverse_active,False)
                        brake_cmd = 0.5 
                    else:
                        sa_sens = self.control_config.get('steer_angle_sensitivity', 0.7)
                        sa_dz = self.control_config.get('steering_angle_deadzone', 0.2)
                        max_veh_steer_deg = self.control_config.get('max_vehicle_steer_angle_deg', 70.0)
                        max_ay_scale = self.control_config.get('max_ay_for_scaling', 9.8)
                        
                        if abs(processed_ay) > sa_dz:
                            scale_factor = max(0.001, max_ay_scale - sa_dz)
                            scaled_input = (abs(processed_ay) - sa_dz) / scale_factor
                            target_angle_deg = math.copysign(scaled_input, processed_ay) * sa_sens * max_veh_steer_deg
                            desired_steer_angle_rad = np.radians(np.clip(target_angle_deg, -max_veh_steer_deg, max_veh_steer_deg))

                        sr_sens = self.control_config.get('steer_rate_sensitivity', 0.8)
                        sr_dz = self.control_config.get('steering_rate_deadzone', 0.05)
                        max_veh_steer_rate_deg_s = self.control_config.get('max_vehicle_steer_rate_deg_s', 100.0)
                        max_gz_scale_rps = self.control_config.get('max_gz_for_scaling_rps', 5.0)
                        max_veh_steer_rate_rps = np.radians(max_veh_steer_rate_deg_s)

                        if abs(processed_gz) > sr_dz:
                            scale_factor_rate = max(0.001, max_gz_scale_rps - sr_dz)
                            scaled_input_rate = (abs(processed_gz) - sr_dz) / scale_factor_rate
                            desired_steer_rate_rps = math.copysign(scaled_input_rate, -processed_gz) * sr_sens * max_veh_steer_rate_rps
                            desired_steer_rate_rps = np.clip(desired_steer_rate_rps, -max_veh_steer_rate_rps, max_veh_steer_rate_rps)
                        
                        throttle_cmd = 1.0 if accel_pressed == 1 else 0.0
                        brake_cmd = 1.0 if brake_pressed == 1 else 0.0
                        if brake_cmd > 0.0: throttle_cmd = 0.0
                    
                    optimal_target_steer_angle_rad = self.mpc_controller.compute_steering_command(
                        processed_vehicle_state, 
                        desired_steer_angle_rad, 
                        desired_steer_rate_rps   
                    )

                    normalized_steer_cmd = 0.0
                    if abs(max_physical_steer_angle_rad) > 1e-4:
                        normalized_steer_cmd = np.clip(optimal_target_steer_angle_rad / max_physical_steer_angle_rad, -1.0, 1.0)
                    
                    control_output = {
                        'steer': normalized_steer_cmd,
                        'throttle': throttle_cmd,
                        'brake': brake_cmd,
                        'reverse': reverse_active
                    }
                    self.control_cmd_pipe_send.send(control_output)
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt: print("MPC_Process: KeyboardInterrupt, stopping.")
        except Exception as e: print(f"MPC_Process: Error in run loop: {e}"); traceback.print_exc()
        finally:
            print("MPC_Process: Exiting run loop.")
            self.stop()

    def stop(self):
        print("MPC_Process: stop() called.")
        self.running = False
        self._listener_running_flag.clear()
        if self.phone_listener.is_alive():
            self.phone_listener.join(timeout=1.0)
        print("MPC_Process: Fully stopped.")

