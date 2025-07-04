# mpc_controller.py
import multiprocessing
import socket
import threading
import time
import traceback
import numpy as np
import math
import collections # For deque
import json
import cv2
import queue
import matplotlib.pyplot as plt
# In mpc_controller.py

class CameraStreamListener(threading.Thread):
    """Handles UDP communication to receive and reassemble camera frames, putting them into a queue."""
    def __init__(self, host, port, image_queue, running_flag_event): # Modified: image_queue instead of data_lock/callback
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.image_queue = image_queue # Queue to put reassembled images
        self.running_flag = running_flag_event
        self.socket = None
        self.frame_buffer = {}
        self.last_frame_id = -1
        print(f"CameraStreamListener (UDP): Initialized for {host}:{port}")

    def run(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(1.0) # Timeout for recvfrom
            self.socket.bind((self.host, self.port))
            print(f"CameraStreamListener (UDP): Socket bound and listening on {self.host}:{self.port}")
        except Exception as e:
            print(f"CameraStreamListener (UDP): Error setting up socket: {e}"); traceback.print_exc()
            if self.socket: self.socket.close()
            self.running_flag.clear(); return

        while self.running_flag.is_set():
            try:
    
                packet, _ = self.socket.recvfrom(65535)
                
                header = packet[:11]
                chunk_data = packet[11:]

                frame_id = int.from_bytes(header[:4], 'big')
                chunk_idx = header[4]
                total_chunks = header[5]
                img_h = int.from_bytes(header[6:8], 'big')
                img_w = int.from_bytes(header[8:10], 'big')
                img_c = int.from_bytes(header[10:11], 'big')
                shape = (img_h, img_w, img_c)
                

                if frame_id < self.last_frame_id:
                     # Discard old frames, but be careful with wrap-around if frame_id is small
                    # For simplicity, assuming frame_id is always increasing or wraps around rarely
                    # If frame_id wraps around, a more robust check (e.g., timestamp) might be needed.
                    if self.last_frame_id - frame_id > 1000: # Arbitrary large gap, might be a wrap-around
                        print(f"CameraStreamListener (UDP): Frame ID wrapped around or large jump detected. Resetting last_frame_id from {self.last_frame_id} to {frame_id}.")
                        self.last_frame_id = frame_id
                        self.frame_buffer = {} # Clear buffer to avoid mixing old/new frames
                    else:
                        continue # Discard genuinely old frames

                if frame_id not in self.frame_buffer:
                    if len(self.frame_buffer) > 5:
                        keys_to_del = sorted(list(self.frame_buffer.keys()))[:-5]
                        for key in keys_to_del:
                            del self.frame_buffer[key]
                    
                    # Store the shape when a new frame is detected
                    self.frame_buffer[frame_id] = {'total': total_chunks, 'chunks': {}, 'received_time': time.time(), 'shape': shape}

                self.frame_buffer[frame_id]['chunks'][chunk_idx] = chunk_data

                if len(self.frame_buffer[frame_id]['chunks']) == total_chunks:
                    # Reassemble the frame
                    sorted_chunks = sorted(self.frame_buffer[frame_id]['chunks'].items())
                    
                    img_data_bytes = b''.join([chunk for _, chunk in sorted_chunks])
                    frame_shape = self.frame_buffer[frame_id]['shape']
                    
                    # Decode and process the image
                    try:
                        # No more cv2.imdecode. Reconstruct the numpy array directly.
                        frame = np.frombuffer(img_data_bytes, dtype=np.uint8).reshape(frame_shape)
                        if frame is not None:
                            try:
                                self.image_queue.put_nowait(frame) # Put reassembled image into queue
                            except queue.Full:
                                # print("CameraStreamListener (UDP): Image queue full, dropping frame.")
                                pass # Drop frame if processing thread can't keep up
                    except ValueError as e:
                        print(f"CameraStreamListener (UDP): Could not reshape image for frame {frame_id}. "
                              f"Received {len(img_data_bytes)} bytes, expected {frame_shape[0]*frame_shape[1]*frame_shape[2]}. Error: {e}")
                    except Exception as e:
                        print(f"CameraStreamListener (UDP): Could not decode image for frame {frame_id}: {e}")
                    

                    del self.frame_buffer[frame_id]
                    self.last_frame_id = frame_id

            except socket.timeout:
                stale_time = 2.0 # seconds
                stale_keys = [k for k, v in self.frame_buffer.items() if time.time() - v.get('received_time', 0) > stale_time]
                for k in stale_keys:
                    del self.frame_buffer[k]
                continue
            except Exception as e:
                print(f"CameraStreamListener (UDP): Error in receive loop: {e}")
                # traceback.print_exc() # Can be noisy, enable for deep debugging
                time.sleep(0.1)

        print("CameraStreamListener (UDP): Thread stopping.")
        if self.socket: self.socket.close()

# from scipy.optimize import minimize # Example using SciPy (Commented out)
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, dt, output_limits=(-1.0, 1.0), integral_limits=(-5.0, 5.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint  # Target speed
        self.dt = dt              # Time step

        self.integral = 0.0
        self.previous_error = 0.0
        
        self.output_min, self.output_max = output_limits
        self.integral_min, self.integral_max = integral_limits # Anti-windup for integral term
        
        print(f"PID Initialized: Kp={Kp}, Ki={Ki}, Kd={Kd}, Setpoint={setpoint}, dt={dt}, OutLimits={output_limits}, IntLimits={integral_limits}")


    def update(self, current_value):
        if self.dt <= 0: return 0.0 # Avoid division by zero if dt is invalid
        error = self.setpoint - current_value
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        D_out = self.Kd * derivative
        
        # Total output
        output = P_out + I_out + D_out
        
        # Update previous error
        self.previous_error = error
        
        # Clip output
        return np.clip(output, self.output_min, self.output_max)

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        # print(f"PID Reset. Setpoint: {self.setpoint}")

    def set_setpoint(self, setpoint):
        if self.setpoint != setpoint:
            print(f"PID Setpoint Changed: Old={self.setpoint}, New={setpoint}")
            self.setpoint = setpoint
            self.reset() # Reset PID when setpoint changes

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



class LaneKeepingAssist:
    def __init__(self, dt, kp=0.1, ki=0.0008, kd=0.0001):
        """
        Initializes the LKA system using the generic PIDController.
        :param dt: The time step (delta time) for the PID derivative calculation.
        :param kp, ki, kd: PID gains tuned for steering correction based on pixel error.
        """
        # The "setpoint" for our LKA's PID controller is an error of 0.
        # We want the difference between the lane center and the vehicle center to be zero.
        setpoint = 0.0
        
        self.pid_controller = PIDController(
            Kp=kp,
            Ki=ki,
            Kd=kd,
            setpoint=setpoint,
            dt=dt,
            output_limits=(-1.0, 1.0),  # Output is a normalized steering command
            integral_limits=(-150.0, 150.0) # Integral limit based on pixel error magnitude
        )
        
        # --- MODIFICATION START ---
        # Store the last known-good lane polynomials
        self.last_poly_left = None
        self.last_poly_right = None
        # Add a timer to invalidate old lines after a certain period of no detection
        self.last_detection_time = 0
        self.max_line_age_seconds = 1.0 # Invalidate lines if not seen for 1 second
        # --- MODIFICATION END ---
        
        print(f"LKA Initialized using shared PIDController class. Kp={kp}, Ki={ki}, Kd={kd}")

    def process_frame(self, frame):
        """
        Processes a single camera frame to find lane lines and calculate steering correction.
        If current lines aren't found, it uses the last known lines for a short duration.
        Returns a steering correction value between -1.0 (steer left) and 1.0 (steer right).
        """
        if frame is None:
            self.pid_controller.reset() # Reset PID if we lose sight of lines
            return 0.0,None
 
        h, w, _ = frame.shape
        
        # 1. Image processing (ROI, grayscale, blur, Canny, Hough)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced_gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(contrast_enhanced_gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        roi_vertices = np.array([[(0, h), (w/2, h/2), (w, h)]], dtype=np.int32)
        mask = np.zeros_like(dilated_edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_image = cv2.bitwise_and(dilated_edges, mask)
        
        lines = cv2.HoughLinesP(masked_image, 1, np.pi/180, 120, minLineLength=30, maxLineGap=200)
        
        # --- MODIFICATION START ---
        # Logic to detect, store, and recall lane lines
        
        lines_detected_this_frame = False
        poly_left_to_use = None
        poly_right_to_use = None

        if lines is not None:
            # 2. Separate and fit lines using polyfit
            left_line_x, left_line_y = [], []
            right_line_x, right_line_y = [], []
    
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 == x1: 
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < 0.6: # Filter out horizontal lines
                        continue
                    if slope <= 0: # Left lane
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else: # Right lane
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
            
            # If we have points for both lanes, we have a successful detection
            if left_line_y and right_line_y:
                # Fit a 1st degree polynomial (a line) to the points
                self.last_poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
                self.last_poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
                self.last_detection_time = time.time()
                lines_detected_this_frame = True

        # Determine which polynomials to use for calculation
        if lines_detected_this_frame:
            poly_left_to_use = self.last_poly_left
            poly_right_to_use = self.last_poly_right
        else:
            # Check if we can use the last known lines
            is_last_lines_valid = (self.last_poly_left is not None and 
                                   self.last_poly_right is not None and
                                   (time.time() - self.last_detection_time) < self.max_line_age_seconds)
            if is_last_lines_valid:
                # Use the stored polynomials
                poly_left_to_use = self.last_poly_left
                poly_right_to_use = self.last_poly_right
            else:
                # No new lines and no valid old lines. Reset and exit.
                self.pid_controller.reset()
                self.last_poly_left = None # Invalidate them now
                self.last_poly_right = None
                return 0.0, None # Return no correction and no display image

        # --- MODIFICATION END ---
            
        # 3. Calculate Error (using the chosen polynomials)
        # Evaluate the line equations at a specific y-coordinate to find the lane center
        y_eval = int(h * 0.8)
        left_x = int(poly_left_to_use(y_eval))
        right_x = int(poly_right_to_use(y_eval))
        
        # Calculate lane center with WMA
        wma_size = 10 # Keep a history of last 5 lane center values
        if not hasattr(self, 'lane_center_history'): self.lane_center_history = collections.deque(maxlen=wma_size)
        self.lane_center_history.append((left_x + right_x) / 2)
        lane_center_x = np.average(self.lane_center_history, weights=np.arange(1, len(self.lane_center_history) + 1))
        vehicle_center_x = w / 2

        # --- Visualization (using the chosen polynomials) ---
        min_y = int(h * (3 / 5))
        max_y = int(h)
        left_x_start = int(poly_left_to_use(max_y))
        left_x_end = int(poly_left_to_use(min_y))
        right_x_start = int(poly_right_to_use(max_y))
        right_x_end = int(poly_right_to_use(min_y))
        
        # Determine line color for visualization: Green for new, Orange for recalled
        line_color = [0, 255, 0] if lines_detected_this_frame else [255, 165, 0]

        display = LaneKeepingAssist.draw_lines(
                    frame,
                    [[
                        [left_x_start, max_y, left_x_end, min_y],
                        [right_x_start, max_y, right_x_end, min_y],
                    ]],
                    color=line_color,
                    thickness=5,
                        )

        # The error is the deviation from the center in pixels.
        error = lane_center_x - vehicle_center_x
        
        # 4. Get Correction from the PID Controller
        current_value_for_pid = -error
        steering_correction = self.pid_controller.update(current_value_for_pid)

        return steering_correction, display
    
    @staticmethod
    def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8
        )
        img = np.copy(img)
        if lines is None:
            return
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)   

        img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)    
        return img
        
class MPCController:
    """
    Core MPC logic.
    Calculates an optimal steering angle based on predictive control.
    Note: This is a simplified MPC, primarily focused on steering angle and rate.
    A full MPC would typically involve a more complex optimization problem.
    """
    def __init__(self, horizon_N=10, dt=0.1, wheelbase=2.5):
        self.horizon_N = horizon_N
        self.dt = dt
        self.wheelbase = wheelbase 

        # --- MPC Tuning Parameters (Weights for the cost function) ---
        self.q_angle_error = 1.5   # Weight for tracking desired steering angle error
        self.q_rate_error = 0.5    # Weight for tracking desired steering rate error
        self.r_steer_effort = 0.1  # Weight for steering command magnitude (penalizes steer_rate_cmd)
        self.s_steer_change = 0.5  # Weight for change in steering command (smoothness of steer_rate_cmd)

        # Constraints (used for clipping the output)
        self.max_vehicle_steer_angle_rad = np.radians(70.0) # Max physical steering angle of vehicle wheels
        self.max_steer_rate_rps = np.radians(30.0) # Max desired steering rate (rad/s)

    def _cost_function(self, control_sequence_flat, current_vehicle_state,
                       target_steer_angle_rad, target_steer_rate_rps):
        """
        Calculates the total cost for a given sequence of steering rate commands.
        Assumes control_sequence_flat contains only steer_rate_cmd for each step.
        """
        total_cost = 0.0
        predicted_state = np.copy(current_vehicle_state)
        steer_rate_cmds = control_sequence_flat
        
        accel_cmd_placeholder = 0.0 
        last_steer_rate_cmd = 0.0 

        for k in range(self.horizon_N):
            steer_rate_cmd_k = steer_rate_cmds[k]
            
            # Predict next state using the vehicle model
            predicted_state = vehicle_model(predicted_state, (steer_rate_cmd_k, accel_cmd_placeholder), self.dt)
            predicted_steer_angle_k_rad = predicted_state[4] # Current steering angle in the predicted state

            # 1. Angle Tracking Error: Difference between predicted angle and target angle
            angle_error = (predicted_steer_angle_k_rad - target_steer_angle_rad)**2

            # 2. Rate Tracking Error: Difference between the commanded rate and target rate
            rate_error = (steer_rate_cmd_k - target_steer_rate_rps)**2
            
            # 3. Control Effort Cost (magnitude of steer rate command)
            effort_cost = steer_rate_cmd_k**2

            # 4. Control Smoothness Cost (change in steer rate command)
            smoothness_cost = (steer_rate_cmd_k - last_steer_rate_cmd)**2
            
            # Combine costs for this stage
            stage_cost = (self.q_angle_error * angle_error +
                          self.q_rate_error * rate_error +
                          self.r_steer_effort * effort_cost +
                          self.s_steer_change * smoothness_cost)
            
            total_cost += stage_cost
            last_steer_rate_cmd = steer_rate_cmd_k # Update for next iteration's smoothness cost
            
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
    def __init__(self, mpc_params, phone_tcp_config, control_config,
                 vehicle_data_server_address=('127.0.0.1', 6003),
                 control_cmd_client_address=('127.0.0.1', 6004),
                 image_data_server_address = ('127.0.0.1', 6005)):
        
        super().__init__()
        self.mpc_controller = MPCController(**mpc_params)
        self.control_config = control_config
        
        self.vehicle_data_server_address = vehicle_data_server_address
        self.control_cmd_client_address = control_cmd_client_address
        self.vehicle_data_socket_server = None # Will be created in run()
        self.control_cmd_socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.HISTORY_SIZE = self.control_config.get('data_history_size', 10)
        self.ay_data_phone = 0.0; self.gz_data_phone = 0.0
        self.ay_history = collections.deque(maxlen=self.HISTORY_SIZE)
        self.gz_history = collections.deque(maxlen=self.HISTORY_SIZE)
        self.vehicle_state_history = collections.deque(maxlen=self.HISTORY_SIZE)
        self.accelerate_pressed_phone = 0; self.brake_pressed_phone = 0
        self.reverse_enabled_phone = False
        self._is_phone_connected = False
        self._last_phone_update_time = time.time()
        self._phone_data_lock = threading.Lock()

        self._listener_running_flag = threading.Event(); self._listener_running_flag.set()
        self.phone_listener = PhoneSensorListener(
            host=phone_tcp_config.get("host", "0.0.0.0"),
            port=phone_tcp_config.get("port", 6002),
            data_lock=self._phone_data_lock,
            data_update_callback=self._update_local_phone_sensor_data,
            running_flag_event=self._listener_running_flag
        )
        self.running = True # Controls the main run loop of MPCProcess

        # --- New for LKA and Camera Processing ---
        self._image_receiving_queue = queue.Queue(maxsize=5) # Queue for raw images from receiver
        self._lka_processing_running_flag = threading.Event(); self._lka_processing_running_flag.set()
        self._lka_enabled_state = False # Shared state for LKA enable/disable
        self._lka_state_lock = threading.Lock() # Lock for _lka_enabled_state
        self._lka_steering_correction_norm = 0.0 # Shared output from LKA processing thread
        self._lka_output_lock = threading.Lock() # Lock for _lka_steering_correction_norm

        self.camera_listener = CameraStreamListener(
            host=image_data_server_address[0],
            port=image_data_server_address[1],
            image_queue=self._image_receiving_queue, # Pass the queue directly
            running_flag_event=self._lka_processing_running_flag # Use the same flag for both LKA threads
        )
        self.lka_controller = LaneKeepingAssist(dt=self.mpc_controller.dt) # Initialize LKA controller

        # New thread for LKA image processing
        self._lka_processing_thread = threading.Thread(
            target=self._run_lka_processing_thread,
            daemon=True
        )
        # ---- New for Gradual Throttle ----
        self.current_applied_throttle_user = 0.0 # User's desired throttle after ramping
        self.throttle_increase_rate = self.control_config.get('throttle_increase_rate', 0.5) # units/sec
        self.throttle_decrease_rate = self.control_config.get('throttle_decrease_rate', 1.0) # units/sec

        # ---- New for PID Speed Control ----
        self.pid_speed_controller = None
        self.target_speed_mps = self.control_config.get('max_speed_mps', 0.0) # Target speed for PID

        pid_kp = self.control_config.get('pid_kp', 0.5)
        pid_ki = self.control_config.get('pid_ki', 0.1)
        pid_kd = self.control_config.get('pid_kd', 0.05)
        pid_dt = self.mpc_controller.dt # Use MPC's dt for PID updates

        if self.target_speed_mps > 0 and pid_dt > 0:
            self.pid_speed_controller = PIDController(
                Kp=pid_kp, Ki=pid_ki, Kd=pid_kd,
                setpoint=self.target_speed_mps,
                dt=pid_dt,
                output_limits=(-1.0, 1.0), # PID output: -1 (max brake) to 1 (max throttle)
                integral_limits=(-2.0, 2.0) # Example integral limits for anti-windup
            )
            print(f"MPCProcess: PID Speed Controller enabled for target {self.target_speed_mps:.2f} m/s.")
        else:
            print(f"MPCProcess: PID Speed Controller disabled (target_speed_mps={self.target_speed_mps}, pid_dt={pid_dt}).")

        self.running = True
    def _run_lka_processing_thread(self):
        """
        Dedicated thread for processing camera frames for Lane Keeping Assist.
        Receives images from a queue, processes them, and updates a shared steering correction.
        """
        print("MPC_Process: LKA Processing Thread started.")
        while self._lka_processing_running_flag.is_set():
            try:
                # Get image from the queue (blocking with timeout)
                image = self._image_receiving_queue.get(timeout=0.1)
                if image is None: # Sentinel value to stop the thread
                    break

                # Check if LKA is currently enabled
                with self._lka_state_lock:
                    if not self._lka_enabled_state:
                        # If LKA is disabled, just discard the image and continue
                        continue

                # Process the image with LKA controller
                correction, display_frame = self.lka_controller.process_frame(image)
                
                # Update the shared steering correction value
                with self._lka_output_lock:
                    self._lka_steering_correction_norm = correction
                
                # Display the processed frame (optional, for debugging)
                if display_frame is not None:
                    cv2.imshow('LKA Lines', display_frame)
                    cv2.waitKey(1) # Small delay to allow display update

            except queue.Empty:
                continue # No image in queue, loop again
            except Exception as e:
                print(f"MPC_Process: LKA Processing Thread Error: {e}")
                traceback.print_exc()
                # Reset correction on error
                with self._lka_output_lock:
                    self._lka_steering_correction_norm = 0.0
                time.sleep(0.01) # Small sleep to prevent busy-waiting on error

        print("MPC_Process: LKA Processing Thread stopping.")
        cv2.destroyAllWindows() # Close any open OpenCV windows


    def _receive_vehicle_data(self):
        """Receives and deserializes vehicle data from Android_control.py via UDP."""
        if not self.vehicle_data_socket_server:
            return None
        
        try:
            data, _ = self.vehicle_data_socket_server.recvfrom(4096)
            if not data:
                return None
            
            vehicle_data = json.loads(data.decode('utf-8'))
            return vehicle_data
        except socket.timeout:
            return None # No data received
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"MPCProcess (UDP): Error decoding vehicle data: {e}.")
            return None
        except Exception as e:
            print(f"MPCProcess (UDP): Error receiving vehicle data: {e}")
            return None

    def _send_control_command(self, command):
        """Serializes and sends control commands to Android_control.py via UDP."""
        if not self.control_cmd_socket_client:
            print("MPCProcess: Control command UDP socket not available.")
            return False
        try:
            message_json = json.dumps(command) # No newline needed for UDP datagrams
            self.control_cmd_socket_client.sendto(message_json.encode('utf-8'), self.control_cmd_client_address)
            return True
        except socket.error as e:
            print(f"MPCProcess (UDP): Socket error sending control command: {e}.")
            return False 
        except Exception as e:
            print(f"MPCProcess (UDP): Unexpected error sending control command: {e}")
            return False

    def _update_local_phone_sensor_data(self, ay, gz, accel, brake, reverse, connected):
        with self._phone_data_lock:
            self.ay_data_phone = ay; self.gz_data_phone = gz
            if connected: self.ay_history.append(ay); self.gz_history.append(gz)
            self.accelerate_pressed_phone = accel; self.brake_pressed_phone = brake
            self.reverse_enabled_phone = bool(reverse)
            self._is_phone_connected = connected
            self._last_phone_update_time = time.time()

    def _calculate_wma(self, data_deque, default_value=0.0):
        if not data_deque: return default_value
        data_list = list(data_deque); n = len(data_list)
        if n == 0: return default_value
        weights = np.arange(1, n + 1)
        weighted_sum = np.sum(np.array(data_list) * weights)
        sum_of_weights = np.sum(weights)
        return weighted_sum / sum_of_weights if sum_of_weights != 0 else default_value

    def _calculate_vector_wma(self, data_deque_of_vectors, default_vector=None):
        if not data_deque_of_vectors: return default_vector
        data_list_of_vectors = list(data_deque_of_vectors); n = len(data_list_of_vectors)
        if n == 0: return default_vector
        if default_vector is None and data_list_of_vectors: default_vector = np.zeros_like(data_list_of_vectors[0])
        weights = np.arange(1, n + 1)
        vector_array = np.array(data_list_of_vectors)
        try: return np.average(vector_array, axis=0, weights=weights)
        except ZeroDivisionError: return default_vector if default_vector is not None else np.zeros(vector_array.shape[1])

    def run(self):
        print("MPC_Process: Starting PhoneSensorListener thread...")
        self.phone_listener.start()
                # --- NEW: Start camera listener thread ---
        print("MPC_Process: Starting CameraStreamListener thread (for receiving images)...")
        self.camera_listener.start()
        
        print("MPC_Process: Starting LKA Processing Thread...")
        self._lka_processing_thread.start() # Start the new LKA processing thread


        # Setup UDP server for vehicle data
        try:
            self.vehicle_data_socket_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.vehicle_data_socket_server.settimeout(0.1) # Short timeout for recvfrom
            self.vehicle_data_socket_server.bind(self.vehicle_data_server_address)
            print(f"MPCProcess: Vehicle data UDP server listening on {self.vehicle_data_server_address}")
        except Exception as e:
            print(f"MPCProcess: Failed to set up vehicle data UDP server: {e}")
            self.running = False

        dt = self.mpc_controller.dt # Get time step for physics updates
            # --- LKA Setup ---
        # FIX: Pass the dt to the constructor
        lka_controller = LaneKeepingAssist(dt=dt) 

        # --- Timing Measurement Setup ---
        processing_times = collections.deque(maxlen=100) # Store last 100 processing times
        last_print_time = time.time()
        print("MPC_Process: MPC run loop started.")
        # --- End Timing Setup ---



        lka_enabled = False
        lka_strength = 0.4 # Increased strength for better testing
        # Debugging counter
        frame_process_count = 0
        last_print_time = time.time()


        try:
            while self.running:
                # --- Start Timing ---
                loop_start_time = time.perf_counter()

                vehicle_data_from_main = self._receive_vehicle_data()
                if vehicle_data_from_main is None:
                    time.sleep(0.005) 
                    # Safety: if phone disconnects, send brake command
                    with self._phone_data_lock:
                        is_phone_currently_connected = self._is_phone_connected
                        last_phone_update_current = self._last_phone_update_time
                    if not is_phone_currently_connected or time.time() - last_phone_update_current > 2.0:
                        control_output = {'steer': 0.0, 'throttle': 0.0, 'brake': 0.8, 'reverse': False}
                        self._send_control_command(control_output)
                    continue
                # --- LKA Toggle Logic ---
                if vehicle_data_from_main and vehicle_data_from_main.get('lka_toggle_request'):
                    with self._lka_state_lock:
                        self._lka_enabled_state = not self._lka_enabled_state
                        current_lka_state = self._lka_enabled_state
                    print(f"--- LKA TOGGLE RECEIVED --- New state: {'ENABLED' if current_lka_state else 'DISABLED'} ---")
                    self.phone_listener.send_haptic_feedback(f"LKA,{'1' if current_lka_state else '0'}\n")

                # Get LKA steering correction from the processing thread
                with self._lka_output_lock:
                    lka_steering_correction_norm = self._lka_steering_correction_norm



                raw_vehicle_state = np.array(vehicle_data_from_main['state'])
                self.vehicle_state_history.append(raw_vehicle_state)
                processed_vehicle_state = self._calculate_vector_wma(self.vehicle_state_history, default_vector=raw_vehicle_state)
                current_speed_mps = processed_vehicle_state[3] # Speed is at index 3
                max_physical_steer_angle_rad = vehicle_data_from_main.get('max_steer_rad', np.radians(70.0))
                haptic_collision_intensity = vehicle_data_from_main.get('collision_intensity', 0.0)
                haptic_accel_magnitude = vehicle_data_from_main.get('accel_magnitude', 0.0)
                

                with self._phone_data_lock:
                    processed_ay = self._calculate_wma(self.ay_history, default_value=self.ay_data_phone)
                    processed_gz = self._calculate_wma(self.gz_history, default_value=self.gz_data_phone)
                    accel_button_pressed = self.accelerate_pressed_phone == 1
                    brake_button_pressed = self.brake_pressed_phone == 1
                    reverse_active = self.reverse_enabled_phone
                    is_phone_currently_connected = self._is_phone_connected
                    last_phone_update_current = self._last_phone_update_time
                
                # Haptic feedback logic (same as before)
                COLLISION_THRESHOLD = self.control_config.get('haptic_collision_threshold', 0.5)
                ACCEL_HAPTIC_THRESHOLD = self.control_config.get('haptic_accel_threshold', 7.0) # m/s^2
                haptic_msg = None
                if haptic_collision_intensity > COLLISION_THRESHOLD:
                    norm_intensity = min(haptic_collision_intensity / 5000.0, 1.0) # Normalize
                    haptic_msg = f"COLLISION,{norm_intensity:.2f}\n"
                elif haptic_accel_magnitude > ACCEL_HAPTIC_THRESHOLD:
                    norm_accel = np.clip(haptic_accel_magnitude / 20.0, 0, 1) # Normalize and clip
                    haptic_msg = f"FORCE,{norm_accel:.2f}\n"
                if haptic_msg:
                    self.phone_listener.send_haptic_feedback(haptic_msg)


                desired_steer_angle_rad = 0.0; desired_steer_rate_rps = 0.0
                throttle_cmd = 0.0; brake_cmd = 0.0

                if not is_phone_currently_connected or time.time() - last_phone_update_current > 2.0: # 2 second timeout for phone data
                    if is_phone_currently_connected: # Was connected, now stale
                        print("MPC_Process: Phone data stale. Applying brakes.")
                        self._update_local_phone_sensor_data(0.0,0.0,0,0,reverse_active,False) # Update status internally
                    # Removed automatic brake application on stale phone data
                else:
                    # Control logic based on phone data (same as before)
                    sa_sens = self.control_config.get('steer_angle_sensitivity', 3)
                    sa_dz = self.control_config.get('steering_angle_deadzone', 2)
                    max_veh_steer_deg = self.control_config.get('max_vehicle_steer_angle_deg', 70.0)
                    max_ay_scale = self.control_config.get('max_ay_for_scaling', 9.8)
                    if abs(processed_ay) > sa_dz:
                        scale_factor = max(0.001, max_ay_scale - sa_dz)
                        scaled_input = (abs(processed_ay) - sa_dz) / scale_factor
                        target_angle_deg = math.copysign(scaled_input, processed_ay) * sa_sens * max_veh_steer_deg
                        desired_steer_angle_rad = np.radians(np.clip(target_angle_deg, -max_veh_steer_deg, max_veh_steer_deg))

                    sr_sens = self.control_config.get('steer_rate_sensitivity', 0.8)
                    sr_dz = self.control_config.get('steering_rate_deadzone', 1)
                    max_veh_steer_rate_deg_s = self.control_config.get('max_vehicle_steer_rate_deg_s', 100.0)
                    max_gz_scale_rps = self.control_config.get('max_gz_for_scaling_rps', 5.0) # rad/s
                    max_veh_steer_rate_rps = np.radians(max_veh_steer_rate_deg_s)
                    if abs(processed_gz) > sr_dz:
                        scale_factor_rate = max(0.001, max_gz_scale_rps - sr_dz)
                        scaled_input_rate = (abs(processed_gz) - sr_dz) / scale_factor_rate
                        desired_steer_rate_rps = math.copysign(scaled_input_rate, -processed_gz) * sr_sens * max_veh_steer_rate_rps
                        desired_steer_rate_rps = np.clip(desired_steer_rate_rps, -max_veh_steer_rate_rps, max_veh_steer_rate_rps)
                    
                    throttle_cmd = 1.0 if accel_button_pressed == 1 else 0.0
                    brake_cmd = 1.0 if brake_button_pressed == 1 else 0.0
                    if brake_cmd > 0.0: throttle_cmd = 0.0 # Prioritize braking

                # --- Combine User Input and LKA Correction ---
                # Convert LKA's normalized correction to a radian angle
                # Note: max_physical_steer_angle_rad is received from Android_control
                lka_steering_correction_rad = lka_steering_correction_norm * max_physical_steer_angle_rad
                
                # Blend the inputs
                final_desired_steer_angle_rad = (1 - lka_strength) * desired_steer_angle_rad + \
                                                (lka_strength) * lka_steering_correction_rad
                
                # Ensure the blended angle is still within physical limits
                final_desired_steer_angle_rad = np.clip(final_desired_steer_angle_rad, -max_physical_steer_angle_rad, max_physical_steer_angle_rad)

                # Now, feed this combined angle into the MPC
                optimal_target_steer_angle_rad = self.mpc_controller.compute_steering_command(
                    processed_vehicle_state, 
                    final_desired_steer_angle_rad, # Use the blended angle
                    desired_steer_rate_rps   
            )

                normalized_steer_cmd = 0.0
                if abs(max_physical_steer_angle_rad) > 1e-4: # Avoid division by zero
                    normalized_steer_cmd = np.clip(optimal_target_steer_angle_rad / max_physical_steer_angle_rad, -1.0, 1.0)

                
                # --- Gradual Throttle Logic ---
                if accel_button_pressed and not brake_button_pressed:
                    self.current_applied_throttle_user += self.throttle_increase_rate * dt
                else: # Ramp down if accel not pressed OR if brake is pressed
                    self.current_applied_throttle_user -= self.throttle_decrease_rate * dt
                self.current_applied_throttle_user = np.clip(self.current_applied_throttle_user, 0.0, 1.0)


                # User's direct intention for throttle and brake
                user_throttle_request = self.current_applied_throttle_user
                user_brake_request = 1.0 if brake_button_pressed else 0.0
                if user_brake_request > 0: # If user brakes, ramped throttle is cut
                    user_throttle_request = 0.0
                    self.current_applied_throttle_user = 0.0 # Reset ramp

                # Initialize final commands
                final_throttle_cmd = user_throttle_request
                final_brake_cmd = user_brake_request
                
                # --- PID Speed Control Logic (acts as a limiter) ---
                if self.pid_speed_controller and self.target_speed_mps > 0:
                    if self.pid_speed_controller.setpoint != self.target_speed_mps:
                        self.pid_speed_controller.set_setpoint(self.target_speed_mps)
                    
                    pid_output = self.pid_speed_controller.update(current_speed_mps)

                    if user_brake_request > 0: 
                        # User braking always overrides PID.
                        # final_throttle_cmd is already 0 from above.
                        # final_brake_cmd is already user_brake_request.
                        self.pid_speed_controller.reset() 
                        # self.current_applied_throttle_user was already reset.
                    else: # No user braking, PID can influence.
                        # Condition for PID to actively limit/control:
                        # - If speed is over the target OR
                        # - If speed is very close to target and user is still trying to accelerate OR
                        # - If user is not providing throttle input (coasting) and speed needs adjustment by PID.
                        
                        # Let PID primarily act if speed is above or very near the setpoint.
                        if current_speed_mps > self.target_speed_mps: # Overspeeding
                            final_throttle_cmd = 0.0 # Definitely cut throttle
                            if pid_output < 0: # PID wants to brake
                                final_brake_cmd = max(final_brake_cmd, np.clip(-pid_output, 0.0, 1.0))
                            self.current_applied_throttle_user = 0.0 # Reflect that throttle is cut
                            
                        elif current_speed_mps > self.target_speed_mps - 1.0: # Approaching limit (e.g., within 1 m/s)
                            # If PID suggests throttling down or braking, respect that.
                            if pid_output < user_throttle_request : # PID wants less throttle than user or wants to brake
                                if pid_output >= 0: # PID wants some throttle, but less than user
                                    final_throttle_cmd = np.clip(pid_output, 0.0, 1.0)
                                else: # PID wants to brake
                                    final_throttle_cmd = 0.0
                                    final_brake_cmd = max(final_brake_cmd, np.clip(-pid_output, 0.0, 1.0))
                                # Sync user ramp if PID is reducing throttle
                                self.current_applied_throttle_user = final_throttle_cmd
                        
                        # If user is not accelerating (user_throttle_request is 0 or very low)
                        # and speed is below target, PID can apply gentle throttle to maintain speed (e.g., uphill/downhill).
                        # This prevents unwanted strong acceleration if user is just coasting.
                        # elif user_throttle_request < 0.1 and current_speed_mps < self.target_speed_mps :
                        #      if pid_output > 0: # PID wants to apply throttle to maintain speed
                        #          # Apply only a small portion of PID's throttle if user isn't actively accelerating
                        #          # This is to counteract drag/hills, not to aggressively accelerate.
                        #          maintenance_throttle = np.clip(pid_output, 0.0, 0.3) # Max 30% throttle for maintenance
                        #          final_throttle_cmd = max(final_throttle_cmd, maintenance_throttle) # Ensure it doesn't override user's small throttle
                
                
                # --- Handle Phone Disconnection / Stale Data (Overrides PID/User) ---
                if not is_phone_currently_connected or time.time() - last_phone_update_current > 2.0:
                    if is_phone_currently_connected: 
                        print("MPC_Process: Phone data stale. Applying emergency brakes.")
                        self._update_local_phone_sensor_data(0.0,0.0,0,0,reverse_active,False) 
                    # Removed automatic throttle cut and brake application on stale phone data
                    self.current_applied_throttle_user = 0.0 
                    if self.pid_speed_controller: self.pid_speed_controller.reset() 

                
                
                control_output = {
                    'steer': normalized_steer_cmd, 
                    'throttle': final_throttle_cmd,
                    'brake': final_brake_cmd, 
                    'reverse': reverse_active 
                }
                if not self._send_control_command(control_output):
                    print("MPC_Process: Failed to send control command via UDP.")
                
                # --- End Timing and Report ---
                loop_end_time = time.perf_counter()
                processing_times.append((loop_end_time - loop_start_time) * 1000) # Add time in ms

                current_time = time.time()
                if current_time - last_print_time >= 1.0: # Print every second
                    if processing_times:
                        avg_time = np.mean(processing_times)
                        print(f"MPC Loop Avg Processing Time: {avg_time:.2f} ms")
                    last_print_time = current_time
                # --- End Timing and Report ---

        except KeyboardInterrupt: print("MPC_Process: KeyboardInterrupt, stopping.")
        except Exception as e: print(f"MPC_Process: Error in run loop: {e}"); traceback.print_exc()
        finally:
            print("MPC_Process: Exiting run loop.")
            self.stop() # Ensure cleanup

    def stop(self):
        print("MPC_Process: stop() called.")
        self.running = False # Signal main loop to stop
        self._listener_running_flag.clear() # Signal phone listener thread to stop
        self._lka_processing_running_flag.clear() # Signal LKA processing thread to stop

        # Unblock image queue for LKA processing thread
        try: self._image_receiving_queue.put_nowait(None)
        except queue.Full: pass

        if self.phone_listener.is_alive():
            self.phone_listener.join(timeout=1.0) # Wait for phone listener

        if self.camera_listener.is_alive():
            self.camera_listener.join(timeout=1.0) # Wait for camera listener
         
        if self._lka_processing_thread.is_alive():
            self._lka_processing_thread.join(timeout=1.0) # Wait for LKA processing thread
        
        # Close server socket for vehicle data
        if self.vehicle_data_socket_server:
            try: self.vehicle_data_socket_server.close()
            except: pass
            self.vehicle_data_socket_server = None
        
        # Close client socket for control commands
        if self.control_cmd_socket_client:
            try: self.control_cmd_socket_client.close()
            except: pass
            self.control_cmd_socket_client = None
        print("MPC_Process: Sockets closed. Fully stopped.")