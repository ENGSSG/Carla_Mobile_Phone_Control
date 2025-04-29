# mpc_controller.py
import numpy as np
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