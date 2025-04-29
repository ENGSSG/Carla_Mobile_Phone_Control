# mpc_controller.py
import numpy as np
from scipy.optimize import minimize
import cvxpy
import carla

#Replace the wheel value with carla.Wheel.PhysicsControl.radius bare that in mind its in cm

# Placeholder dynamic model
def vehicle_model(state, control_input,dt):
    """Predicts the next state based on current state, control input, and time step.
    Replace this with my actual model"""

    # --- Place holder Model ---
    x, y, yaw, v, delta = state # Unpack current state
    steer_rate_cmd, accel_cmd = control_input # Unpack control input
    
    # Simple Kinematic Bicycle Model Example (needs physical parameter like L - Wheelbase) Maybe make this into a variable
    L = 0.6

    # Update steering angle (simplistic, assumes direct control over angle change rate)
    # A real model might limit the rate or angle itself.
    # This example assumes steer_rate_cmd is directly the target angle change per second
    new_delta = delta + steer_rate_cmd * dt
    # Clamp steering angle (example limits)
    max_steer_angle = np.radians(35.0)
    new_delta = np.clip(new_delta, -max_steer_angle, max_steer_angle)

    # Update state using kinematic bicycle model equations
    new_x = x + v * np.cos(yaw) * dt # vehicle x position
    new_y = y + v * np.sin(yaw) * dt # vehicle y position
    new_yaw = yaw + v / L * np.tan(new_delta) * dt # angle data
    new_v = v + accel_cmd * dt # adjusted v

    # Normalize yaw angle (Optional)
    new_yaw = (new_yaw + np.pi) % (2*np.pi) - np.pi

    next_state = np.array([new_x, new_y, new_yaw, new_v, new_delta])

    return next_state

class MPCController:
    def __init__(self, horizon_N=10, dt=0.1, wheelbase=0.6):
        """
        Initialize MPC parameters
        horizon_N: Prediction horizon (number of prediction steps)
        dt: Time step duration
        wheelbase = wheel diameter
        """
        self.horizon_N = horizon_N
        self.dt = dt
        self.wheelbase = wheelbase = wheelbase

        # ---MPC Tuning Parameters ---
        # Cost function weights (These need caregul tuning!)
        self.q_track =1.0 # Weight for trav
        self.q_lane = 0.1 # Weight for lane centering will be added later
        self.r_cmd = 0.1 # Weight for steering command magnitude
        self.s_delta_cmd = 0.5 # Weight for change in steering command (smoothness)

        # Contraints(example)
        self.max_steer_angle = np.radians(70.0) # Max sterring wheel angels
        self.max_steer_rate = np.radians(90.0) # Max steering wheel rate (deg/s)


    def _cost_function(self, control_sequence, current_state, desired_steer_rate):
        """
        Calculates the toal cost for a given control sequence.
        :param control_sequence: Falttened array of control inputs [steer_rate_cmd_0, accel_cmd_0, steer_rate_cmd_1, ...]
        :param current_state: Intial state [x, y, yaw, v ,delta]
        :param desired_steer_rate: Target steering rate from user input
        :return: Total cost"""
        total_cost = 0.0
        state = np.copy(current_state)
        # Reshape contol sequence (Assuming 2inputs: steer_rate, accel)
        control_inputs = control_sequence.reshape((self.horizon_N, 2))
        last_steer_cmd = current_state[4] # Initial sttering angle

        for k in range(self.horizon_N):
            steer_rate_cmd = control_inputs[k, 0]
            accel_cmd = control_inputs[k,1] # Even if not controlling accel now, model requires it

            # Predict next state using the vehicle model
            state = vehicle_model(state, (steer_rate_cmd, accel_cmd), self.dt)
                        # --- Calculate stage cost for step k ---
            # 1. Tracking error cost (steering rate error)
            # This assumes your model state includes steering rate, or you calculate it.
            # Simpler: Penalize difference between predicted angle and angle derived from desired rate
            predicted_steer_angle = state[4] # Get predicted steering angle
            # Simplistic: Assume desired angle ramps up/down based on rate
            # A better approach might track the desired rate directly if model allows
            # This needs refinement based on how desired_steer_rate is defined
            # Let's penalize deviation from the *desired rate* itself for simplicity now
            # NEED TO REFINE HOW TO MAP desired_steer_rate to state error properly
            # Placeholder: Penalize the steering rate command directly for now
            # tracking_error = (steer_rate_cmd - desired_steer_rate) ** 2
            # Better Placeholder: Assume desired_steer_rate implies a target angle change
            target_delta_change = desired_steer_rate * self.dt
            actual_delta_change = state[4] - current_state[4] # How much angle *actually* changed in simulation step
            tracking_error = (actual_delta_change - target_delta_change)**2 # Needs refinement
            
            # 2. Lateral error cost (Needs lane center info - skip for now if just steering)
            # lateral_error = state[1] # Assuming y is lateral position relative to lane center
            # lane_cost = lateral_error ** 2 
                
            # 3. Control effort cost (magnitude)
            # Penalize steer rate command magnitude
            cmd_cost = steer_rate_cmd ** 2

            # 4. Control effort cost (rate of change/smoothness)
            delta_cmd_cost = (steer_rate_cmd - last_steer_cmd / self.dt)**2 # Approx rate change

            # Combine costs for this stage
            stage_cost = (self.q_track * tracking_error +
                        # self.q_lane * lane_cost + # Add back when lane info available
                        self.r_cmd * cmd_cost +
                        self.s_delta_cmd * delta_cmd_cost)

            total_cost += stage_cost
            last_steer_cmd = state[4] # Update last command for next delta calc
            current_state = state # Update state for next prediction step

        return total_cost
    
    def compute_steering_command(self, current_state, desired_steer_rate):
        """ Computes the optimal steering command using MPC
        :param current_state: Current vehicle state [x, y, yaw, v, delta]
        :param desired_steer_rate: Target steering rate from user
        :return: Optimal steering rate command for the current step"""
        print(f"MPC Input: State={current_state}, DesiredRate={desired_steer_rate:.4f}")

        # Define bouhnds for control inputs (steer_rate,accel)
        # We only care about steer_rate bounds now, accel is placeholder
        steer_rate_bounds = (-self.max_steer_rate, self.max_steer_rate)
        accel_bounds = (-5.0,5.0) # Placeholder bounds for accelration
        bounds = []
        for _ in range(self.horizon_N):
            bounds.extend([steer_rate_bounds, accel_bounds])

            # Initial guess for the control sequence (e.g., all zeroes)
            initial_control_sequence = np.zeros(self.horizon_N * 2)
            
            # --- Call Optimization Solver ---
            # This is a simplified example using scipy.optimize.minimize
            # Real MPC often uses specialized QP/NLP solvers (OSQPm IPOPT via CasADi/CVXPY)
            # whicj handle contstarints more directly and efficiently.
            # You'll likely need to adapt this part significantly.

            # Example using scipy.optimize.minimize (basic, might be slow, constraints are tricky)
            # You'd typically formulate this as a constrained optimization problem.
            # result = minimize(self._cost_function,
            #                   initial_control_sequence)
            #                   args=(current_state, desired_steer_rate),
            #                   method='SLSQP', # Example solver supporting bounds
            #                   bounds=bounds,
            #                   options={'maxiter': 50, 'ftol': 1e-4}) # Limit iterations
            if True: #Replace 'True' with 'result.success' if using a solver
                # optimal_sequence = result.x
                # optimal_steer_rate_cmd = optimal_sequence[0] # First steering command

                # --- Temporary placeholder ---
                # Simple proportional control as a placeholder until solver is implemented
                # This will NOT have the benefits of prediction or smoothness from MPC cost function
                gain = 0.5
                optimal_steer_rate_cmd = np.clip(desired_steer_rate * gain,
                                                    -self.max_steer_rate,
                                                    self.max_steer_rate)
                print(f"MPC Placeholder Output: Cmd={optimal_steer_rate_cmd:.4f}")
                # --- End Placeholder ---

            else:
                print("MPC Optimizatino failed!")
                optimal_steer_rate_cmd = 0.0 # Failsafe command
            # --- End Solver Call ---
            

            # Return only the first steering command of the optimal sequence
            return optimal_steer_rate_cmd





 
 
