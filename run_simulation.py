import carla
# Define the host and port of your CARLA server
HOST = '127.0.0.1'
PORT = 2000

# Name of the log file you want to inspect
log_file = "/home/engssg/CARLA_0.9.15/PythonAPI/Recordings/FollowLeadingVehicle_1.log"

# --- FIX: Create an instance of the CARLA client ---
try:
    client = carla.Client(HOST, PORT)
    client.set_timeout(5.0)  # Set a timeout for server connection

    # --- FIX: Call the method on the client instance ---
    # The second argument, 'show_all', is optional and defaults to False.
    # Set it to True if you want to see every single event (can be very long).
    file_info = client.show_recorder_file_info(log_file, show_all=False)
    
    # The function returns a string with the formatted information.
    print(file_info)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure the CARLA server is running and the log file exists in the correct directory.")
client.replay_file("/home/engssg/CARLA_0.9.15/PythonAPI/Recordings/FollowLeadingVehicle_1.log", 1, 50, 122)
# print(client.show_recorder_file_info(log_file,show_all=False))