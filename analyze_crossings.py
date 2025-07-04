import carla
import argparse
import time
import logging

def analyze_lane_crossings(args):
    """
    Connects to a CARLA server, replays a log file, and counts the number
    of times the ego vehicle crosses a lane line.
    """
    client = None
    world = None
    total_crossings = 0
    
    try:
        # 1. Connect to the CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        # 2. Get the log file's map and hero actor ID
        # We need this to set up the world correctly before replaying.
        try:
            log_info = client.show_recorder_file_info(args.file, False)
            
            # Extract map name from the info string
            map_name = log_info.split("Map: ")[1].split("\n")[0]
            
            # Find the actor ID for the vehicle with role_name 'hero'
            hero_id = -1
            for line in log_info.split('\n'):
                if 'role_name = hero' in line:
                    hero_id = int(line.split(' id = ')[1].split(' ')[0])
                    break
            
            if hero_id == -1:
                raise RuntimeError("Could not find an actor with role_name='hero' in the log file.")

            print(f"Log Info: Map='{map_name}', Hero Actor ID='{hero_id}'")

        except Exception as e:
            logging.error(f"Error parsing log file info. Make sure the file exists and is valid.")
            logging.error(f"Details: {e}")
            return

        # 3. Load the correct world map
        world = client.load_world(map_name)
        carla_map = world.get_map()
        print(f"World loaded with map: {carla_map.name}")

        # Put the world into synchronous mode for controlled replay analysis
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # Replay at 20 FPS, can be adjusted
        world.apply_settings(settings)

        # 4. Start the replay
        # The duration=0 means it will play the entire file.
        # The follow_id=0 means the server won't move the spectator camera.
        client.replay_file(args.file, start=0, duration=0, follow_id=0)
        print("Replay started. Analyzing frames...")

        # We need to wait a moment for the replay to spawn the actors
        world.tick()
        time.sleep(1)

        ego_vehicle = world.get_actor(hero_id)
        if ego_vehicle is None:
            raise RuntimeError(f"Could not find replayed hero vehicle with ID {hero_id} in the world.")

        print(f"Successfully attached to replayed ego vehicle: {ego_vehicle.type_id}")

        previous_lane_id = None
        frame_count = 0

        # 5. Main analysis loop
        # We tick the world to advance the replay one frame at a time.
        # The loop will end when the replay is finished and the actor is destroyed.
        while world.get_actor(hero_id) is not None:
            world.tick()
            frame_count += 1

            location = ego_vehicle.get_location()
            
            # Get the waypoint corresponding to the vehicle's location
            # Using LaneType.Any allows us to detect crossings from driving lanes to shoulders, etc.
            waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
            
            if waypoint is None:
                # Vehicle might be off-road, skip this frame
                continue
                
            current_lane_id = waypoint.lane_id

            # Initialize the previous_lane_id on the first valid frame
            if previous_lane_id is None:
                previous_lane_id = current_lane_id

            # Check if the lane ID has changed since the last frame
            if current_lane_id != previous_lane_id:
                total_crossings += 1
                
                # Get details about the line that was crossed
                # We check the marking of the lane we *came from*
                # To do this, we need to find the waypoint for the previous lane
                # Note: This is a simplified check. A robust check would need to handle road topology.
                # Here, we just check the current waypoint's marking based on direction.
                
                marking_type = "Unknown"
                if current_lane_id < previous_lane_id: # Moved to a lane to the right
                    marking = waypoint.get_left_lane_marking() # We crossed the line to our left
                else: # Moved to a lane to the left
                    marking = waypoint.get_right_lane_marking() # We crossed the line to our right
                
                if marking:
                    marking_type = marking.type.name
                
                print(f"Frame {frame_count}: Lane crossing detected! "
                      f"From lane {previous_lane_id} to {current_lane_id}. "
                      f"Line type: {marking_type}. "
                      f"Total crossings: {total_crossings}")

            previous_lane_id = current_lane_id

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()

    finally:
        # Clean up
        if world is not None and original_settings is not None:
            print("\nResetting world settings to asynchronous mode.")
            world.apply_settings(original_settings)
        
        print("\n--- Analysis Complete ---")
        print(f"Total lane crossings detected: {total_crossings}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Log File Lane Crossing Analyzer')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host CARLA server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of the CARLA server (default: 2000)')
    argparser.add_argument(
        '-f', '--file',
        required=True,
        help='Path to the CARLA log file to be analyzed')
    
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        analyze_lane_crossings(args)
    except KeyboardInterrupt:
        print('\nAnalysis cancelled by user.')