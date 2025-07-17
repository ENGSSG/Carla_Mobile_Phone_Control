#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
# ... (rest of the copyright notice) ...

"""Allows controlling a vehicle with an android phone's accelerometer."""

# ... (rest of the docstring/help text - consider updating if needed) ...

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import os
import weakref
import socket
import threading
import time
import sys
import traceback
import multiprocessing
import json
import queue
from mpc_controller import MPCProcess
import cv2


try:
    import pygame
    # ... (Keep all necessary pygame imports) ...
    from pygame.locals import KMOD_CTRL, KMOD_SHIFT, K_ESCAPE, K_q, K_SPACE # etc.
    # Ensure all required K_ constants are imported
    from pygame.locals import K_0,K_k, K_9, K_BACKQUOTE, K_BACKSPACE, K_COMMA, K_DOWN, K_F1, K_LEFT, K_PERIOD, K_RIGHT, K_SLASH, K_TAB, K_UP, K_a, K_b, K_c, K_d, K_f, K_g, K_h, K_i, K_l, K_m, K_n, K_o, K_p, K_r, K_s, K_t, K_v, K_w, K_x, K_z, K_MINUS, K_EQUALS

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ... (OBJECT_TO_COLOR list remains the same) ...

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

# ... (find_weather_presets, get_actor_display_name, get_actor_blueprints remain the same) ...
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3, 4]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

# ... (World class remains largely the same, ensure IMUSensor is still used for HUD display if needed) ...
# Minor change: Remove gyroscope display from HUD later.
class World(object):
    def __init__(self, carla_world, hud, traffic_manager, args):
        self.world = carla_world
        self.sync = args.sync
        self.traffic_manager = traffic_manager
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self._args = args
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None # Keep IMU for HUD display if desired (accel data)
        self.radar_sensor = None
        self.camera_manager = None
        self.lka_camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self._shutdown_requested = False
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE, carla.MapLayer.Buildings, carla.MapLayer.Decals,
            carla.MapLayer.Foliage, carla.MapLayer.Ground, carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles, carla.MapLayer.Props, carla.MapLayer.StreetLights,
            carla.MapLayer.Walls, carla.MapLayer.All
        ]

    def restart(self): # Removed args from here, should be passed during init
            # If a player exists, destroy it cleanly before restarting.
        if self.player is not None:
            self.destroy() # This will call destroy_sensors and destroy the player actor

        # --- NEW CONDITIONAL LOGIC ---
        if self._args.waitForEgo:
            # --- "WAITING" MODE ---
            # This block is for when running with scenario_runner
            print("Waiting for ego vehicle spawned by scenario_runner...")
            self.player = None
            while self.player is None:
                time.sleep(0.5)
                possible_vehicles = self.world.get_actors().filter('vehicle.*')
                for vehicle in possible_vehicles:
                    if vehicle.attributes['role_name'] == self._args.rolename:
                        print("Ego vehicle found!")
                        self.player = vehicle
                        break
                if self._shutdown_requested: # Add a check to exit if the script is closing
                    return
        else:
            # --- "SPAWNING" MODE ---
            # This is your original logic for running standalone
            print("Spawning new ego vehicle...")
            cam_index = self.camera_manager.index if self.camera_manager is not None else 0
            cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
            
            blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
            if not blueprint_list:
                raise ValueError("Couldn't find any blueprints with the specified filters")
            
            blueprint = random.choice(blueprint_list)
            blueprint.set_attribute('role_name', self._args.rolename) # Use rolename from args
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))

            # Spawn the vehicle
            self.player = None
            if self.map.get_spawn_points():
                spawn_point = random.choice(self.map.get_spawn_points())
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
                while self.player is None:
                    spawn_point = random.choice(self.map.get_spawn_points())
                    self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            else:
                print("Map has no spawn points! Cannot spawn vehicle.")
                sys.exit(1)

         # --- SENSOR SETUP (This runs for both modes) ---
        # This part of the code is the same for both cases, as sensors need to be
        # attached to the 'self.player' vehicle, regardless of how it was obtained.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)

        # Keep previous camera index if it exists
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        
        # We manually create the LKA camera manager here with different settings
        lka_hud_placeholder = HUD(640, 480) # Placeholder HUD for camera attributes
        lka_camera_manager = CameraManager(self.player, lka_hud_placeholder, self._gamma, is_for_display=False)
        lka_camera_manager.transform_index = 1 # A fixed forward view
        
        # Set a low resolution specifically for LKA
        lka_bp = lka_camera_manager.sensors[0][-1] # Get the RGB blueprint
        lka_bp.set_attribute('image_size_x', '640')
        lka_bp.set_attribute('image_size_y', '480')
        # --- MODIFICATION START ---
        # Also control the data rate of the LKA camera
        lka_bp.set_attribute('sensor_tick', '0.022222') # 20 FPS is plenty for LKA
        # --- MODIFICATION END ---
        lka_camera_manager.set_sensor(0, notify=False)
        self.lka_camera_manager = lka_camera_manager # Attach it to the world object
        # LKA camera setup
        # self.lka_camera_manager = CameraManager(self.player, self.hud, self._gamma)
        # self.lka_camera_manager.transform_index = 1 # A fixed forward view
        # self.lka_camera_manager.set_sensor(0, notify=False) # Use the first sensor (RGB)

        self.hud.notification(get_actor_display_name(self.player))
        if self.traffic_manager:
            self.traffic_manager.update_vehicle_lights(self.player, True)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception: pass # Actor might not be a vehicle

    def tick(self, clock):
        if self.camera_manager:
            self.camera_manager.tick()
        if self.lka_camera_manager:
            self.lka_camera_manager.tick()

        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        if self.camera_manager and self.camera_manager.sensor:
             self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None: self.toggle_radar()
        sensors = [
            self.camera_manager.sensor if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor if self.gnss_sensor else None,
            self.imu_sensor.sensor if self.imu_sensor else None # Keep IMU sensor
        ]
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception as e:
                    print(f"Error destroying sensor: {e}")
        if self.player is not None:
            try:
                self.player.destroy()
            except Exception as e:
                 print(f"Error destroying player: {e}")
        self.player = None # Ensure player is None after destruction

class SensorControl(object):
    def __init__(self, world, start_in_autopilot,
                 mpc_vehicle_data_address=('127.0.0.1', 6003),
                 control_cmd_server_address=('127.0.0.1', 6004),
                 mpc_image_data_address=('127.0.0.1', 6005)):
        self._world_ref = weakref.ref(world)
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE

        # IPC Sockets Configuration
        self.mpc_vehicle_data_address = mpc_vehicle_data_address
        self.control_cmd_server_address = control_cmd_server_address
        self.mpc_image_data_address = mpc_image_data_address

        # Sockets for sending data to MPC (client) - UDP
        self.vehicle_data_socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_data_socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_frame_id = 0

        # Server socket for receiving control commands from MPC - UDP
        self.control_cmd_socket_server = None

        self._data_lock = threading.Lock()
        self._last_mpc_control_update_time = time.time()

        # --- Queues for outgoing data to MPC ---
        self.vehicle_data_send_queue = queue.Queue(maxsize=10)
        # NEW: Queue for images to be encoded and sent
        self.image_send_queue = queue.Queue(maxsize=50) # Small buffer to prevent memory bloat

        # --- LKA Sending Throttle Configuration ---
        self.LKA_TARGET_FPS = 20.0  # Send images to MPC at this rate
        self.LKA_SEND_INTERVAL = 1.0 / self.LKA_TARGET_FPS
        self._last_lka_send_time = 0.0

        # --- Thread for Vehicle Data Sender ---
        self._vehicle_data_sender_running_flag = threading.Event(); self._vehicle_data_sender_running_flag.set()
        self._vehicle_data_sender_thread = threading.Thread(target=self._run_vehicle_data_sender, daemon=True)
        self._vehicle_data_sender_thread.start()
        print(f"Android_control: Vehicle data sender thread started for MPC @ {self.mpc_vehicle_data_address}")

        # --- NEW: Thread for Image Encoder/Sender ---
        self._image_sender_running_flag = threading.Event(); self._image_sender_running_flag.set()
        self._image_sender_thread = threading.Thread(target=self._run_image_sender, daemon=True)
        self._image_sender_thread.start()
        print(f"Android_control: Image sender thread started for MPC @ {self.mpc_image_data_address}")

        # --- Thread for Control Command Server ---
        self._control_server_running_flag = threading.Event(); self._control_server_running_flag.set()
        self._control_server_thread = threading.Thread(target=self._run_control_command_server, daemon=True)
        self._control_server_thread.start()
        print(f"Android_control: Control command server thread started on {self.control_cmd_server_address}")

        world_instance = self._get_world()
        if world_instance and world_instance.player:
            player = world_instance.player
            if isinstance(player, carla.Vehicle):
                player.set_autopilot(self._autopilot_enabled)
                player.set_light_state(self._lights)
                player.apply_control(self._control)
            else: self._control = None # Not a vehicle
            if world_instance.hud: world_instance.hud.notification("Sensor Control (IPC via Sockets). Press 'H'.", seconds=4.0)
        else: self._control = None; print("Error: SensorControl initialized with invalid world or player.")

    def _send_image_data(self, image_array):
        if image_array is None:
            return False
        
        
        try:
            # --- MODIFICATION START ---
            # We no longer encode to JPEG. We send the raw numpy array bytes.
            image_bytes = image_array.tobytes()
            image_shape = image_array.shape # (height, width, channels)
            shape_header_bytes = image_shape[0].to_bytes(2, 'big') + \
                                 image_shape[1].to_bytes(2, 'big') + \
                                 image_shape[2].to_bytes(1, 'big')
            # --- MODIFICATION END ---

            CHUNK_SIZE = 65000
            self.image_frame_id = (self.image_frame_id + 1) % 4294967295
            frame_id_bytes = self.image_frame_id.to_bytes(4, 'big')
            
            # --- MODIFICATION START ---
            # Use the raw image_bytes instead of encoded_image_bytes
            total_chunks = math.ceil(len(image_bytes) / CHUNK_SIZE)
            # --- MODIFICATION END ---

            if total_chunks > 255:
                print(f"Android_control: Image too large to chunk for UDP ({total_chunks} chunks). Skipping frame.")
                return False

            # --- MODIFICATION START ---
            for i, chunk_start in enumerate(range(0, len(image_bytes), CHUNK_SIZE)):
                chunk_data = image_bytes[chunk_start:chunk_start + CHUNK_SIZE]
                # Packet Header: [frame_id(4b)] [chunk_idx(1b)] [total_chunks(1b)]
                udp_header = frame_id_bytes + i.to_bytes(1, 'big') + total_chunks.to_bytes(1, 'big')
                # Prepend the shape header to every packet for simplicity and robustness
                packet = udp_header + shape_header_bytes + chunk_data
            # --- MODIFICATION END ---
                self.image_data_socket_client.sendto(packet, self.mpc_image_data_address)
            
            return True
        except Exception as e:
            print(f"Android_control (ImageSender): Unexpected error sending image data: {e}")
            traceback.print_exc()
            return False

    # NEW: Worker thread function for encoding and sending images
    def _run_image_sender(self):
        """Dedicated thread to get images from a queue, encode, and send them at a throttled rate."""
        while self._image_sender_running_flag.is_set():
            try:
                # Block until an image is available in the queue
                image_to_send = self.image_send_queue.get(timeout=1.0)
                if image_to_send is None: # Sentinel value to stop the thread
                    break

                # --- LKA SENDING THROTTLE LOGIC ---
                current_time = time.time()
                if (current_time - self._last_lka_send_time) > self.LKA_SEND_INTERVAL:
                    # If enough time has passed, encode and send the image
                    if self._send_image_data(image_to_send):
                        self._last_lka_send_time = current_time
                # If not enough time has passed, the frame is simply dropped,
                # preventing the network stack from being overwhelmed.

            except queue.Empty:
                continue # Loop back to wait for more data
            except Exception as e:
                print(f"Android_control (ImageSender): Unexpected error: {e}")
                traceback.print_exc()

        print("Android_control: Image sender thread stopping.")

    def _actual_send_vehicle_data(self, data_to_mpc_json_str):
        if not self.vehicle_data_socket_client:
            return False
        try:
            self.vehicle_data_socket_client.sendto(data_to_mpc_json_str.encode('utf-8'), self.mpc_vehicle_data_address)
            return True
        except socket.error as e:
            # Suppress frequent errors in async mode if needed
            # print(f"Android_control (SenderThread-UDP): Error sending vehicle data: {e}.")
            return False
        except Exception as e:
            print(f"Android_control (SenderThread-UDP): Unexpected error sending vehicle data: {e}")
            return False
        
    def _run_vehicle_data_sender(self):
        while self._vehicle_data_sender_running_flag.is_set():
            try:
                data_to_send = self.vehicle_data_send_queue.get(timeout=1.0) 
                if data_to_send is None:
                    break
                
                message_json = json.dumps(data_to_send)
                self._actual_send_vehicle_data(message_json) # Removed newline for UDP

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Android_control (SenderThread): Unexpected error: {e}")
                traceback.print_exc()
        
        print("Android_control: Vehicle data sender thread stopping.")

    def _run_control_command_server(self):
        try:
            self.control_cmd_socket_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.control_cmd_socket_server.settimeout(1)
            self.control_cmd_socket_server.bind(self.control_cmd_server_address)
            print(f"Android_control: Control command UDP server listening on {self.control_cmd_server_address}")
        except Exception as e:
            print(f"Android_control: Failed to set up control command UDP server: {e}")
            self.control_cmd_socket_server = None; return

        while self._control_server_running_flag.is_set():
            try:
                data, _ = self.control_cmd_socket_server.recvfrom(1024)
                if not data:
                    continue
                
                mpc_command = json.loads(data.decode('utf-8'))
                with self._data_lock: 
                    self._control.steer = mpc_command.get('steer', self._control.steer)
                    self._control.throttle = mpc_command.get('throttle', self._control.throttle)
                    self._control.brake = mpc_command.get('brake', self._control.brake)
                    self._control.reverse = mpc_command.get('reverse', self._control.reverse)
                    self._last_mpc_control_update_time = time.time()
            except socket.timeout:
                continue
            except Exception as e:
                if self._control_server_running_flag.is_set():
                    # print(f"Android_control (UDP): Error in control command server loop: {e}")
                    pass
        
        print("Android_control: Control command server thread stopping.")
            
    def _get_world(self):
        return self._world_ref()

    def parse_events(self, client, world, clock, sync_mode):
        world_instance = self._get_world()
        if not world_instance or not world_instance.player or not world_instance.player.is_alive:
            return False # Simplified exit condition

        player = world_instance.player
        
        transform = player.get_transform(); velocity = player.get_velocity(); physics_control = player.get_physics_control()
        current_steer_angle_rad = np.radians(player.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
        max_physical_steer_angle_rad = np.radians(physics_control.wheels[0].max_steer_angle) if physics_control and physics_control.wheels else np.radians(70.0)
        current_vehicle_state_list = [
            transform.location.x, transform.location.y, np.radians(transform.rotation.yaw),
            np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2), current_steer_angle_rad ]
        collision_intensity = world_instance.collision_sensor.get_collision_intensity() if world_instance.collision_sensor else 0.0
        accel_vec = player.get_acceleration(); accel_mag = np.sqrt(accel_vec.x**2 + accel_vec.y**2 + accel_vec.z**2)
        data_to_mpc = {'state': current_vehicle_state_list, 'max_steer_rad': max_physical_steer_angle_rad,
                       'collision_intensity': collision_intensity, 'accel_magnitude': accel_mag, 'lka_toggle_request': False}
        
        # --- MODIFIED: Put image on queue instead of sending directly ---
        if world_instance and world_instance.lka_camera_manager:
            image_to_send = world_instance.lka_camera_manager.raw_image_array
            if image_to_send is not None:
                try:
                    # Put raw numpy array on the queue. Non-blocking.
                    self.image_send_queue.put_nowait(image_to_send)
                except queue.Full:
                    # This is okay, it just means we're dropping a frame because the
                    # encoder/sender can't keep up. The main loop is not blocked.
                    pass

        # --- Process Keyboard Events (Unchanged) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYUP and self._is_quit_shortcut(event.key)):
                return True # Signal to exit the game loop
            # ... (rest of the keyboard event logic remains the same)
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key): 
                    return True
                elif event.key == K_k: 
                    data_to_mpc['lka_toggle_request'] = True
                elif event.key == K_BACKSPACE:
                    world_instance.restart()
                    self._control = carla.VehicleControl(); self._lights = carla.VehicleLightState.NONE
                    if world_instance.player and isinstance(world_instance.player, carla.Vehicle):
                        world_instance.player.apply_control(self._control); world_instance.player.set_light_state(self._lights)
                        world_instance.player.set_autopilot(self._autopilot_enabled)
                elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                    self._autopilot_enabled = not self._autopilot_enabled
                    world_instance.player.set_autopilot(self._autopilot_enabled)
                    world_instance.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                # Other key events...
        
        try:
            self.vehicle_data_send_queue.put_nowait(data_to_mpc)
        except queue.Full:
            pass 
        
        with self._data_lock:
            time_since_last_mpc_update = time.time() - self._last_mpc_control_update_time
            control_to_apply_this_frame = carla.VehicleControl()
            control_to_apply_this_frame.steer = self._control.steer
            control_to_apply_this_frame.throttle = self._control.throttle
            control_to_apply_this_frame.brake = self._control.brake
            control_to_apply_this_frame.reverse = self._control.reverse
            
            # In async mode, a stale command can be dangerous. Apply brake if MPC is silent.
            if not sync_mode and time_since_last_mpc_update > 2.0: # Increased timeout to 2 seconds
                # If MPC commands are stale, continue using the last received control values.
                # This introduces control input delay without stopping the vehicle.
                pass

        if not self._autopilot_enabled and player.is_alive:
            keys = pygame.key.get_pressed()
            control_to_apply_this_frame.hand_brake = keys[K_SPACE]
            player.apply_control(control_to_apply_this_frame)
        
        return False # Continue game loop

    def stop(self):
        print("Android_control: SensorControl stop called.")
        # Signal and join all threads
        self._vehicle_data_sender_running_flag.clear()
        self._image_sender_running_flag.clear() # Signal new thread
        self._control_server_running_flag.clear()

        # Unblock queues with sentinel values
        try: self.vehicle_data_send_queue.put_nowait(None)
        except queue.Full: pass
        try: self.image_send_queue.put_nowait(None) # Unblock new queue
        except queue.Full: pass

        # Join threads
        if hasattr(self, '_vehicle_data_sender_thread') and self._vehicle_data_sender_thread.is_alive():
            self._vehicle_data_sender_thread.join(timeout=1.0)
        if hasattr(self, '_image_sender_thread') and self._image_sender_thread.is_alive():
            self._image_sender_thread.join(timeout=1.0)
        if hasattr(self, '_control_server_thread') and self._control_server_thread.is_alive():
            self._control_server_thread.join(timeout=1.0)
        
        # Close sockets
        if self.vehicle_data_socket_client: self.vehicle_data_socket_client.close()
        if self.image_data_socket_client: self.image_data_socket_client.close()
        if self.control_cmd_socket_server: self.control_cmd_socket_server.close()
        print("Android_control: SensorControl sockets and threads stopped/closed.")

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)




# ==============================================================================
# -- HUD (Largely Unchanged) ---------------------------------------------------
# ==============================================================================
class HUD(object): # Definition remains the same as previous refactor
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._show_ackermann_info = False 
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame # Use timestamp.frame directly
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock): # world is an instance of World class
        self._notifications.tick(world, clock) # Pass world for consistency, though not used by FadingText
        if not self._show_info: return
        
        # Ensure world and player are valid before accessing attributes
        if not world or not world.player or not world.player.is_alive:
             self._info_text = ["Player not available"] 
             return

        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control() 

        compass = world.imu_sensor.compass if world.imu_sensor and world.imu_sensor.sensor else 0.0
        accelerometer = world.imu_sensor.accelerometer if world.imu_sensor and world.imu_sensor.sensor else (0.0,0.0,0.0)
        
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''

        colhist = world.collision_sensor.get_collision_history() if world.collision_sensor and world.collision_sensor.sensor else {}
        collision = [colhist.get(x + self.frame - 200, 0.0) for x in range(0, 200)]
        max_col = max(1.0, max(collision) if collision else 1.0) 
        collision = [x / max_col for x in collision]

        vehicles = world.world.get_actors().filter('vehicle.*')
        gnss_lat = world.gnss_sensor.lat if world.gnss_sensor and world.gnss_sensor.sensor else 0.0
        gnss_lon = world.gnss_sensor.lon if world.gnss_sensor and world.gnss_sensor.sensor else 0.0

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(), '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)), '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % accelerometer,
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (gnss_lat, gnss_lon)),
            'Height:  % 18.0f m' % t.location.z, '' ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0), ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0), ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake), ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear) ]
            if self._show_ackermann_info:
                 self._info_text += ['', 'Ackermann Controller:', '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [('Speed:', c.speed, 0.0, 5.556), ('Jump:', c.jump)]
        self._info_text += ['', 'Collision:', collision, '', 'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0: break
                self._info_text.append('% 4dm %s' % (d, get_actor_display_name(vehicle, truncate=22)))

    def show_ackermann_info(self, enabled): self._show_ackermann_info = enabled
    def update_ackermann_control(self, ackermann_control): self._ackermann_control = ackermann_control
    def toggle_info(self): self._show_info = not self._show_info
    def notification(self, text, seconds=2.0): self._notifications.set_text(text, seconds=seconds)
    def error(self, text): self._notifications.set_text('Error: %s' % text, (255, 0, 0))
    def render(self, display): # Identical to previous version
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1])); info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0)); v_offset = 4; bar_h_offset = 100; bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]: break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None; v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2]) if (item[3] - item[2]) != 0 else 0
                        f = max(0.0, min(f, 1.0))
                        if item[2] < 0.0:
                             bar_offset = (f - 0.5) * (bar_width - 6)
                             rect = pygame.Rect((bar_h_offset + (bar_width/2) + bar_offset - 3 , v_offset + 8), (6, 6))
                        else: rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: surface = self._font_mono.render(item, True, (255, 255, 255)); display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display); self.help.render(display)


# ==============================================================================
# -- FadingText, HelpText (Keep as is) -----------------------------------------
# ==============================================================================
class FadingText(object):
    # ... (Keep class implementation as is) ...
    def __init__(self, font, dim, pos):
        self.font = font; self.dim = dim; self.pos = pos
        self.seconds_left = 0; self.surface = pygame.Surface(self.dim)
    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim); self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0)); self.surface.blit(text_texture, (10, 11))
    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)
    def render(self, display): display.blit(self.surface, self.pos)

class HelpText(object):
    # ... (Keep class implementation as is) ...
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font; self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0; self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
        self._render = False; self.surface.set_alpha(220)
    def toggle(self): self._render = not self._render
    def render(self, display):
        if self._render: display.blit(self.surface, self.pos)


# ==============================================================================
# -- Sensor Classes (Collision, LaneInvasion, Gnss, IMU, Radar, CameraManager) -
# ==============================================================================

# --- CollisionSensor, LaneInvasionSensor, GnssSensor ---
# Keep these classes as they are. They don't depend on the gyroscope.
class CollisionSensor(object):
    # ... (Keep class implementation as is) ...
    def __init__(self, parent_actor, hud):
        self.sensor = None; self.history = []; self._parent = parent_actor; self.hud = hud
        world = self._parent.get_world(); bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self); self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
    
    def get_collision_history(self): # For HUD
        history = collections.defaultdict(int); [history.__setitem__(frame, history[frame]+intensity) for frame, intensity in self.history]
        return history

    def get_collision_intensity(self): # For MPC haptics
        if not self.history: return 0.0
        # Return intensity of the most recent collision, or average over a very short window
        return self.history[-1][1] if self.history else 0.0

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self();
        if not self: return
        actor_type = get_actor_display_name(event.other_actor); self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse; intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity));
        if len(self.history) > 200: self.history.pop(0) 

class LaneInvasionSensor(object):
    # ... (Keep class implementation as is) ...
     def __init__(self, parent_actor, hud):
        self.sensor = None
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor; self.hud = hud; world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            weak_self = weakref.ref(self); self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
     @staticmethod
     def _on_invasion(weak_self, event):
        self = weak_self();
        if not self: return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

class GnssSensor(object):
    # ... (Keep class implementation as is) ...
    def __init__(self, parent_actor):
        self.sensor = None; self._parent = parent_actor; self.lat = 0.0; self.lon = 0.0
        world = self._parent.get_world(); bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        weak_self = weakref.ref(self); self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))
    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self();
        if not self: return
        self.lat = event.latitude; self.lon = event.longitude

# --- IMUSensor ---
# Keep this class, but the HUD will no longer display gyroscope data.
# The accelerometer data is still useful for the HUD.
class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None; self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0) # Keep variable, but won't be used by control/HUD
        self.compass = 0.0
        world = self._parent.get_world(); bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self();
        if not self: return
        limits = (-99.9, 99.9)
        # Store accelerometer data
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        # Store gyroscope data (even if not used by control)
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)

# --- RadarSensor, CameraManager ---
# Keep these classes as they are.
class RadarSensor(object):
    # ... (Keep class implementation as is) ...
    def __init__(self, parent_actor):
        self.sensor = None; self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x; bound_y = 0.5 + self._parent.bounding_box.extent.y; bound_z = 0.5 + self._parent.bounding_box.extent.z
        self.velocity_range = 7.5; world = self._parent.get_world(); self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35)); bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=bound_x + 0.05, z=bound_z+0.05), carla.Rotation(pitch=5)), attach_to=self._parent)
        weak_self = weakref.ref(self); self.sensor.listen(lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))
    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self();
        if not self: return
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth); alt = math.degrees(detect.altitude)
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(carla.Location(), carla.Rotation(pitch=current_rot.pitch + alt, yaw=current_rot.yaw + azi, roll=current_rot.roll)).transform(fw_vec)
            def clamp(min_v, max_v, value): return max(min_v, min(value, max_v))
            norm_velocity = detect.velocity / self.velocity_range
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0); g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0); b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(radar_data.transform.location + fw_vec, size=0.075, life_time=0.06, persistent_lines=False, color=carla.Color(r, g, b))

class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, is_for_display=True):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.raw_image_array = None
        self._latest_image = None
        self._image_lock = threading.Lock()
        self._is_for_display = is_for_display # New flag

        bound_x = 0.5 + parent_actor.bounding_box.extent.x
        bound_y = 0.5 + parent_actor.bounding_box.extent.y
        bound_z = 0.5 + parent_actor.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-0.05, z=bound_z + 0.5)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9 * bound_x, y=+1.0 * bound_y, z=1.2 * bound_z)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8 * bound_x, y=+0.0 * bound_y, z=4.6 * bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0 * bound_y, z=0.4 * bound_z)), Attachment.Rigid)
        ]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.lidar.ray_cast_semantic', None, 'Semantic Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    # --- MODIFICATION START ---
                # Set the sensor to capture at a fixed rate (e.g., 20 FPS)
                # This drastically reduces network traffic in async mode.
                # A value of 0.0 means it captures every simulation tick.
                # A value of 0.05 means it captures every 0.05s (20 FPS).
                bp.set_attribute('sensor_tick', '0.016')
                # --- MODIFICATION END ---
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def tick(self):
        """
        New method to be called once per game loop.
        It processes the latest image received from the callback.
        """
        with self._image_lock:
            image = self._latest_image
        
        if image:
            if self.sensors[self.index][0].startswith('sensor.lidar'):
                points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                lidar_data = np.array(points[:, :2])
                lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
                lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
                lidar_data = np.fabs(lidar_data)
                lidar_data = lidar_data.astype(np.int32)
                lidar_data = np.reshape(lidar_data, (-1, 2))
                lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
                lidar_img = np.zeros(lidar_img_size, dtype=np.uint8)
                lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
                self.surface = pygame.surfarray.make_surface(lidar_img)
            elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
                # Example of displaying DVS events
                dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                    ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
                dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                # Blue is positive, red is negative
                dvs_img[dvs_events['y'], dvs_events['x'], 2 * (1 - dvs_events['pol'])] = 255
                self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
            elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
                image = image.get_color_coded_flow()
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            else:
                image.convert(self.sensors[self.index][1])
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                if self._is_for_display:
                    self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
            
            # This is now updated only once per frame, making it stable
            self.raw_image_array = array
            if self.recording:
                image.save_to_disk('_out/%08d' % image.frame)

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            
            # The callback is now lightweight
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        """
        This callback is now extremely lightweight.
        It just stores the latest image from CARLA in a thread-safe way.
        """
        self = weak_self()
        if not self:
            return
        with self._image_lock:
            self._latest_image = image


FIXED_SIMULATION_STEP_TIME = 0.017
# ==============================================================================
# -- game_loop() (Updated for MPC Process) -------------------------------------
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    controller = None
    mpc_process_instance = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        sim_world = client.get_world()

        # Get a client handle to the Traffic Manager.
        # This does not cause a conflict. Scenario Runner is the master.
        traffic_manager = client.get_trafficmanager()

        if args.sync:
            # We only need to set the world to sync mode.
            # Scenario Runner will handle setting the TM to sync mode.
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.delta_seconds # Use the arg
            sim_world.apply_settings(settings)
            print(f"Android_control.py: World set to synchronous mode with dt={args.delta_seconds}")

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0)); pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, traffic_manager, args)
        # MPC Process Setup
        mpc_dt = FIXED_SIMULATION_STEP_TIME if args.sync else 0.016
        mpc_params = {'horizon_N': 10, 'dt': mpc_dt, 'wheelbase': 2.5}
        phone_tcp_config = {"host": args.mpc_phone_host, "port": args.mpc_phone_port}
        
        max_speed_mps_limit = 0.0
        if args.max_speed_kph > 0:
            max_speed_mps_limit = args.max_speed_kph / 3.6

        # MPC Process Setup
        mpc_params = {'horizon_N': 10, 'dt': args.delta_seconds if args.sync else 0.016, 'wheelbase': 2.5}
        phone_tcp_config = {"host": args.mpc_phone_host, "port": args.mpc_phone_port}
        control_config = { # Pass all relevant control tuning parameters from args
            "steer_angle_sensitivity": args.steer_angle_sensitivity,
            "steering_angle_deadzone": args.steering_angle_deadzone,
            "max_vehicle_steer_angle_deg": args.max_vehicle_steer_angle_deg,
            "steer_rate_sensitivity": args.steer_rate_sensitivity,
            "steering_rate_deadzone": args.steering_rate_deadzone,
            "max_vehicle_steer_rate_deg_s": args.max_vehicle_steer_rate_deg_s,
            "max_ay_for_scaling": args.max_ay_for_scaling,
            "max_gz_for_scaling_rps": args.max_gz_for_scaling_rps,
            "haptic_collision_threshold": args.haptic_collision_threshold,
            "haptic_accel_threshold": args.haptic_accel_threshold,
            "data_history_size": args.data_history_size,
            "max_speed_mps": max_speed_mps_limit,
            # ---- New Config for Gradual Throttle & PID ----
            "throttle_increase_rate": args.throttle_increase_rate,
            "throttle_decrease_rate": args.throttle_decrease_rate,
            "pid_kp": args.pid_kp,
            "pid_ki": args.pid_ki,
            "pid_kd": args.pid_kd
        }
        
        vehicle_data_server_addr = (args.ipc_host, args.ipc_vehicle_data_port)
        control_cmd_client_addr = (args.ipc_host, args.ipc_control_cmd_port)
        image_data_server_addr = (args.ipc_host, args.ipc_image_data_port)

       

        controller = SensorControl(world, args.autopilot,
                                   mpc_vehicle_data_address=vehicle_data_server_addr,
                                   control_cmd_server_address=control_cmd_client_addr,
                                   mpc_image_data_address=image_data_server_addr)
        time.sleep(0.5) 

        mpc_process_instance = MPCProcess(
            mpc_params, phone_tcp_config, control_config,
            vehicle_data_server_address=vehicle_data_server_addr,
            control_cmd_client_address=control_cmd_client_addr,
            image_data_server_address=image_data_server_addr # Pass new address
            )
        mpc_process_instance.start()
        print("Game_loop: MPC Process started.")

        # SensorControl now uses socket addresses
        
        
        if args.sync: sim_world.tick()
        else: sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync: sim_world.tick(); clock.tick() 
            else: clock.tick_busy_loop(60) # Limit client FPS if async

            if not mpc_process_instance.is_alive() and mpc_process_instance.exitcode is not None:
                print(f"Game_loop: MPC process has exited with code {mpc_process_instance.exitcode}. Stopping.")
                break
            
            if controller.parse_events(client, world, clock, args.sync): break 
            
            if world: world.tick(clock)
            if world: world.render(display)
            pygame.display.flip()

    except Exception as e:
         logging.error("Error during game loop: %s", e); traceback.print_exc()
    finally:
        print("Game_loop: Initiating cleanup...")
        if isinstance(controller, SensorControl): controller.stop()

        if mpc_process_instance:
            print("Game_loop: Signaling MPC process to stop...")
            mpc_process_instance.stop() 
            if mpc_process_instance.is_alive(): mpc_process_instance.join(timeout=2.0)
            if mpc_process_instance.is_alive():
                print("Game_loop: MPC process did not exit cleanly, terminating.")
                mpc_process_instance.terminate(); mpc_process_instance.join(timeout=1.0)
        print("Game_loop: MPC Process cleanup attempt finished.")

        if original_settings and sim_world:
             try: sim_world.apply_settings(original_settings)
             except Exception as e: print(f"Error restoring world settings: {e}")
        if world and world.recording_enabled:
            try: client.stop_recorder()
            except Exception as e: print(f"Error stopping recorder: {e}")
        if world is not None:
            try: world.destroy()
            except Exception as e: print(f"Error destroying world: {e}")
        pygame.quit()
        print("Game_loop: Cleanup finished.")

# ==============================================================================
# -- main() (Updated with new args) --------------------------------------------
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client (Accelerometer via MPC)')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.tesla.model3', help='actor filter (default: "vehicle.tesla.model3")')
    argparser.add_argument('--generation', metavar='G', default='All', help='restrict to certain actor generation (values: "1","2","All" - default: "All")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument('--sync', action='store_true', help='Activate synchronous mode execution')
    # argparser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port for Traffic Manager (default: 8000)')
    argparser.add_argument('--delta_seconds', metavar='S', default=0.0017, type=float, help='Fixed delta seconds for synchronous mode (default: 0.05)')
    
    # Args for MPC phone listener (used by MPCProcess)
    argparser.add_argument('--mpc_phone_host', default='0.0.0.0', help='Host for MPC to listen for phone data (default: 0.0.0.0 to listen on all interfaces)')
    argparser.add_argument('--mpc_phone_port', default=6002, type=int, help='Port for MPC to listen for phone data (default: 6002)')

    # NEW ARGS for IPC Sockets
    argparser.add_argument('--ipc_host', default='127.0.0.1', help='Host for IPC sockets between Android_control and MPC (default: 127.0.0.1)')
    argparser.add_argument('--ipc_vehicle_data_port', default=6003, type=int, help='Port for Android_control to send vehicle data to MPC (default: 6003)')
    argparser.add_argument('--ipc_control_cmd_port', default=6004, type=int, help='Port for MPC to send control commands to Android_control (default: 6004)')
    argparser.add_argument('--ipc_image_data_port', default=6005, type=int, help='Port for Android_control to send camera images to MPC (default: 6005)')

    # Args for control_config (passed to MPCProcess)
    argparser.add_argument('--steer_angle_sensitivity', default=0.7, type=float, help='Sensitivity for AY to desired steer angle.')
    argparser.add_argument('--steering_angle_deadzone', default=0.2, type=float, help='Deadzone for AY for steering angle (m/s^2).')
    argparser.add_argument('--max_vehicle_steer_angle_deg', default=70.0, type=float, help='Max physical steer angle of vehicle wheels (deg) for normalization.')
    argparser.add_argument('--steer_rate_sensitivity', default=0.8, type=float, help='Sensitivity for GZ to desired steer rate.')
    argparser.add_argument('--steering_rate_deadzone', default=0.16, type=float, help='Deadzone for GZ for steering rate (rad/s).')
    argparser.add_argument('--max_vehicle_steer_rate_deg_s', default=100.0, type=float, help='Max desired steer rate (deg/s).')
    argparser.add_argument('--max_ay_for_scaling', default=9.8, type=float, help='Assumed max relevant AY for scaling input.')
    argparser.add_argument('--max_gz_for_scaling_rps', default=5.0, type=float, help='Assumed max relevant GZ for scaling input (rad/s).')
    argparser.add_argument('--max_speed_kph', default=0, type=float, help='Vehicle speed limit (kph). 0 for blueprint default. (REMOVED from MPCProcess, handled by World/Player)')
    argparser.add_argument('--haptic_collision_threshold', default=0.5, type=float, help='Collision intensity threshold for haptic feedback.')
    argparser.add_argument('--haptic_accel_threshold', default=7.0, type=float, help='Vehicle acceleration magnitude threshold for haptic feedback (m/s^2).')
    argparser.add_argument('--data_history_size', default=5, type=int, help="Number of samples for WMA filter in MPC.")

    # ---- New Args for Gradual Throttle & PID ----
    argparser.add_argument('--throttle_increase_rate', default=0.7, type=float, help='Rate at which throttle increases (units/sec) when accel is pressed.')
    argparser.add_argument('--throttle_decrease_rate', default=1.0, type=float, help='Rate at which throttle decreases (units/sec) when accel is released.')
    argparser.add_argument('--pid_kp', default=0.6, type=float, help='PID Proportional gain for speed control.') # Tuned Kp
    argparser.add_argument('--pid_ki', default=0.15, type=float, help='PID Integral gain for speed control.')    # Tuned Ki
    argparser.add_argument('--pid_kd', default=0.08, type=float, help='PID Derivative gain for speed control.')  # Tuned Kd

    # --- NEW ARGUMENT ---
    argparser.add_argument(
        '--waitForEgo',
        action='store_true',
        help='Instead of spawning a new ego vehicle, wait for one with the correct rolename to be spawned by scenario_runner.'
    )

    args = argparser.parse_args()

    try: args.width, args.height = [int(x) for x in args.res.split('x')]
    except ValueError: print("Error: Invalid resolution format."); sys.exit(1)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    
    try: game_loop(args)
    except KeyboardInterrupt: print('\nCancelled by user. Bye!')
    except Exception as e: logging.critical("Unhandled exception in main: %s", e); traceback.print_exc()

if __name__ == '__main__':
    # multiprocessing.freeze_support() 
    main()
