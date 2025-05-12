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
from mpc_controller import MPCProcess


try:
    import pygame
    # ... (Keep all necessary pygame imports) ...
    from pygame.locals import KMOD_CTRL, KMOD_SHIFT, K_ESCAPE, K_q, K_SPACE # etc.
    # Ensure all required K_ constants are imported
    from pygame.locals import K_0, K_9, K_BACKQUOTE, K_BACKSPACE, K_COMMA, K_DOWN, K_F1, K_LEFT, K_PERIOD, K_RIGHT, K_SLASH, K_TAB, K_UP, K_a, K_b, K_c, K_d, K_f, K_g, K_h, K_i, K_l, K_m, K_n, K_o, K_p, K_r, K_s, K_t, K_v, K_w, K_x, K_z, K_MINUS, K_EQUALS

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
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None # Keep IMU for HUD display if desired (accel data)
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
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

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'): blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'): blueprint.set_attribute('is_invincible', 'true')
        if blueprint.has_attribute('speed'):
            try: # Handle potential missing values or format issues
                rec_values = blueprint.get_attribute('speed').recommended_values
                if len(rec_values) >= 3:
                    self.player_max_speed = float(rec_values[1])
                    self.player_max_speed_fast = float(rec_values[2])
                elif len(rec_values) == 2: # Fallback if only two values
                     self.player_max_speed = float(rec_values[1])
                     self.player_max_speed_fast = float(rec_values[1]) # Use same value
            except (ValueError, IndexError, AttributeError) as e:
                 print(f"Warning: Could not set max speed from blueprint: {e}. Using defaults.")


        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player) # Keep for accel display in HUD
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        self.traffic_manager.update_vehicle_lights(self.player, True)

        if self.sync: self.world.tick()
        else: self.world.wait_for_tick()

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


# ==============================================================================
# -- SensorControl -------------------------------------------------------------
# ==============================================================================

class SensorControl(object):
    """
    Class that handles input events for CARLA.
    - Uses phone accelerometer data (received via integrated TCP listener) for driving.
    - Uses keyboard for auxiliary actions (HUD, camera, lights, etc.).
    - Manages its own background thread for receiving sensor data.
    """
    def __init__(self, world, start_in_autopilot,  mpc_conn_send, mpc_conn_recv):
        self._world_ref = weakref.ref(world)
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
                # IPC Pipes
        self.mpc_vehicle_data_pipe_send = mpc_conn_send
        self.mpc_control_cmd_pipe_recv = mpc_conn_recv

        # Initialize player state
        world_instance = self._get_world()
        if world_instance and world_instance.player:
            player = world_instance.player
            if isinstance(player, carla.Vehicle):
                player.set_autopilot(self._autopilot_enabled)
                player.set_light_state(self._lights)
                player.apply_control(self._control)
            else:
                print("Warning: SensorControl initialized with a non-vehicle player.")
                self._control = None
            if world_instance.hud:
                world_instance.hud.notification("Sensor Control Enabled. Press 'H' for help.", seconds=4.0)
        else:
            print("Error: SensorControl initialized with invalid world or player.")
            self._control = None

    def _get_world(self):
        return self._world_ref()

    def _get_sensor_data(self):
        with self._data_lock:
            # return self.accelerometer_data, self.reverse_enabled, self._is_connected
            return self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, self._is_connected
        self._is_connected

    def _update_sensor_data(self, ay_val, gz_val, accel_state, brake_state, reverse_state, is_connected):
         
        with self._data_lock:
        # self.accelerometer_data = accel # REMOVE
            self.ay_data = ay_val
            self.gz_data = gz_val
            self.accelerate_pressed = accel_state
            self.brake_pressed = brake_state
            self.reverse_enabled = bool(reverse_state) # Ensure boolean
            self._last_update_time = time.time()
            self._is_connected = is_connected
        
            # print(f"Data updated: Accel={self.ay_data}, accel={self.accelerate_pressed},accel={self.brake_pressed}, Connected={is_connected}") # Debug

    def _run_sensor_listener(self):
        server_socket = None
        client_conn = None
        client_addr = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self._tcp_host, self._tcp_port))
            server_socket.listen(1)
            print(f"SensorControl Listener: Socket bound and listening on {self._tcp_host}:{self._tcp_port}")
        except socket.error as msg:
            print(f"SensorControl Listener: Error setting up socket: {msg}")
            traceback.print_exc()
            if server_socket: server_socket.close()
            self._running_flag.clear() # Stop if setup fails
            return
        except Exception as e:
            print(f"SensorControl Listener: Unexpected setup error: {e}")
            traceback.print_exc()
            if server_socket: server_socket.close()
            self._running_flag.clear()
            return

        while self._running_flag.is_set():
            client_conn = None # Reset connection for each attempt
            self.client_conn = None
            try:
                # Update status to disconnected before waiting
                if not self._is_connected:
                     self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)

                server_socket.settimeout(1.0) # Timeout to check running_flag
                try:
                    # print("SensorControl Listener: Waiting to accept...") # Debug
                    client_conn, client_addr = server_socket.accept()
                    self.client_conn = client_conn
                    client_conn.settimeout(5.0) # Set a timeout for receiving data
                    print(f"SensorControl Listener: Connection from {client_addr}")
                    self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, True) # Mark as connected
                except socket.timeout:
                    self.client_conn = None # Ensure None on timeout
                    continue # Loop back to check running_flag
                except socket.error as e_accept:
                    print(f"SnnesorControl Listener: Error accepting connection: {e_accept}")
                    self.client_conn = None
                    time.sleep(0.5)
                    continue

                buffer = ""
                while self._running_flag.is_set():
                    try:
                        data = client_conn.recv(self._buffer_size)
                        if not data:
                            print(f"SensorControl Listener: Client {client_addr} disconnected.")
                            self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                            self.client_conn = None
                            break # Break inner loop

                        buffer += data.decode('utf-8')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line: continue

                            parts = line.split(',')
                            # Expecting 5 values: ay, gz, accelerate_state, brake_state, reverse_state
                            if len(parts) == 5:
                                try:
                                    ay = float(parts[0])
                                    gz = float(parts[1])
                                    accel_state = int(parts[2]) # 0 or 1
                                    brake_state = int(parts[3]) # 0 or 1
                                    reverse_state = int(parts[4]) # 0 or 1
                                    # Update internal data safely
                                    self._update_sensor_data(ay, gz, accel_state, brake_state, reverse_state, True)
                                except ValueError:
                                    print(f"SensorControl Listener: Invalid values from {client_addr}: {line}")
                                except Exception as parse_e:
                                        print(f"SensorControl Listener: Error parsing values from {client_addr}: {parse_e} - Data: {line}")
                            else:
                                print(f"SensorControl Listener: Malformed data from {client_addr}: Got {len(parts)} vals. Expected 5 Data: '{line}'")

                    except socket.timeout:
                         print(f"SensorControl Listener: Timeout receiving data from {client_addr}. Closing connection.")
                         self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                         self.client_conn = None
                         break # Break inner loop, will try to accept again
                    except socket.error as e:
                        print(f"SensorControl Listener: Socket error recv from {client_addr}: {e}")
                        self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        self.client_conn = None
                        traceback.print_exc()
                        break
                    except UnicodeDecodeError:
                        print(f"SensorControl Listener: Non-UTF-8 data from {client_addr}.")
                        buffer = "" # Clear buffer on decode error
                    except Exception as e:
                        print(f"SensorControl Listener: Processing error from {client_addr}: {e}")
                        self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                        traceback.print_exc()
                        break # Break inner loop

            except socket.error as e:
                 print(f"SensorControl Listener: Error accepting connection: {e}")
                 self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                 time.sleep(1)
            except Exception as e:
                 print(f"SensorControl Listener: Unexpected error in accept loop: {e}")
                 self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                 traceback.print_exc()
                 time.sleep(1)
            finally:
                if client_conn:
                    print(f"SensorControl Listener: Closing connection to {client_addr}")
                    try: client_conn.shutdown(socket.SHUT_RDWR)
                    except socket.error: pass
                    try: client_conn.close()
                    except socket.error: pass
                    client_conn = None
                    # Ensure status is updated if connection closed cleanly or abnormally
                    self._update_sensor_data(self.ay_data, self.gz_data, self.accelerate_pressed, self.brake_pressed, self.reverse_enabled, False)
                    self.client_conn = None


        print("SensorControl Listener: Stopping...")
        if server_socket:
            try: server_socket.close()
            except socket.error: pass
            print("SensorControl Listener: Server socket closed.")
        self.client_conn = None


    def parse_events(self, client, world, clock, sync_mode):
        """
        Parses input events (keyboard for auxiliary actions, sensors for driving)
        and applies controls to the player vehicle. Includes speed limit.
        """
        world_instance = self._get_world()
        if not world_instance or not self._control or not world_instance.player or not world_instance.player.is_alive:
            # Handle quit events even if world/control is invalid
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and self._is_quit_shortcut(event.key)): return True
            return False
        
        player = world_instance.player
        
        # --- Get Vehicle Data for MPC and Haptics ---
        transform = player.get_transform()
        velocity = player.get_velocity()
        physics_control = player.get_physics_control()
        current_steer_angle_rad = 0.0
        try:
            current_steer_angle_rad = np.radians(player.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
        except Exception as e:
            print(f"Android_control: Warning: Could not get wheel steer angle: {e}. Using 0.")

        current_location = transform.location
        current_rotation = transform.rotation
        current_speed_mps = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        current_vehicle_state = np.array([
            current_location.x,
            current_location.y,
            np.radians(current_rotation.yaw),
            current_speed_mps,
            current_steer_angle_rad
        ])

        # --- Haptic-related data ---
        collision_intensity_for_haptics = 0.0
        if world_instance.collision_sensor:
            collision_history = world_instance.collision_sensor.get_collision_history()
            current_frame_num = world_instance.hud.frame # Ensure hud.frame is updated
            collision_intensity_for_haptics = collision_history.get(current_frame_num, 0.0)

        vehicle_accel_for_haptics = player.get_acceleration()
        accel_magnitude_for_haptics = np.sqrt(vehicle_accel_for_haptics.x**2 + vehicle_accel_for_haptics.y**2 + vehicle_accel_for_haptics.z**2)


        # --- Send Vehicle Data to MPC Process ---
        data_to_mpc = {
            'state': current_vehicle_state,
            'collision_intensity': collision_intensity_for_haptics, # Send for haptics
            'accel_magnitude': accel_magnitude_for_haptics      # Send for haptics
        }
        try:
            if self.mpc_vehicle_data_pipe_send:
                 self.mpc_vehicle_data_pipe_send.send(data_to_mpc)
        except Exception as e:
            print(f"Android_control: Error sending data to MPC: {e}")


        # --- Receive Control Command from MPC Process ---
        # Use poll for non-blocking receive with a timeout
        if self.mpc_control_cmd_pipe_recv and self.mpc_control_cmd_pipe_recv.poll(timeout=0.005): # Short timeout
            try:
                mpc_command = self.mpc_control_cmd_pipe_recv.recv()
                if mpc_command:
                    # --- Apply MPC Command ---
                    optimal_steer_angle_cmd_rad = mpc_command.get('steer_angle_rad', 0.0)
                    self._control.throttle = mpc_command.get('throttle', 0.0)
                    self._control.brake = mpc_command.get('brake', 0.0)
                    self._control.reverse = mpc_command.get('reverse', False)

                    # Convert optimal angle command (radians) back to normalized CARLA input [-1, 1]
                    max_phys_steer_angle_rad = np.radians(70.0) # Fallback
                    try:
                        max_phys_steer_angle_deg_val = physics_control.wheels[0].max_steer_angle
                        max_phys_steer_angle_rad = np.radians(max_phys_steer_angle_deg_val)
                    except (AttributeError, IndexError) as e:
                        print(f"Android_control: Warning getting max steer angle: {e}. Using default.")

                    if max_phys_steer_angle_rad > 1e-4:
                        self._control.steer = np.clip(optimal_steer_angle_cmd_rad / max_phys_steer_angle_rad, -1.0, 1.0)
                    else:
                        self._control.steer = 0.0
            except EOFError:
                print("Android_control: MPC control pipe closed.")
            except Exception as e:
                print(f"Android_control: Error receiving/processing control from MPC: {e}")
                # Fallback: Zero out controls if MPC comm fails
                self._control.steer = 0.0
                self._control.throttle = 0.0
                self._control.brake = 0.5 # Apply some brake
        # else:
            # Optional: If no command received, maintain last control or apply a default
            # print("Android_control: No command from MPC this tick.")
            # pass


        # --- Speed Limit (still handled locally) ---
        v = player.get_velocity()
        speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        # Max speed from arguments or a default
        max_speed_kph_limit = world_instance.player_max_speed_fast * 3.6 if world_instance.player_max_speed_fast else 60.0 # Example
        if hasattr(self, 'max_speed_kph'): # If SensorControl has its own max_speed_kph
            max_speed_kph_limit = self.max_speed_kph

        if speed_kmh >= max_speed_kph_limit and self._control.throttle > 0:
            self._control.throttle = 0.0

        # --- Process Keyboard Events for Auxiliary Actions ---
        current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key): return True
                # --- Keep other keyboard shortcuts ---
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled: player.set_autopilot(False); world_instance.restart(); player = world_instance.player; player.set_autopilot(True)
                    else: world_instance.restart(); player = world_instance.player
                    # Re-initialize control state after restart
                    self._control = carla.VehicleControl(); self._lights = carla.VehicleLightState.NONE
                    if player and isinstance(player, carla.Vehicle): player.apply_control(self._control); player.set_light_state(self._lights); player.set_autopilot(self._autopilot_enabled)
                    return False # Continue after restart
                elif event.key == K_F1: world_instance.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT): world_instance.hud.help.toggle()
                elif event.key == K_TAB: world_instance.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT: world_instance.next_weather(reverse=True)
                elif event.key == K_c: world_instance.next_weather()
                elif event.key == K_BACKQUOTE: world_instance.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9: world_instance.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL): world_instance.camera_manager.toggle_recording()
                # ... (rest of recorder, playback, map layer keys) ...
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if world_instance.recording_enabled: client.stop_recorder(); world_instance.recording_enabled = False; world_instance.hud.notification("Recorder is OFF")
                    else:
                        try: client.start_recorder("manual_recording.rec"); world_instance.recording_enabled = True; world_instance.hud.notification("Recorder is ON")
                        except Exception as e: print(f"Error starting recorder: {e}"); world_instance.hud.error("Could not start recorder")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL): print("CTRL+P (Playback start) not implemented.") # Placeholder
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL): print("CTRL+- (Playback time adjust) not implemented.") # Placeholder
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL): print("CTRL+= (Playback time adjust) not implemented.") # Placeholder
                elif event.key == K_PERIOD and sync_mode: world_instance.world.tick() # Manual tick
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT: world_instance.next_map_layer(reverse=True)
                elif event.key == K_v: world_instance.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT: world_instance.load_map_layer(unload=True)
                elif event.key == K_b: world_instance.load_map_layer()

                # Vehicle Specific Controls (Keep most, but remove Q for reverse)
                if isinstance(self._control, carla.VehicleControl):
                    # REMOVED: elif event.key == K_q: self._control.reverse = not self._control.reverse
                    if event.key == K_m: # Manual Gear
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = player.get_control().gear
                        world_instance.hud.notification('%s Transmission' % ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA: self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD and not sync_mode: self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL: # Autopilot
                        self._autopilot_enabled = not self._autopilot_enabled
                        player.set_autopilot(self._autopilot_enabled)
                        world_instance.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    # Lights (Keep light controls)
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL: current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT: current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l: current_lights ^= carla.VehicleLightState.LowBeam
                    elif event.key == K_i: current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z: current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x: current_lights ^= carla.VehicleLightState.RightBlinker

        # --- Apply Control if not in Autopilot ---
        if not self._autopilot_enabled and player.is_alive:
            # Hand brake still controlled by keyboard SPACE
            keys = pygame.key.get_pressed()
            self._control.hand_brake = keys[K_SPACE] if isinstance(self._control, carla.VehicleControl) else False

            try:
                player.apply_control(self._control)

                # Update brake/reverse lights based on final control state
                if isinstance(self._control, carla.VehicleControl):
                    # Update Brake Light
                    if self._control.brake > 0.1: current_lights |= carla.VehicleLightState.Brake
                    else: current_lights &= ~carla.VehicleLightState.Brake
                    # Update Reverse Light (based on phone state via self._control.reverse)
                    if self._control.reverse: current_lights |= carla.VehicleLightState.Reverse
                    else: current_lights &= ~carla.VehicleLightState.Reverse

                    # Apply lights if changed
                    if current_lights != self._lights:
                        player.set_light_state(carla.VehicleLightState(current_lights))
                        self._lights = current_lights
            except Exception as e:
                # Log error but don't crash
                print(f"Error applying control/lights to player {player.id}: {e}")
                # Attempt to reset control to safe state? Optional.
                # self._control = carla.VehicleControl()
                # player.apply_control(self._control)


        

        return False # Continue loop

    def stop(self):
        # This method is now primarily for cleaning up IPC if necessary,
        # the TCP listener is in the MPC process.
        print("Android_control: SensorControl stop called.")
        # Signal MPC process to stop if this process is shutting down game_loop
        if self.mpc_vehicle_data_pipe_send:
            try:
                self.mpc_vehicle_data_pipe_send.send(None) # Sentinel
                self.mpc_vehicle_data_pipe_send.close()
            except Exception as e:
                print(f"Android_control: Error closing MPC send pipe: {e}")
        if self.mpc_control_cmd_pipe_recv:
             try:
                self.mpc_control_cmd_pipe_recv.close()
             except Exception as e:
                print(f"Android_control: Error closing MPC recv pipe: {e}")


    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
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
        self._show_ackermann_info = False # Keep Ackermann info if KeyboardControl is used
        self._ackermann_control = carla.VehicleAckermannControl()


    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return

        # Ensure world and player are valid before accessing attributes
        if not world or not world.player or not world.player.is_alive:
             self._info_text = ["Player not available"] # Show basic message
             return

        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control() # Get current control state

        # IMU data (check if sensor exists)
        compass = 0.0
        accelerometer = (0.0, 0.0, 0.0)
        if world.imu_sensor and world.imu_sensor.sensor is not None:
             compass = world.imu_sensor.compass
             accelerometer = world.imu_sensor.accelerometer
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''

        # Collision data (check if sensor exists)
        collision = [0.0] * 200 # Default empty collision
        if world.collision_sensor and world.collision_sensor.sensor is not None:
            colhist = world.collision_sensor.get_collision_history()
            collision = [colhist.get(x + self.frame - 200, 0) for x in range(0, 200)] # Use get with default
            max_col = max(1.0, max(collision) if collision else 1.0) # Handle empty collision list
            collision = [x / max_col for x in collision]

        vehicles = world.world.get_actors().filter('vehicle.*')

        # GNSS data (check if sensor exists)
        gnss_lat = 0.0
        gnss_lon = 0.0
        if world.gnss_sensor and world.gnss_sensor.sensor is not None:
             gnss_lat = world.gnss_sensor.lat
             gnss_lon = world.gnss_sensor.lon


        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % accelerometer, # Display accelerometer
            # Removed: 'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (gnss_lat, gnss_lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse), # Show current reverse state
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            # Keep Ackermann info display logic if KeyboardControl is still used
            if self._show_ackermann_info:
                 self._info_text += [
                     '',
                     'Ackermann Controller:',
                     '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                 ]

        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0: break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    # --- Rest of HUD methods remain the same ---
    def show_ackermann_info(self, enabled): self._show_ackermann_info = enabled
    def update_ackermann_control(self, ackermann_control): self._ackermann_control = ackermann_control
    def toggle_info(self): self._show_info = not self._show_info
    def notification(self, text, seconds=2.0): self._notifications.set_text(text, seconds=seconds)
    def error(self, text): self._notifications.set_text('Error: %s' % text, (255, 0, 0))
    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]: break
                if isinstance(item, list): # Collision history graph
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None # Reset item
                    v_offset += 18 # Adjust vertical offset
                elif isinstance(item, tuple): # Control bars or boolean indicators
                    if isinstance(item[1], bool): # Boolean like Reverse, Handbrake, Manual
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else: # Float like Throttle, Steer, Brake
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        # Normalize the value within its range [min_val, max_val] -> [0, 1]
                        f = (item[1] - item[2]) / (item[3] - item[2]) if (item[3] - item[2]) != 0 else 0
                        f = max(0.0, min(f, 1.0)) # Clamp normalized value to [0, 1]
                        # Calculate bar position and size based on range
                        if item[2] < 0.0: # Centered bar (like steering)
                             # Map [0, 1] normalized value to [-bar_width/2, bar_width/2] offset
                             bar_offset = (f - 0.5) * (bar_width - 6)
                             rect = pygame.Rect((bar_h_offset + (bar_width/2) + bar_offset - 3 , v_offset + 8), (6, 6))
                        else: # Left-aligned bar (like throttle, brake)
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0] # Get the label string
                if item:  # Render string item
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18 # Move to next line
        self._notifications.render(display)
        self.help.render(display)


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
    def get_collision_history(self):
        history = collections.defaultdict(int); [history.__setitem__(frame, history[frame]+intensity) for frame, intensity in self.history]
        return history
    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self();
        if not self: return
        actor_type = get_actor_display_name(event.other_actor); self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse; intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity));
        if len(self.history) > 4000: self.history.pop(0)

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
    # ... (Keep class implementation as is) ...
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None; self.surface = None; self._parent = parent_actor; self.hud = hud; self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x; bound_y = 0.5 + self._parent.bounding_box.extent.y; bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType
        if not self._parent.type_id.startswith("walker.pedestrian"):
             self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else: # Walker transforms
             self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [ # Sensor definitions remain the same
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}], ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}], ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}], ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}], ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.lidar.ray_cast_semantic', None, 'Semantic Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted', {'lens_circle_multiplier': '3.0', 'lens_circle_falloff': '3.0', 'chromatic_aberration_intensity': '0.5', 'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}], ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world(); bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0])); bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'): bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items(): bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range': self.lidar_range = float(attr_value)
            item.append(bp)
        self.index = None
    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)
    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None: self.sensor.destroy(); self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(self.sensors[index][-1], self._camera_transforms[self.transform_index][0], attach_to=self._parent, attachment_type=self._camera_transforms[self.transform_index][1])
            weak_self = weakref.ref(self); self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify: self.hud.notification(self.sensors[index][2])
        self.index = index
    def next_sensor(self): self.set_sensor(self.index + 1)
    def toggle_recording(self): self.recording = not self.recording; self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))
    def render(self, display):
        if self.surface is not None: display.blit(self.surface, (0, 0))
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self();
        if not self: return
        # Lidar and Camera parsing logic remains the same
        if self.sensors[self.index][0] == 'sensor.lidar.ray_cast':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4')); points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2]); lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range); lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data).astype(np.int32); lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3); lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255); self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0] == 'sensor.lidar.ray_cast_semantic':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4')); points = np.reshape(points, (int(points.shape[0] / 6), 6))
            lidar_data = np.array(points[:, :2]); lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range); lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = lidar_data.astype(np.int32); lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3); lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            for i in range(len(image)): lidar_img[tuple(lidar_data[i].T)] = OBJECT_TO_COLOR[int(image[i].object_tag)]
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow(); array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4)); array = array[:, :, :3]; array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else: # Default camera handling
            image.convert(self.sensors[self.index][1]); array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4)); array = array[:, :, :3]; array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording: image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    controller = None # Initialize controller variable
    mpc_process_instance = None
    # Pipes for communication: one for Android_control -> MPC, one for MPC -> Android_control
    ac_to_mpc_pipe_parent, ac_to_mpc_pipe_child = multiprocessing.Pipe()
    mpc_to_ac_pipe_parent, mpc_to_ac_pipe_child = multiprocessing.Pipe()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0) # Increased timeout slightly

        sim_world = client.get_world()

        # Setup sync mode if requested
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = args.delta_seconds # Use an arg for this
            sim_world.apply_settings(settings)
            traffic_manager = client.get_trafficmanager(args.tm_port)
            traffic_manager.set_synchronous_mode(True)
        else:
            traffic_manager = client.get_trafficmanager(args.tm_port)



        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: Autopilot requested in asynchronous mode. May lead to issues.")

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, traffic_manager, args) # Pass traffic_manager to World
        # --- MPC Process Setup ---
        mpc_params = {'horizon_N': 10, 'dt': args.delta_seconds if args.sync else 0.05} # Pass delta_seconds
        # Ensure phone_tcp_config uses a *different* port if Android_control still has any TCP server
        phone_tcp_config = {"host": "127.0.0.1", "port": 6002} # Port for MPC to listen to phone

        mpc_process_instance = MPCProcess(ac_to_mpc_pipe_child, mpc_to_ac_pipe_parent, mpc_params, phone_tcp_config)
        mpc_process_instance.start()
        print("Game_loop: MPC Process started.")
        # --- End MPC Process Setup ---


        # Instantiate SensorControl as the primary controller
        controller = SensorControl(world, args.autopilot, ac_to_mpc_pipe_parent, mpc_to_ac_pipe_child)

        # controller = KeyboardControl(world, args.autopilot) # Use this line instead if you want keyboard fallback

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                # Important: Tick the world *before* parsing events in sync mode
                # to ensure sensor data used for control is from the *current* frame.
                sim_world.tick()
                clock.tick() # Tick the Pygame clock as well
            else:
                 # In async mode, wait for tick happens implicitly,
                 # but we still tick Pygame clock for frame rate limiting.
                 clock.tick_busy_loop(60) # Limit client FPS

            if not mpc_process_instance.is_alive() and mpc_process_instance.exitcode is not None :
                print(f"Game_loop: MPC process has exited with code {mpc_process_instance.exitcode}. Stopping.")
                break

            if controller.parse_events(client, world, clock, args.sync):
                return # Exit loop if parse_events signals quit

            # Tick the game world state (mainly HUD updates)
            # This should happen *after* controls are potentially applied by parse_events
            if world: # Check if world exists
                 world.tick(clock)

            # Render the world (camera + HUD)
            if world: # Check if world exists
                 world.render(display)

            pygame.display.flip() # Update the display

    except Exception as e:
         # Log any unexpected exceptions during the game loop
         logging.error("Error during game loop: %s", e)
         traceback.print_exc()

    finally:
        print("Game_loop: Cleaning up resources...")
        if isinstance(controller, SensorControl):
            controller.stop() # This will now primarily close its ends of the pipes

        if mpc_process_instance and mpc_process_instance.is_alive():
            print("Game_loop: Signaling MPC process to stop...")
            if ac_to_mpc_pipe_parent: # Check if pipe still valid before sending
                try:
                    ac_to_mpc_pipe_parent.send(None) # Sentinel to stop MPC process
                except Exception as e:
                    print(f"Game_loop: Error sending stop signal to MPC: {e}")
            mpc_process_instance.join(timeout=2.0)
            if mpc_process_instance.is_alive():
                print("Game_loop: MPC process did not exit cleanly, terminating.")
                mpc_process_instance.terminate()
                mpc_process_instance.join(timeout=1.0)
        print("Game_loop: MPC Process stopped.")

        # Close parent ends of pipes
        if ac_to_mpc_pipe_parent: ac_to_mpc_pipe_parent.close()
        if mpc_to_ac_pipe_child: mpc_to_ac_pipe_child.close() # Child for MPC, parent for AC
        # Child ends are closed by the MPCProcess when it exits or its __del__ is called.


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
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client (Accelerometer)') # Updated description
    # Keep existing arguments
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot (overridden by sensor control)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--generation', metavar='G', default='All', help='restrict to certain actor generation (values: "1","2","All" - default: "All")') # Corrected values
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction of the camera (default: 2.2)') # Default gamma is often 2.2
    argparser.add_argument('--sync', action='store_true', help='Activate synchronous mode execution')
    # Add TM port argument
    argparser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port for Traffic Manager (default: 8000)')
    argparser.add_argument('--delta_seconds', metavar='S', default=0.05, type=float, help='Fixed delta seconds for synchronous mode (default: 0.05)')
    argparser.add_argument('--mpc_phone_port', metavar='P', default=6002, type=int, help='TCP port for MPC process to listen to phone sensors (default: 6002)')



    args = argparser.parse_args()

    # Validate resolution format
    try:
        args.width, args.height = [int(x) for x in args.res.split('x')]
    except ValueError:
        print("Error: Invalid resolution format. Use WIDTHxHEIGHT (e.g., 1280x720)")
        sys.exit(1)


    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # Update help text if needed
    # print(__doc__) # You might want to update the help text at the top

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
         # Catch exceptions happening outside the main game loop (e.g., during setup)
         logging.critical("Unhandled exception: %s", e)
         traceback.print_exc()


if __name__ == '__main__':
    main()
