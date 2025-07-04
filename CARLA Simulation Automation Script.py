#!/usr/bin/env python3
import subprocess
import os
import time
import shlex

# --- Configuration ---
# This dictionary holds the commands for each "terminal".
# The keys are descriptive names for the tasks.
# Each command is a string. For multi-line commands, they are combined with '&&'.
# NOTE: The '****' in the scenario_runner.py command is a placeholder.
# You will need to replace it with the actual scenario name.
COMMANDS = {
    "carla_server": [
        "pyenv activate carla",
        "cd ~/CARLA_0.9.15",
        "./CarlaUE4.sh"
    ],
    "scenario_runner": [
        "pyenv activate carla",
        "export PYTHONPATH=/home/engssg/CARLA_0.9.15/PythonAPI/carla/:$PYTHONPATH",
        "cd ~/CARLA_0.9.15/PythonAPI/examples/scenario_runner",
        "python scenario_runner.py --scenario FollowLeadingVehicle_1 --reloadWorld --record /home/engssg/CARLA_0.9.15/PythonAPI/Recordings"
    ],
    "android_control": [
        "pyenv activate carla",
        "cd ~/CARLA_0.9.15/PythonAPI/examples",
        "python Android_control.py --max_speed_kph 60 --rolename hero --waitForEgo"
    ],
    "network_config": [
        "sudo tc qdisc del dev lo root 2>/dev/null || true",
        "sudo tc qdisc add dev lo root handle 1: prio",
        "sudo tc qdisc add dev lo parent 1:1 handle 10: netem delay 40ms 10ms 25%",
        "sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 6002 0xffff flowid 1:1",
        "sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 6001 0xffff flowid 1:1",
        "sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 6003 0xffff flowid 1:1",
        "sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 6004 0xffff flowid 1:1",
        "sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 6005 0xffff flowid 1:1",
        "tc qdisc show dev lo"
    ]
}

def run_commands():
    """
    Executes the commands defined in the COMMANDS dictionary.
    Each top-level key in COMMANDS is treated as a separate process,
    simulating multiple terminal windows.
    """
    # Check for sudo privileges if network configuration is being run
    if os.geteuid() != 0 and "network_config" in COMMANDS:
        print("---------------------------------------------------------")
        print("WARNING: Network configuration commands require sudo.")
        print("Please run this script with 'sudo python3 your_script_name.py'")
        print("---------------------------------------------------------")
        return

    processes = []
    
    # --- Execute all command blocks ---
    for name, command_list in COMMANDS.items():
        print(f"--- Starting: {name} ---")
        
        # Join the list of commands into a single string to be executed in a shell
        # We use '&&' to ensure that if a command fails, the subsequent ones in the block don't run.
        full_command = " && ".join(command_list)
        
        # To handle 'pyenv activate', we need to run the command in a shell
        # that has the pyenv environment properly initialized.
        # We source the pyenv init scripts before running our actual commands.
        # Note: This assumes pyenv is installed in the standard location.
        shell_command = f"""
        bash -c '
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
        {full_command}
        '
        """
        
        # We use Popen to run the command in a new non-blocking process.
        # This allows all our "terminals" to run concurrently.
        # We pipe stdout and stderr to the main script's console.
        try:
            process = subprocess.Popen(
                shell_command,
                shell=True,
                text=True,
                executable='/bin/bash' # Explicitly use bash
            )
            processes.append((name, process))
            print(f"Successfully started '{name}'. PID: {process.pid}")
            # Add a small delay to allow processes to initialize, especially servers like Carla
            time.sleep(5) 
        except Exception as e:
            print(f"ERROR: Could not start '{name}'. Reason: {e}")

    print("\n--- All processes have been started. ---")
    print("--- Press Ctrl+C to terminate all running processes. ---")

    try:
        # Keep the main script alive while the subprocesses are running.
        # We can wait for them to complete or just wait for user interruption.
        for name, process in processes:
            process.wait() # Wait for each process to finish
            print(f"--- Process '{name}' (PID: {process.pid}) has finished. ---")

    except KeyboardInterrupt:
        print("\n--- Termination signal received. Stopping all processes. ---")
        for name, process in reversed(processes):
            try:
                if process.poll() is None: # Check if the process is still running
                    print(f"Terminating '{name}' (PID: {process.pid})...")
                    process.terminate()
                    process.wait(timeout=5) # Wait for graceful termination
            except subprocess.TimeoutExpired:
                print(f"Process '{name}' did not terminate gracefully. Forcing kill.")
                process.kill()
            except Exception as e:
                print(f"Error terminating process '{name}': {e}")
        print("--- All processes terminated. ---")

if __name__ == "__main__":
    run_commands()
