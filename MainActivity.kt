package com.example.sensorreaderapp // Replace with your actual package name

import android.content.Context
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.switchmaterial.SwitchMaterial // Import SwitchMaterial
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException
import java.io.PrintWriter
import java.net.InetAddress
import java.net.Socket
import java.net.SocketException
import java.util.Locale

// Constants for ADB communication
private const val ADB_HOST = "127.0.0.1"
private const val ADB_PORT = 6001 // Ensure this port matches adb forward and python script

class MainActivity : AppCompatActivity(), SensorEventListener {

    // Sensor management
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    // Removed: private var gyroscope: Sensor? = null

    // UI Elements
    private lateinit var accelXValueTextView: TextView
    private lateinit var accelYValueTextView: TextView
    private lateinit var accelZValueTextView: TextView
    // Removed: private lateinit var gyroXValueTextView: TextView
    // Removed: private lateinit var gyroYValueTextView: TextView
    // Removed: private lateinit var gyroZValueTextView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var reverseSwitch: SwitchMaterial // Declare the switch

    // Latest sensor data
    private var lastAccelerometerData = FloatArray(3) { 0.0f }
    // Removed: private var lastGyroscopeData = FloatArray(3) { 0.0f }
    private var lastReverseState: Int = 0 // Store last reverse state

    // Network components
    private var clientSocket: Socket? = null
    private var printWriter: PrintWriter? = null
    private var connectionJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main) // Use the layout with the switch

        // --- Initialize UI Elements ---
        accelXValueTextView = findViewById(R.id.accelXValueTextView)
        accelYValueTextView = findViewById(R.id.accelYValueTextView)
        accelZValueTextView = findViewById(R.id.accelZValueTextView)
        // Removed gyroscope TextView initializations
        statusTextView = findViewById(R.id.statusTextView)
        reverseSwitch = findViewById(R.id.reverseSwitch) // Initialize the switch
        // --- End UI Initialization ---

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        // Removed: gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        if (accelerometer == null) Log.e("SensorError", "Accelerometer not available.")
        // Removed: if (gyroscope == null) Log.e("SensorError", "Gyroscope not available.")

        // Add listener to switch to immediately send state change
        reverseSwitch.setOnCheckedChangeListener { _, isChecked ->
            lastReverseState = if (isChecked) 1 else 0
            sendCurrentState() // Send data immediately when switch is toggled
        }

        // Initialize lastReverseState based on initial switch state
        lastReverseState = if (reverseSwitch.isChecked) 1 else 0

        updateUI()
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        // Removed: gyroscope?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        Log.i("SensorLifecycle", "Accelerometer listener registered.")
        connectToServer()
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        Log.i("SensorLifecycle", "Sensor listeners unregistered.")
        disconnectFromServer()
    }

    // --- Connection Management (Identical to previous version) ---
    private fun connectToServer() {
        connectionJob?.cancel()
        connectionJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
                withContext(Dispatchers.Main) { updateStatus("Connecting...") }
                Log.i("NetworkTCP", "Attempting to connect to $ADB_HOST:$ADB_PORT...")
                clientSocket?.close()
                clientSocket = Socket(ADB_HOST, ADB_PORT)
                printWriter = PrintWriter(clientSocket!!.getOutputStream(), true)
                Log.i("NetworkTCP", "TCP Connection established successfully.")
                withContext(Dispatchers.Main) { updateStatus("Connected") }
                // Send initial state upon connection
                sendCurrentState()
            } catch (e: IOException) {
                Log.e("NetworkTCP", "Connection failed: ${e.message}", e)
                withContext(Dispatchers.Main) { updateStatus("Connection Failed") }
                clientSocket = null
                printWriter = null
            } catch (e: Exception) {
                Log.e("NetworkTCP", "Unexpected error during connection: ${e.message}", e)
                withContext(Dispatchers.Main) { updateStatus("Error") }
                clientSocket = null
                printWriter = null
            }
        }
    }

    private fun disconnectFromServer() {
        connectionJob?.cancel()
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                printWriter?.close()
                clientSocket?.close()
                Log.i("NetworkTCP", "TCP Connection closed.")
            } catch (e: IOException) {
                Log.e("NetworkTCP", "Error closing socket: ${e.message}", e)
            } finally {
                printWriter = null
                clientSocket = null
                withContext(Dispatchers.Main) {
                    updateStatus("Disconnected")
                }
            }
        }
    }

    // --- Sensor Event Handling ---
    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        var sensorDataChanged = false
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                // Check if data actually changed significantly to avoid flooding
                // (Optional: Add a threshold check if needed)
                if (!lastAccelerometerData.contentEquals(event.values)) {
                    lastAccelerometerData = event.values.clone()
                    sensorDataChanged = true
                }
            }
            // Removed: Gyroscope case
        }

        if (sensorDataChanged) {
            updateUI() // Update local display elements
            sendCurrentState() // Send the latest state including switch
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* Optional log */ }

    // --- Data Sending ---
    // Helper function to send the current state (accel + switch)
    private fun sendCurrentState() {
        if (clientSocket?.isConnected == true && printWriter != null) {
            // Send combined data: Accelerometer + last known reverse state
            sendDataOverTCP(lastAccelerometerData, lastReverseState)
        }
    }

    // Updated function to send only accelerometer and reverse state
    private fun sendDataOverTCP(accelData: FloatArray, reverseState: Int) {
        val writer = printWriter
        if (writer == null || clientSocket?.isConnected != true) return

        // Format data: ax,ay,az,reverse_state (4 values total)
        val messageStr = String.format(Locale.US, "%.4f,%.4f,%.4f,%d",
            accelData[0], accelData[1], accelData[2], reverseState)

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                writer.println(messageStr)
                if (writer.checkError()) {
                    Log.e("NetworkTCP", "PrintWriter error occurred during send.")
                    // Consider adding automatic reconnection logic here
                    withContext(Dispatchers.Main) { updateStatus("Send Error - Reconnecting...") }
                    disconnectFromServer()
                    kotlinx.coroutines.delay(500) // Wait before reconnecting
                    connectToServer()
                }
                // else { Log.d("NetworkTCP", "Sent TCP data: $messageStr") } // Optional log
            } catch (e: Exception) { // Catch more general exceptions during send
                Log.e("NetworkTCP", "Error sending TCP data: ${e.message}", e)
                withContext(Dispatchers.Main) { updateStatus("Send Error") }
                // Consider adding automatic reconnection logic here as well
                disconnectFromServer()
                kotlinx.coroutines.delay(500)
                connectToServer()
            }
        }
    }

    // --- UI Update ---
    private fun updateUI() {
        // Update Accelerometer display
        accelXValueTextView.text = String.format(Locale.US, "%+6.2f", lastAccelerometerData[0])
        accelYValueTextView.text = String.format(Locale.US, "%+6.2f", lastAccelerometerData[1])
        accelZValueTextView.text = String.format(Locale.US, "%+6.2f", lastAccelerometerData[2])

        // Removed Gyroscope display updates

        // Update status text (no change needed here)
        updateStatus(null) // Pass null to determine status based on connection state
    }

    // updateStatus function remains largely the same, handles connection status display
    private fun updateStatus(newStatusText: String?) {
        val currentStatusText = when {
            newStatusText != null -> "Status: $newStatusText"
            connectionJob?.isActive == true -> "Status: Connecting..." // Check if connection job is running
            clientSocket?.isConnected == true -> "Status: Connected"
            else -> "Status: Disconnected"
        }
        statusTextView.text = currentStatusText
        // Set text color based on status
        val colorRes = when {
            currentStatusText.contains("Connected") -> android.R.color.holo_green_dark
            currentStatusText.contains("Connecting") || currentStatusText.contains("Reconnecting") -> android.R.color.holo_orange_light
            else -> android.R.color.holo_red_dark // Disconnected, Failed, Error
        }
        statusTextView.setTextColor(ContextCompat.getColor(this, colorRes))
    }
}
