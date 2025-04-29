package com.example.sensorreaderapp // Replace with your actual package name

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
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
    private var gyroscope: Sensor? = null // Gyroscope sensor variable

    // UI Elements
    private lateinit var accelXValueTextView: TextView
    private lateinit var accelYValueTextView: TextView
    private lateinit var accelZValueTextView: TextView
    private lateinit var gyroZValueTextView: TextView // ADDED: Gyroscope Z value TextView
    private lateinit var statusTextView: TextView
    private lateinit var reverseSwitch: SwitchMaterial
    private lateinit var accelerateButton: MaterialButton
    private lateinit var brakeButton: MaterialButton

    // Button states
    private var isAcceleratePressed = false
    private var isBrakePressed = false

    // Latest sensor data
    private var lastAccelerometerData = FloatArray(3) { 0.0f }
    private var lastGyroscopeData = FloatArray(3) { 0.0f } // Gyroscope data storage
    private var lastReverseState: Int = 0

    // Network components
    private var clientSocket: Socket? = null
    private var printWriter: PrintWriter? = null
    private var connectionJob: Job? = null

    // Threshold for detecting significant sensor changes
    private val ACCEL_CHANGE_THRESHOLD = 0.05f // Adjust as needed
    private val GYRO_CHANGE_THRESHOLD = 0.02f  // Adjust as needed


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Make sure your layout file (e.g., activity_main.xml) includes a TextView with id gyroZValueTextView
        setContentView(R.layout.activity_main)

        // --- Initialize UI Elements ---
        accelXValueTextView = findViewById(R.id.accelXValueTextView)
        accelYValueTextView = findViewById(R.id.accelYValueTextView)
        accelZValueTextView = findViewById(R.id.accelZValueTextView)
        // Find the new TextView for Gyro Z (Make sure this ID exists in your XML)
        gyroZValueTextView = findViewById(R.id.gyroZValueTextView) // Ensure this ID exists in activity_main.xml
        statusTextView = findViewById(R.id.statusTextView)
        reverseSwitch = findViewById(R.id.reverseSwitch)
        accelerateButton = findViewById(R.id.accelerateButton)
        brakeButton = findViewById(R.id.brakeButton)
        // --- End UI Initialization ---

        // Touch listeners for buttons remain the same
        accelerateButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> { isAcceleratePressed = true; sendCurrentState(); true }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { isAcceleratePressed = false; sendCurrentState(); true }
                else -> false
            }
        }

        brakeButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> { isBrakePressed = true; sendCurrentState(); true }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { isBrakePressed = false; sendCurrentState(); true }
                else -> false
            }
        }

        // --- Sensor Initialization ---
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        if (accelerometer == null) Log.e("SensorError", "Accelerometer not available.")
        if (gyroscope == null) Log.e("SensorError", "Gyroscope not available.")

        // Reverse switch listener remains the same
        reverseSwitch.setOnCheckedChangeListener { _, isChecked ->
            lastReverseState = if (isChecked) 1 else 0
            sendCurrentState() // Send data immediately when switch is toggled
        }
        lastReverseState = if (reverseSwitch.isChecked) 1 else 0

        updateUI() // Initial UI update
    }


    override fun onResume() {
        super.onResume()
        // Register listeners for both sensors
        accelerometer?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyroscope?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        Log.i("SensorLifecycle", "Sensor listeners registered.")
        connectToServer()
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        Log.i("SensorLifecycle", "Sensor listeners unregistered.")
        disconnectFromServer()
    }

    // --- Connection Management (No changes needed) ---
    private fun connectToServer() {
        // ... (Connection logic remains the same) ...
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
                sendCurrentState() // Send initial state
            } catch (e: IOException) {
                Log.e("NetworkTCP", "Connection failed: ${e.message}", e)
                withContext(Dispatchers.Main) { updateStatus("Connection Failed") }
                clientSocket = null; printWriter = null
            } catch (e: Exception) {
                Log.e("NetworkTCP", "Unexpected error during connection: ${e.message}", e)
                withContext(Dispatchers.Main) { updateStatus("Error") }
                clientSocket = null; printWriter = null
            }
        }
    }

    private fun disconnectFromServer() {
        // ... (Disconnection logic remains the same) ...
        connectionJob?.cancel()
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                printWriter?.close(); clientSocket?.close()
                Log.i("NetworkTCP", "TCP Connection closed.")
            } catch (e: IOException) {
                Log.e("NetworkTCP", "Error closing socket: ${e.message}", e)
            } finally {
                printWriter = null; clientSocket = null
                withContext(Dispatchers.Main) { updateStatus("Disconnected") }
            }
        }
    }

    // --- Sensor Event Handling ---
    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        var significantChange = false
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                // Check if 'ay' (index 1) changed significantly
                if (Math.abs(lastAccelerometerData[1] - event.values[1]) > ACCEL_CHANGE_THRESHOLD) {
                    significantChange = true
                }
                // Always update the stored data for UI display
                lastAccelerometerData = event.values.clone()
            }
            Sensor.TYPE_GYROSCOPE -> {
                // Check if 'gz' (index 2) changed significantly
                if (Math.abs(lastGyroscopeData[2] - event.values[2]) > GYRO_CHANGE_THRESHOLD) {
                    significantChange = true
                }
                // Always update stored data for UI display
                lastGyroscopeData = event.values.clone()
            }
        }

        updateUI() // Update UI with latest data from both sensors

        // Send data ONLY if either ay or gz changed significantly.
        // Button/Switch changes trigger sends via their own listeners.
        if (significantChange) {
            sendCurrentState()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* Optional log */ }

    // --- Data Sending ---
    // Helper function to send the current state
    private fun sendCurrentState() {
        // Ensure we have the latest button/switch states included when sending
        if (clientSocket?.isConnected == true && printWriter != null) {
            // Pass all necessary data to the sending function
            sendDataOverTCP(lastAccelerometerData,lastGyroscopeData, isAcceleratePressed, isBrakePressed, lastReverseState)
        }
    }
//lastGyroscopeData add later
    // MODIFIED: Function now accepts gyroData and sends gz
    private fun sendDataOverTCP(accelData: FloatArray, gyroData: FloatArray,accelState: Boolean, brakeState: Boolean, reverseState: Int) {
        val writer = printWriter
        if (writer == null || clientSocket?.isConnected != true) {
            // Log.w("NetworkTCP", "Send attempt failed: Not connected or writer is null.") // Optional log
            return
        }
//gyroData: FloatArray, add later
        val accelInt = if (accelState) 1 else 0
        val brakeInt = if (brakeState) 1 else 0

        // MODIFIED: Format data: ay, gz, accelerate_state, brake_state, reverse_state (5 values total)
        // Accel Y is index 1, Gyro Z is index 2
        val messageStr = String.format(Locale.US, "%.4f,%.4f,%d,%d,%d",
            accelData[1], // Send 'ay'
            gyroData[2],  // Send 'gz' (Gyro Z-axis)
            accelInt,     // Send accelerate button state
            brakeInt,     // Send brake button state
            reverseState) // Send reverse switch state
    //
        // Launch sending in IO scope
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                writer.println(messageStr)
                // Log.d("NetworkTCP", "Sent: $messageStr") // Optional: Log sent data
                if (writer.checkError()) { // Check for errors after sending
                    Log.e("NetworkTCP", "PrintWriter error occurred during send.")
                    // Attempt to reconnect on error
                    withContext(Dispatchers.Main) { updateStatus("Send Error - Reconnecting...") }
                    disconnectFromServer()
                    kotlinx.coroutines.delay(500) // Short delay before reconnecting
                    connectToServer()
                }
            } catch (e: SocketException) {
                Log.e("NetworkTCP", "SocketException sending TCP data (likely broken pipe): ${e.message}")
                withContext(Dispatchers.Main) { updateStatus("Send Error - Reconnecting...") }
                disconnectFromServer()
                kotlinx.coroutines.delay(500)
                connectToServer()
            } catch (e: Exception) { // Catch other potential exceptions
                Log.e("NetworkTCP", "Error sending TCP data: ${e.message}", e)
                withContext(Dispatchers.Main) { updateStatus("Send Error") }
                // Optionally try reconnecting on other errors too
                // disconnectFromServer()
                // kotlinx.coroutines.delay(500)
                // connectToServer()
            }
        }
    }


    // --- UI Update ---
    private fun updateUI() {
        // Update Accelerometer display (Showing only Y for now)
        // accelXValueTextView.text = String.format(Locale.US, "%+6.2f", lastAccelerometerData[0])
        accelYValueTextView.text = String.format(Locale.US, "%+6.2f", lastAccelerometerData[1])
        // accelZValueTextView.text = String.format(Locale.US, "%+6.2f", lastAccelerometerData[2])

        // Update Gyroscope display (Showing Y and Z)

        gyroZValueTextView.text = String.format(Locale.US, "%+6.2f", lastGyroscopeData[2]) // Display Gyro Z

        updateStatus(null)
    }

    // updateStatus function remains the same
    private fun updateStatus(newStatusText: String?) {
        // ... (updateStatus logic remains the same) ...
        val currentStatusText = when {
            newStatusText != null -> "Status: $newStatusText"
            connectionJob?.isActive == true -> "Status: Connecting..."
            clientSocket?.isConnected == true -> "Status: Connected"
            else -> "Status: Disconnected"
        }
        statusTextView.text = currentStatusText
        val colorRes = when {
            currentStatusText.contains("Connected") -> android.R.color.holo_green_dark
            currentStatusText.contains("Connecting") || currentStatusText.contains("Reconnecting") -> android.R.color.holo_orange_light
            else -> android.R.color.holo_red_dark
        }
        statusTextView.setTextColor(ContextCompat.getColor(this, colorRes))
    }
}
