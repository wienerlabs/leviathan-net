package run.leviathan.multitude

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.PowerManager
import uniffi.leviathan_multitude_mobile.DeviceSnapshot
import uniffi.leviathan_multitude_mobile.NetworkKind
import uniffi.leviathan_multitude_mobile.thermalFromAndroid

class DeviceReader(context: Context) {
    private val app = context.applicationContext
    private val battery = app.getSystemService(BatteryManager::class.java)
    private val power = app.getSystemService(PowerManager::class.java)
    private val connectivity = app.getSystemService(ConnectivityManager::class.java)
    private val activities = app.getSystemService(ActivityManager::class.java)

    fun snapshot(): DeviceSnapshot {
        val status = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val memory = ActivityManager.MemoryInfo().also { activities.getMemoryInfo(it) }
        return DeviceSnapshot(
            charging = (status?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0,
            batteryPercent = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY).coerceIn(0, 100).toUByte(),
            network = network(),
            thermal = thermalFromAndroid(power.currentThermalStatus),
            userActive = false,
            freeMemoryMb = (memory.availMem / (1024 * 1024)).toUInt(),
        )
    }

    private fun network(): NetworkKind {
        val capabilities = connectivity.getNetworkCapabilities(connectivity.activeNetwork)
        return when {
            capabilities == null || !capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) -> NetworkKind.OFFLINE
            capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) -> NetworkKind.UNMETERED
            else -> NetworkKind.METERED
        }
    }
}
