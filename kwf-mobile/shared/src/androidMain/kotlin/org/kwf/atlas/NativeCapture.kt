package org.kwf.atlas

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream

actual fun nowMillis(): Long = System.currentTimeMillis()

@Composable
actual fun NativeCapture(
    onPosition: (Vertex) -> Unit,
    onPhoto: (String) -> Unit,
    onError: (String) -> Unit
) {
    val context = LocalContext.current
    val manager = remember {
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    }

    val handler = remember { Handler(Looper.getMainLooper()) }
    var listener by remember { mutableStateOf<LocationListener?>(null) }

    val positionCallback by rememberUpdatedState(onPosition)
    val errorCallback by rememberUpdatedState(onError)

    fun captureLocation() {
        listener?.let { manager.removeUpdates(it) }

        val provider = when {
            manager.isProviderEnabled(LocationManager.GPS_PROVIDER) ->
                LocationManager.GPS_PROVIDER

            manager.isProviderEnabled(LocationManager.NETWORK_PROVIDER) ->
                LocationManager.NETWORK_PROVIDER

            else -> {
                errorCallback("Enable location services on this phone.")
                return
            }
        }

        val active = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                manager.removeUpdates(this)
                listener = null

                positionCallback(
                    Vertex(
                        longitude = location.longitude,
                        latitude = location.latitude,
                        source = "phone",
                        accuracyMeters = location.accuracy.toDouble()
                    )
                )
            }

            override fun onProviderEnabled(provider: String) {}

            override fun onProviderDisabled(provider: String) {}

            @Deprecated("Legacy Android callback")
            override fun onStatusChanged(
                provider: String?,
                status: Int,
                extras: Bundle?
            ) {}
        }

        listener = active

        try {
            @Suppress("DEPRECATION")
            manager.requestSingleUpdate(
                provider,
                active,
                Looper.getMainLooper()
            )

            handler.postDelayed({
                if (listener === active) {
                    manager.removeUpdates(active)
                    listener = null
                    errorCallback(
                        "Location timed out. Try outdoors or enter coordinates."
                    )
                }
            }, 30_000)
        } catch (error: Exception) {
            listener = null
            errorCallback(error.message ?: "Location failed.")
        }
    }

    val permissions = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { granted ->
        if (
            granted[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
            granted[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        ) {
            captureLocation()
        } else {
            errorCallback("Location permission was not granted.")
        }
    }

    val camera = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap == null) {
            onError("No photograph captured.")
        } else {
            val output = ByteArrayOutputStream()
            bitmap.compress(
                android.graphics.Bitmap.CompressFormat.JPEG,
                85,
                output
            )

            onPhoto(
                Base64.encodeToString(
                    output.toByteArray(),
                    Base64.NO_WRAP
                )
            )
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            listener?.let { manager.removeUpdates(it) }
        }
    }

    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(
            onClick = {
                val fine = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED

                val coarse = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED

                if (fine || coarse) captureLocation()
                else permissions.launch(
                    arrayOf(
                        Manifest.permission.ACCESS_FINE_LOCATION,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    )
                )
            }
        ) {
            Text("Phone location")
        }

        OutlinedButton(onClick = { camera.launch(null) }) {
            Text("Take photo")
        }
    }
}
