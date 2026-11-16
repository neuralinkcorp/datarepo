package org.kwf.atlas

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
    onError: (String) -> Unit,
    showLocation: Boolean,
    showCamera: Boolean
) {
    val context = LocalContext.current
    val manager = remember {
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    }

    val handler = remember { Handler(Looper.getMainLooper()) }
    var listener by remember { mutableStateOf<LocationListener?>(null) }
    var pendingPoorerFix by remember { mutableStateOf<Vertex?>(null) }

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

        val startedAt = System.currentTimeMillis()
        val selector = BestFixSelector(startedAtEpochMs = startedAt, desiredAccuracyM = 10.0)

        val active = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                val now = System.currentTimeMillis()
                val vAccuracy = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && location.hasVerticalAccuracy()) {
                    location.verticalAccuracyMeters.toDouble()
                } else null

                val vertex = SourceVertex(
                    x = location.longitude,
                    y = location.latitude,
                    z = if (location.hasAltitude()) location.altitude else null,
                    crs = "EPSG:4326",
                    vcrs = "unknown", // Do not assign a vertical reference without API evidence
                    horizontalAccuracyM = if (location.hasAccuracy()) location.accuracy.toDouble() else null,
                    verticalAccuracyM = vAccuracy,
                    source = when (location.provider) {
                        LocationManager.GPS_PROVIDER -> "gps"
                        LocationManager.NETWORK_PROVIDER -> "network"
                        else -> "unknown"
                    },
                    capturedAtEpochMs = location.time,
                    provider = location.provider
                )

                val acceptable = selector.offer(vertex, now)
                if (acceptable != null) {
                    manager.removeUpdates(this)
                    listener = null
                    positionCallback(
                        Vertex(
                            longitude = acceptable.x,
                            latitude = acceptable.y,
                            source = acceptable.source,
                            accuracyMeters = acceptable.horizontalAccuracyM,
                            z = acceptable.z,
                            crs = acceptable.crs,
                            vcrs = acceptable.vcrs,
                            verticalAccuracyM = acceptable.verticalAccuracyM,
                            capturedAtEpochMs = acceptable.capturedAtEpochMs,
                            provider = acceptable.provider
                        )
                    )
                }
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
            manager.requestLocationUpdates(
                provider,
                500L,
                0f,
                active,
                Looper.getMainLooper()
            )

            handler.postDelayed({
                if (listener === active) {
                    manager.removeUpdates(active)
                    listener = null
                    val poorer = selector.bestAtTimeout()
                    if (poorer != null) {
                        pendingPoorerFix = Vertex(
                            longitude = poorer.x,
                            latitude = poorer.y,
                            source = poorer.source,
                            accuracyMeters = poorer.horizontalAccuracyM,
                            z = poorer.z,
                            crs = poorer.crs,
                            vcrs = poorer.vcrs,
                            verticalAccuracyM = poorer.verticalAccuracyM,
                            capturedAtEpochMs = poorer.capturedAtEpochMs,
                            provider = poorer.provider
                        )
                    } else {
                        errorCallback(
                            "Location timed out without an acceptable fix. Try outdoors or enter coordinates."
                        )
                    }
                }
            }, 15_000)
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

    // Cancel location subscriptions when leaving the screen
    DisposableEffect(Unit) {
        onDispose {
            listener?.let { manager.removeUpdates(it) }
        }
    }

    if (pendingPoorerFix != null) {
        val fix = pendingPoorerFix!!
        val accText = fix.accuracyMeters?.let { "±${it.toInt()} m" } ?: "unknown accuracy"
        AlertDialog(
            onDismissRequest = { pendingPoorerFix = null },
            title = { Text("Accept Poorer Fix?") },
            text = {
                Text(
                    "Desired GPS accuracy is ≤10 m. Timeout returned a fix with $accText accuracy. Accept this position?"
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        positionCallback(fix)
                        pendingPoorerFix = null
                    }
                ) {
                    Text("Accept Fix")
                }
            },
            dismissButton = {
                TextButton(onClick = { pendingPoorerFix = null }) {
                    Text("Cancel")
                }
            }
        )
    }

    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        if (showLocation) OutlinedButton(
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

        if (showCamera) OutlinedButton(onClick = { camera.launch(null) }) {
            Text("Take photo (low-res preview)")
        }
    }
}
