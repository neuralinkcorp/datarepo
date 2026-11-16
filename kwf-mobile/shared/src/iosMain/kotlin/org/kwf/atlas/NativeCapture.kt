@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package org.kwf.atlas

import androidx.compose.foundation.layout.*
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.ComposeUIViewController
import kotlinx.cinterop.useContents
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import platform.CoreLocation.*
import platform.Foundation.*
import platform.UIKit.*
import platform.darwin.NSObject

actual fun nowMillis(): Long =
    (NSDate().timeIntervalSince1970 * 1000).toLong()

fun MainViewController(): UIViewController =
    ComposeUIViewController { App() }

private class LocationDelegate : NSObject(), CLLocationManagerDelegateProtocol {
    var onPosition: (Vertex) -> Unit = {}
    var onError: (String) -> Unit = {}
    var onAuthorized: () -> Unit = {}
    var selector: BestFixSelector? = null

    override fun locationManager(
        manager: CLLocationManager,
        didUpdateLocations: List<*>
    ) {
        val activeSelector = selector ?: return
        for (location in didUpdateLocations.filterIsInstance<CLLocation>()) {
            if (location.horizontalAccuracy < 0) continue
            val latitude = location.coordinate.useContents { latitude }
            val longitude = location.coordinate.useContents { longitude }
            val vertex = Vertex(
                longitude = longitude,
                latitude = latitude,
                source = "platform-derived",
                accuracyMeters = location.horizontalAccuracy,
                z = if (location.verticalAccuracy >= 0) location.altitude else null,
                crs = "EPSG:4326",
                vcrs = "unknown",
                verticalAccuracyM = if (location.verticalAccuracy >= 0) {
                    location.verticalAccuracy
                } else null,
                capturedAtEpochMs =
                    (location.timestamp.timeIntervalSince1970 * 1000).toLong(),
                provider = null
            )
            val acceptable = activeSelector.offer(
                vertex.toSourceVertex(),
                nowMillis()
            ) ?: continue
            selector = null
            manager.stopUpdatingLocation()
            onPosition(acceptable.toVertex())
            return
        }
    }

    override fun locationManager(
        manager: CLLocationManager,
        didFailWithError: NSError
    ) {
        manager.stopUpdatingLocation()
        onError(didFailWithError.localizedDescription)
    }

    override fun locationManagerDidChangeAuthorization(
        manager: CLLocationManager
    ) {
        when (manager.authorizationStatus) {
            kCLAuthorizationStatusAuthorizedWhenInUse,
            kCLAuthorizationStatusAuthorizedAlways ->
                onAuthorized()

            kCLAuthorizationStatusDenied,
            kCLAuthorizationStatusRestricted ->
                onError("Location access is denied. Change it in Settings.")
        }
    }
}

private class PhotoDelegate :
    NSObject(),
    UIImagePickerControllerDelegateProtocol,
    UINavigationControllerDelegateProtocol {

    var onPhoto: (String) -> Unit = {}
    var onError: (String) -> Unit = {}

    override fun imagePickerController(
        picker: UIImagePickerController,
        didFinishPickingMediaWithInfo: Map<Any?, *>
    ) {
        val image =
            didFinishPickingMediaWithInfo[UIImagePickerControllerOriginalImage]
                as? UIImage

        val data = image?.let { UIImageJPEGRepresentation(it, 0.35) }

        if (data != null) {
            onPhoto(data.base64EncodedStringWithOptions(0u))
        } else {
            onError("Unable to read photograph.")
        }

        picker.dismissViewControllerAnimated(true, null)
    }

    override fun imagePickerControllerDidCancel(
        picker: UIImagePickerController
    ) {
        picker.dismissViewControllerAnimated(true, null)
    }
}

@Composable
actual fun NativeCapture(
    onPosition: (Vertex) -> Unit,
    onPhoto: (String) -> Unit,
    onError: (String) -> Unit,
    showLocation: Boolean,
    showCamera: Boolean
) {
    val locationDelegate = remember { LocationDelegate() }
    val photoDelegate = remember { PhotoDelegate() }
    val manager = remember { CLLocationManager() }
    val scope = rememberCoroutineScope()
    var captureAttempt by remember { mutableStateOf(0) }
    var captureInProgress by remember { mutableStateOf(false) }
    var pendingPoorerFix by remember { mutableStateOf<Vertex?>(null) }

    val positionCallback by rememberUpdatedState(onPosition)
    val errorCallback by rememberUpdatedState(onError)

    fun startLocationCapture() {
        manager.stopUpdatingLocation()
        val selector = BestFixSelector(startedAtEpochMs = nowMillis())
        locationDelegate.selector = selector
        captureInProgress = true
        val attempt = ++captureAttempt
        manager.startUpdatingLocation()
        scope.launch {
            delay(15_000)
            if (
                captureAttempt == attempt &&
                locationDelegate.selector === selector
            ) {
                manager.stopUpdatingLocation()
                locationDelegate.selector = null
                captureInProgress = false
                val poorer = selector.bestAtTimeout()
                if (poorer != null) {
                    pendingPoorerFix = poorer.toVertex()
                } else {
                    errorCallback(
                        "Location timed out without an acceptable fix. Try outdoors or enter coordinates."
                    )
                }
            }
        }
    }

    locationDelegate.onPosition = {
        captureAttempt += 1
        captureInProgress = false
        positionCallback(it)
    }
    locationDelegate.onError = onError
    locationDelegate.onAuthorized = { startLocationCapture() }
    photoDelegate.onPhoto = onPhoto
    photoDelegate.onError = onError

    DisposableEffect(manager) {
        manager.delegate = locationDelegate
        manager.desiredAccuracy = kCLLocationAccuracyBest

        onDispose {
            captureAttempt += 1
            locationDelegate.selector = null
            manager.stopUpdatingLocation()
            manager.delegate = null
        }
    }

    if (pendingPoorerFix != null) {
        val fix = pendingPoorerFix!!
        val accuracy = fix.accuracyMeters
            ?.let { "±${it.toInt()} m" } ?: "unknown accuracy"
        AlertDialog(
            onDismissRequest = { pendingPoorerFix = null },
            title = { Text("Accept Poorer Fix?") },
            text = {
                Text(
                    "Desired location accuracy is ≤10 m. Timeout returned $accuracy. Accept this position?"
                )
            },
            confirmButton = {
                Button(onClick = {
                    positionCallback(fix)
                    pendingPoorerFix = null
                }) {
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
            enabled = !captureInProgress,
            onClick = {
                when (manager.authorizationStatus) {
                    kCLAuthorizationStatusNotDetermined ->
                        manager.requestWhenInUseAuthorization()

                    kCLAuthorizationStatusAuthorizedWhenInUse,
                    kCLAuthorizationStatusAuthorizedAlways ->
                        startLocationCapture()

                    else ->
                        onError("Allow location access in iPhone Settings.")
                }
            }
        ) {
            Text(if (captureInProgress) "Waiting for location…" else "Phone location")
        }

        if (showCamera) OutlinedButton(
            onClick = {
                if (!UIImagePickerController.isSourceTypeAvailable(
                        UIImagePickerControllerSourceTypeCamera
                    )
                ) {
                    onError("Camera unavailable. Use a physical iPhone.")
                } else {
                    val scene = UIApplication.sharedApplication.connectedScenes
                        .firstOrNull { it is UIWindowScene } as? UIWindowScene

                    var controller = scene?.windows
                        ?.filterIsInstance<UIWindow>()
                        ?.firstOrNull { it.isKeyWindow }
                        ?.rootViewController

                    while (controller?.presentedViewController != null) {
                        controller = controller.presentedViewController
                    }

                    if (controller == null) {
                        onError("Unable to open camera.")
                    } else {
                        val picker = UIImagePickerController()
                        picker.sourceType =
                            UIImagePickerControllerSourceTypeCamera
                        picker.delegate = photoDelegate

                        controller.presentViewController(
                            picker,
                            true,
                            null
                        )
                    }
                }
            }
        ) {
            Text("Take photo (low-res preview)")
        }
    }
}
