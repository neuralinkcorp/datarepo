package org.kwf.atlas

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.russhwolf.settings.Settings
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*
import kotlin.random.Random

private val codec = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
}

private val storage = Settings()

@Serializable
data class Vertex(
    val longitude: Double,
    val latitude: Double,
    val source: String,
    val accuracyMeters: Double? = null
)

@Serializable
data class Observation(
    val id: String,
    val project: String,
    val type: String,
    val subtype: String,
    val shape: String,
    val vertices: List<Vertex>,
    val capturedAtEpochMs: Long,
    val notes: String,
    val soilPh: Double? = null,
    val moisture: Double? = null,
    val surveyed: Boolean = false,
    val surveyReference: String = "",
    val elevationMeters: Double? = null,
    val verticalDatum: String = "",
    val photosBase64: List<String> = emptyList()
)

expect fun nowMillis(): Long

@Composable
expect fun NativeCapture(
    onPosition: (Vertex) -> Unit,
    onPhoto: (String) -> Unit,
    onError: (String) -> Unit
)

fun uniqueId(): String = buildString {
    repeat(16) {
        append(
            Random.nextInt(256)
                .toString(16)
                .padStart(2, '0')
        )
    }
}

private fun readRecords(): List<Observation> =
    storage.keys
        .filter { it.startsWith("record:") }
        .mapNotNull { key ->
            runCatching {
                codec.decodeFromString<Observation>(
                    storage.getString(key, "")
                )
            }.getOrNull()
        }
        .sortedByDescending { it.capturedAtEpochMs }

private fun saveRecord(record: Observation) {
    val records = readRecords()

    require(records.size < 500) {
        "Pilot limit reached: 500 local records. Back up the dataset."
    }

    val encoded = codec.encodeToString(record)
    val existingSize = storage.keys
        .filter { it.startsWith("record:") }
        .sumOf { storage.getString(it, "").length.toLong() }

    require(existingSize + encoded.length <= 12_000_000) {
        "Pilot storage limit reached. Back up the dataset."
    }

    storage.putString("record:${record.id}", encoded)
}

fun Observation.toFeature(assetPrefix: String): JsonObject {
    fun coordinate(vertex: Vertex) = buildJsonArray {
        add(vertex.longitude)
        add(vertex.latitude)
    }

    val coordinates = when (shape) {
        "Point" -> coordinate(vertices.first())

        "LineString" -> buildJsonArray {
            vertices.forEach { add(coordinate(it)) }
        }

        else -> buildJsonArray {
            add(buildJsonArray {
                vertices.forEach { add(coordinate(it)) }
                add(coordinate(vertices.first()))
            })
        }
    }

    return buildJsonObject {
        put("type", "Feature")
        put("id", id)

        putJsonObject("geometry") {
            put("type", shape)
            put("coordinates", coordinates)
        }

        putJsonObject("properties") {
            put("project", project)
            put("type", type)
            put("subtype", subtype)
            put("capturedAtEpochMs", capturedAtEpochMs)
            put("notes", notes)
            put("surveyed", surveyed)
            put("surveyReference", surveyReference)
            put("verticalDatum", verticalDatum)

            soilPh?.let { put("soilPh", it) }
            moisture?.let { put("moisture", it) }
            elevationMeters?.let { put("elevationMeters", it) }

            putJsonArray("vertexMetadata") {
                vertices.forEach {
                    add(buildJsonObject {
                        put("source", it.source)
                        it.accuracyMeters?.let { accuracy ->
                            put("accuracyMeters", accuracy)
                        }
                    })
                }
            }

            putJsonArray("assets") {
                photosBase64.indices.forEach { index ->
                    add("$assetPrefix/$id-$index.jpg")
                }
            }
        }
    }
}

private fun validate(record: Observation) {
    require(record.project.isNotBlank()) { "Project is required." }
    require(record.type.isNotBlank()) { "Type is required." }
    require(record.subtype.isNotBlank()) { "Subtype is required." }

    require(record.vertices.all {
        it.latitude.isFinite() &&
        it.longitude.isFinite() &&
        it.latitude in -90.0..90.0 &&
        it.longitude in -180.0..180.0
    }) { "Invalid coordinates." }

    val distinct = record.vertices
        .map { it.longitude to it.latitude }
        .distinct()
        .size

    when (record.shape) {
        "Point" -> require(record.vertices.size == 1) {
            "A point requires one vertex."
        }

        "LineString" -> require(distinct >= 2) {
            "A line requires two distinct vertices."
        }

        "Polygon" -> require(distinct >= 3) {
            "An area requires three distinct vertices."
        }

        else -> error("Unsupported geometry.")
    }

    require(record.vertices.size <= 500) {
        "Maximum 500 vertices."
    }

    record.soilPh?.let {
        require(it.isFinite() && it in 0.0..14.0) {
            "Soil pH must be between 0 and 14."
        }
    }

    record.moisture?.let {
        require(it.isFinite()) { "Invalid moisture value." }
    }

    record.elevationMeters?.let {
        require(it.isFinite() && record.verticalDatum.isNotBlank()) {
            "Elevation requires a vertical datum."
        }
    }

    if (record.surveyed) {
        require(
            record.surveyReference.isNotBlank() &&
            record.vertices.all { it.source == "manual" }
        ) {
            "Surveyed positions require manual coordinates and a survey reference."
        }
    }

    require(record.photosBase64.size <= 4) {
        "Maximum four photographs."
    }

    require(record.photosBase64.all { it.length <= 1_500_000 }) {
        "A photograph exceeds this pilot's size limit."
    }
}

@Composable
fun App() {
    val scope = rememberCoroutineScope()

    var step by remember { mutableStateOf(0) }
    var screen by remember { mutableStateOf("capture") }

    var project by remember {
        mutableStateOf(
            storage.getString("project", "Arkansas Field Survey")
        )
    }

    var type by remember { mutableStateOf("GCP") }
    var subtype by remember { mutableStateOf("Survey target") }
    var shape by remember { mutableStateOf("Point") }

    var latitude by remember { mutableStateOf("") }
    var longitude by remember { mutableStateOf("") }
    var vertices by remember { mutableStateOf(emptyList<Vertex>()) }

    var notes by remember { mutableStateOf("") }
    var soilPh by remember { mutableStateOf("") }
    var moisture by remember { mutableStateOf("") }

    var surveyed by remember { mutableStateOf(false) }
    var surveyReference by remember { mutableStateOf("") }
    var elevation by remember { mutableStateOf("") }
    var datum by remember { mutableStateOf("") }

    var photos by remember { mutableStateOf(emptyList<String>()) }
    var records by remember { mutableStateOf(readRecords()) }

    // Intentionally session-only. Never persisted to ordinary preferences.
    var token by remember { mutableStateOf("") }

    var message by remember { mutableStateOf("") }
    var uploading by remember { mutableStateOf(false) }

    fun number(value: String, label: String): Double? {
        if (value.isBlank()) return null
        return value.toDoubleOrNull()
            ?.takeIf { it.isFinite() }
            ?: error("$label must be a number.")
    }

    fun observation(): Observation = Observation(
        id = uniqueId(),
        project = project.trim(),
        type = type.trim(),
        subtype = subtype.trim(),
        shape = shape,
        vertices = vertices,
        capturedAtEpochMs = nowMillis(),
        notes = notes,
        soilPh = number(soilPh, "Soil pH"),
        moisture = number(moisture, "Moisture"),
        surveyed = surveyed,
        surveyReference = surveyReference.trim(),
        elevationMeters = number(elevation, "Elevation"),
        verticalDatum = datum.trim(),
        photosBase64 = photos
    )

    fun addVertex(vertex: Vertex) {
        if (shape == "Point") vertices = listOf(vertex)
        else if (vertices.size < 500) vertices = vertices + vertex
        else message = "Maximum 500 vertices."
    }

    val colors = lightColorScheme(
        primary = Color(0xFF246B51),
        background = Color(0xFFF4F7F2),
        surface = Color.White
    )

    MaterialTheme(colorScheme = colors) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = colors.background
        ) {
            Column(
                Modifier
                    .fillMaxSize()
                    .safeDrawingPadding()
                    .imePadding()
                    .verticalScroll(rememberScrollState())
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    "FIELD ATLAS",
                    style = MaterialTheme.typography.titleLarge
                )

                Text(
                    "Native field capture · saved on this device",
                    style = MaterialTheme.typography.bodySmall
                )

                OutlinedTextField(
                    value = project,
                    onValueChange = {
                        project = it.take(120)
                        storage.putString("project", project)
                    },
                    label = { Text("Project") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("capture", "records", "upload").forEach { destination ->
                        OutlinedButton(
                            onClick = {
                                screen = destination
                                records = readRecords()
                            }
                        ) {
                            Text(destination.replaceFirstChar { it.uppercase() })
                        }
                    }
                }

                if (message.isNotBlank()) {
                    Card {
                        Text(message, Modifier.padding(14.dp))
                    }
                }

                if (screen == "capture") {
                    Text(
                        "Step ${step + 1} of 4",
                        style = MaterialTheme.typography.labelLarge
                    )

                    LinearProgressIndicator(
                        progress = { (step + 1) / 4f },
                        modifier = Modifier.fillMaxWidth()
                    )

                    when (step) {
                        0 -> {
                            Text(
                                "Classify",
                                style = MaterialTheme.typography.headlineMedium
                            )

                            Row(
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                listOf("GCP", "Soil", "Observation").forEach { preset ->
                                    OutlinedButton(onClick = { type = preset }) {
                                        Text(preset)
                                    }
                                }
                            }

                            OutlinedTextField(
                                type,
                                { type = it.take(120) },
                                label = { Text("Type — select or enter a new type") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                subtype,
                                { subtype = it.take(120) },
                                label = { Text("Subtype") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            listOf(
                                "Point" to "Point",
                                "LineString" to "Line",
                                "Polygon" to "Area"
                            ).forEach { (value, label) ->
                                OutlinedButton(
                                    onClick = {
                                        shape = value
                                        vertices = emptyList()
                                    },
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text(
                                        if (shape == value) "✓ $label" else label
                                    )
                                }
                            }
                        }

                        1 -> {
                            Text(
                                "Locate",
                                style = MaterialTheme.typography.headlineMedium
                            )

                            NativeCapture(
                                onPosition = {
                                    addVertex(it)
                                    message = "Phone position recorded."
                                },
                                onPhoto = {},
                                onError = { message = it }
                            )

                            OutlinedTextField(
                                latitude,
                                { latitude = it },
                                label = { Text("Latitude") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                longitude,
                                { longitude = it },
                                label = { Text("Longitude") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            Button(
                                onClick = {
                                    runCatching {
                                        val lat = number(latitude, "Latitude")
                                            ?: error("Latitude is required.")
                                        val lng = number(longitude, "Longitude")
                                            ?: error("Longitude is required.")

                                        require(lat in -90.0..90.0 && lng in -180.0..180.0) {
                                            "Coordinates are out of range."
                                        }

                                        addVertex(Vertex(lng, lat, "manual"))
                                    }.onFailure { message = it.message.orEmpty() }
                                }
                            ) {
                                Text("Add manual vertex")
                            }

                            vertices.forEachIndexed { index, vertex ->
                                Text(
                                    "${index + 1}. ${vertex.latitude}, " +
                                    "${vertex.longitude} · ${vertex.source}" +
                                    (vertex.accuracyMeters?.let { " ±$it m" } ?: "")
                                )
                            }

                            OutlinedButton(
                                onClick = { vertices = vertices.dropLast(1) },
                                enabled = vertices.isNotEmpty()
                            ) {
                                Text("Undo last vertex")
                            }

                            Text(
                                "Phone GPS does not establish survey-grade GCP accuracy."
                            )

                            Row {
                                Checkbox(
                                    checked = surveyed,
                                    onCheckedChange = { surveyed = it }
                                )
                                Text(
                                    "Surveyed coordinates",
                                    Modifier.padding(top = 12.dp)
                                )
                            }

                            OutlinedTextField(
                                surveyReference,
                                { surveyReference = it },
                                label = { Text("Survey reference") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                elevation,
                                { elevation = it },
                                label = { Text("Elevation in meters — optional") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                datum,
                                { datum = it },
                                label = { Text("Vertical datum") },
                                modifier = Modifier.fillMaxWidth()
                            )
                        }

                        2 -> {
                            Text(
                                "Document",
                                style = MaterialTheme.typography.headlineMedium
                            )

                            OutlinedTextField(
                                soilPh,
                                { soilPh = it },
                                label = { Text("Soil pH — optional") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                moisture,
                                { moisture = it },
                                label = { Text("Moisture — optional; document units in notes") },
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                notes,
                                { notes = it.take(10000) },
                                label = { Text("Notes") },
                                modifier = Modifier.fillMaxWidth(),
                                minLines = 3
                            )

                            NativeCapture(
                                onPosition = {},
                                onPhoto = {
                                    if (photos.size >= 4) {
                                        message = "Maximum four photographs."
                                    } else if (it.length > 1_500_000) {
                                        message = "Photograph too large for this pilot."
                                    } else {
                                        photos = photos + it
                                        message = "Photograph attached."
                                    }
                                },
                                onError = { message = it }
                            )

                            Text("${photos.size} photographs attached")

                            OutlinedButton(
                                onClick = { photos = photos.dropLast(1) },
                                enabled = photos.isNotEmpty()
                            ) {
                                Text("Remove last photograph")
                            }
                        }

                        3 -> {
                            Text(
                                "Review",
                                style = MaterialTheme.typography.headlineMedium
                            )

                            Card(Modifier.fillMaxWidth()) {
                                Column(
                                    Modifier.padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Text("$type · $subtype")
                                    Text("Project: $project")
                                    Text("Shape: $shape")
                                    Text("Vertices: ${vertices.size}")
                                    Text("Photographs: ${photos.size}")
                                    Text("Soil pH: ${soilPh.ifBlank { "Not entered" }}")
                                    Text("Moisture: ${moisture.ifBlank { "Not entered" }}")
                                    Text(notes.ifBlank { "No notes" })
                                }
                            }

                            Button(
                                onClick = {
                                    runCatching {
                                        val record = observation()
                                        validate(record)
                                        saveRecord(record)

                                        vertices = emptyList()
                                        photos = emptyList()
                                        notes = ""
                                        soilPh = ""
                                        moisture = ""
                                        elevation = ""
                                        datum = ""
                                        surveyReference = ""
                                        surveyed = false
                                        latitude = ""
                                        longitude = ""
                                        step = 0

                                        records = readRecords()
                                        message = "Saved on this device."
                                    }.onFailure {
                                        message = it.message ?: "Save failed."
                                    }
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("Save observation")
                            }
                        }
                    }

                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        OutlinedButton(
                            onClick = { step-- },
                            enabled = step > 0
                        ) {
                            Text("Back")
                        }

                        if (step < 3) {
                            Button(
                                onClick = {
                                    runCatching {
                                        if (step == 0) {
                                            require(
                                                project.isNotBlank() &&
                                                type.isNotBlank() &&
                                                subtype.isNotBlank()
                                            ) {
                                                "Project, type and subtype are required."
                                            }
                                        }

                                        if (step >= 1) validate(observation())

                                        message = ""
                                        step++
                                    }.onFailure {
                                        message = it.message.orEmpty()
                                    }
                                }
                            ) {
                                Text("Continue")
                            }
                        }
                    }
                }

                if (screen == "records") {
                    val visible = records.filter { it.project == project.trim() }

                    Text(
                        "${visible.size} saved records",
                        style = MaterialTheme.typography.headlineSmall
                    )

                    visible.take(100).forEach { record ->
                        Card(Modifier.fillMaxWidth()) {
                            Column(Modifier.padding(14.dp)) {
                                Text("${record.type} · ${record.subtype}")
                                Text("${record.shape} · ${record.vertices.size} vertices")
                                Text("${record.photosBase64.size} photographs")
                                Text(record.notes)
                                Text(
                                    record.id,
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }

                    if (visible.size > 100) {
                        Text("Showing the latest 100 records.")
                    }
                }

                if (screen == "upload") {
                    Text(
                        "Repository snapshot",
                        style = MaterialTheme.typography.headlineMedium
                    )

                    Text("Destination: KashmirWorld/datarepo")
                    Text("Folder: kwf-field-data/captures/")

                    Text(
                        "Enter your own fine-grained GitHub PAT. " +
                        "It remains in memory for this app session only."
                    )

                    OutlinedTextField(
                        value = token,
                        onValueChange = { token = it.trim() },
                        label = { Text("GitHub Personal Access Token") },
                        visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true
                    )

                    Button(
                        onClick = {
                            val snapshotRecords = readRecords()
                                .filter { it.project == project.trim() }

                            uploading = true
                            message = "Preparing snapshot…"

                            scope.launch {
                                try {
                                    pushSnapshot(
                                        token = token,
                                        project = project.trim(),
                                        records = snapshotRecords,
                                        onProgress = { message = it }
                                    )

                                    message = "Snapshot committed to GitHub."
                                } catch (error: Exception) {
                                    message = error.message ?: "Upload failed."
                                } finally {
                                    uploading = false
                                }
                            }
                        },
                        enabled = !uploading && token.isNotBlank(),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(if (uploading) "Uploading…" else "Upload project snapshot")
                    }

                    OutlinedButton(onClick = { token = "" }) {
                        Text("Forget token")
                    }

                    Text(
                        "Upload requires internet. Local records remain on the " +
                        "device if upload fails."
                    )
                }
            }
        }
    }
}
