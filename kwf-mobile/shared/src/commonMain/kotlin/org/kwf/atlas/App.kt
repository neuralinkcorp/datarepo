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
    val accuracyMeters: Double? = null,
    val z: Double? = null,
    val crs: String = "EPSG:4326",
    val vcrs: String = "unknown",
    val verticalAccuracyM: Double? = null,
    val capturedAtEpochMs: Long? = null,
    val provider: String? = null
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
    val photosBase64: List<String> = emptyList(),
    val projectId: String = "",
    val revision: Int = 1,
    val moistureUnit: String? = null,
    val gcpLabel: String? = null,
    val sourceLabel: String? = null
)

expect fun nowMillis(): Long

@Composable
expect fun NativeCapture(
    onPosition: (Vertex) -> Unit,
    onPhoto: (String) -> Unit,
    onError: (String) -> Unit,
    showLocation: Boolean,
    showCamera: Boolean
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

fun Vertex.toSourceVertex(): SourceVertex = SourceVertex(
    x = longitude,
    y = latitude,
    z = z,
    crs = crs,
    vcrs = vcrs,
    horizontalAccuracyM = accuracyMeters,
    verticalAccuracyM = verticalAccuracyM,
    source = source,
    capturedAtEpochMs = capturedAtEpochMs,
    provider = provider
)

fun SourceVertex.toVertex(): Vertex = Vertex(
    longitude = x,
    latitude = y,
    source = source,
    accuracyMeters = horizontalAccuracyM,
    z = z,
    crs = crs,
    vcrs = vcrs,
    verticalAccuracyM = verticalAccuracyM,
    capturedAtEpochMs = capturedAtEpochMs,
    provider = provider
)

fun Observation.toSourceObservation(currentProjectId: String): SourceObservation {
    return SourceObservation(
        id = id,
        revision = revision,
        projectId = if (projectId.isNotBlank()) projectId else currentProjectId,
        type = type,
        subtype = subtype,
        shape = shape,
        vertices = vertices.map { it.toSourceVertex() },
        notes = notes,
        soilPh = soilPh,
        moisture = moisture,
        moistureUnit = moistureUnit,
        surveyed = surveyed,
        surveyReference = if (surveyReference.isBlank()) null else surveyReference,
        gcpLabel = gcpLabel,
        sourceLabel = sourceLabel ?: vertices.firstOrNull()?.source
    )
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

/**
 * Returns valid WGS 84 GeoJSON Feature, or null if any vertex is not EPSG:4326.
 * Projected coordinates are never placed directly into GeoJSON.
 */
fun Observation.toFeatureOrNull(assetPrefix: String): JsonObject? {
    val allWgs84 = vertices.isNotEmpty() && vertices.all { it.crs == "EPSG:4326" }
    if (!allWgs84) return null

    fun coordinate(vertex: Vertex) = buildJsonArray {
        add(vertex.longitude)
        add(vertex.latitude)
        vertex.z?.let { add(it) }
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
            put("projectId", projectId)
            put("type", type)
            put("subtype", subtype)
            put("capturedAtEpochMs", capturedAtEpochMs)
            put("notes", notes)
            put("surveyed", surveyed)
            put("surveyReference", surveyReference)
            put("verticalDatum", verticalDatum)
            gcpLabel?.let { put("gcpLabel", it) }

            soilPh?.let { put("soilPh", it) }
            moisture?.let { put("moisture", it) }
            moistureUnit?.let { put("moistureUnit", it) }
            elevationMeters?.let { put("elevationMeters", it) }

            putJsonArray("vertexMetadata") {
                vertices.forEach {
                    add(buildJsonObject {
                        put("source", it.source)
                        it.accuracyMeters?.let { accuracy ->
                            put("accuracyMeters", accuracy)
                        }
                        it.z?.let { z -> put("z", z) }
                        put("crs", it.crs)
                        put("vcrs", it.vcrs)
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
    require(record.vertices.isNotEmpty()) { "At least one coordinate vertex is required." }

    val allWgs84 = record.vertices.all { it.crs == "EPSG:4326" }
    if (allWgs84) {
        require(record.vertices.all {
            it.latitude.isFinite() &&
            it.longitude.isFinite() &&
            it.latitude in -90.0..90.0 &&
            it.longitude in -180.0..180.0
        }) { "WGS-84 coordinates must be in valid ranges (latitude [-90, 90], longitude [-180, 180])." }
    } else {
        require(record.vertices.all {
            it.latitude.isFinite() && it.longitude.isFinite()
        }) { "Coordinates must be finite numbers." }
    }

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

        else -> error("Unsupported geometry shape.")
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
        require(!record.moistureUnit.isNullOrBlank()) {
            "Moisture requires a unit (e.g. %, m³/m³)."
        }
    }

    record.elevationMeters?.let {
        require(it.isFinite() && record.verticalDatum.isNotBlank()) {
            "Elevation requires a vertical datum / height reference."
        }
    }

    if (record.surveyed) {
        require(record.surveyReference.isNotBlank()) {
            "Surveyed positions require a survey reference."
        }
        require(record.vertices.all { it.source == "manual" || it.source == "survey" }) {
            "Surveyed positions must retain manual/survey source."
        }
    }

    if (record.type.equals("GCP", ignoreCase = true)) {
        require(record.shape == "Point") { "GCP must be a point." }
        require(!record.gcpLabel.isNullOrBlank()) { "GCP requires a target label." }
        if (record.surveyed) {
            val v = record.vertices.single()
            require(record.elevationMeters != null || v.z != null) {
                "Surveyed GCP requires elevation."
            }
            val vcrs = if (record.verticalDatum.isNotBlank()) record.verticalDatum else v.vcrs
            require(vcrs != "unknown") {
                "Surveyed GCP requires a known height reference."
            }
            require(v.accuracyMeters != null && v.verticalAccuracyM != null) {
                "Surveyed GCP requires horizontal and vertical accuracy."
            }
        }
    }

    require(record.photosBase64.size <= 4) {
        "Maximum four photographs."
    }

    require(record.photosBase64.all { it.length <= 1_500_000 }) {
        "A photograph exceeds this pilot's size limit."
    }
}

private fun defaultProjectTypes(): List<ObservationTypeEntry> = listOf(
    ObservationTypeEntry("GCP", listOf("Survey target", "Aerial marker")),
    ObservationTypeEntry("Soil", listOf("Surface sample", "Core sample", "Moisture sensor")),
    ObservationTypeEntry("Vegetation", listOf("Tree", "Shrub", "Ground cover")),
    ObservationTypeEntry("Observation", listOf("Wildlife", "Infrastructure", "Water resource"))
)

private fun loadProjectTypes(): List<ObservationTypeEntry> {
    val raw = storage.getString("project_types_json", "")
    if (raw.isNotBlank()) {
        val parsed = runCatching {
            codec.decodeFromString<List<ObservationTypeEntry>>(raw)
        }.getOrNull()
        if (parsed != null && parsed.isNotEmpty()) return parsed
    }
    val defaults = defaultProjectTypes()
    storage.putString("project_types_json", codec.encodeToString(defaults))
    return defaults
}

private fun saveProjectTypes(entries: List<ObservationTypeEntry>) {
    storage.putString("project_types_json", codec.encodeToString(entries))
}

@Composable
fun App() {
    val scope = rememberCoroutineScope()
    val snackbars = remember { SnackbarHostState() }

    var screen by remember { mutableStateOf("home") }
    var step by remember { mutableStateOf(0) }

    // Project UUID is persisted separately from editable project name.
    // Project rename NEVER modifies the project ID or upload path.
    val projectId = remember {
        var id = storage.getString("project_id", "")
        if (id.isBlank()) {
            id = uniqueId()
            storage.putString("project_id", id)
        }
        id
    }

    var project by remember {
        mutableStateOf(storage.getString("project", "Arkansas Field Survey"))
    }
    var defaultCrs by remember {
        mutableStateOf(storage.getString("project_default_crs", "EPSG:4326"))
    }
    var defaultVcrs by remember {
        mutableStateOf(storage.getString("project_default_vcrs", "unknown"))
    }
    var studyExtentText by remember {
        mutableStateOf(storage.getString("project_study_extent", ""))
    }

    var projectTypes by remember { mutableStateOf(loadProjectTypes()) }
    var showAddTypeDialog by remember { mutableStateOf(false) }
    var newTypeName by remember { mutableStateOf("") }
    var newSubtypeName by remember { mutableStateOf("") }

    var type by remember { mutableStateOf("GCP") }
    var subtype by remember { mutableStateOf("Survey target") }
    var shape by remember { mutableStateOf("Point") }
    var gcpLabel by remember { mutableStateOf("") }

    var manualX by remember { mutableStateOf("") }
    var manualY by remember { mutableStateOf("") }
    var manualZ by remember { mutableStateOf("") }
    var manualCrs by remember { mutableStateOf("EPSG:4326") }
    var manualVcrs by remember { mutableStateOf("unknown") }
    var manualAccuracyM by remember { mutableStateOf("") }
    var manualVAccuracyM by remember { mutableStateOf("") }

    var vertices by remember { mutableStateOf(emptyList<Vertex>()) }

    var notes by remember { mutableStateOf("") }
    var soilPh by remember { mutableStateOf("") }
    var moisture by remember { mutableStateOf("") }
    var moistureUnit by remember { mutableStateOf("%") }

    var surveyed by remember { mutableStateOf(false) }
    var surveyReference by remember { mutableStateOf("") }
    var elevation by remember { mutableStateOf("") }
    var datum by remember { mutableStateOf("unknown") }

    var photos by remember { mutableStateOf(emptyList<String>()) }
    var records by remember { mutableStateOf(readRecords()) }

    // Session only. Never persisted to ordinary preferences.
    var token by remember { mutableStateOf("") }
    var uploading by remember { mutableStateOf(false) }
    var uploadStatus by remember { mutableStateOf("") }

    var query by remember { mutableStateOf("") }
    var selectedRecord by remember { mutableStateOf<Observation?>(null) }

    fun notify(text: String) {
        scope.launch { snackbars.showSnackbar(text) }
    }

    fun parsedNumber(value: String, label: String): Double? {
        if (value.isBlank()) return null
        return value.toDoubleOrNull()?.takeIf { it.isFinite() }
            ?: error("$label must be a valid number.")
    }

    fun parseStudyExtent(text: String): List<Double>? {
        if (text.isBlank()) return null
        val parts = text.split(",").map { it.trim().toDoubleOrNull() }
        if (parts.size == 4 && parts.all { it != null && it.isFinite() }) {
            return parts.filterNotNull()
        }
        return null
    }

    val currentProjectConfig = ProjectConfig(
        id = projectId,
        name = project.trim(),
        defaultCrs = defaultCrs,
        defaultVcrs = defaultVcrs,
        studyExtent = parseStudyExtent(studyExtentText),
        types = projectTypes
    )

    fun draft(): Observation = Observation(
        id = uniqueId(),
        projectId = projectId,
        project = project.trim(),
        type = type.trim(),
        subtype = subtype.trim(),
        shape = shape,
        vertices = vertices,
        capturedAtEpochMs = nowMillis(),
        notes = notes,
        soilPh = parsedNumber(soilPh, "Soil pH"),
        moisture = parsedNumber(moisture, "Moisture"),
        moistureUnit = if (moisture.isNotBlank()) moistureUnit.trim() else null,
        surveyed = surveyed,
        surveyReference = surveyReference.trim(),
        elevationMeters = parsedNumber(elevation, "Elevation"),
        verticalDatum = datum.trim(),
        photosBase64 = photos,
        gcpLabel = if (type.equals("GCP", ignoreCase = true)) gcpLabel.trim() else null
    )

    fun attempt(action: () -> Unit) {
        try {
            action()
        } catch (error: Exception) {
            notify(error.message ?: "Please check the entered information.")
        }
    }

    fun addVertex(vertex: Vertex) {
        if (shape == "Point") {
            vertices = listOf(vertex)
        } else {
            require(vertices.size < 500) { "Maximum 500 vertices." }
            vertices = vertices + vertex
        }
    }

    fun resetDraft() {
        vertices = emptyList()
        photos = emptyList()
        manualX = ""
        manualY = ""
        manualZ = ""
        notes = ""
        soilPh = ""
        moisture = ""
        moistureUnit = "%"
        surveyReference = ""
        elevation = ""
        datum = "unknown"
        gcpLabel = ""
        surveyed = false
        step = 0
    }

    val projectRecords = records.filter {
        (it.projectId.isNotBlank() && it.projectId == projectId) ||
        (it.projectId.isBlank() && it.project == project.trim())
    }
    val hasDraft = vertices.isNotEmpty() || photos.isNotEmpty() || notes.isNotBlank()

    val colors = lightColorScheme(
        primary = Color(0xFF235F49),
        onPrimary = Color.White,
        primaryContainer = Color(0xFFE0EEDC),
        onPrimaryContainer = Color(0xFF244532),
        background = Color(0xFFF5F6F1),
        surface = Color(0xFFFFFFFF),
        surfaceVariant = Color(0xFFEAEFE5),
        onSurface = Color(0xFF20382D),
        onSurfaceVariant = Color(0xFF647164),
        outline = Color(0xFFC8D1C4)
    )

    MaterialTheme(colorScheme = colors) {
        Scaffold(
            modifier = Modifier.safeDrawingPadding(),
            containerColor = colors.background,
            snackbarHost = { SnackbarHost(snackbars) },
            bottomBar = {
                if (screen != "capture") {
                    NavigationBar(
                        containerColor = colors.surface,
                        tonalElevation = 0.dp
                    ) {
                        listOf(
                            Triple("home", "Home", "⌂"),
                            Triple("records", "Records", "▤"),
                            Triple("repository", "Repository", "↥")
                        ).forEach { (destination, label, symbol) ->
                            NavigationBarItem(
                                selected = screen == destination,
                                onClick = {
                                    records = readRecords()
                                    screen = destination
                                },
                                icon = { Text(symbol) },
                                label = { Text(label) }
                            )
                        }
                    }
                }
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .imePadding()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 22.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(18.dp)
            ) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text(
                            "FIELD ATLAS",
                            style = MaterialTheme.typography.labelLarge,
                            color = colors.primary
                        )
                        Text(
                            if (screen == "capture") "New observation"
                            else "Field workspace",
                            style = MaterialTheme.typography.titleMedium
                        )
                    }
                    Surface(
                        shape = androidx.compose.foundation.shape.RoundedCornerShape(50.dp),
                        color = colors.primaryContainer
                    ) {
                        Text(
                            "ON DEVICE",
                            Modifier.padding(horizontal = 12.dp, vertical = 7.dp),
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }

                when (screen) {
                    "home" -> {
                        AtlasHeading(
                            "Know your field.",
                            "Capture locations, observations and evidence in one place."
                        )

                        AtlasCard {
                            Text("ACTIVE PROJECT", style = MaterialTheme.typography.labelSmall)
                            AtlasField(
                                project,
                                {
                                    project = it.take(120)
                                    storage.putString("project", project)
                                },
                                "Project display name"
                            )
                            Text(
                                "Permanent Project ID: $projectId",
                                style = MaterialTheme.typography.bodySmall,
                                color = colors.onSurfaceVariant
                            )
                            AtlasField(
                                defaultCrs,
                                {
                                    defaultCrs = it.trim()
                                    storage.putString("project_default_crs", defaultCrs)
                                },
                                "Default Horizontal CRS (e.g. EPSG:4326)"
                            )
                            AtlasField(
                                studyExtentText,
                                {
                                    studyExtentText = it.trim()
                                    storage.putString("project_study_extent", studyExtentText)
                                },
                                "Study Extent [minLon, minLat, maxLon, maxLat] · optional"
                            )
                            Text(
                                "${projectRecords.size} observations saved",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }

                        Button(
                            onClick = {
                                screen = "capture"
                                if (!hasDraft) resetDraft()
                            },
                            modifier = Modifier.fillMaxWidth().heightIn(min = 58.dp),
                            shape = androidx.compose.foundation.shape.RoundedCornerShape(16.dp)
                        ) {
                            Text(if (hasDraft) "Continue observation" else "+ New observation")
                        }

                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            AtlasMetric(
                                "Records",
                                projectRecords.size.toString(),
                                Modifier.weight(1f)
                            )
                            AtlasMetric(
                                "Photos",
                                projectRecords.sumOf { it.photosBase64.size }.toString(),
                                Modifier.weight(1f)
                            )
                        }

                        AtlasCard {
                            Text("Ready without a connection",
                                style = MaterialTheme.typography.titleMedium)
                            Text(
                                "Capture and save on this device. Repository uploads can happen later.",
                                color = colors.onSurfaceVariant
                            )
                        }

                        Text("RECENT OBSERVATIONS",
                            style = MaterialTheme.typography.labelLarge)

                        if (projectRecords.isEmpty()) {
                            AtlasEmpty(
                                "Your field log starts here",
                                "Create an observation to record a location and attach evidence."
                            )
                        } else {
                            projectRecords.take(3).forEach { record ->
                                AtlasRecord(record) { selectedRecord = record }
                            }
                            TextButton(onClick = { screen = "records" }) {
                                Text("View all records")
                            }
                        }
                    }

                    "capture" -> {
                        TextButton(onClick = { screen = "home" }) {
                            Text("← Workspace")
                        }

                        Text(
                            "STEP ${step + 1} OF 4",
                            style = MaterialTheme.typography.labelLarge,
                            color = colors.primary
                        )

                        LinearProgressIndicator(
                            progress = { (step + 1) / 4f },
                            modifier = Modifier.fillMaxWidth()
                        )

                        when (step) {
                            0 -> {
                                AtlasHeading(
                                    "Classify the observation",
                                    "Choose a category from your project configuration, or add a new one."
                                )

                                AtlasCard {
                                    Text("CATEGORY TYPE", style = MaterialTheme.typography.labelSmall)
                                    Row(
                                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        projectTypes.forEach { entry ->
                                            FilterChip(
                                                selected = type == entry.type,
                                                onClick = {
                                                    type = entry.type
                                                    subtype = entry.subtypes.firstOrNull() ?: ""
                                                },
                                                label = { Text(entry.type) }
                                            )
                                        }
                                    }

                                    val currentEntry = projectTypes.firstOrNull { it.type == type }
                                    if (currentEntry != null && currentEntry.subtypes.isNotEmpty()) {
                                        Text("SUBTYPE", style = MaterialTheme.typography.labelSmall)
                                        Row(
                                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                                        ) {
                                            currentEntry.subtypes.forEach { sub ->
                                                FilterChip(
                                                    selected = subtype == sub,
                                                    onClick = { subtype = sub },
                                                    label = { Text(sub) }
                                                )
                                            }
                                        }
                                    }

                                    AtlasField(type, { type = it.take(120) }, "Type name")
                                    AtlasField(subtype, { subtype = it.take(120) }, "Subtype name")

                                    TextButton(onClick = { showAddTypeDialog = true }) {
                                        Text("+ Add new category / subtype to project")
                                    }

                                    if (type.equals("GCP", ignoreCase = true)) {
                                        AtlasField(
                                            gcpLabel,
                                            { gcpLabel = it.take(80) },
                                            "GCP Target Label · required for GCP"
                                        )
                                    }

                                    Text(
                                        "GEOGRAPHIC SHAPE",
                                        style = MaterialTheme.typography.labelSmall
                                    )

                                    listOf(
                                        Triple("Point", "Point", "One location · target, sample or observation"),
                                        Triple("LineString", "Line", "Connected vertices · fence or route"),
                                        Triple("Polygon", "Area", "Closed boundary · habitat or survey area")
                                    ).forEach { (value, label, description) ->
                                        OutlinedButton(
                                            onClick = {
                                                if (shape != value) {
                                                    shape = value
                                                    vertices = emptyList()
                                                }
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
                                        ) {
                                            Column(
                                                Modifier.fillMaxWidth().padding(vertical = 5.dp)
                                            ) {
                                                Text(
                                                    "${if (shape == value) "●" else "○"} $label",
                                                    style = MaterialTheme.typography.titleSmall
                                                )
                                                Text(
                                                    description,
                                                    style = MaterialTheme.typography.bodySmall
                                                )
                                            }
                                        }
                                    }
                                }
                            }

                            1 -> {
                                AtlasHeading(
                                    "Capture the location",
                                    if (shape == "Point")
                                        "Record one position using GPS or manual coordinates."
                                    else
                                        "Record each vertex in order as you move around the feature."
                                )

                                AtlasCard {
                                    Text("GPS Position Capture", style = MaterialTheme.typography.titleMedium)
                                    Text(
                                        "Filters native fixes through BestFixSelector target (≤10 m).",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = colors.onSurfaceVariant
                                    )
                                    AtlasLocationCapture(
                                        onPosition = {
                                            attempt {
                                                addVertex(it)
                                                notify("Location added (${it.source}).")
                                            }
                                        },
                                        onError = { notify(it) }
                                    )

                                    Text(
                                        "${vertices.size} ${if (vertices.size == 1) "vertex" else "vertices"} recorded",
                                        style = MaterialTheme.typography.titleMedium
                                    )

                                    vertices.takeLast(8).forEachIndexed { index, vertex ->
                                        Text(
                                            "${vertices.size - minOf(vertices.size, 8) + index + 1}. " +
                                            "x=${vertex.longitude}, y=${vertex.latitude} (${vertex.crs})\n" +
                                            "${vertex.source}" +
                                            (vertex.accuracyMeters?.let { " · ±${it.toInt()} m" } ?: "") +
                                            (vertex.provider?.let { " · provider: $it" } ?: ""),
                                            style = MaterialTheme.typography.bodySmall
                                        )
                                    }

                                    if (vertices.isNotEmpty()) {
                                        TextButton(onClick = { vertices = vertices.dropLast(1) }) {
                                            Text("Undo last vertex")
                                        }
                                    }
                                }

                                AtlasCard {
                                    Text("Manual coordinates", style = MaterialTheme.typography.titleMedium)
                                    Text(
                                        "x means easting/longitude and y means northing/latitude. " +
                                        "EPSG:4326 is supported directly on the phone. " +
                                        "Other CRSs are captured as source observations and normalized by Python.",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = colors.onSurfaceVariant
                                    )

                                    AtlasField(manualX, { manualX = it }, "x (Easting / Longitude)")
                                    AtlasField(manualY, { manualY = it }, "y (Northing / Latitude)")
                                    AtlasField(manualZ, { manualZ = it }, "z (Elevation / Height) · optional")
                                    AtlasField(manualCrs, { manualCrs = it.trim() }, "Horizontal CRS (e.g. EPSG:4326)")

                                    Text("HEIGHT REFERENCE (VCRS)", style = MaterialTheme.typography.labelSmall)
                                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                        listOf("unknown", "ellipsoidal", "EPSG:5703", "EPSG:5714").forEach { ref ->
                                            FilterChip(
                                                selected = manualVcrs == ref,
                                                onClick = { manualVcrs = ref },
                                                label = { Text(ref) }
                                            )
                                        }
                                    }

                                    if (surveyed) {
                                        AtlasField(manualAccuracyM, { manualAccuracyM = it }, "Horizontal Accuracy (meters)")
                                        AtlasField(manualVAccuracyM, { manualVAccuracyM = it }, "Vertical Accuracy (meters)")
                                    }

                                    OutlinedButton(
                                        onClick = {
                                            attempt {
                                                val x = parsedNumber(manualX, "x (Easting/Longitude)")
                                                    ?: error("x coordinate is required.")
                                                val y = parsedNumber(manualY, "y (Northing/Latitude)")
                                                    ?: error("y coordinate is required.")
                                                val z = parsedNumber(manualZ, "z (Elevation)")
                                                val hAcc = parsedNumber(manualAccuracyM, "Horizontal accuracy")
                                                val vAcc = parsedNumber(manualVAccuracyM, "Vertical accuracy")

                                                if (manualCrs == "EPSG:4326") {
                                                    require(x in -180.0..180.0 && y in -90.0..90.0) {
                                                        "WGS-84 coordinates must be: x in [-180, 180], y in [-90, 90]."
                                                    }
                                                }

                                                addVertex(
                                                    Vertex(
                                                        longitude = x,
                                                        latitude = y,
                                                        source = if (surveyed) "survey" else "manual",
                                                        accuracyMeters = hAcc,
                                                        z = z,
                                                        crs = manualCrs.ifBlank { "EPSG:4326" },
                                                        vcrs = manualVcrs.ifBlank { "unknown" },
                                                        verticalAccuracyM = vAcc,
                                                        capturedAtEpochMs = nowMillis(),
                                                        provider = null
                                                    )
                                                )
                                                manualX = ""
                                                manualY = ""
                                                manualZ = ""
                                                notify("Manual vertex added.")
                                            }
                                        },
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Text("Add manual vertex")
                                    }
                                }

                                AtlasCard {
                                    Text("Survey & Vertical Reference", style = MaterialTheme.typography.titleMedium)

                                    Row {
                                        Checkbox(surveyed, { surveyed = it })
                                        Text(
                                            "These are surveyed coordinates",
                                            Modifier.padding(top = 12.dp)
                                        )
                                    }

                                    if (surveyed) {
                                        AtlasField(
                                            surveyReference,
                                            { surveyReference = it },
                                            "Survey reference identifier · required"
                                        )
                                    }

                                    AtlasField(
                                        elevation,
                                        { elevation = it },
                                        "Elevation in meters · optional"
                                    )

                                    Text("VERTICAL REFERENCE CHOICE", style = MaterialTheme.typography.labelSmall)
                                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                        listOf("unknown", "ellipsoidal", "EPSG:5703", "EPSG:5714", "EPSG:3855").forEach { choice ->
                                            FilterChip(
                                                selected = datum == choice,
                                                onClick = { datum = choice },
                                                label = { Text(choice) }
                                            )
                                        }
                                    }
                                }
                            }

                            2 -> {
                                AtlasHeading(
                                    "Add field evidence",
                                    "Attach photographs and record the context or measurements."
                                )

                                AtlasCard {
                                    Text("Photographs (Low-Resolution Preview)",
                                        style = MaterialTheme.typography.titleMedium)
                                    Text(
                                        "Low-resolution inline photo capture for field validation. Large raw imagery is not stored in Git.",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = colors.onSurfaceVariant
                                    )
                                    AtlasPhotoCapture(
                                        onPhoto = {
                                            when {
                                                photos.size >= 4 ->
                                                    notify("Maximum four photographs.")
                                                it.length > 1_500_000 ->
                                                    notify("Photograph exceeds this pilot's size limit.")
                                                else -> {
                                                    photos = photos + it
                                                    notify("Photograph attached.")
                                                }
                                            }
                                        },
                                        onError = { notify(it) }
                                    )
                                    Text("${photos.size} of 4 attached")
                                    if (photos.isNotEmpty()) {
                                        TextButton(onClick = { photos = photos.dropLast(1) }) {
                                            Text("Remove last photograph")
                                        }
                                    }
                                }

                                AtlasCard {
                                    AtlasField(
                                        notes,
                                        { notes = it.take(10000) },
                                        "Field notes",
                                        multiline = true
                                    )

                                    if (type.equals("Soil", ignoreCase = true)) {
                                        AtlasField(soilPh, { soilPh = it }, "Soil pH (0–14)")
                                        AtlasField(moisture, { moisture = it }, "Moisture value")
                                        if (moisture.isNotBlank()) {
                                            AtlasField(
                                                moistureUnit,
                                                { moistureUnit = it },
                                                "Moisture unit (e.g. %, m³/m³, cb) · required"
                                            )
                                        }
                                    } else {
                                        TextButton(onClick = { type = "Soil" }) {
                                            Text("Record soil measurements instead")
                                        }
                                    }
                                }
                            }

                            3 -> {
                                AtlasHeading(
                                    "Review your observation",
                                    "Confirm details before saving into your local field log."
                                )

                                AtlasCard {
                                    AtlasSummary("Project", project)
                                    AtlasSummary("Project ID", projectId)
                                    AtlasSummary("Category", "$type · $subtype")
                                    if (gcpLabel.isNotBlank()) AtlasSummary("GCP Label", gcpLabel)
                                    AtlasSummary("Shape", shape)
                                    AtlasSummary("Vertices", vertices.size.toString())
                                    AtlasSummary("Photographs", photos.size.toString())
                                    AtlasSummary("Position Mode", if (surveyed) "Surveyed ($surveyReference)" else "Unsurveyed")
                                    if (soilPh.isNotBlank()) AtlasSummary("Soil pH", soilPh)
                                    if (moisture.isNotBlank()) AtlasSummary("Moisture", "$moisture $moistureUnit")
                                    if (elevation.isNotBlank()) AtlasSummary("Elevation", "$elevation m ($datum)")
                                    if (notes.isNotBlank()) AtlasSummary("Notes", notes)
                                }

                                Text(
                                    "Saved locally on device. Repository uploads can be performed in the Repository tab.",
                                    style = MaterialTheme.typography.bodySmall
                                )

                                Button(
                                    onClick = {
                                        attempt {
                                            val record = draft()
                                            validate(record)
                                            saveRecord(record)
                                            records = readRecords()
                                            resetDraft()
                                            screen = "home"
                                            notify("Observation added to your field log.")
                                        }
                                    },
                                    modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp)
                                ) {
                                    Text("Save observation")
                                }
                            }
                        }

                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            OutlinedButton(
                                onClick = { step-- },
                                enabled = step > 0,
                                modifier = Modifier.weight(1f)
                            ) {
                                Text("Back")
                            }

                            if (step < 3) {
                                Button(
                                    onClick = {
                                        attempt {
                                            if (step == 0) {
                                                require(
                                                    project.isNotBlank() &&
                                                    type.isNotBlank() &&
                                                    subtype.isNotBlank()
                                                ) {
                                                    "Project, type and subtype are required."
                                                }
                                                if (type.equals("GCP", ignoreCase = true)) {
                                                    require(gcpLabel.isNotBlank()) {
                                                        "GCP target label is required for GCP observations."
                                                    }
                                                }
                                            }
                                            if (step == 1) {
                                                require(vertices.isNotEmpty()) {
                                                    "At least one coordinate vertex must be captured."
                                                }
                                            }
                                            if (step >= 1) validate(draft())
                                            step++
                                        }
                                    },
                                    modifier = Modifier.weight(1f)
                                ) {
                                    Text("Continue")
                                }
                            }
                        }

                        TextButton(
                            onClick = {
                                resetDraft()
                                screen = "home"
                            }
                        ) {
                            Text("Discard unsaved observation")
                        }
                    }

                    "records" -> {
                        AtlasHeading(
                            "Your field log",
                            "${projectRecords.size} observations in ${project.ifBlank { "this project" }}"
                        )

                        AtlasField(query, { query = it }, "Search type, subtype or notes")

                        val filtered = projectRecords.filter {
                            query.isBlank() ||
                            "${it.type} ${it.subtype} ${it.notes} ${it.gcpLabel ?: ""}".contains(query, ignoreCase = true)
                        }

                        if (filtered.isEmpty()) {
                            AtlasEmpty(
                                "No matching observations",
                                "Try a different search or create a new observation."
                            )
                        } else {
                            filtered.take(100).forEach { record ->
                                AtlasRecord(record) { selectedRecord = record }
                            }

                            if (filtered.size > 100) {
                                Text(
                                    "Showing latest 100 matching observations.",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }

                    "repository" -> {
                        AtlasHeading(
                            "Repository connection",
                            "Upload bounded batches of new or modified observations to GitHub."
                        )

                        val unacknowledgedRecords = projectRecords.filter { record ->
                            val ackRev = storage.getInt("ack_rev_${record.id}", 0)
                            record.revision > ackRev
                        }

                        AtlasCard {
                            AtlasSummary("Repository", "KashmirWorld/datarepo")
                            AtlasSummary("Project ID", projectId)
                            AtlasSummary("Upload path", "kwf-field-data/captures/$projectId/{batchId}/")
                            AtlasSummary("Total project records", projectRecords.size.toString())
                            AtlasSummary("Pending upload", unacknowledgedRecords.size.toString())

                            Text(
                                "Observations remain pending until successfully committed to GitHub. " +
                                "Batches are bounded to ≤100 records and will not re-upload unchanged acknowledged records.",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }

                        AtlasCard {
                            OutlinedTextField(
                                value = token,
                                onValueChange = { token = it.trim() },
                                label = { Text("GitHub access token") },
                                visualTransformation = PasswordVisualTransformation(),
                                modifier = Modifier.fillMaxWidth(),
                                singleLine = true
                            )

                            Text(
                                "Session-only: token is never stored in persistent preferences.",
                                style = MaterialTheme.typography.bodySmall
                            )

                            val pendingBatchKey = "pending_batch_$projectId"
                            val pendingBatchId = storage.getString(pendingBatchKey, "")
                            if (pendingBatchId.isNotBlank()) {
                                Text(
                                    "A previous batch ($pendingBatchId) is pending. Retrying will reuse this batch ID.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = colors.primary
                                )
                            }

                            Button(
                                enabled = !uploading &&
                                    token.isNotBlank() &&
                                    unacknowledgedRecords.isNotEmpty(),
                                onClick = {
                                    val batchRecords = unacknowledgedRecords.take(50)
                                    val batchId = storage.getString(pendingBatchKey, "").ifBlank {
                                        val newId = uniqueId()
                                        storage.putString(pendingBatchKey, newId)
                                        newId
                                    }

                                    uploading = true
                                    uploadStatus = "Preparing batch $batchId…"

                                    scope.launch {
                                        try {
                                            val manifest = pushSnapshot(
                                                token = token,
                                                projectId = projectId,
                                                projectName = project.trim(),
                                                projectConfig = currentProjectConfig,
                                                records = batchRecords,
                                                batchId = batchId
                                            ) {
                                                uploadStatus = it
                                            }

                                            // Acknowledge records only after branch reference update succeeds
                                            batchRecords.forEach { record ->
                                                storage.putInt("ack_rev_${record.id}", record.revision)
                                            }
                                            // Clear pending batch ID
                                            storage.remove(pendingBatchKey)

                                            records = readRecords()
                                            uploadStatus = "Batch ${manifest.batchId} acknowledged and committed (${manifest.observationIds.size} records)."
                                            notify("Repository snapshot saved.")
                                        } catch (error: Exception) {
                                            uploadStatus = error.message ?: "Upload failed. Batch remains pending."
                                        } finally {
                                            uploading = false
                                        }
                                    }
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(
                                    if (uploading) "Uploading…"
                                    else if (unacknowledgedRecords.isEmpty()) "All observations acknowledged"
                                    else "Upload pending batch (${minOf(unacknowledgedRecords.size, 50)})"
                                )
                            }

                            if (uploading) {
                                LinearProgressIndicator(Modifier.fillMaxWidth())
                            }

                            if (uploadStatus.isNotBlank()) {
                                Text(uploadStatus, style = MaterialTheme.typography.bodySmall)
                            }

                            TextButton(onClick = { token = "" }) {
                                Text("Forget token")
                            }
                        }

                        AtlasCard {
                            Text("Protect your field log",
                                style = MaterialTheme.typography.titleMedium)
                            Text(
                                "Records remain on this device until uploaded. " +
                                "Uninstalling the application or clearing its storage can remove them.",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            }
        }

        if (showAddTypeDialog) {
            AlertDialog(
                onDismissRequest = { showAddTypeDialog = false },
                title = { Text("Add Category to Project") },
                text = {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        AtlasField(newTypeName, { newTypeName = it }, "Type Name")
                        AtlasField(newSubtypeName, { newSubtypeName = it }, "Subtype Name")
                    }
                },
                confirmButton = {
                    Button(
                        onClick = {
                            if (newTypeName.isNotBlank() && newSubtypeName.isNotBlank()) {
                                val existing = projectTypes.firstOrNull { it.type.equals(newTypeName, ignoreCase = true) }
                                val updated = if (existing != null) {
                                    projectTypes.map {
                                        if (it.type.equals(newTypeName, ignoreCase = true)) {
                                            it.copy(subtypes = (it.subtypes + newSubtypeName).distinct())
                                        } else it
                                    }
                                } else {
                                    projectTypes + ObservationTypeEntry(newTypeName.trim(), listOf(newSubtypeName.trim()))
                                }
                                projectTypes = updated
                                saveProjectTypes(updated)
                                type = newTypeName.trim()
                                subtype = newSubtypeName.trim()
                                newTypeName = ""
                                newSubtypeName = ""
                                showAddTypeDialog = false
                                notify("New category added to project configuration.")
                            }
                        }
                    ) {
                        Text("Add")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showAddTypeDialog = false }) {
                        Text("Cancel")
                    }
                }
            )
        }

        selectedRecord?.let { record ->
            AlertDialog(
                onDismissRequest = { selectedRecord = null },
                title = { Text("${record.type} · ${record.subtype}") },
                text = {
                    Column(
                        Modifier.verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        AtlasSummary("Project", record.project)
                        if (record.projectId.isNotBlank()) AtlasSummary("Project ID", record.projectId)
                        record.gcpLabel?.let { AtlasSummary("GCP Target Label", it) }
                        AtlasSummary("Geometry", record.shape)
                        AtlasSummary("Photographs", "${record.photosBase64.size} (low-res preview)")
                        AtlasSummary("Position Mode", if (record.surveyed) "Surveyed (${record.surveyReference})" else "Unsurveyed")
                        record.soilPh?.let { AtlasSummary("Soil pH", it.toString()) }
                        record.moisture?.let {
                            AtlasSummary("Moisture", "$it ${record.moistureUnit ?: ""}")
                        }

                        Text("Vertices", style = MaterialTheme.typography.titleSmall)
                        record.vertices.forEachIndexed { index, vertex ->
                            Text(
                                "${index + 1}. x=${vertex.longitude}, y=${vertex.latitude} (${vertex.crs})\n" +
                                "${vertex.source}" +
                                (vertex.accuracyMeters?.let { " · ±${it.toInt()} m" } ?: "") +
                                (vertex.z?.let { " · z=$it (${vertex.vcrs})" } ?: ""),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }

                        if (record.notes.isNotBlank()) {
                            Text("Notes", style = MaterialTheme.typography.titleSmall)
                            Text(record.notes)
                        }

                        val ackRev = storage.getInt("ack_rev_${record.id}", 0)
                        AtlasSummary("Sync Status", if (record.revision <= ackRev) "Acknowledged (v$ackRev)" else "Pending upload (v${record.revision})")
                        Text("ID: ${record.id}", style = MaterialTheme.typography.labelSmall)
                    }
                },
                confirmButton = {
                    TextButton(onClick = { selectedRecord = null }) {
                        Text("Done")
                    }
                }
            )
        }
    }
}

@Composable
private fun AtlasHeading(title: String, subtitle: String) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            title,
            style = MaterialTheme.typography.headlineMedium
        )
        Text(
            subtitle,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

@Composable
private fun AtlasCard(content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = androidx.compose.foundation.shape.RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            content = content
        )
    }
}

@Composable
private fun AtlasField(
    value: String,
    onChange: (String) -> Unit,
    label: String,
    multiline: Boolean = false
) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        label = { Text(label) },
        modifier = Modifier.fillMaxWidth(),
        singleLine = !multiline,
        minLines = if (multiline) 3 else 1,
        shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
    )
}

@Composable
private fun AtlasMetric(label: String, value: String, modifier: Modifier) {
    Card(
        modifier = modifier,
        shape = androidx.compose.foundation.shape.RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(Modifier.padding(18.dp)) {
            Text(value, style = MaterialTheme.typography.headlineMedium)
            Text(label, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun AtlasSummary(label: String, value: String) {
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun AtlasEmpty(title: String, description: String) {
    AtlasCard {
        Text(title, style = MaterialTheme.typography.titleMedium)
        Text(description, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun AtlasRecord(record: Observation, onOpen: () -> Unit) {
    Card(
        onClick = onOpen,
        modifier = Modifier.fillMaxWidth(),
        shape = androidx.compose.foundation.shape.RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                "${record.type} · ${record.subtype}",
                style = MaterialTheme.typography.titleMedium
            )
            Text(
                "${record.shape}  ·  ${record.vertices.size} vertices  ·  " +
                "${record.photosBase64.size} photos (low-res preview)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (record.notes.isNotBlank()) {
                Text(
                    record.notes,
                    maxLines = 2,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            Text(
                "Open details →",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

@Composable
private fun AtlasLocationCapture(
    onPosition: (Vertex) -> Unit,
    onError: (String) -> Unit
) {
    NativeCapture(
        onPosition = onPosition,
        onPhoto = {},
        onError = onError,
        showLocation = true,
        showCamera = false
    )
}

@Composable
private fun AtlasPhotoCapture(
    onPhoto: (String) -> Unit,
    onError: (String) -> Unit
) {
    NativeCapture(
        onPosition = {},
        onPhoto = onPhoto,
        onError = onError,
        showLocation = false,
        showCamera = true
    )
}
