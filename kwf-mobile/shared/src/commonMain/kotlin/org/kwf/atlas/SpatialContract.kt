package org.kwf.atlas

import kotlinx.serialization.Serializable

/**
 * Shared location contract for KWF Field Atlas.
 * Platform adapters must satisfy this contract before writing a snapshot.
 *
 * This file is additive. It does not replace existing observation models.
 * Existing records migrate by tagging their stored longitude/latitude as EPSG:4326
 * and marking height reference as "unknown" where it was not recorded.
 */

/**
 * A single coordinate vertex from a field observation.
 *
 * [x]   Easting or longitude. For EPSG:4326 this is longitude in decimal degrees.
 * [y]   Northing or latitude. For EPSG:4326 this is latitude in decimal degrees.
 * [z]   Optional ellipsoidal or orthometric height, depending on [vcrs].
 * [crs] Horizontal CRS as an authority:code string, e.g. "EPSG:4326".
 *       Other CRSs are captured as source observations and normalized by Python.
 * [vcrs] Vertical reference: "unknown", "ellipsoidal", or an authority:code string.
 *        Do not assign a vertical reference without API evidence.
 * [horizontalAccuracyM] Horizontal accuracy in metres. Do not invent this value.
 * [verticalAccuracyM]   Vertical accuracy in metres. Do not invent this value.
 * [source] How this coordinate was collected: "gps", "manual", "survey", "platform-derived".
 * [capturedAtEpochMs] Epoch milliseconds when the fix was recorded. Null for manual entry.
 * [provider] Android location provider name, if available. Null on iOS and manual entry.
 */
@Serializable
data class SourceVertex(
    val x: Double,
    val y: Double,
    val z: Double? = null,
    val crs: String,
    val vcrs: String = "unknown",
    val horizontalAccuracyM: Double? = null,
    val verticalAccuracyM: Double? = null,
    val source: String,
    val capturedAtEpochMs: Long? = null,
    val provider: String? = null,
    val coordinateEpoch: Double? = null,
)

/**
 * A complete field observation as captured on device.
 * This is written to source-observations.json.
 *
 * [id]       Stable UUID for this observation. Never changes on edit.
 * [revision] Increments on every edit. Used to detect conflicts.
 * [projectId] UUID of the project this observation belongs to.
 * [type]      Observation type, from project configuration.
 * [subtype]   Observation subtype, from project configuration.
 * [shape]     "Point", "LineString", or "Polygon".
 * [vertices]  One or more [SourceVertex] instances.
 * [notes]     Free-text notes.
 * [soilPh]    Soil pH 0–14, or null.
 * [moisture]  Moisture measurement, or null.
 * [moistureUnit] Required when [moisture] is non-null.
 * [surveyed]  True only when coordinates are from a verified survey instrument.
 * [surveyReference] Required when [surveyed] is true.
 * [gcpLabel]  Required when [type] is "GCP".
 * [sourceLabel] Original source label, preserved during migration.
 */
@Serializable
data class SourceObservation(
    val id: String,
    val revision: Int = 1,
    val projectId: String,
    val type: String,
    val subtype: String = "",
    val shape: String = "Point",
    val vertices: List<SourceVertex> = emptyList(),
    val notes: String = "",
    val soilPh: Double? = null,
    val moisture: Double? = null,
    val moistureUnit: String? = null,
    val surveyed: Boolean = false,
    val surveyReference: String? = null,
    val gcpLabel: String? = null,
    val sourceLabel: String? = null,
    val observedAtEpochMs: Long? = null,
    val updatedAtEpochMs: Long? = null,
    val assets: List<String> = emptyList(),
)

/**
 * Project configuration persisted on device.
 *
 * [id]           Stable UUID. Never changes when the project is renamed.
 * [name]         Editable display name. Does not affect upload path.
 * [defaultCrs]   Default horizontal CRS for new observations, e.g. "EPSG:4326".
 * [defaultVcrs]  Default vertical reference for new observations.
 * [studyExtent]  Optional bounding box [minLon, minLat, maxLon, maxLat] in WGS-84.
 * [types]        Observation type/subtype pairs from project configuration.
 */
@Serializable
data class ProjectConfig(
    val id: String,
    val name: String,
    val defaultCrs: String = "EPSG:4326",
    val defaultVcrs: String = "unknown",
    val studyExtent: List<Double>? = null,
    val types: List<ObservationTypeEntry> = emptyList(),
)

@Serializable
data class ObservationTypeEntry(
    val type: String,
    val subtypes: List<String> = emptyList(),
)

/**
 * Manifest written alongside source-observations.json.
 * Used to detect which batches have already been acknowledged.
 */
@Serializable
data class BatchManifest(
    val projectId: String,
    val batchId: String,
    val observationIds: List<String>,
    val assetReferences: List<String> = emptyList(),
    val acknowledgedRevision: String? = null,
    val acknowledgedHash: String? = null,
)

/**
 * Contract validator. Call before writing a snapshot.
 * Throws [IllegalArgumentException] with a descriptive message on violation.
 */
object SpatialContract {

    private val crsPattern = Regex("""^[A-Z]+:\d+$""")

    fun validate(
        observation: SourceObservation,
        project: ProjectConfig? = null
    ) {
        require(observation.id.isNotBlank()) { "Observation ID must not be blank." }
        require(observation.projectId.isNotBlank()) { "Project ID must not be blank." }
        require(observation.revision >= 1) { "Revision must be positive." }
        require(observation.shape in listOf("Point", "LineString", "Polygon")) {
            "Shape must be Point, LineString, or Polygon."
        }
        require(observation.vertices.isNotEmpty()) { "At least one vertex is required." }
        require(observation.vertices.size <= 500) {
            "Maximum 500 vertices."
        }

        val sourceCrs = observation.vertices.first().crs
        require(observation.vertices.all { it.crs == sourceCrs }) {
            "All vertices in one geometry must use the same horizontal CRS."
        }
        project?.studyExtent?.let { extent ->
            require(
                extent.size == 4 &&
                    extent.all { it.isFinite() } &&
                    extent[0] in -180.0..180.0 &&
                    extent[2] in -180.0..180.0 &&
                    extent[1] in -90.0..90.0 &&
                    extent[3] in -90.0..90.0 &&
                    extent[0] < extent[2] &&
                    extent[1] < extent[3]
            ) { "Invalid project study extent." }
        }

        for (vertex in observation.vertices) {
            require(vertex.x.isFinite()) { "Vertex x must be finite." }
            require(vertex.y.isFinite()) { "Vertex y must be finite." }
            vertex.z?.let { require(it.isFinite()) { "Vertex z must be finite." } }
            vertex.coordinateEpoch?.let {
                require(it.isFinite()) { "Coordinate epoch must be finite." }
            }
            vertex.horizontalAccuracyM?.let {
                require(it.isFinite() && it >= 0.0) {
                    "Horizontal accuracy cannot be negative or non-finite."
                }
            }
            vertex.verticalAccuracyM?.let {
                require(it.isFinite() && it >= 0.0) {
                    "Vertical accuracy cannot be negative or non-finite."
                }
            }
            require(
                vertex.crs == "EPSG:4326" ||
                crsPattern.matches(vertex.crs)
            ) { "Invalid horizontal CRS: ${vertex.crs}" }
            if (vertex.crs == "EPSG:4326") {
                require(vertex.x in -180.0..180.0 && vertex.y in -90.0..90.0) {
                    "WGS 84 coordinates must be longitude [-180, 180] and latitude [-90, 90]."
                }
                project?.studyExtent?.let { extent ->
                    require(extent.size == 4) { "Invalid project study extent." }
                    require(
                        vertex.x in extent[0]..extent[2] &&
                            vertex.y in extent[1]..extent[3]
                    ) { "WGS 84 coordinate is outside the project study extent." }
                }
            }
            require(
                vertex.vcrs == "unknown" ||
                vertex.vcrs == "ellipsoidal" ||
                crsPattern.matches(vertex.vcrs)
            ) { "Invalid height reference." }
        }

        val distinct = observation.vertices
            .map { it.x to it.y }
            .distinct()
            .size

        when (observation.shape) {
            "Point" -> require(observation.vertices.size == 1) {
                "Point requires exactly one vertex."
            }
            "LineString" -> require(distinct >= 2) {
                "Line requires two distinct vertices."
            }
            "Polygon" -> require(distinct >= 3) {
                "Polygon requires three distinct vertices."
            }
        }

        observation.soilPh?.let {
            require(it.isFinite() && it in 0.0..14.0) {
                "Soil pH must be between 0 and 14."
            }
        }

        observation.moisture?.let {
            require(it.isFinite()) { "Invalid moisture measurement." }
            require(!observation.moistureUnit.isNullOrBlank()) {
                "Moisture requires a unit."
            }
        }

        if (observation.surveyed) {
            require(!observation.surveyReference.isNullOrBlank()) {
                "Surveyed coordinates require a survey reference."
            }
            require(observation.vertices.all {
                it.source == "manual" || it.source == "survey"
            }) { "Surveyed positions must retain their survey source." }
        }

        if (observation.type.equals("GCP", ignoreCase = true)) {
            require(observation.shape == "Point") {
                "GCP must be a point."
            }
            require(!observation.gcpLabel.isNullOrBlank()) {
                "GCP requires a target label."
            }

            // Unsurveyed GCP candidates can be saved, but cannot
            // be accepted by the ground-coordinate exporter.
            if (observation.surveyed) {
                val vertex = observation.vertices.single()

                require(vertex.z != null) { "Surveyed GCP requires elevation." }
                require(vertex.vcrs != "unknown") {
                    "Surveyed GCP requires a known height reference."
                }
                require(
                    vertex.horizontalAccuracyM != null &&
                    vertex.verticalAccuracyM != null
                ) { "Surveyed GCP requires horizontal and vertical accuracy." }
            }
        }
    }

    fun validate(record: Observation, project: ProjectConfig) {
        require(record.project.isNotBlank()) { "Project is required." }
        require(record.type.isNotBlank()) { "Type is required." }
        require(record.subtype.isNotBlank()) { "Subtype is required." }
        validate(record.toSourceObservation(project.id), project)
        record.elevationMeters?.let {
            require(it.isFinite() && record.verticalDatum.isNotBlank()) {
                "Elevation requires a vertical datum / height reference."
            }
        }
        require(record.photosBase64.size <= 4) {
            "Maximum four photographs."
        }
        require(record.photosBase64.all { it.length <= 1_500_000 }) {
            "A photograph exceeds this pilot's size limit."
        }
    }
}
