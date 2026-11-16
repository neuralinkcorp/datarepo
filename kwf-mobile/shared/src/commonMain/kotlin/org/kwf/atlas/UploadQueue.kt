package org.kwf.atlas

import com.russhwolf.settings.Settings
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

@Serializable
data class UploadAcknowledgement(
    val revision: Int,
    val contentHash: String
)

@Serializable
data class PendingUploadBatch(
    val batchId: String,
    val projectId: String,
    val projectName: String,
    val projectConfig: ProjectConfig,
    val records: List<Observation>,
    val recordHashes: Map<String, String>,
    val createdAtEpochMs: Long = nowMillis()
)

class UploadQueue(
    private val settings: Settings = Settings()
) {
    private val json = Json {
        encodeDefaults = true
        ignoreUnknownKeys = true
    }

    private fun pendingKey(projectId: String) =
        "upload_pending_v2:$projectId"

    private fun acknowledgementKey(projectId: String, recordId: String) =
        "upload_ack_v2:$projectId:$recordId"

    fun pending(projectId: String): PendingUploadBatch? {
        val value = settings.getString(pendingKey(projectId), "")
        if (value.isBlank()) return null
        return json.decodeFromString<PendingUploadBatch>(value).also {
            require(it.projectId == projectId) {
                "Persisted upload batch belongs to another project."
            }
        }
    }

    fun acknowledgement(
        projectId: String,
        recordId: String
    ): UploadAcknowledgement? {
        val value = settings.getString(
            acknowledgementKey(projectId, recordId),
            ""
        )
        if (value.isBlank()) return null
        return json.decodeFromString<UploadAcknowledgement>(value)
    }

    fun needsUpload(
        projectId: String,
        record: Observation,
        contentHash: String
    ): Boolean {
        val ack = acknowledgement(projectId, record.id) ?: return true
        return ack.revision != record.revision ||
            ack.contentHash != contentHash
    }

    fun persist(batch: PendingUploadBatch) {
        require(batch.records.isNotEmpty()) { "Batch is empty." }
        require(batch.records.map { it.id }.distinct().size == batch.records.size) {
            "Batch contains duplicate observation IDs."
        }
        val recordIds = batch.records.map { it.id }.toSet()
        require(batch.recordHashes.keys == recordIds) {
            "Batch record hashes do not match its observation IDs."
        }
        require(batch.recordHashes.values.all {
            it.matches(Regex("^[0-9a-f]{64}$"))
        }) { "Batch contains an invalid SHA-256 hash." }
        val existing = pending(batch.projectId)
        require(existing == null || existing == batch) {
            "An immutable pending batch already exists."
        }
        settings.putString(
            pendingKey(batch.projectId),
            json.encodeToString(batch)
        )
    }

    fun acknowledge(batch: PendingUploadBatch) {
        require(pending(batch.projectId) == batch) {
            "Cannot acknowledge a batch that is not the persisted pending batch."
        }
        batch.records.forEach { record ->
            val acknowledgement = UploadAcknowledgement(
                revision = record.revision,
                contentHash = batch.recordHashes.getValue(record.id)
            )
            settings.putString(
                acknowledgementKey(batch.projectId, record.id),
                json.encodeToString(acknowledgement)
            )
        }
        settings.remove(pendingKey(batch.projectId))
    }
}

expect fun sha256Hex(bytes: ByteArray): String

fun observationContentHash(record: Observation): String =
    sha256Hex(Json.encodeToString(record).encodeToByteArray())
