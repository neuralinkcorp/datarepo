package org.kwf.atlas

import com.russhwolf.settings.Settings
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

private class MemorySettings : Settings {
    private val values = mutableMapOf<String, Any>()

    override val keys: Set<String> get() = values.keys
    override val size: Int get() = values.size
    override fun clear() = values.clear()
    override fun remove(key: String) { values.remove(key) }
    override fun hasKey(key: String): Boolean = values.containsKey(key)
    override fun putInt(key: String, value: Int) { values[key] = value }
    override fun getInt(key: String, defaultValue: Int): Int =
        values[key] as? Int ?: defaultValue
    override fun getIntOrNull(key: String): Int? = values[key] as? Int
    override fun putLong(key: String, value: Long) { values[key] = value }
    override fun getLong(key: String, defaultValue: Long): Long =
        values[key] as? Long ?: defaultValue
    override fun getLongOrNull(key: String): Long? = values[key] as? Long
    override fun putString(key: String, value: String) { values[key] = value }
    override fun getString(key: String, defaultValue: String): String =
        values[key] as? String ?: defaultValue
    override fun getStringOrNull(key: String): String? = values[key] as? String
    override fun putFloat(key: String, value: Float) { values[key] = value }
    override fun getFloat(key: String, defaultValue: Float): Float =
        values[key] as? Float ?: defaultValue
    override fun getFloatOrNull(key: String): Float? = values[key] as? Float
    override fun putDouble(key: String, value: Double) { values[key] = value }
    override fun getDouble(key: String, defaultValue: Double): Double =
        values[key] as? Double ?: defaultValue
    override fun getDoubleOrNull(key: String): Double? = values[key] as? Double
    override fun putBoolean(key: String, value: Boolean) { values[key] = value }
    override fun getBoolean(key: String, defaultValue: Boolean): Boolean =
        values[key] as? Boolean ?: defaultValue
    override fun getBooleanOrNull(key: String): Boolean? =
        values[key] as? Boolean
}

class UploadQueueTest {
    private fun observation(notes: String = "field note", photo: String = "jpeg") =
        Observation(
            id = "record-1",
            projectId = "project-1",
            project = "Test project",
            type = "Observation",
            subtype = "Vegetation",
            shape = "Point",
            vertices = listOf(
                Vertex(
                    longitude = -94.1,
                    latitude = 35.1,
                    source = "manual"
                )
            ),
            capturedAtEpochMs = 1_700_000_000_000,
            notes = notes,
            photosBase64 = listOf(photo)
        )

    private fun batch(record: Observation): PendingUploadBatch {
        val config = ProjectConfig("project-1", "Test project")
        return PendingUploadBatch(
            batchId = "batch-1",
            projectId = config.id,
            projectName = config.name,
            projectConfig = config,
            records = listOf(record),
            recordHashes = mapOf(record.id to observationContentHash(record)),
            createdAtEpochMs = 1_700_000_000_001
        )
    }

    @Test
    fun persistedBatchIsReusedUnchangedWhenRecordsChange() {
        val settings = MemorySettings()
        val queue = UploadQueue(settings)
        val pending = batch(observation())
        queue.persist(pending)

        val newlyCaptured = observation(notes = "new record").copy(id = "record-2")
        val changedBatch = pending.copy(
            records = pending.records + newlyCaptured,
            recordHashes = pending.recordHashes +
                (newlyCaptured.id to observationContentHash(newlyCaptured))
        )

        assertFailsWith<IllegalArgumentException> { queue.persist(changedBatch) }
        assertEquals(pending, UploadQueue(settings).pending("project-1"))
    }

    @Test
    fun acknowledgementSkipsOnlyUnchangedRecordContent() {
        val settings = MemorySettings()
        val queue = UploadQueue(settings)
        val record = observation()
        val pending = batch(record)
        queue.persist(pending)
        queue.acknowledge(pending)

        assertFalse(
            queue.needsUpload(
                "project-1",
                record,
                observationContentHash(record)
            )
        )
        assertTrue(
            queue.needsUpload(
                "project-1",
                record.copy(notes = "edited"),
                observationContentHash(record.copy(notes = "edited"))
            )
        )
        assertEquals(null, queue.pending("project-1"))
    }

    @Test
    fun observationHashIncludesPhotoContents() {
        val withoutPhoto = observation(photo = "photo-one")
        val changedPhoto = observation(photo = "photo-two")
        assertNotEquals(
            observationContentHash(withoutPhoto),
            observationContentHash(changedPhoto)
        )
    }
}
