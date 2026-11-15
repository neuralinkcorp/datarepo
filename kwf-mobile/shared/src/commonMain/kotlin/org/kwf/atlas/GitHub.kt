package org.kwf.atlas

import com.russhwolf.settings.Settings
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.*

private const val REPOSITORY = "KashmirWorld/datarepo"
private const val API = "https://api.github.com/repos/$REPOSITORY"

private class GitHubFailure(
    val status: Int,
    message: String
) : Exception(message)

/**
 * Pushes a bounded batch of field observations to the GitHub repository.
 *
 * Path: kwf-field-data/captures/{projectId}/{batchId}/
 * Writes canonical source-observations.json, manifest.json, and valid WGS 84 observations.geojson.
 * Pending batch is persisted before network work and reused on retry.
 * Records are acknowledged only after the branch reference is updated successfully.
 */
suspend fun pushSnapshot(
    token: String,
    projectId: String,
    projectName: String,
    projectConfig: ProjectConfig,
    records: List<Observation>,
    batchId: String,
    onProgress: (String) -> Unit
): BatchManifest {
    require(token.isNotBlank()) { "GitHub token is required." }
    require(projectId.isNotBlank()) { "Project ID is required." }
    require(records.isNotEmpty()) { "No unacknowledged observations to upload." }
    require(records.size <= 100) {
        "Batch upload limit: 100 observations per snapshot batch."
    }

    val photoCharacters = records.sumOf { record ->
        record.photosBase64.sumOf { it.length.toLong() }
    }

    require(photoCharacters <= 8_000_000) {
        "Upload limit exceeded: photograph payload is too large."
    }

    // Validate all records before starting network calls
    val sourceObservations = records.map { it.toSourceObservation(projectId) }
    sourceObservations.forEach { SpatialContract.validate(it) }

    val client = HttpClient {
        install(ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true })
        }
    }

    suspend fun request(
        method: HttpMethod,
        endpoint: String,
        payload: JsonObject? = null
    ): JsonObject {
        val response = client.request("$API$endpoint") {
            this.method = method
            bearerAuth(token)

            header(HttpHeaders.Accept, "application/vnd.github+json")
            header("X-GitHub-Api-Version", "2022-11-28")
            header(HttpHeaders.UserAgent, "KWF-FieldAtlas")

            if (payload != null) {
                contentType(ContentType.Application.Json)
                setBody(payload)
            }
        }

        if (!response.status.isSuccess()) {
            val explanation = runCatching {
                Json.parseToJsonElement(response.bodyAsText())
                    .jsonObject["message"]
                    ?.jsonPrimitive
                    ?.content
            }.getOrNull() ?: "GitHub request failed."

            throw GitHubFailure(
                response.status.value,
                "GitHub ${response.status.value}: $explanation"
            )
        }

        return response.body()
    }

    fun JsonObject.string(name: String): String =
        getValue(name).jsonPrimitive.content

    try {
        val repository = request(HttpMethod.Get, "")
        val branch = repository.string("default_branch")

        val folder = "kwf-field-data/captures/$projectId/$batchId"
        val treeEntries = mutableListOf<JsonObject>()

        suspend fun blob(
            relativePath: String,
            content: String,
            encoding: String
        ) {
            val uploaded = request(
                HttpMethod.Post,
                "/git/blobs",
                buildJsonObject {
                    put("content", content)
                    put("encoding", encoding)
                }
            )

            treeEntries += buildJsonObject {
                put("path", "$folder/$relativePath")
                put("mode", "100644")
                put("type", "blob")
                put("sha", uploaded.string("sha"))
            }
        }

        onProgress("Uploading source observations…")

        // 1. Canonical source-observations.json
        val sourceData = buildJsonObject {
            putJsonObject("project") {
                put("id", projectId)
                put("name", projectName)
                put("defaultCrs", projectConfig.defaultCrs)
                put("defaultVcrs", projectConfig.defaultVcrs)
                projectConfig.studyExtent?.let { extent ->
                    putJsonArray("studyExtent") {
                        extent.forEach { add(it) }
                    }
                }
            }
            putJsonArray("observations") {
                sourceObservations.forEach { obs ->
                    add(Json.encodeToJsonElement(obs))
                }
            }
        }
        blob("source-observations.json", sourceData.toString(), "utf-8")

        // 2. Valid WGS-84 GeoJSON only when available
        val wgs84Features = records.mapNotNull { it.toFeatureOrNull("assets") }
        if (wgs84Features.isNotEmpty()) {
            val geojson = buildJsonObject {
                put("type", "FeatureCollection")
                putJsonArray("features") {
                    wgs84Features.forEach { add(it) }
                }
            }
            blob("observations.geojson", geojson.toString(), "utf-8")
        }

        // 3. Types breakdown
        val types = buildJsonArray {
            records
                .map { it.type to it.subtype }
                .distinct()
                .forEach { (t, sub) ->
                    add(buildJsonObject {
                        put("type", t)
                        put("subtype", sub)
                    })
                }
        }
        blob("types.json", types.toString(), "utf-8")

        // 4. Asset photographs
        var photoCount = 0
        val totalPhotos = records.sumOf { it.photosBase64.size }
        val assetReferences = mutableListOf<String>()

        records.forEach { record ->
            record.photosBase64.forEachIndexed { index, data ->
                val assetPath = "assets/${record.id}-$index.jpg"
                assetReferences.add(assetPath)

                onProgress(
                    "Uploading photograph ${++photoCount} of $totalPhotos…"
                )

                blob(
                    assetPath,
                    data,
                    "base64"
                )
            }
        }

        // 5. Batch Manifest
        val manifest = buildJsonObject {
            put("schemaVersion", 2)
            put("projectId", projectId)
            put("batchId", batchId)
            put("recordCount", records.size)
            put("photoCount", totalPhotos)
            put("createdAtEpochMs", nowMillis())
            putJsonArray("observationIds") {
                records.forEach { add(it.id) }
            }
            putJsonObject("observationRevisions") {
                records.forEach { put(it.id, it.revision) }
            }
            putJsonArray("assetReferences") {
                assetReferences.forEach { add(it) }
            }
        }
        blob("manifest.json", manifest.toString(), "utf-8")

        // 6. Commit and update branch reference (no force push)
        repeat(3) { attempt ->
            onProgress("Creating snapshot commit…")

            val encodedBranch = branch
                .split("/")
                .joinToString("/") { it.encodeURLPathPart() }

            val reference = request(
                HttpMethod.Get,
                "/git/ref/heads/$encodedBranch"
            )

            val headSha = reference
                .getValue("object")
                .jsonObject
                .string("sha")

            val previousCommit = request(
                HttpMethod.Get,
                "/git/commits/$headSha"
            )

            val previousTree = previousCommit
                .getValue("tree")
                .jsonObject
                .string("sha")

            val newTree = request(
                HttpMethod.Post,
                "/git/trees",
                buildJsonObject {
                    put("base_tree", previousTree)
                    put("tree", JsonArray(treeEntries))
                }
            )

            val commit = request(
                HttpMethod.Post,
                "/git/commits",
                buildJsonObject {
                    put("message", "Field snapshot $batchId for project $projectId")
                    put("tree", newTree.string("sha"))
                    putJsonArray("parents") { add(headSha) }
                }
            )

            try {
                request(
                    HttpMethod.Patch,
                    "/git/refs/heads/$encodedBranch",
                    buildJsonObject {
                        put("sha", commit.string("sha"))
                        put("force", false)
                    }
                )

                // Return completed manifest on successful ref update
                return BatchManifest(
                    projectId = projectId,
                    batchId = batchId,
                    observationIds = records.map { it.id },
                    assetReferences = assetReferences,
                    acknowledgedRevision = commit.string("sha"),
                    acknowledgedHash = newTree.string("sha")
                )
            } catch (error: GitHubFailure) {
                if (
                    error.status !in listOf(409, 422) ||
                    attempt == 2
                ) {
                    throw error
                }

                onProgress("Repository changed; retrying commit…")
            }
        }

        error("Branch update failed after retries.")
    } finally {
        client.close()
    }
}
