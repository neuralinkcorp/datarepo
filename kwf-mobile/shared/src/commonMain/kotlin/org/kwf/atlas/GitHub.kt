package org.kwf.atlas

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

suspend fun pushSnapshot(
    token: String,
    project: String,
    records: List<Observation>,
    onProgress: (String) -> Unit
) {
    require(token.isNotBlank()) { "GitHub token is required." }
    require(records.isNotEmpty()) { "No observations in this project." }
    require(records.size <= 100) {
        "Pilot upload limit: 100 observations per project snapshot."
    }

    val photoCharacters = records.sumOf { record ->
        record.photosBase64.sumOf { it.length.toLong() }
    }

    require(photoCharacters <= 8_000_000) {
        "Pilot upload limit exceeded: photograph payload is too large."
    }

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

        val projectFolder = project
            .lowercase()
            .replace(Regex("[^a-z0-9_-]+"), "-")
            .trim('-')
            .take(80)
            .ifBlank { "project" }

        val snapshotId = uniqueId()
        val folder =
            "kwf-field-data/captures/$projectFolder/$snapshotId"

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

        onProgress("Uploading observation data…")

        val geojson = buildJsonObject {
            put("type", "FeatureCollection")
            putJsonArray("features") {
                records.forEach {
                    add(it.toFeature("assets"))
                }
            }
        }

        blob("observations.geojson", geojson.toString(), "utf-8")

        val types = buildJsonArray {
            records
                .map { it.type to it.subtype }
                .distinct()
                .forEach { (type, subtype) ->
                    add(buildJsonObject {
                        put("type", type)
                        put("subtype", subtype)
                    })
                }
        }

        blob("types.json", types.toString(), "utf-8")

        var photoCount = 0
        val totalPhotos = records.sumOf { it.photosBase64.size }

        records.forEach { record ->
            record.photosBase64.forEachIndexed { index, data ->
                onProgress(
                    "Uploading photograph ${++photoCount} of $totalPhotos…"
                )

                blob(
                    "assets/${record.id}-$index.jpg",
                    data,
                    "base64"
                )
            }
        }

        val manifest = buildJsonObject {
            put("formatVersion", 1)
            put("snapshotId", snapshotId)
            put("project", project)
            put("recordCount", records.size)
            put("photoCount", totalPhotos)
            put("createdAtEpochMs", nowMillis())
            put("coordinateSystem", "EPSG:4326")
            put("coordinateOrder", "longitude,latitude")
            put("gcpExport", false)
        }

        blob("manifest.json", manifest.toString(), "utf-8")

        // Retry only branch-update conflicts. Existing repository files
        // remain in the tree through base_tree.
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
                    put("message", "Field snapshot $snapshotId")
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

                return
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
    } finally {
        client.close()
    }
}
