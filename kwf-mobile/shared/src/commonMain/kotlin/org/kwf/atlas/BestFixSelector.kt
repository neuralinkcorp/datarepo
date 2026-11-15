package org.kwf.atlas

/**
 * Platform adapters supply fixes until:
 * - offer() returns an acceptable fix; or
 * - the adapter timeout expires, then calls bestAtTimeout().
 *
 * This class does not infer altitude reference or provider identity.
 */
class BestFixSelector(
    private val startedAtEpochMs: Long,
    private val desiredAccuracyM: Double = 10.0,
    private val maximumFixAgeMs: Long = 15_000
) {
    private var best: SourceVertex? = null

    fun offer(
        fix: SourceVertex,
        nowEpochMs: Long
    ): SourceVertex? {
        val accuracy = fix.horizontalAccuracyM ?: return null
        val capturedAt = fix.capturedAtEpochMs ?: return null

        if (
            !fix.x.isFinite() ||
            !fix.y.isFinite() ||
            fix.crs != "EPSG:4326" ||
            fix.x !in -180.0..180.0 ||
            fix.y !in -90.0..90.0 ||
            !accuracy.isFinite() ||
            accuracy < 0
        ) return null

        if (
            capturedAt < startedAtEpochMs - maximumFixAgeMs ||
            capturedAt > nowEpochMs + 5_000 ||
            nowEpochMs - capturedAt > maximumFixAgeMs
        ) return null

        if (
            best == null ||
            accuracy < best!!.horizontalAccuracyM!!
        ) best = fix

        return best?.takeIf {
            it.horizontalAccuracyM!! <= desiredAccuracyM
        }
    }

    /**
     * A poorer fix is returned for explicit user acceptance.
     * null means no usable fix was received.
     */
    fun bestAtTimeout(): SourceVertex? = best
}
