@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package org.kwf.atlas

import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.usePinned
import platform.CommonCrypto.CC_SHA256
import platform.CommonCrypto.CC_SHA256_DIGEST_LENGTH

actual fun sha256Hex(bytes: ByteArray): String {
    val digest = ByteArray(CC_SHA256_DIGEST_LENGTH.toInt())
    bytes.usePinned { input ->
        digest.usePinned { output ->
            CC_SHA256(
                input.addressOf(0),
                bytes.size.convert(),
                output.addressOf(0)
            )
        }
    }
    return digest.joinToString("") { byte ->
        (byte.toInt() and 0xff).toString(16).padStart(2, '0')
    }
}
