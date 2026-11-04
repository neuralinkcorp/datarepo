plugins {
    id("com.android.application") version "8.7.3" apply false
    id("com.android.library") version "8.7.3" apply false

    kotlin("multiplatform") version "2.1.20" apply false
    kotlin("android") version "2.1.20" apply false
    kotlin("plugin.serialization") version "2.1.20" apply false
    kotlin("plugin.compose") version "2.1.20" apply false

    id("org.jetbrains.compose") version "1.8.0" apply false
}
