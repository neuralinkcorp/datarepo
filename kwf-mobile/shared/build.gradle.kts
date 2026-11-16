plugins {
    kotlin("multiplatform")
    kotlin("plugin.serialization")
    kotlin("plugin.compose")
    id("org.jetbrains.compose")
    id("com.android.library")
}

kotlin {
    androidTarget {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    iosX64()

    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget>()
        .configureEach {
            binaries.framework {
                baseName = "FieldAtlas"
                isStatic = true
            }
        }

    sourceSets {
        commonMain.dependencies {
            implementation(compose.runtime)
            implementation(compose.foundation)
            implementation(compose.material3)
            implementation(compose.ui)

            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
            implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")

            implementation("io.ktor:ktor-client-core:3.1.2")
            implementation("io.ktor:ktor-client-content-negotiation:3.1.2")
            implementation("io.ktor:ktor-serialization-kotlinx-json:3.1.2")

            implementation(
                "com.russhwolf:multiplatform-settings-no-arg:1.3.0"
            )
        }

        androidMain.dependencies {
            implementation("androidx.activity:activity-compose:1.10.1")
            implementation("androidx.core:core-ktx:1.15.0")
            implementation("io.ktor:ktor-client-okhttp:3.1.2")
        }

        androidUnitTest.dependencies {
            implementation(kotlin("test"))
        }

        iosMain.dependencies {
            implementation("io.ktor:ktor-client-darwin:3.1.2")
        }
    }
}

android {
    namespace = "org.kwf.atlas.shared"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

