plugins {
    id("com.android.application")
}

val buildNative = providers.gradleProperty("smalllm.native").map(String::toBoolean).orElse(false).get()

android {
    namespace = "com.hanaml.smalllm"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.hanaml.smalllm"
        minSdk = 29
        targetSdk = 36
        versionCode = 600
        versionName = "0.6.0"
        if (buildNative) {
            ndk { abiFilters += listOf("arm64-v8a") }
            externalNativeBuild {
                cmake { arguments += listOf("-DGGML_VULKAN=ON", "-DANDROID_STL=c++_shared") }
            }
        }
    }

    if (buildNative) {
        externalNativeBuild {
            cmake { path = file("../../native/CMakeLists.txt") }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}
