plugins {
    id("com.android.application") version "9.4.0" apply false
    // AGP 9 compiles Kotlin itself (android.builtInKotlin); this pins the Kotlin compiler it uses.
    id("org.jetbrains.kotlin.jvm") version "2.4.20" apply false
}
