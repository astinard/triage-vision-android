# Triage Vision Android - ProGuard Rules
# HIPAA-compliant medical app - security-focused configuration

# Keep all native method names
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep NativeBridge for JNI
-keep class com.triage.vision.native.NativeBridge { *; }

# Keep data classes for JSON serialization
-keep class com.triage.vision.data.** { *; }
-keep class com.triage.vision.pipeline.** { *; }
-keep class com.triage.vision.charting.** { *; }

# Keep Room database entities
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-keep @androidx.room.Dao class *

# Kotlinx Serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}
-keep,includedescriptorclasses class com.triage.vision.**$$serializer { *; }
-keepclassmembers class com.triage.vision.** {
    *** Companion;
}
-keepclasseswithmembers class com.triage.vision.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# CameraX
-keep class androidx.camera.** { *; }

# Security: Remove logging in release
-assumenosideeffects class android.util.Log {
    public static int v(...);
    public static int d(...);
    public static int i(...);
}

# Security: Obfuscate aggressively
-repackageclasses ''
-allowaccessmodification
-optimizationpasses 5

# Keep Compose
-dontwarn androidx.compose.**

# Remove debug info
-renamesourcefileattribute SourceFile
-keepattributes SourceFile,LineNumberTable
