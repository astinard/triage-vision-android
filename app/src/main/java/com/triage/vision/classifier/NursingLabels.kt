package com.triage.vision.classifier

/**
 * Nursing-specific classification labels for patient monitoring.
 *
 * These labels are designed for CLIP zero-shot classification.
 * Each category has descriptive phrases that CLIP can match against.
 */
object NursingLabels {

    /**
     * Patient position categories
     */
    enum class Position(val label: String, val clipPrompt: String) {
        LYING_SUPINE("lying_supine", "a patient lying flat on their back in a hospital bed"),
        LYING_LEFT("lying_left_lateral", "a patient lying on their left side in a hospital bed"),
        LYING_RIGHT("lying_right_lateral", "a patient lying on their right side in a hospital bed"),
        LYING_PRONE("lying_prone", "a patient lying face down on their stomach in a hospital bed"),
        SITTING_BED("sitting_in_bed", "a patient sitting up in a hospital bed"),
        SITTING_CHAIR("sitting_in_chair", "a patient sitting in a chair or wheelchair"),
        STANDING("standing", "a patient standing upright"),
        ON_FLOOR("on_floor", "a person lying on the floor, possibly fallen");

        companion object {
            val allPrompts: List<String> = entries.map { it.clipPrompt }
            fun fromIndex(index: Int): Position = entries[index]
        }
    }

    /**
     * Patient alertness/consciousness level
     */
    enum class Alertness(val label: String, val clipPrompt: String) {
        AWAKE_ALERT("awake_alert", "a patient who is awake and alert, eyes open, looking around"),
        AWAKE_DROWSY("awake_drowsy", "a patient who appears drowsy or sleepy, eyes half-closed"),
        SLEEPING("sleeping", "a patient who is sleeping peacefully with eyes closed"),
        EYES_CLOSED("eyes_closed", "a patient with eyes closed, resting"),
        UNRESPONSIVE("unresponsive", "an unresponsive patient, not reacting to surroundings");

        companion object {
            val allPrompts: List<String> = entries.map { it.clipPrompt }
            fun fromIndex(index: Int): Alertness = entries[index]
        }
    }

    /**
     * Patient activity/movement level
     */
    enum class Activity(val label: String, val clipPrompt: String) {
        STILL("still", "a patient lying completely still with no movement"),
        MINIMAL("minimal_movement", "a patient with slight movement, small gestures"),
        MODERATE("moderate_movement", "a patient moving moderately, shifting position"),
        ACTIVE("active_movement", "a patient moving actively, gesturing or repositioning"),
        RESTLESS("restless", "a patient who appears restless or agitated");

        companion object {
            val allPrompts: List<String> = entries.map { it.clipPrompt }
            fun fromIndex(index: Int): Activity = entries[index]
        }
    }

    /**
     * Comfort/distress assessment
     */
    enum class Comfort(val label: String, val clipPrompt: String) {
        COMFORTABLE("comfortable", "a patient who appears comfortable and relaxed"),
        MILD_DISCOMFORT("mild_discomfort", "a patient showing signs of mild discomfort"),
        MODERATE_DISCOMFORT("moderate_discomfort", "a patient showing moderate discomfort or pain"),
        DISTRESSED("distressed", "a patient who appears distressed or in significant pain"),
        PAIN_INDICATED("pain_indicated", "a patient with facial expressions indicating pain");

        companion object {
            val allPrompts: List<String> = entries.map { it.clipPrompt }
            fun fromIndex(index: Int): Comfort = entries[index]
        }
    }

    /**
     * Safety concern categories
     */
    enum class SafetyConcern(val label: String, val clipPrompt: String) {
        NONE("none", "a patient in a safe, normal hospital room setting"),
        FALL_RISK("fall_risk", "a patient at risk of falling, near edge of bed"),
        FALLEN("fallen", "a patient who has fallen on the floor"),
        LEAVING_BED("leaving_bed", "a patient attempting to get out of bed"),
        EQUIPMENT_ISSUE("equipment_issue", "medical equipment that appears disconnected or problematic");

        companion object {
            val allPrompts: List<String> = entries.map { it.clipPrompt }
            fun fromIndex(index: Int): SafetyConcern = entries[index]
        }
    }

    /**
     * All classification prompts for pre-computing embeddings
     */
    fun getAllPrompts(): Map<String, List<String>> = mapOf(
        "position" to Position.allPrompts,
        "alertness" to Alertness.allPrompts,
        "activity" to Activity.allPrompts,
        "comfort" to Comfort.allPrompts,
        "safety" to SafetyConcern.allPrompts
    )

    /**
     * Get total number of labels across all categories
     */
    fun getTotalLabelCount(): Int =
        Position.entries.size +
        Alertness.entries.size +
        Activity.entries.size +
        Comfort.entries.size +
        SafetyConcern.entries.size
}
