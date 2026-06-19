package com.hanaml.smalllm

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p

/**
 * The on-device half of small_lm.router: the same features, weights, temperature and coverage
 * thresholds, read from the router.json that `small-lm calibrate` writes.
 */
class Router(
    private val weights: DoubleArray,
    private val temperature: Double,
    private val mean: DoubleArray,
    private val std: DoubleArray,
    private val thresholds: Map<Double, Double>,
) {
    enum class Route { LOCAL, ESCALATE, ABSTAIN }

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun fromJson(text: String): Router {
            val m = Json.parse(text) as Map<String, Any?>
            fun arr(k: String) = (m[k] as List<Double>).toDoubleArray()
            val th = (m["thresholds"] as Map<String, Double>).mapKeys { it.key.toDouble() }
            return Router(arr("weights"), m["temperature"] as Double, arr("mean"), arr("std"), th)
        }

        /** Top first-token probability, entropy of the top-k distribution, log question length. */
        fun features(question: String, firstTokenLogprobs: DoubleArray): DoubleArray {
            val probs = firstTokenLogprobs.map { exp(it) }
            val total = probs.sum().takeIf { it > 0 } ?: 1.0
            val entropy = -probs.map { it / total }.filter { it > 0 }.sumOf { it * ln(it) }
            val words = question.trim().split(Regex("\\s+")).count { it.isNotEmpty() }
            return doubleArrayOf(probs.max(), entropy, ln1p(words.toDouble()))
        }
    }

    fun confidence(features: DoubleArray): Double {
        var z = weights.last()
        for (j in features.indices) z += weights[j] * (features[j] - mean[j]) / std[j]
        return 1.0 / (1.0 + exp(-z / temperature))
    }

    fun threshold(coverage: Double): Double {
        thresholds[coverage]?.let { return it }
        val keys = thresholds.keys.sorted()
        val hi = keys.firstOrNull { it >= coverage } ?: return thresholds.getValue(keys.last())
        val lo = keys.lastOrNull { it <= coverage } ?: return thresholds.getValue(keys.first())
        val t = (coverage - lo) / (hi - lo)
        return thresholds.getValue(lo) + t * (thresholds.getValue(hi) - thresholds.getValue(lo))
    }

    fun route(question: String, firstTokenLogprobs: DoubleArray, coverage: Double, escalate: Boolean): Route =
        when {
            confidence(features(question, firstTokenLogprobs)) >= threshold(coverage) -> Route.LOCAL
            escalate -> Route.ESCALATE
            else -> Route.ABSTAIN
        }
}
