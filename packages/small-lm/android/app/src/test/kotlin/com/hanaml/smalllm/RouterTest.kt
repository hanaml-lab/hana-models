package com.hanaml.smalllm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.ln

class RouterTest {
    // Written by `small-lm calibrate`; the confidences below were computed with small_lm.calibrate.
    private val json = """
        {"weights": [1.8, -0.6, -0.3, 0.4], "temperature": 1.25, "mean": [0.6, 0.8, 1.2], "std": [0.2, 0.4, 0.3],
         "thresholds": {"0.50": 0.71, "0.80": 0.52, "1.00": 0.0}}
    """.trimIndent()

    @Test
    fun confidenceMatchesThePythonRouter() {
        val r = Router.fromJson(json)
        assertEquals(0.6637, r.confidence(doubleArrayOf(0.7, 0.9, 1.5)), 1e-4)
    }

    @Test
    fun thresholdsInterpolateBetweenCoverages() {
        val r = Router.fromJson(json)
        assertEquals(0.52, r.threshold(0.8), 1e-9)
        assertEquals(0.615, r.threshold(0.65), 1e-9)
    }

    @Test
    fun confidentAnswersStayLocal() {
        val r = Router.fromJson(json)
        val sure = doubleArrayOf(ln(0.95), ln(0.02), ln(0.01), ln(0.01), ln(0.01))
        val unsure = doubleArrayOf(ln(0.25), ln(0.2), ln(0.2), ln(0.2), ln(0.15))
        assertEquals(Router.Route.LOCAL, r.route("When is the insurance due?", sure, 0.8, true))
        assertEquals(Router.Route.ABSTAIN, r.route("When is the insurance due?", unsure, 0.8, false))
        assertTrue(Router.features("a b", sure)[1] > 0)
    }
}
