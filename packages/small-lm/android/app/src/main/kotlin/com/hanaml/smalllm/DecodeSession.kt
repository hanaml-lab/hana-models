package com.hanaml.smalllm

import android.content.Context
import android.os.PowerManager

/**
 * Runs a question through retrieval, the local model and the router. The wake lock is held only
 * while the model decodes: retrieval is short, and holding the lock through it cost battery for
 * nothing.
 */
class DecodeSession(context: Context, private val handle: Long, private val router: Router) {
    private val wakeLock = (context.getSystemService(Context.POWER_SERVICE) as PowerManager)
        .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "small-lm:decode")

    data class Result(val text: String, val route: Router.Route)

    fun ask(question: String, notes: List<String>, coverage: Double = 0.8, escalate: Boolean = true): Result {
        val prompt = if (notes.isEmpty()) question
        else "Notes:\n" + notes.joinToString("\n") { "- $it" } + "\n\nUsing only the notes above, answer: $question"
        val reply = decode(prompt)
        @Suppress("UNCHECKED_CAST")
        val m = Json.parse(reply) as Map<String, Any?>
        val logprobs = (m["first_logprobs"] as List<Double>).toDoubleArray()
        val route = router.route(question, logprobs, coverage, escalate)
        return Result(m["text"] as String, route)
    }

    private fun decode(prompt: String): String {
        wakeLock.acquire(60_000L)
        try {
            return LlamaBridge.generate(handle, prompt, 256)
        } finally {
            if (wakeLock.isHeld) wakeLock.release()
        }
    }
}
