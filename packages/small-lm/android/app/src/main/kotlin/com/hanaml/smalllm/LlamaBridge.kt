package com.hanaml.smalllm

/** JNI bindings to native/llama_bridge.cpp (llama.cpp with Vulkan). */
object LlamaBridge {
    init {
        System.loadLibrary("llama_bridge")
    }

    /** Loads a GGUF model; returns a handle to pass to [generate] and [free]. */
    external fun load(path: String, nCtx: Int, nThreads: Int): Long

    /** Greedy generation. Returns JSON: {"text": "...", "first_logprobs": [..top 5..], "tokens_per_s": x}. */
    external fun generate(handle: Long, prompt: String, maxTokens: Int): String

    external fun free(handle: Long)
}
