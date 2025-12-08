// JNI bridge between com.hanaml.smalllm.LlamaBridge and llama.cpp.
//
// generate() decodes greedily and returns JSON with the text, the top-5 log-probabilities of
// the first generated token (what the router reads), and the decode speed.

#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

namespace {

struct Session {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    int n_ctx = 0;
};

std::string json_escape(const std::string &s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    o << buf;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

std::vector<float> top_logprobs(const float *logits, int n_vocab, int k) {
    float max_logit = *std::max_element(logits, logits + n_vocab);
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp(double(logits[i] - max_logit));
    const double log_z = max_logit + std::log(sum);
    std::vector<float> lp(logits, logits + n_vocab);
    std::partial_sort(lp.begin(), lp.begin() + k, lp.end(), std::greater<float>());
    std::vector<float> out(k);
    for (int i = 0; i < k; ++i) out[i] = float(lp[i] - log_z);
    return out;
}

std::string chat_prompt(const llama_model *model, const std::string &user) {
    const char *tmpl = llama_model_chat_template(model, nullptr);
    llama_chat_message msg{"user", user.c_str()};
    std::vector<char> buf(user.size() * 2 + 512);
    int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), int(buf.size()));
    if (n > int(buf.size())) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), int(buf.size()));
    }
    return n < 0 ? user : std::string(buf.data(), n);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_hanaml_smalllm_LlamaBridge_load(JNIEnv *env, jobject, jstring jpath, jint n_ctx, jint n_threads) {
    llama_backend_init();
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;  // Vulkan offload when the build has it; ignored otherwise
    llama_model *model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!model) return 0;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 0;
    }
    return reinterpret_cast<jlong>(new Session{model, ctx, n_ctx});
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_hanaml_smalllm_LlamaBridge_generate(JNIEnv *env, jobject, jlong handle, jstring jprompt, jint max_tokens) {
    auto *s = reinterpret_cast<Session *>(handle);
    const char *raw = env->GetStringUTFChars(jprompt, nullptr);
    const std::string prompt = chat_prompt(s->model, raw);
    env->ReleaseStringUTFChars(jprompt, raw);

    const llama_vocab *vocab = llama_model_get_vocab(s->model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int n = -llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), tokens.data(), n, true, true);
    if (n >= s->n_ctx) tokens.erase(tokens.begin(), tokens.begin() + (n - s->n_ctx + max_tokens));

    llama_memory_clear(llama_get_memory(s->ctx), true);
    std::string text;
    std::vector<float> first;
    int generated = 0;
    auto t0 = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(), int(tokens.size()));
    llama_token tok = 0;
    for (; generated < max_tokens; ++generated) {
        if (llama_decode(s->ctx, batch) != 0) break;
        const float *logits = llama_get_logits_ith(s->ctx, -1);
        if (generated == 0) first = top_logprobs(logits, n_vocab, 5);
        tok = llama_token(std::max_element(logits, logits + n_vocab) - logits);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[256];
        int len = llama_token_to_piece(vocab, tok, piece, sizeof piece, 0, true);
        if (len > 0) text.append(piece, len);
        batch = llama_batch_get_one(&tok, 1);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::ostringstream out;
    out << "{\"text\":\"" << json_escape(text) << "\",\"first_logprobs\":[";
    for (size_t i = 0; i < first.size(); ++i) out << (i ? "," : "") << first[i];
    out << "],\"tokens_per_s\":" << (secs > 0 ? generated / secs : 0.0) << "}";
    return env->NewStringUTF(out.str().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_hanaml_smalllm_LlamaBridge_free(JNIEnv *, jobject, jlong handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s) return;
    llama_free(s->ctx);
    llama_model_free(s->model);
    delete s;
}
