package com.hanaml.smalllm

/** A small JSON reader for router.json and the bridge's replies; keeps the app free of dependencies. */
internal class Json private constructor(private val s: String) {
    private var i = 0

    companion object {
        fun parse(text: String): Any? = Json(text).run { val v = value(); ws(); require(i == s.length) { "trailing data at $i" }; v }
    }

    private fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }

    private fun value(): Any? {
        ws()
        require(i < s.length) { "unexpected end of JSON" }
        return when (val c = s[i]) {
            '{' -> obj()
            '[' -> arr()
            '"' -> str()
            't' -> lit("true", true)
            'f' -> lit("false", false)
            'n' -> lit("null", null)
            else -> if (c == '-' || c.isDigit()) num() else error("unexpected '$c' at $i")
        }
    }

    private fun lit(word: String, v: Any?): Any? {
        require(s.startsWith(word, i)) { "bad literal at $i" }
        i += word.length
        return v
    }

    private fun num(): Double {
        val start = i
        while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
        return s.substring(start, i).toDouble()
    }

    private fun str(): String {
        val out = StringBuilder()
        i++
        while (s[i] != '"') {
            if (s[i] == '\\') {
                i++
                when (val e = s[i]) {
                    'n' -> out.append('\n'); 't' -> out.append('\t'); 'r' -> out.append('\r')
                    'b' -> out.append('\b'); 'f' -> out.append('\u000c')
                    'u' -> { out.append(s.substring(i + 1, i + 5).toInt(16).toChar()); i += 4 }
                    else -> out.append(e)
                }
            } else out.append(s[i])
            i++
        }
        i++
        return out.toString()
    }

    private fun arr(): List<Any?> {
        val out = mutableListOf<Any?>()
        i++; ws()
        if (s[i] == ']') { i++; return out }
        while (true) {
            out += value(); ws()
            if (s[i++] == ']') return out
        }
    }

    private fun obj(): Map<String, Any?> {
        val out = linkedMapOf<String, Any?>()
        i++; ws()
        if (s[i] == '}') { i++; return out }
        while (true) {
            ws(); val k = str(); ws()
            require(s[i++] == ':') { "expected ':' at $i" }
            out[k] = value(); ws()
            if (s[i++] == '}') return out
        }
    }
}
