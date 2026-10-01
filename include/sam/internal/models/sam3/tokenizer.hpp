#ifndef SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TOKENIZER_HPP
#define SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TOKENIZER_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sam::internal::sam3 {

struct TokenizerData {
    std::unordered_map<std::string, std::int32_t> vocabulary;
    std::vector<std::pair<std::string, std::string>> merges;
};

inline bool ascii_letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline bool ascii_digit(char c) { return c >= '0' && c <= '9'; }
inline bool ascii_space(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

inline std::string clean_prompt(const std::string& text) {
    std::string result;
    bool pending_space = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if (byte >= 128 || (byte < 32 && !ascii_space(text[i])) || byte == 127) {
            throw std::invalid_argument("plain-text prompts support printable ASCII and ASCII whitespace; use token IDs for Unicode");
        }
        if (text[i] == '&') {
            std::size_t end = i + 1;
            while (end < text.size() && (ascii_letter(text[end]) || ascii_digit(text[end]) ||
                                        text[end] == '#')) {
                ++end;
            }
            bool entity = end > i + 1 && end < text.size() && text[end] == ';';
            entity = entity || (i + 2 < text.size() && text[i + 1] == '#' &&
                                 (ascii_digit(text[i + 2]) || text[i + 2] == 'x' || text[i + 2] == 'X'));
            // HTML permits these legacy names without a semicolon; case matters.
            for (const char* name : {
                "AElig", "AMP", "Aacute", "Acirc", "Agrave", "Aring", "Atilde", "Auml",
                "COPY", "Ccedil", "ETH", "Eacute", "Ecirc", "Egrave", "Euml", "GT",
                "Iacute", "Icirc", "Igrave", "Iuml", "LT", "Ntilde", "Oacute", "Ocirc",
                "Ograve", "Oslash", "Otilde", "Ouml", "QUOT", "REG", "THORN", "Uacute",
                "Ucirc", "Ugrave", "Uuml", "Yacute", "aacute", "acirc", "acute", "aelig",
                "agrave", "amp", "aring", "atilde", "auml", "brvbar", "ccedil", "cedil",
                "cent", "copy", "curren", "deg", "divide", "eacute", "ecirc", "egrave",
                "eth", "euml", "frac12", "frac14", "frac34", "gt", "iacute", "icirc",
                "iexcl", "igrave", "iquest", "iuml", "laquo", "lt", "macr", "micro",
                "middot", "nbsp", "not", "ntilde", "oacute", "ocirc", "ograve", "ordf",
                "ordm", "oslash", "otilde", "ouml", "para", "plusmn", "pound", "quot",
                "raquo", "reg", "sect", "shy", "sup1", "sup2", "sup3", "szlig",
                "thorn", "times", "uacute", "ucirc", "ugrave", "uml", "uuml", "yacute",
                "yen", "yuml"
            }) {
                entity = entity || text.compare(i + 1, std::char_traits<char>::length(name), name) == 0;
            }
            if (entity) {
                throw std::invalid_argument("HTML entity-encoded prompts are unsupported; supply cleaned plain text or token IDs");
            }
        }
        // ftfy removes vertical-tab; form-feed remains for whitespace cleaning.
        if (text[i] == '\v') { continue; }
        if (ascii_space(text[i])) {
            pending_space = !result.empty();
            continue;
        }
        if (pending_space) {
            result.push_back(' ');
            pending_space = false;
        }
        result.push_back(text[i] >= 'A' && text[i] <= 'Z' ? static_cast<char>(text[i] + ('a' - 'A')) : text[i]);
    }
    return result;
}

class Tokenizer {
public:
    static constexpr std::int32_t sot_token = 49406;
    static constexpr std::int32_t eot_token = 49407;
    static constexpr std::int32_t vocabulary_size = 49408;
    static constexpr std::size_t context_length = 32;

    explicit Tokenizer(const TokenizerData& data) : data_(&data) {
        const auto sot = data.vocabulary.find("<start_of_text>");
        const auto eot = data.vocabulary.find("<end_of_text>");
        if (sot == data.vocabulary.end() || sot->second != sot_token ||
            eot == data.vocabulary.end() || eot->second != eot_token) {
            throw std::runtime_error("SAM 3 tokenizer special tokens do not match the official vocabulary");
        }
        for (std::size_t rank = 0; rank < data.merges.size(); ++rank) {
            const auto& pair = data.merges[rank];
            if (pair.first.empty() || pair.second.empty() ||
                pair.first.find('\0') != std::string::npos || pair.second.find('\0') != std::string::npos ||
                !ranks_.emplace(pair.first + '\0' + pair.second, rank).second) {
                throw std::runtime_error("duplicate tokenizer merge");
            }
        }
    }

    inline void validate_tokens(const std::vector<std::int32_t>& ids) const {
        if (ids.size() != context_length || ids.front() != sot_token) {
            throw std::invalid_argument("token IDs must contain 32 entries beginning with the official start token");
        }
        for (auto id : ids) {
            if (id < 0 || id >= vocabulary_size) {
                throw std::invalid_argument("token ID is outside the SAM 3 vocabulary");
            }
        }
        auto last = ids.size();
        while (last > 0 && ids[last - 1] == 0) {
            --last;
        }
        if (last < 2 || ids[last - 1] != eot_token) {
            throw std::invalid_argument("token IDs must end with the official end token followed by zero padding");
        }
    }

    inline std::vector<std::int32_t> encode(const std::string& text) {
        const auto cleaned = clean_prompt(text);
        if (has_last_ && cleaned == last_prompt_) {
            return last_tokens_;
        }
        std::vector<std::int32_t> ids{sot_token};
        for (std::size_t position = 0; position < cleaned.size();) {
            if (cleaned[position] == ' ') {
                ++position;
                continue;
            }
            std::size_t end = position;
            std::string special;
            for (const char* token : {"<start_of_text>", "<end_of_text>"}) {
                if (cleaned.compare(position, std::char_traits<char>::length(token), token) == 0) {
                    special = token;
                    end = position + special.size();
                    break;
                }
            }
            if (!special.empty()) {
                ids.push_back(data_->vocabulary.at(special));
            } else {
                if (cleaned[position] == '\'') {
                    for (const char* contraction : {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}) {
                        const auto length = std::char_traits<char>::length(contraction);
                        if (cleaned.compare(position, length, contraction) == 0) {
                            end = position + length;
                            break;
                        }
                    }
                }
                if (end == position) {
                    end = position + 1;
                    if (ascii_letter(cleaned[position])) {
                        while (end < cleaned.size() && ascii_letter(cleaned[end])) { ++end; }
                    } else if (!ascii_digit(cleaned[position])) {
                        while (end < cleaned.size() && cleaned[end] != ' ' &&
                               !ascii_letter(cleaned[end]) && !ascii_digit(cleaned[end])) { ++end; }
                    }
                }
                for (const auto& symbol : bpe(cleaned.substr(position, end - position))) {
                    const auto token = data_->vocabulary.find(symbol);
                    if (token == data_->vocabulary.end()) {
                        throw std::runtime_error("tokenizer vocabulary is missing BPE token: " + symbol);
                    }
                    ids.push_back(token->second);
                }
            }
            position = end;
            if (ids.size() >= context_length) {
                break;
            }
        }
        ids.push_back(eot_token);
        if (ids.size() > context_length) {
            ids.resize(context_length);
            ids.back() = eot_token;
        }
        ids.resize(context_length, 0);
        validate_tokens(ids);
        last_prompt_ = cleaned;
        last_tokens_ = ids;
        has_last_ = true;
        return ids;
    }

private:
    inline std::vector<std::string> bpe(const std::string& token) const {
        // All accepted prompt bytes are printable ASCII and map to themselves
        // in the official CLIP byte encoder. Whitespace never enters BPE.
        std::vector<std::string> word;
        for (char character : token) {
            word.emplace_back(1, character);
        }
        word.back() += "</w>";
        // ponytail: quadratic scans match the reference BPE; use a ranked-pair
        // queue if unusually long individual words become a measured bottleneck.
        while (word.size() > 1) {
            std::size_t best = std::numeric_limits<std::size_t>::max();
            std::pair<std::string, std::string> selected;
            for (std::size_t i = 0; i + 1 < word.size(); ++i) {
                const auto rank = ranks_.find(word[i] + '\0' + word[i + 1]);
                if (rank != ranks_.end() && rank->second < best) {
                    best = rank->second;
                    selected = {word[i], word[i + 1]};
                }
            }
            if (best == std::numeric_limits<std::size_t>::max()) {
                break;
            }
            std::vector<std::string> merged;
            for (std::size_t i = 0; i < word.size(); ++i) {
                if (i + 1 < word.size() && word[i] == selected.first && word[i + 1] == selected.second) {
                    merged.push_back(word[i] + word[i + 1]);
                    ++i;
                } else {
                    merged.push_back(word[i]);
                }
            }
            word = std::move(merged);
        }
        return word;
    }

    const TokenizerData* data_;
    std::unordered_map<std::string, std::size_t> ranks_;
    std::string last_prompt_;
    std::vector<std::int32_t> last_tokens_;
    bool has_last_ = false;
};

} // namespace sam::internal::sam3

#endif // SAM_CPP_SAM_INTERNAL_MODELS_SAM3_TOKENIZER_HPP
