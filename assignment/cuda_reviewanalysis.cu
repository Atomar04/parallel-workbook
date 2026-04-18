#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "cJSON.h"
#include "cJSON.c"
}

#define INPUT_FILE   "Movies_and_TV_5.json"
#define LEXICON_FILE "vader_lexicon.txt"
#define OUTPUT_FILE  "review_sentiment_labels.txt"

#define BLOCK_SIZE 256
#define BATCH_REVIEWS 20000
#define BATCH_MAX_TOKEN_IDS 4000000

/* sized for VADER (~7380 entries, max phrase length observed = 4) */
#define MAX_LEXICON_ENTRIES 7800
#define MAX_LEXICON_TOKENS  11000
#define UNKNOWN_TOKEN 65535u

#define CUDA_CHECK(x)                                                      \
    do {                                                                   \
        cudaError_t err = (x);                                             \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)         \
                      << " at line " << __LINE__ << std::endl;             \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

/*
 * Full lexicon in constant memory:
 *   c_phrase_offsets[pid]     -> start of phrase pid in c_phrase_tokens
 *   c_phrase_lengths[pid]     -> number of token IDs in phrase pid
 *   c_phrase_scores_x1000[pid]-> score * 1000 for phrase pid
 *   c_phrase_tokens[]         -> flat token-ID array for all phrases
 *
 * Candidate offsets/list stay in global memory as an auxiliary index.
 */
__constant__ uint16_t c_phrase_offsets[MAX_LEXICON_ENTRIES];
__constant__ uint8_t  c_phrase_lengths[MAX_LEXICON_ENTRIES];
__constant__ int16_t  c_phrase_scores_x1000[MAX_LEXICON_ENTRIES];
__constant__ uint16_t c_phrase_tokens[MAX_LEXICON_TOKENS];
__constant__ int      c_num_phrases;

struct Phrase {
    std::vector<uint16_t> tokens;
    int16_t score_x1000;
};

/* ------------------------------------------------------------
 * Tokenize text into lowercase token IDs.
 * Keeps letters, digits, apostrophe.
 * Unknown words become UNKNOWN_TOKEN.
 * ------------------------------------------------------------ */
void tokenize_to_ids(const std::string &text,
                     const std::unordered_map<std::string, uint16_t> &word_to_id,
                     std::vector<uint16_t> &out_ids) {
    out_ids.clear();
    std::string cur;
    cur.reserve(32);

    for (unsigned char ch : text) {
        char c = (char)std::tolower(ch);

        if (std::isalnum((unsigned char)c) || c == '\'') {
            cur.push_back(c);
        } else {
            if (!cur.empty()) {
                auto it = word_to_id.find(cur);
                if (it == word_to_id.end()) out_ids.push_back(UNKNOWN_TOKEN);
                else out_ids.push_back(it->second);
                cur.clear();
            }
        }
    }

    if (!cur.empty()) {
        auto it = word_to_id.find(cur);
        if (it == word_to_id.end()) out_ids.push_back(UNKNOWN_TOKEN);
        else out_ids.push_back(it->second);
    }
}

/* ------------------------------------------------------------
 * One CUDA thread processes one review.
 *
 * review_tokens[start ... start+len-1] is the token-ID sequence of the review.
 * candidate_offsets[tok] .. candidate_offsets[tok+1]-1 gives the phrase IDs
 * whose first token is tok.
 *
 * Every matched phrase occurrence contributes its lexicon score once.
 * So score = frequency * lexicon score naturally.
 * ------------------------------------------------------------ */
__global__ void review_label_kernel(const uint16_t *review_tokens,
                                    const int *review_offsets,
                                    const int *review_lengths,
                                    const int *candidate_offsets,
                                    const uint16_t *candidate_list,
                                    int vocab_size,
                                    int *scores_x1000,
                                    signed char *labels,
                                    int num_reviews) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_reviews) return;

    int start = review_offsets[tid];
    int len   = review_lengths[tid];

    int score = 0;

    for (int pos = 0; pos < len; pos++) {
        uint16_t tok = review_tokens[start + pos];
        if (tok == UNKNOWN_TOKEN || tok >= vocab_size) continue;

        int begin = candidate_offsets[tok];
        int end   = candidate_offsets[tok + 1];

        for (int idx = begin; idx < end; idx++) {
            uint16_t pid = candidate_list[idx];
            int plen = (int)c_phrase_lengths[pid];

            if (pos + plen > len) continue;

            int poff = c_phrase_offsets[pid];
            bool match = true;

            /* first token already matches because we selected candidate bucket by first token */
            for (int k = 1; k < plen; k++) {
                if (review_tokens[start + pos + k] != c_phrase_tokens[poff + k]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                score += (int)c_phrase_scores_x1000[pid];
            }
        }
    }

    scores_x1000[tid] = score;

    if (score > 0) labels[tid] = 1;
    else if (score < 0) labels[tid] = -1;
    else labels[tid] = 0;
}

/* ------------------------------------------------------------
 * Process one batch of reviews on GPU
 * ------------------------------------------------------------ */
void process_batch(const std::vector<uint16_t> &h_review_tokens,
                   const std::vector<int> &h_review_offsets,
                   const std::vector<int> &h_review_lengths,
                   long long review_index_base,
                   const int *d_candidate_offsets,
                   const uint16_t *d_candidate_list,
                   int vocab_size,
                   std::ofstream &out,
                   long long &positive,
                   long long &negative,
                   long long &neutral,
                   float &kernel_ms_total) {
    int num_reviews = (int)h_review_offsets.size();
    if (num_reviews == 0) return;

    uint16_t *d_review_tokens = nullptr;
    int *d_review_offsets = nullptr;
    int *d_review_lengths = nullptr;
    int *d_scores_x1000 = nullptr;
    signed char *d_labels = nullptr;

    CUDA_CHECK(cudaMalloc(&d_review_tokens,
                          h_review_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_review_offsets,
                          num_reviews * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_review_lengths,
                          num_reviews * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scores_x1000,
                          num_reviews * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels,
                          num_reviews * sizeof(signed char)));

    CUDA_CHECK(cudaMemcpy(d_review_tokens, h_review_tokens.data(),
                          h_review_tokens.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_review_offsets, h_review_offsets.data(),
                          num_reviews * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_review_lengths, h_review_lengths.data(),
                          num_reviews * sizeof(int),
                          cudaMemcpyHostToDevice));

    int grid = (num_reviews + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    review_label_kernel<<<grid, BLOCK_SIZE>>>(
        d_review_tokens,
        d_review_offsets,
        d_review_lengths,
        d_candidate_offsets,
        d_candidate_list,
        vocab_size,
        d_scores_x1000,
        d_labels,
        num_reviews
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    kernel_ms_total += ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<int> h_scores_x1000(num_reviews);
    std::vector<signed char> h_labels(num_reviews);

    CUDA_CHECK(cudaMemcpy(h_scores_x1000.data(), d_scores_x1000,
                          num_reviews * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                          num_reviews * sizeof(signed char),
                          cudaMemcpyDeviceToHost));

    /* buffered output to avoid slow line-by-line file writes */
    std::string buffer;
    buffer.reserve(8 * 1024 * 1024);

    for (int i = 0; i < num_reviews; i++) {
        const char *label_str;
        if (h_labels[i] > 0) {
            label_str = "POSITIVE";
            positive++;
        } else if (h_labels[i] < 0) {
            label_str = "NEGATIVE";
            negative++;
        } else {
            label_str = "NEUTRAL";
            neutral++;
        }

        float score = h_scores_x1000[i] / 1000.0f;

        buffer += std::to_string(review_index_base + i);
        buffer += '\t';
        buffer += std::to_string(score);
        buffer += '\t';
        buffer += label_str;
        buffer += '\n';

        if ((i + 1) % 100000 == 0) {
            out << buffer;
            buffer.clear();
        }
    }

    if (!buffer.empty()) out << buffer;

    CUDA_CHECK(cudaFree(d_review_tokens));
    CUDA_CHECK(cudaFree(d_review_offsets));
    CUDA_CHECK(cudaFree(d_review_lengths));
    CUDA_CHECK(cudaFree(d_scores_x1000));
    CUDA_CHECK(cudaFree(d_labels));
}

int main() {
    auto program_start = std::chrono::high_resolution_clock::now();

    /* --------------------------------------------------------
     * Step 1: Load lexicon, tokenize phrases, assign token IDs
     * -------------------------------------------------------- */
    std::ifstream lex_file(LEXICON_FILE);
    if (!lex_file.is_open()) {
        std::cerr << "Error opening lexicon file\n";
        return 1;
    }

    std::unordered_map<std::string, uint16_t> word_to_id;
    std::vector<Phrase> phrases;

    std::string line;
    int max_phrase_len = 0;

    while (std::getline(lex_file, line)) {
        std::stringstream ss(line);
        std::string phrase_str, score_str;

        std::getline(ss, phrase_str, '\t');
        std::getline(ss, score_str, '\t');

        if (phrase_str.empty() || score_str.empty()) continue;

        std::vector<uint16_t> ids;
        std::string cur;
        cur.reserve(32);

        for (unsigned char ch : phrase_str) {
            char c = (char)std::tolower(ch);

            if (std::isalnum((unsigned char)c) || c == '\'') {
                cur.push_back(c);
            } else {
                if (!cur.empty()) {
                    auto it = word_to_id.find(cur);
                    if (it == word_to_id.end()) {
                        uint16_t new_id = (uint16_t)word_to_id.size();
                        if (new_id == UNKNOWN_TOKEN) {
                            std::cerr << "Lexicon vocabulary too large for uint16 token IDs\n";
                            return 1;
                        }
                        word_to_id[cur] = new_id;
                        ids.push_back(new_id);
                    } else {
                        ids.push_back(it->second);
                    }
                    cur.clear();
                }
            }
        }

        if (!cur.empty()) {
            auto it = word_to_id.find(cur);
            if (it == word_to_id.end()) {
                uint16_t new_id = (uint16_t)word_to_id.size();
                if (new_id == UNKNOWN_TOKEN) {
                    std::cerr << "Lexicon vocabulary too large for uint16 token IDs\n";
                    return 1;
                }
                word_to_id[cur] = new_id;
                ids.push_back(new_id);
            } else {
                ids.push_back(it->second);
            }
        }

        if (ids.empty()) continue;

        Phrase p;
        p.tokens = ids;
        p.score_x1000 = (int16_t)std::lrint(std::stof(score_str) * 1000.0f);
        phrases.push_back(p);

        if ((int)ids.size() > max_phrase_len) {
            max_phrase_len = (int)ids.size();
        }
    }
    lex_file.close();

    int num_phrases = (int)phrases.size();
    int vocab_size = (int)word_to_id.size();

    std::cout << "Loaded lexicon entries: " << num_phrases << '\n';
    std::cout << "Maximum phrase length: " << max_phrase_len << '\n';

    if (num_phrases > MAX_LEXICON_ENTRIES) {
        std::cerr << "Too many lexicon entries for constant-memory arrays\n";
        return 1;
    }

    /* --------------------------------------------------------
     * Step 2: Flatten lexicon into constant-memory arrays
     * -------------------------------------------------------- */
    std::vector<uint16_t> h_phrase_offsets(num_phrases);
    std::vector<uint8_t> h_phrase_lengths(num_phrases);
    std::vector<int16_t> h_phrase_scores_x1000(num_phrases);
    std::vector<uint16_t> h_phrase_tokens;
    h_phrase_tokens.reserve(MAX_LEXICON_TOKENS);

    for (int pid = 0; pid < num_phrases; pid++) {
        h_phrase_offsets[pid] = (uint16_t)h_phrase_tokens.size();
        h_phrase_lengths[pid] = (uint8_t)phrases[pid].tokens.size();
        h_phrase_scores_x1000[pid] = phrases[pid].score_x1000;

        if ((int)h_phrase_tokens.size() + (int)phrases[pid].tokens.size() > MAX_LEXICON_TOKENS) {
            std::cerr << "Too many lexicon token IDs for constant memory. "
                      << "Increase MAX_LEXICON_TOKENS carefully.\n";
            return 1;
        }

        for (uint16_t x : phrases[pid].tokens) {
            h_phrase_tokens.push_back(x);
        }
    }

    int h_num_phrases = num_phrases;

    CUDA_CHECK(cudaMemcpyToSymbol(c_phrase_offsets,
                                  h_phrase_offsets.data(),
                                  num_phrases * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_phrase_lengths,
                                  h_phrase_lengths.data(),
                                  num_phrases * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_phrase_scores_x1000,
                                  h_phrase_scores_x1000.data(),
                                  num_phrases * sizeof(int16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_phrase_tokens,
                                  h_phrase_tokens.data(),
                                  h_phrase_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_num_phrases,
                                  &h_num_phrases,
                                  sizeof(int)));

    /* --------------------------------------------------------
     * Step 3: Build candidate index by first token
     *
     * This keeps matching efficient:
     * for a review token t, only phrases starting with t are checked.
     * -------------------------------------------------------- */
    std::vector<std::vector<uint16_t>> buckets(vocab_size);

    for (int pid = 0; pid < num_phrases; pid++) {
        uint16_t first = phrases[pid].tokens[0];
        buckets[first].push_back((uint16_t)pid);
    }

    std::vector<int> h_candidate_offsets(vocab_size + 1, 0);
    std::vector<uint16_t> h_candidate_list;

    for (int t = 0; t < vocab_size; t++) {
        h_candidate_offsets[t] = (int)h_candidate_list.size();
        for (uint16_t pid : buckets[t]) {
            h_candidate_list.push_back(pid);
        }
    }
    h_candidate_offsets[vocab_size] = (int)h_candidate_list.size();

    int *d_candidate_offsets = nullptr;
    uint16_t *d_candidate_list = nullptr;

    CUDA_CHECK(cudaMalloc(&d_candidate_offsets,
                          h_candidate_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_list,
                          h_candidate_list.size() * sizeof(uint16_t)));

    CUDA_CHECK(cudaMemcpy(d_candidate_offsets, h_candidate_offsets.data(),
                          h_candidate_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidate_list, h_candidate_list.data(),
                          h_candidate_list.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    /* --------------------------------------------------------
     * Step 4: Open review dataset and output file
     * -------------------------------------------------------- */
    std::ifstream in(INPUT_FILE);
    if (!in.is_open()) {
        std::cerr << "Error opening review file\n";
        return 1;
    }

    std::ofstream out(OUTPUT_FILE);
    if (!out.is_open()) {
        std::cerr << "Error opening output file\n";
        return 1;
    }
    out << "review_index\tscore\tlabel\n";

    /* --------------------------------------------------------
     * Step 5: Parse JSON with cJSON, tokenize reviews, batch them
     * -------------------------------------------------------- */
    std::vector<uint16_t> h_review_tokens;
    std::vector<int> h_review_offsets;
    std::vector<int> h_review_lengths;

    h_review_tokens.reserve(BATCH_MAX_TOKEN_IDS);
    h_review_offsets.reserve(BATCH_REVIEWS);
    h_review_lengths.reserve(BATCH_REVIEWS);

    long long total_lines = 0;
    long long valid_reviews = 0;
    long long parse_errors = 0;
    long long missing_text = 0;
    long long positive = 0, negative = 0, neutral = 0;
    float kernel_ms_total = 0.0f;

    std::vector<uint16_t> review_ids;
    review_ids.reserve(512);

    while (std::getline(in, line)) {
        total_lines++;

        cJSON *root = cJSON_Parse(line.c_str());
        if (!root) {
            parse_errors++;
            continue;
        }

        cJSON *txt = cJSON_GetObjectItemCaseSensitive(root, "reviewText");
        if (!cJSON_IsString(txt) || txt->valuestring == nullptr) {
            missing_text++;
            cJSON_Delete(root);
            continue;
        }

        tokenize_to_ids(txt->valuestring, word_to_id, review_ids);
        cJSON_Delete(root);

        if (!h_review_offsets.empty() &&
            ((int)h_review_offsets.size() >= BATCH_REVIEWS ||
             (int)(h_review_tokens.size() + review_ids.size()) > BATCH_MAX_TOKEN_IDS)) {

            process_batch(h_review_tokens,
                          h_review_offsets,
                          h_review_lengths,
                          valid_reviews - (long long)h_review_offsets.size(),
                          d_candidate_offsets,
                          d_candidate_list,
                          vocab_size,
                          out,
                          positive, negative, neutral,
                          kernel_ms_total);

            h_review_tokens.clear();
            h_review_offsets.clear();
            h_review_lengths.clear();
        }

        h_review_offsets.push_back((int)h_review_tokens.size());
        h_review_lengths.push_back((int)review_ids.size());

        for (uint16_t x : review_ids) {
            h_review_tokens.push_back(x);
        }

        valid_reviews++;
    }
    in.close();

    /* process final batch */
    if (!h_review_offsets.empty()) {
        process_batch(h_review_tokens,
                      h_review_offsets,
                      h_review_lengths,
                      valid_reviews - (long long)h_review_offsets.size(),
                      d_candidate_offsets,
                      d_candidate_list,
                      vocab_size,
                      out,
                      positive, negative, neutral,
                      kernel_ms_total);
    }

    out.close();

    CUDA_CHECK(cudaFree(d_candidate_offsets));
    CUDA_CHECK(cudaFree(d_candidate_list));

    auto program_end = std::chrono::high_resolution_clock::now();
    double total_program_time =
        std::chrono::duration<double>(program_end - program_start).count();

    std::cout << "\nSummary\n";
    std::cout << "Total JSON lines read: " << total_lines << '\n';
    std::cout << "Valid reviews processed: " << valid_reviews << '\n';
    std::cout << "JSON parse errors: " << parse_errors << '\n';
    std::cout << "Missing reviewText: " << missing_text << '\n';
    std::cout << "Positive: " << positive << '\n';
    std::cout << "Negative: " << negative << '\n';
    std::cout << "Neutral: " << neutral << '\n';
    std::cout << "Total kernel time: " << kernel_ms_total << " ms\n";
    std::cout << "Total program time: " << total_program_time << " s\n";
    std::cout << "Labels written to: " << OUTPUT_FILE << '\n';

    return 0;
}