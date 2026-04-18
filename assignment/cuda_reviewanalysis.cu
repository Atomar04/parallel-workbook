#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
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

#define BLOCK_SIZE 256
#define BATCH_REVIEWS 20000
#define BATCH_MAX_TOKEN_IDS 4000000

/* constant-memory limits */
#define MAX_LEXICON_ENTRIES 8192
#define MAX_PHRASE_TOKEN_IDS 65535

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
 * Lexicon metadata stored in constant memory for fast access.
 * phrase_offsets[i] -> start index of phrase i in flat phrase token array
 * phrase_lengths[i] -> number of tokens in phrase i
 * phrase_scores[i]  -> sentiment score of phrase i
 */
__constant__ uint16_t c_phrase_offsets[MAX_LEXICON_ENTRIES];
__constant__ uint8_t  c_phrase_lengths[MAX_LEXICON_ENTRIES];
__constant__ float    c_phrase_scores[MAX_LEXICON_ENTRIES];

struct Phrase {
    std::vector<uint16_t> tokens;
    float score;
};

/*
 * Convert text into lowercase word tokens.
 * Keep letters, digits, and apostrophe inside words.
 * Treat everything else as a separator.
 */
std::vector<std::string> tokenize_text(const std::string &text) {
    std::vector<std::string> tokens;
    std::string cur;

    for (unsigned char ch : text) {
        char c = (char)std::tolower(ch);

        if (std::isalnum((unsigned char)c) || c == '\'') {
            cur.push_back(c);
        } else {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        }
    }

    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

/*
 * CUDA kernel:
 * Each thread processes exactly one review.
 *
 * For each token position in the review:
 * - look up candidate phrases that start with that token
 * - compare phrase token IDs with review token IDs
 * - if matched, add lexicon score
 *
 * Final label:
 *   score > 0  -> positive
 *   score < 0  -> negative
 *   score == 0 -> neutral
 */
__global__ void review_label_kernel(
    const uint16_t *review_tokens,
    const int *review_offsets,
    const int *review_lengths,
    const uint16_t *phrase_tokens,
    const int *candidate_offsets,
    const int *candidate_list,
    int vocab_size,
    signed char *labels,
    int num_reviews)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_reviews) return;

    int start = review_offsets[tid];
    int len   = review_lengths[tid];

    float score = 0.0f;

    for (int pos = 0; pos < len; pos++) {
        uint16_t first_tok = review_tokens[start + pos];

        if (first_tok == UNKNOWN_TOKEN || first_tok >= vocab_size) continue;

        int begin = candidate_offsets[first_tok];
        int end   = candidate_offsets[first_tok + 1];

        for (int idx = begin; idx < end; idx++) {
            int pid = candidate_list[idx];
            int plen = (int)c_phrase_lengths[pid];

            if (pos + plen > len) continue;

            uint16_t poff = c_phrase_offsets[pid];
            bool match = true;

            for (int k = 1; k < plen; k++) {
                if (review_tokens[start + pos + k] != phrase_tokens[poff + k]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                score += c_phrase_scores[pid];
            }
        }
    }

    if (score > 0.0f) labels[tid] = 1;
    else if (score < 0.0f) labels[tid] = -1;
    else labels[tid] = 0;
}

/*
 * Process one batch of reviews on the GPU.
 * Only labels are copied back, since no output file is required.
 */
void process_batch(
    const std::vector<uint16_t> &h_review_tokens,
    const std::vector<int> &h_review_offsets,
    const std::vector<int> &h_review_lengths,
    const uint16_t *d_phrase_tokens,
    const int *d_candidate_offsets,
    const int *d_candidate_list,
    int vocab_size,
    long long &positive,
    long long &negative,
    long long &neutral,
    float &kernel_ms_total)
{
    int num_reviews = (int)h_review_offsets.size();
    if (num_reviews == 0) return;

    uint16_t *d_review_tokens = nullptr;
    int *d_review_offsets = nullptr;
    int *d_review_lengths = nullptr;
    signed char *d_labels = nullptr;

    CUDA_CHECK(cudaMalloc(&d_review_tokens,
                          h_review_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_review_offsets,
                          num_reviews * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_review_lengths,
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
        d_phrase_tokens,
        d_candidate_offsets,
        d_candidate_list,
        vocab_size,
        d_labels,
        num_reviews
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    kernel_ms_total += ms;

    std::vector<signed char> h_labels(num_reviews);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                          num_reviews * sizeof(signed char),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_reviews; i++) {
        if (h_labels[i] > 0) positive++;
        else if (h_labels[i] < 0) negative++;
        else neutral++;
    }

    cudaFree(d_review_tokens);
    cudaFree(d_review_offsets);
    cudaFree(d_review_lengths);
    cudaFree(d_labels);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    auto total_start = std::chrono::high_resolution_clock::now();

    /* --------------------------------------------------------
     * Step 1: Load lexicon and tokenize phrases
     * -------------------------------------------------------- */
    std::ifstream lex_file(LEXICON_FILE);
    if (!lex_file.is_open()) {
        std::cerr << "Error opening lexicon file\n";
        return 1;
    }

    std::unordered_map<std::string, uint16_t> word_to_id;
    std::vector<Phrase> phrases;

    std::string line;
    while (std::getline(lex_file, line)) {
        std::stringstream ss(line);
        std::string phrase_str, score_str;

        std::getline(ss, phrase_str, '\t');
        std::getline(ss, score_str, '\t');

        if (phrase_str.empty() || score_str.empty()) continue;

        std::vector<std::string> toks = tokenize_text(phrase_str);
        if (toks.empty()) continue;

        Phrase p;
        for (const std::string &tok : toks) {
            auto it = word_to_id.find(tok);
            if (it == word_to_id.end()) {
                uint16_t new_id = (uint16_t)word_to_id.size();
                if (new_id == UNKNOWN_TOKEN) {
                    std::cerr << "Vocabulary too large for uint16_t token IDs\n";
                    return 1;
                }
                word_to_id[tok] = new_id;
                p.tokens.push_back(new_id);
            } else {
                p.tokens.push_back(it->second);
            }
        }

        p.score = std::stof(score_str);
        phrases.push_back(p);
    }
    lex_file.close();

    int num_phrases = (int)phrases.size();
    int vocab_size  = (int)word_to_id.size();

    std::cout << "Loaded lexicon entries: " << num_phrases << std::endl;

    if (num_phrases > MAX_LEXICON_ENTRIES) {
        std::cerr << "Too many lexicon entries for constant memory arrays\n";
        return 1;
    }

    /* Flatten phrase token IDs */
    std::vector<uint16_t> h_phrase_tokens;
    std::vector<uint16_t> h_phrase_offsets(num_phrases);
    std::vector<uint8_t>  h_phrase_lengths(num_phrases);
    std::vector<float>    h_phrase_scores(num_phrases);

    int max_phrase_len = 0;
    for (int i = 0; i < num_phrases; i++) {
        if ((int)phrases[i].tokens.size() > max_phrase_len) {
            max_phrase_len = (int)phrases[i].tokens.size();
        }

        if ((int)h_phrase_tokens.size() + (int)phrases[i].tokens.size() > MAX_PHRASE_TOKEN_IDS) {
            std::cerr << "Too many flat phrase token IDs for configured limit\n";
            return 1;
        }

        h_phrase_offsets[i] = (uint16_t)h_phrase_tokens.size();
        h_phrase_lengths[i] = (uint8_t)phrases[i].tokens.size();
        h_phrase_scores[i]  = phrases[i].score;

        for (uint16_t x : phrases[i].tokens) {
            h_phrase_tokens.push_back(x);
        }
    }

    std::cout << "Maximum phrase length: " << max_phrase_len << std::endl;
    std::cout << "Lexicon vocabulary size: " << vocab_size << std::endl;

    /*
     * Build candidate list by first token.
     * candidate_offsets[t]..candidate_offsets[t+1]-1
     * contains phrase IDs whose first token is t.
     */
    std::vector<std::vector<int>> buckets(vocab_size);
    for (int i = 0; i < num_phrases; i++) {
        uint16_t first = phrases[i].tokens[0];
        buckets[first].push_back(i);
    }

    std::vector<int> h_candidate_offsets(vocab_size + 1, 0);
    std::vector<int> h_candidate_list;

    for (int t = 0; t < vocab_size; t++) {
        h_candidate_offsets[t] = (int)h_candidate_list.size();
        for (int pid : buckets[t]) {
            h_candidate_list.push_back(pid);
        }
    }
    h_candidate_offsets[vocab_size] = (int)h_candidate_list.size();

    /* Copy lexicon metadata to constant memory */
    CUDA_CHECK(cudaMemcpyToSymbol(c_phrase_offsets,
                                  h_phrase_offsets.data(),
                                  num_phrases * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_phrase_lengths,
                                  h_phrase_lengths.data(),
                                  num_phrases * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_phrase_scores,
                                  h_phrase_scores.data(),
                                  num_phrases * sizeof(float)));

    /* Copy remaining lexicon data to device global memory */
    uint16_t *d_phrase_tokens = nullptr;
    int *d_candidate_offsets = nullptr;
    int *d_candidate_list = nullptr;

    CUDA_CHECK(cudaMalloc(&d_phrase_tokens,
                          h_phrase_tokens.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_candidate_offsets,
                          h_candidate_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_list,
                          h_candidate_list.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_phrase_tokens, h_phrase_tokens.data(),
                          h_phrase_tokens.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidate_offsets, h_candidate_offsets.data(),
                          h_candidate_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidate_list, h_candidate_list.data(),
                          h_candidate_list.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* --------------------------------------------------------
     * Step 2: Read reviews, parse JSON, tokenize, batch
     * -------------------------------------------------------- */
    std::ifstream in(INPUT_FILE);
    if (!in.is_open()) {
        std::cerr << "Error opening review file\n";
        return 1;
    }

    std::vector<uint16_t> h_review_tokens;
    std::vector<int> h_review_offsets;
    std::vector<int> h_review_lengths;

    long long total_lines = 0;
    long long valid_reviews = 0;
    long long parse_errors = 0;
    long long missing_text = 0;
    long long positive = 0, negative = 0, neutral = 0;
    float kernel_ms_total = 0.0f;

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

        std::vector<std::string> toks = tokenize_text(txt->valuestring);
        cJSON_Delete(root);

        if (!h_review_offsets.empty() &&
            ((int)h_review_offsets.size() >= BATCH_REVIEWS ||
             (int)(h_review_tokens.size() + toks.size()) > BATCH_MAX_TOKEN_IDS)) {

            process_batch(h_review_tokens,
                          h_review_offsets,
                          h_review_lengths,
                          d_phrase_tokens,
                          d_candidate_offsets,
                          d_candidate_list,
                          vocab_size,
                          positive, negative, neutral,
                          kernel_ms_total);

            std::cout << "Processed reviews so far: " << valid_reviews << std::endl;

            h_review_tokens.clear();
            h_review_offsets.clear();
            h_review_lengths.clear();
        }

        h_review_offsets.push_back((int)h_review_tokens.size());
        h_review_lengths.push_back((int)toks.size());

        for (const std::string &tok : toks) {
            auto it = word_to_id.find(tok);
            if (it == word_to_id.end()) {
                h_review_tokens.push_back(UNKNOWN_TOKEN);
            } else {
                h_review_tokens.push_back(it->second);
            }
        }

        valid_reviews++;

        if (valid_reviews % 100000 == 0) {
            std::cout << "Read/tokenized reviews: " << valid_reviews << std::endl;
        }
    }
    in.close();

    /* Process final batch */
    if (!h_review_offsets.empty()) {
        process_batch(h_review_tokens,
                      h_review_offsets,
                      h_review_lengths,
                      d_phrase_tokens,
                      d_candidate_offsets,
                      d_candidate_list,
                      vocab_size,
                      positive, negative, neutral,
                      kernel_ms_total);

        std::cout << "Processed reviews so far: " << valid_reviews << std::endl;
    }

    cudaFree(d_phrase_tokens);
    cudaFree(d_candidate_offsets);
    cudaFree(d_candidate_list);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec =
        std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\nSummary\n";
    std::cout << "Total JSON lines read: " << total_lines << std::endl;
    std::cout << "Valid reviews processed: " << valid_reviews << std::endl;
    std::cout << "Parse errors: " << parse_errors << std::endl;
    std::cout << "Missing reviewText: " << missing_text << std::endl;
    std::cout << "Positive: " << positive << std::endl;
    std::cout << "Negative: " << negative << std::endl;
    std::cout << "Neutral: " << neutral << std::endl;
    std::cout << "Total kernel time: " << kernel_ms_total << " ms" << std::endl;
    std::cout << "Total program time: " << total_sec << " s" << std::endl;

    return 0;
}
