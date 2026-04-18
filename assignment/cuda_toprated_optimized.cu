#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

const int THREADS_PER_BLOCK = 256;
const int NUM_STREAMS = 4;

// ---------------------------------------------------------
// OPTIMIZED CUDA KERNEL
// ---------------------------------------------------------
__global__ void optimizedAggregateRatingsKernel(const int* __restrict__ review_movie_ids, 
                                                const float* __restrict__ review_ratings, 
                                                float* __restrict__ movie_rating_sums, 
                                                int* __restrict__ movie_review_counts, 
                                                int dummy_id) {
    // [Optimization]: Shared memory usage to reduce global memory bandwidth
    __shared__ int s_ids[THREADS_PER_BLOCK];
    __shared__ float s_ratings[THREADS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // [Optimization]: Memory coalescing and aligned global memory accesses
    s_ids[tx] = review_movie_ids[tid];
    s_ratings[tx] = review_ratings[tid];
    __syncthreads();

    int my_id = s_ids[tx];
    float my_rating = s_ratings[tx];

    // [Optimization]: Minimizing thread divergence
    // By padding the input array, we removed the "if (tid < n)" branch.
    // All threads execute identically without warp divergence.

    // [Optimization]: Warp-level primitives and reduction techniques
    int lane_id = tx % 32;
    
    // Step 1: Find the leader thread for this specific movie ID within the warp
    int leader = lane_id; 
    for (int i = 0; i < 32; ++i) {
        int peer_id = __shfl_sync(0xffffffff, my_id, i);
        if (peer_id == my_id && i < leader) {
            leader = i; // The thread with the lowest lane_id for this movie is the leader
        }
    }

    // Step 2: Intra-warp reduction (Accumulate ratings across the warp)
    float warp_sum = 0.0f;
    int warp_count = 0;
    for (int i = 0; i < 32; ++i) {
        int peer_id = __shfl_sync(0xffffffff, my_id, i);
        float peer_rating = __shfl_sync(0xffffffff, my_rating, i);
        if (peer_id == my_id) {
            warp_sum += peer_rating;
            warp_count++;
        }
    }

    // Step 3: Only the warp leader executes the atomic global write
    // This drastically reduces global memory contention and bandwidth
    if (lane_id == leader && my_id != dummy_id) {
        atomicAdd(&movie_rating_sums[my_id], warp_sum);
        atomicAdd(&movie_review_counts[my_id], warp_count);
    }
}

// Struct to help sort the final results
struct MovieResult {
    std::string asin;
    float avg_rating;
    int review_count;
};

bool compareMovies(const MovieResult& a, const MovieResult& b) {
    if (a.avg_rating != b.avg_rating) return a.avg_rating > b.avg_rating;
    return a.review_count > b.review_count; 
}

int main() {
    std::string filename = "ratings_extracted.csv";
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::string> temp_asins;
    std::vector<float> temp_ratings;
    std::unordered_map<std::string, int> asin_to_id;
    std::vector<std::string> id_to_asin;

    std::string line, asin, rating_str;
    std::getline(file, line); // Skip header

    std::cout << "Loading data from CSV..." << std::endl;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        if (std::getline(ss, asin, ',') && std::getline(ss, rating_str, ',')) {
            temp_asins.push_back(asin);
            temp_ratings.push_back(std::stof(rating_str));

            if (asin_to_id.find(asin) == asin_to_id.end()) {
                asin_to_id[asin] = id_to_asin.size();
                id_to_asin.push_back(asin);
            }
        }
    }
    file.close();

    int num_reviews = temp_asins.size();
    int num_unique_movies = id_to_asin.size();
    int dummy_id = num_unique_movies; // Used for padded threads

    // Calculate padding so elements divide perfectly among Streams and Block sizes
    int chunk_size = ((num_reviews + NUM_STREAMS - 1) / NUM_STREAMS);
    // Align chunk size to block size
    chunk_size = ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
    int padded_reviews = chunk_size * NUM_STREAMS;

    // Allocate Pinned Host Memory for Streams overlap
    int *h_review_movie_ids;
    float *h_review_ratings;
    CUDA_CHECK(cudaMallocHost(&h_review_movie_ids, padded_reviews * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_review_ratings, padded_reviews * sizeof(float)));

    // Populate pinned memory and pad the rest
    for (int i = 0; i < padded_reviews; ++i) {
        if (i < num_reviews) {
            h_review_movie_ids[i] = asin_to_id[temp_asins[i]];
            h_review_ratings[i] = temp_ratings[i];
        } else {
            // Padding elements mapped to dummy_id to avoid branch divergence
            h_review_movie_ids[i] = dummy_id;
            h_review_ratings[i] = 0.0f;
        }
    }

    // Allocate Device Memory (Adding 1 to unique movies to hold dummy writes safely)
    int *d_review_movie_ids, *d_movie_review_counts;
    float *d_review_ratings, *d_movie_rating_sums;

    CUDA_CHECK(cudaMalloc(&d_review_movie_ids, padded_reviews * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_review_ratings, padded_reviews * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_movie_rating_sums, (num_unique_movies + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_movie_review_counts, (num_unique_movies + 1) * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_movie_rating_sums, 0, (num_unique_movies + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_movie_review_counts, 0, (num_unique_movies + 1) * sizeof(int)));

    // Create CUDA Streams and Events
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Launching Optimized Kernel with Streams..." << std::endl;
    
    CUDA_CHECK(cudaEventRecord(start));

    // [Optimization]: CUDA Streams for overlapping computation and data transfer
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * chunk_size;
        int blocks = chunk_size / THREADS_PER_BLOCK;

        // Async Memcpy overlaps with kernel execution of other streams
        CUDA_CHECK(cudaMemcpyAsync(&d_review_movie_ids[offset], &h_review_movie_ids[offset], 
                                   chunk_size * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(&d_review_ratings[offset], &h_review_ratings[offset], 
                                   chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        optimizedAggregateRatingsKernel<<<blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
            &d_review_movie_ids[offset], &d_review_ratings[offset], 
            d_movie_rating_sums, d_movie_review_counts, dummy_id
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results back to Host
    std::vector<float> host_movie_rating_sums(num_unique_movies);
    std::vector<int> host_movie_review_counts(num_unique_movies);

    CUDA_CHECK(cudaMemcpy(host_movie_rating_sums.data(), d_movie_rating_sums, num_unique_movies * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_movie_review_counts.data(), d_movie_review_counts, num_unique_movies * sizeof(int), cudaMemcpyDeviceToHost));

    // CPU Post-Processing
    std::vector<MovieResult> results;
    for (int i = 0; i < num_unique_movies; ++i) {
        if (host_movie_review_counts[i] > 0) {
            float avg = host_movie_rating_sums[i] / host_movie_review_counts[i];
            results.push_back({id_to_asin[i], avg, host_movie_review_counts[i]});
        }
    }

    std::sort(results.begin(), results.end(), compareMovies);

    // Print Results
    std::cout << "\n============================================\n";
    std::cout << "Top 10 Rated Movies (Optimized)\n";
    std::cout << "============================================\n";
    for (int i = 0; i < 10 && i < results.size(); ++i) {
        std::cout << i + 1 << ". ASIN: " << results[i].asin 
                  << " | Avg Rating: " << results[i].avg_rating 
                  << " | Reviews: " << results[i].review_count << std::endl;
    }

    std::cout << "\n============================================\n";
    std::cout << "Optimized Execution Time: " << milliseconds << " ms\n";
    std::cout << "============================================\n";

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_review_movie_ids));
    CUDA_CHECK(cudaFreeHost(h_review_ratings));
    CUDA_CHECK(cudaFree(d_review_movie_ids));
    CUDA_CHECK(cudaFree(d_review_ratings));
    CUDA_CHECK(cudaFree(d_movie_rating_sums));
    CUDA_CHECK(cudaFree(d_movie_review_counts));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}