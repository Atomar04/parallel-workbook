#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>

// Definition for error checking
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

// CUDA Kernel
__global__ void optimizedAggregateRatingsKernel(const int* __restrict__ review_movie_ids, 
                                                const float* __restrict__ review_ratings, 
                                                float* __restrict__ movie_rating_sums, 
                                                int* __restrict__ movie_review_counts, 
                                                int dummy_id) {
    __shared__ int s_ids[THREADS_PER_BLOCK];
    __shared__ float s_ratings[THREADS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    s_ids[tx] = review_movie_ids[tid];
    s_ratings[tx] = review_ratings[tid];
    __syncthreads();

    int my_id = s_ids[tx];
    float my_rating = s_ratings[tx];

    int lane_id = tx % 32;
    
    int leader = lane_id; 
    for (int i = 0; i < 32; ++i) {
        int peer_id = __shfl_sync(0xffffffff, my_id, i);
        if (peer_id == my_id && i < leader) {
            leader = i; 
        }
    }

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

    if (lane_id == leader && my_id != dummy_id) {
        atomicAdd(&movie_rating_sums[my_id], warp_sum);
        atomicAdd(&movie_review_counts[my_id], warp_count);
    }
}

struct MovieResult {
    std::string asin;
    float avg_rating;
    int review_count;
};

// Creating a compare function for review counts
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
    std::getline(file, line);

    // Reading from file
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
    int dummy_id = num_unique_movies; 

    int chunk_size = ((num_reviews + NUM_STREAMS - 1) / NUM_STREAMS);
    chunk_size = ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
    int padded_reviews = chunk_size * NUM_STREAMS;

    int *h_review_movie_ids;
    float *h_review_ratings;
    CUDA_CHECK(cudaMallocHost(&h_review_movie_ids, padded_reviews * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_review_ratings, padded_reviews * sizeof(float)));

    // Padding
    for (int i = 0; i < padded_reviews; ++i) {
        if (i < num_reviews) {
            h_review_movie_ids[i] = asin_to_id[temp_asins[i]];
            h_review_ratings[i] = temp_ratings[i];
        } else {
            h_review_movie_ids[i] = dummy_id;
            h_review_ratings[i] = 0.0f;
        }
    }

    int *d_review_movie_ids, *d_movie_review_counts;
    float *d_review_ratings, *d_movie_rating_sums;

    CUDA_CHECK(cudaMalloc(&d_review_movie_ids, padded_reviews * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_review_ratings, padded_reviews * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_movie_rating_sums, (num_unique_movies + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_movie_review_counts, (num_unique_movies + 1) * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_movie_rating_sums, 0, (num_unique_movies + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_movie_review_counts, 0, (num_unique_movies + 1) * sizeof(int)));

    // Creating streams
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start_kernel[NUM_STREAMS], stop_kernel[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&start_kernel[i]));
        CUDA_CHECK(cudaEventCreate(&stop_kernel[i]));
    }

    // Global pipeline events
    cudaEvent_t start_total, stop_total;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));

    std::cout << "Launching Optimized Kernel with Streams..." << std::endl;
    
    // Start global pipeline timer (H2D + Kernel overlap)
    CUDA_CHECK(cudaEventRecord(start_total));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * chunk_size;
        int blocks = chunk_size / THREADS_PER_BLOCK;

        // Async Memcpy (H2D)
        CUDA_CHECK(cudaMemcpyAsync(&d_review_movie_ids[offset], &h_review_movie_ids[offset], 
                                   chunk_size * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(&d_review_ratings[offset], &h_review_ratings[offset], 
                                   chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        // Record start event specifically mapped to this stream
        CUDA_CHECK(cudaEventRecord(start_kernel[i], streams[i]));

        optimizedAggregateRatingsKernel<<<blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
            &d_review_movie_ids[offset], &d_review_ratings[offset], 
            d_movie_rating_sums, d_movie_review_counts, dummy_id
        );

        // Record stop event specifically mapped to this stream
        CUDA_CHECK(cudaEventRecord(stop_kernel[i], streams[i]));
    }

    // Explicitly sync the device before stopping the global timer to ensure all streams are fully completed
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    // Calculate all timing metrics
    float total_h2d_kernel_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_h2d_kernel_time, start_total, stop_total));

    float stream_kernel_times[NUM_STREAMS];
    float sum_kernel_times = 0.0f;
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaEventElapsedTime(&stream_kernel_times[i], start_kernel[i], stop_kernel[i]));
        sum_kernel_times += stream_kernel_times[i];
    }

    std::vector<float> host_movie_rating_sums(num_unique_movies);
    std::vector<int> host_movie_review_counts(num_unique_movies);

    CUDA_CHECK(cudaMemcpy(host_movie_rating_sums.data(), d_movie_rating_sums, num_unique_movies * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_movie_review_counts.data(), d_movie_review_counts, num_unique_movies * sizeof(int), cudaMemcpyDeviceToHost));

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
    std::cout << "             PERFORMANCE METRICS            \n";
    std::cout << "============================================\n";
    for(int i = 0; i < NUM_STREAMS; i++) {
        std::cout << "Stream " << i << " Kernel Time    : " << stream_kernel_times[i] << " ms\n";
    }
    std::cout << "--------------------------------------------\n";
    std::cout << "Sum of Stream Kernels     : " << sum_kernel_times << " ms\n";
    std::cout << "Total Time (H2D + Kernel) : " << total_h2d_kernel_time << " ms\n";
    std::cout << "============================================\n";

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(start_kernel[i]));
        CUDA_CHECK(cudaEventDestroy(stop_kernel[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_review_movie_ids));
    CUDA_CHECK(cudaFreeHost(h_review_ratings));
    CUDA_CHECK(cudaFree(d_review_movie_ids));
    CUDA_CHECK(cudaFree(d_review_ratings));
    CUDA_CHECK(cudaFree(d_movie_rating_sums));
    CUDA_CHECK(cudaFree(d_movie_review_counts));
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));

    return 0;
}