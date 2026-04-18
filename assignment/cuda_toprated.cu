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

__global__ void aggregateRatingsKernel(const int* review_movie_ids, 
                                       const float* review_ratings, 
                                       float* movie_rating_sums, 
                                       int* movie_review_counts, 
                                       int num_reviews) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_reviews) {
        int movie_id = review_movie_ids[tid];
        float rating = review_ratings[tid];

        atomicAdd(&movie_rating_sums[movie_id], rating);
        atomicAdd(&movie_review_counts[movie_id], 1);
    }
}

struct MovieResult {
    std::string asin;
    float avg_rating;
    int review_count;
};

bool compareMovies(const MovieResult& a, const MovieResult& b) {
    if (a.avg_rating != b.avg_rating) {
        return a.avg_rating > b.avg_rating;
    }
    return a.review_count > b.review_count; 
}

int main() {
    std::string filename = "ratings_extracted.csv";
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<int> host_review_movie_ids;
    std::vector<float> host_review_ratings;
    std::unordered_map<std::string, int> asin_to_id;
    std::vector<std::string> id_to_asin;

    std::string line, asin, rating_str;
    
    std::getline(file, line); 

    std::cout << "Loading and parsing data on CPU..." << std::endl;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        if (std::getline(ss, asin, ',') && std::getline(ss, rating_str, ',')) {
            float rating = std::stof(rating_str);

            if (asin_to_id.find(asin) == asin_to_id.end()) {
                asin_to_id[asin] = id_to_asin.size();
                id_to_asin.push_back(asin);
            }
            
            host_review_movie_ids.push_back(asin_to_id[asin]);
            host_review_ratings.push_back(rating);
        }
    }
    file.close();

    int num_reviews = host_review_movie_ids.size();
    int num_unique_movies = id_to_asin.size();

    std::cout << "Total Reviews: " << num_reviews << std::endl;
    std::cout << "Total Unique Movies: " << num_unique_movies << std::endl;

    int *d_review_movie_ids, *d_movie_review_counts;
    float *d_review_ratings, *d_movie_rating_sums;

    CUDA_CHECK(cudaMalloc(&d_review_movie_ids, num_reviews * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_review_ratings, num_reviews * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_movie_rating_sums, num_unique_movies * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_movie_review_counts, num_unique_movies * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_movie_rating_sums, 0, num_unique_movies * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_movie_review_counts, 0, num_unique_movies * sizeof(int)));

    // Create events for the two phases we want to compare against the optimized version
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    
    CUDA_CHECK(cudaEventCreate(&start_h2d)); CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel)); CUDA_CHECK(cudaEventCreate(&stop_kernel));

    // ==========================================
    // Phase 1: Host to Device Memory Copy
    // ==========================================
    CUDA_CHECK(cudaEventRecord(start_h2d));
    
    CUDA_CHECK(cudaMemcpy(d_review_movie_ids, host_review_movie_ids.data(), num_reviews * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_review_ratings, host_review_ratings.data(), num_reviews * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(stop_h2d));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_reviews + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching Kernel..." << std::endl;

    // ==========================================
    // Phase 2: Kernel Execution
    // ==========================================
    CUDA_CHECK(cudaEventRecord(start_kernel));

    aggregateRatingsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_review_movie_ids, d_review_ratings, d_movie_rating_sums, d_movie_review_counts, num_reviews
    );

    CUDA_CHECK(cudaEventRecord(stop_kernel));

    // ==========================================
    // Untimed Phase: Device to Host Memory Copy
    // ==========================================
    std::vector<float> host_movie_rating_sums(num_unique_movies);
    std::vector<int> host_movie_review_counts(num_unique_movies);

    CUDA_CHECK(cudaMemcpy(host_movie_rating_sums.data(), d_movie_rating_sums, num_unique_movies * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_movie_review_counts.data(), d_movie_review_counts, num_unique_movies * sizeof(int), cudaMemcpyDeviceToHost));

    // Synchronize events before reading elapsed times
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate times
    float time_h2d = 0, time_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));
    
    // Total represents H2D + Kernel (matching the timing window of the optimized stream code)
    float total_gpu_time = time_h2d + time_kernel;

    // Post-processing and sorting
    std::vector<MovieResult> results;
    for (int i = 0; i < num_unique_movies; ++i) {
        if (host_movie_review_counts[i] > 0) {
            float avg = host_movie_rating_sums[i] / host_movie_review_counts[i];
            results.push_back({id_to_asin[i], avg, host_movie_review_counts[i]});
        }
    }

    std::sort(results.begin(), results.end(), compareMovies);

    // Output Data
    std::cout << "\n============================================\n";
    std::cout << "Top 10 Rated Movies\n";
    std::cout << "============================================\n";
    for (int i = 0; i < 10 && i < results.size(); ++i) {
        std::cout << i + 1 << ". ASIN: " << results[i].asin 
                  << " | Avg Rating: " << results[i].avg_rating 
                  << " | Reviews: " << results[i].review_count << std::endl;
    }

    std::cout << "\n============================================\n";
    std::cout << "             PERFORMANCE METRICS            \n";
    std::cout << "============================================\n";
    std::cout << "Host-to-Device (H2D) Time : " << time_h2d << " ms\n";
    std::cout << "Kernel Execution Time     : " << time_kernel << " ms\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "Total Time (H2D + Kernel) : " << total_gpu_time << " ms\n";
    std::cout << "============================================\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_review_movie_ids));
    CUDA_CHECK(cudaFree(d_review_ratings));
    CUDA_CHECK(cudaFree(d_movie_rating_sums));
    CUDA_CHECK(cudaFree(d_movie_review_counts));
    CUDA_CHECK(cudaEventDestroy(start_h2d)); CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_kernel)); CUDA_CHECK(cudaEventDestroy(stop_kernel));

    return 0;
}