#include <cuda_runtime.h>
#include <float.h>

// Helper function for atomic max with floats
__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(__int_as_float(assumed), value)));
    } while (assumed != old);
    
    return __int_as_float(old);
}

// Main kernel for sliding window self-attention
__global__ void slidingWindowAttention(
    const float* Q, const float* K, const float* V, float* output,
    int M, int d, int window_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    
    // Determine the window boundaries
    int window_start = max(0, i - window_size);
    int window_end = min(M - 1, i + window_size);
    int window_len = window_end - window_start + 1;
    
    // Allocate memory for scores (max possible window size is 2*32+1=65)
    float scores[65];
    
    // Step 1: Compute attention scores for the window
    float sqrt_d = sqrtf((float)d);
    float max_score = -FLT_MAX;
    
    for (int j_idx = 0; j_idx < window_len; j_idx++) {
        int j = window_start + j_idx;
        float score = 0.0f;
        
        // Compute dot product Q_i · K_j
        for (int k = 0; k < d; k++) {
            score += Q[i * d + k] * K[j * d + k];
        }
        
        score /= sqrt_d;
        scores[j_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Step 2: Apply softmax with numerical stability
    float sum_exp = 0.0f;
    for (int j_idx = 0; j_idx < window_len; j_idx++) {
        float exp_val = expf(scores[j_idx] - max_score);
        scores[j_idx] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    for (int j_idx = 0; j_idx < window_len; j_idx++) {
        scores[j_idx] /= sum_exp;
    }
    
    // Step 3: Compute weighted sum of values
    for (int k = 0; k < d; k++) {
        float result = 0.0f;
        for (int j_idx = 0; j_idx < window_len; j_idx++) {
            int j = window_start + j_idx;
            result += scores[j_idx] * V[j * d + k];
        }
        output[i * d + k] = result;
    }
}

// Optimized kernel using shared memory for small d
__global__ void slidingWindowAttentionShared(
    const float* Q, const float* K, const float* V, float* output,
    int M, int d, int window_size
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x;  // Each block handles one query
    
    if (i >= M) return;
    
    // Shared memory layout:
    // scores[65] + query[d]
    float* scores = shared_mem;
    float* query = shared_mem + 65;
    
    // Load query vector into shared memory
    for (int k = tid; k < d; k += blockDim.x) {
        query[k] = Q[i * d + k];
    }
    __syncthreads();
    
    // Determine window boundaries
    int window_start = max(0, i - window_size);
    int window_end = min(M - 1, i + window_size);
    int window_len = window_end - window_start + 1;
    
    // Step 1: Compute attention scores (parallel over window)
    float sqrt_d = sqrtf((float)d);
    float max_score = -FLT_MAX;
    
    for (int j_idx = tid; j_idx < window_len; j_idx += blockDim.x) {
        int j = window_start + j_idx;
        float score = 0.0f;
        
        // Compute Q_i · K_j
        for (int k = 0; k < d; k++) {
            score += query[k] * K[j * d + k];
        }
        
        score /= sqrt_d;
        scores[j_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Find global max across threads
    __shared__ float max_score_shared;
    if (tid == 0) max_score_shared = -FLT_MAX;
    __syncthreads();
    
    if (max_score > -FLT_MAX) {
        atomicMaxFloat(&max_score_shared, max_score);
    }
    __syncthreads();
    
    // Step 2: Apply softmax
    float local_sum = 0.0f;
    for (int j_idx = tid; j_idx < window_len; j_idx += blockDim.x) {
        float exp_val = expf(scores[j_idx] - max_score_shared);
        scores[j_idx] = exp_val;
        local_sum += exp_val;
    }
    
    // Reduce sum across threads
    __shared__ float sum_exp_shared;
    if (tid == 0) sum_exp_shared = 0.0f;
    __syncthreads();
    
    atomicAdd(&sum_exp_shared, local_sum);
    __syncthreads();
    
    // Normalize
    for (int j_idx = tid; j_idx < window_len; j_idx += blockDim.x) {
        scores[j_idx] /= sum_exp_shared;
    }
    __syncthreads();
    
    // Step 3: Compute weighted sum of values (parallel over d)
    for (int k = tid; k < d; k += blockDim.x) {
        float result = 0.0f;
        for (int j_idx = 0; j_idx < window_len; j_idx++) {
            int j = window_start + j_idx;
            result += scores[j_idx] * V[j * d + k];
        }
        output[i * d + k] = result;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size) {
    // Choose kernel based on problem dimensions
    if (d <= 64 && window_size <= 32) {
        // Use shared memory version for smaller dimensions
        int threadsPerBlock = 128;  // Enough threads to handle d=64 and window operations
        int blocks = M;
        size_t shared_mem_size = (65 + d) * sizeof(float);
        
        slidingWindowAttentionShared<<<blocks, threadsPerBlock, shared_mem_size>>>(
            Q, K, V, output, M, d, window_size
        );
    } else {
        // Use general version for larger dimensions
        int threadsPerBlock = 256;
        int blocks = (M + threadsPerBlock - 1) / threadsPerBlock;
        
        slidingWindowAttention<<<blocks, threadsPerBlock>>>(
            Q, K, V, output, M, d, window_size
        );
    }
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // Handle error silently as we can't print in the test environment
    }
    
    cudaDeviceSynchronize();
}