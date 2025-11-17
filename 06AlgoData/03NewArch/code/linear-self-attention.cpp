#include <cuda_runtime.h>
#include <cfloat>

// Feature map: φ(x) = ELU(x) + 1 = max(0, x) + exp(min(0, x))
__device__ inline float phi(float x) {
    if (x > 0.0f) {
        return x + 1.0f;
    } else {
        return expf(x);
    }
}

// Kernel to apply feature map φ to a matrix
__global__ void applyFeatureMap(const float* input, float* output, int M, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * d;
    
    if (idx < total) {
        output[idx] = phi(input[idx]);
    }
}

// Kernel to compute φ(K)^T @ V (result is d×d)
__global__ void computeKTV(const float* phiK, const float* V, float* KTV, int M, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < d && col < d) {
        float sum = 0.0f;
        
        // K^T[row, :] @ V[:, col]
        // K^T[row, :] is K[:, row] (column row of original K)
        for (int i = 0; i < M; i++) {
            sum += phiK[i * d + row] * V[i * d + col];
        }
        
        KTV[row * d + col] = sum;
    }
}

// Kernel to compute sum of φ(K) rows (result is d-dimensional)
__global__ void computeSumPhiK(const float* phiK, float* sumK, int M, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < d) {
        double sum = 0.0;  // Use double for better precision with large M
        for (int i = 0; i < M; i++) {
            sum += (double)phiK[i * d + col];
        }
        sumK[col] = (float)sum;
    }
}

// Main kernel to compute final output
__global__ void computeLinearAttentionMain(
    const float* phiQ, const float* KTV, const float* sumK, 
    float* output, int M, int d
) {
    // Grid-stride loop pattern for better efficiency
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = M * d;
    
    for (int pos = idx; pos < total; pos += stride) {
        int row = pos / d;
        int col = pos % d;
        
        if (row < M && col < d) {
            double numerator = 0.0;
            double denominator = 0.0;
            
            // Compute numerator: φ(Q_row) @ KTV[:, col]
            for (int k = 0; k < d; k++) {
                numerator += (double)phiQ[row * d + k] * (double)KTV[k * d + col];
            }
            
            // Compute denominator: φ(Q_row) @ sumK
            for (int k = 0; k < d; k++) {
                denominator += (double)phiQ[row * d + k] * (double)sumK[k];
            }
            
            // Avoid division by zero
            if (denominator != 0.0) {
                output[row * d + col] = (float)(numerator / denominator);
            } else {
                output[row * d + col] = 0.0f;
            }
        }
    }
}

// Optimized version with better memory access pattern
__global__ void computeLinearAttentionOpt(
    const float* phiQ, const float* KTV, const float* sumK, 
    float* output, int M, int d
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= M) return;
    
    // Load sumK into shared memory (all threads cooperate)
    for (int i = tid; i < d; i += blockDim.x) {
        shared_mem[i] = sumK[i];
    }
    __syncthreads();
    
    // Compute denominator once for this row
    double denominator = 0.0;
    for (int k = 0; k < d; k++) {
        denominator += (double)phiQ[row * d + k] * (double)shared_mem[k];
    }
    
    // Each thread computes multiple output elements
    for (int col = tid; col < d; col += blockDim.x) {
        double numerator = 0.0;
        
        // Compute φ(Q_row) @ KTV[:, col]
        for (int k = 0; k < d; k++) {
            numerator += (double)phiQ[row * d + k] * (double)KTV[k * d + col];
        }
        
        if (denominator != 0.0) {
            output[row * d + col] = (float)(numerator / denominator);
        } else {
            output[row * d + col] = 0.0f;
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    // Allocate intermediate tensors
    float *phiQ, *phiK, *KTV, *sumK;
    
    cudaMalloc(&phiQ, M * d * sizeof(float));
    cudaMalloc(&phiK, M * d * sizeof(float));
    cudaMalloc(&KTV, d * d * sizeof(float));
    cudaMalloc(&sumK, d * sizeof(float));
    
    // Step 1: Apply feature map to Q and K
    int threadsPerBlock = 256;
    int blocks = (M * d + threadsPerBlock - 1) / threadsPerBlock;
    applyFeatureMap<<<blocks, threadsPerBlock>>>(Q, phiQ, M, d);
    applyFeatureMap<<<blocks, threadsPerBlock>>>(K, phiK, M, d);
    
    // Step 2: Compute φ(K)^T @ V (d×d matrix)
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((d + blockDim2D.x - 1) / blockDim2D.x,
                   (d + blockDim2D.y - 1) / blockDim2D.y);
    computeKTV<<<gridDim2D, blockDim2D>>>(phiK, V, KTV, M, d);
    
    // Step 3: Compute sum of φ(K) rows
    int sumBlocks = (d + threadsPerBlock - 1) / threadsPerBlock;
    computeSumPhiK<<<sumBlocks, threadsPerBlock>>>(phiK, sumK, M, d);
    
    // Wait for previous kernels to complete
    cudaDeviceSynchronize();
    
    // Step 4: Compute final output
    // Choose implementation based on problem size
    if (M <= 1000) {
        // For smaller M, use one block per row
        int threads = min(256, d);
        size_t shared_size = d * sizeof(float);
        computeLinearAttentionOpt<<<M, threads, shared_size>>>(
            phiQ, KTV, sumK, output, M, d
        );
    } else {
        // For larger M, use grid-stride approach
        int totalElements = M * d;
        int threads = 256;
        int blocks = min(65535, (totalElements + threads - 1) / threads);
        computeLinearAttentionMain<<<blocks, threads>>>(
            phiQ, KTV, sumK, output, M, d
        );
    }
    
    // Synchronize and cleanup
    cudaDeviceSynchronize();
    cudaFree(phiQ);
    cudaFree(phiK);
    cudaFree(KTV);
    cudaFree(sumK);
}