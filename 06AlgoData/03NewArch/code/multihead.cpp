#include <cuda_runtime.h>
#include <float.h>

// Kernel to compute Q @ K^T for a specific head
__global__ void computeAttentionScores(
    const float* Q, const float* K, float* scores,
    int N, int d_model, int h, int d_k, int head_idx
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        int head_offset = head_idx * d_k;
        
        // Compute dot product between Q[row] and K[col] for this head
        for (int i = 0; i < d_k; i++) {
            float q_val = Q[row * d_model + head_offset + i];
            float k_val = K[col * d_model + head_offset + i];
            sum += q_val * k_val;
        }
        
        // Scale by 1/sqrt(d_k)
        sum /= sqrtf((float)d_k);
        
        // Store in scores matrix for this head
        scores[head_idx * N * N + row * N + col] = sum;
    }
}

// Kernel to apply softmax along rows
__global__ void applySoftmax(float* scores, int N, int num_heads) {
    int head = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (head < num_heads && row < N) {
        float* row_scores = scores + head * N * N + row * N;
        
        // Find max value in the row for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, row_scores[i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            row_scores[i] = expf(row_scores[i] - max_val);
            sum += row_scores[i];
        }
        
        // Normalize
        for (int i = 0; i < N; i++) {
            row_scores[i] /= sum;
        }
    }
}

// Kernel to multiply attention weights with V and concatenate
__global__ void applyAttentionAndConcat(
    const float* attention, const float* V, float* output,
    int N, int d_model, int h, int d_k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int head = blockIdx.x;
    int col_in_head = threadIdx.x;
    
    if (row < N && head < h && col_in_head < d_k) {
        float sum = 0.0f;
        float* att_row = (float*)attention + head * N * N + row * N;
        
        // Compute weighted sum of V values
        for (int i = 0; i < N; i++) {
            float v_val = V[i * d_model + head * d_k + col_in_head];
            sum += att_row[i] * v_val;
        }
        
        // Write to output in concatenated form
        output[row * d_model + head * d_k + col_in_head] = sum;
    }
}

// Alternative: Single kernel for attention @ V (more efficient for small matrices)
__global__ void applyAttention(
    const float* attention, const float* V, float* output,
    int N, int d_model, int h, int d_k, int head_idx
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < d_k) {
        float sum = 0.0f;
        float* att_row = (float*)attention + head_idx * N * N + row * N;
        int v_col = head_idx * d_k + col;
        
        for (int i = 0; i < N; i++) {
            sum += att_row[i] * V[i * d_model + v_col];
        }
        
        output[row * d_model + v_col] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int d_k = d_model / h;
    
    // Allocate memory for attention scores
    float* attention_scores;
    cudaMalloc(&attention_scores, sizeof(float) * h * N * N);
    
    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (N + blockDim.y - 1) / blockDim.y);
    
    // Step 1: Compute attention scores for each head
    for (int head = 0; head < h; head++) {
        computeAttentionScores<<<gridDim, blockDim>>>(
            Q, K, attention_scores, N, d_model, h, d_k, head
        );
    }
    
    // Step 2: Apply softmax to attention scores
    dim3 softmaxGrid((N + 255) / 256, h);
    applySoftmax<<<softmaxGrid, 256>>>(attention_scores, N, h);
    
    // Step 3: Apply attention to values and concatenate
    if (d_k <= 32) {
        // Use single kernel for small d_k
        dim3 blockDim3(d_k, 16);
        dim3 gridDim3(h, (N + blockDim3.y - 1) / blockDim3.y);
        applyAttentionAndConcat<<<gridDim3, blockDim3>>>(
            attention_scores, V, output, N, d_model, h, d_k
        );
    } else {
        // Use separate kernels for larger d_k
        dim3 blockDim2(16, 16);
        dim3 gridDim2((d_k + blockDim2.x - 1) / blockDim2.x,
                      (N + blockDim2.y - 1) / blockDim2.y);
        
        for (int head = 0; head < h; head++) {
            applyAttention<<<gridDim2, blockDim2>>>(
                attention_scores, V, output, N, d_model, h, d_k, head
            );
        }
    }
    
    // Synchronize and clean up
    cudaDeviceSynchronize();
    cudaFree(attention_scores);
}