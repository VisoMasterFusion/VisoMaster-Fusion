// styleblock.cu — cuBLASLt HGEMM with optional BIAS epilogue.
//
// Computes C[M,N] = A[M,K] * B[K,N]  (+ bias[M] when BIAS epilogue active)
// for InSwapper-128 style-block convolutions.
//
// Row-major / col-major transposition recipe:
//   C_row[M,N] = A_row[M,K] * B_row[K,N]
//   In col-major: C_col[N,M] = B_col[N,K] * A_col[K,M]
//   A row-major tensor X[r,c] is stored identically to a col-major [c,r]:
//     A_row[M,K] → col-major [K,M], ldb=K  (passed as cuBLASLt "B")
//     B_row[K,N] → col-major [N,K], lda=N  (passed as cuBLASLt "A")
//     C_row[M,N] → col-major [N,M], ldc=N
//   BIAS epilogue adds bias[j] to column j of C_col = row j of C_row ✓
//
// Build (from project root with MSVC env active):
//   python custom_kernels/build_kernels.py
// Or manually:
//   run_with_msvc.bat custom_kernels/inswapper_128/_styleblock_build/build.py

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

// -- Per-op state -------------------------------------------------------------
struct GemmOp {
    cublasLtMatmulDesc_t   desc    = nullptr;
    cublasLtMatrixLayout_t lA      = nullptr;
    cublasLtMatrixLayout_t lB      = nullptr;
    cublasLtMatrixLayout_t lC      = nullptr;
    cublasLtMatmulAlgo_t   algo    {};
    bool                   has_algo = false;
    size_t                 ws_sz    = 0;
    int64_t                M = 0, K = 0, N = 0;
};

static std::unordered_map<int64_t, GemmOp*> g_ops;
static std::mutex  g_reg_mx;
static int64_t     g_next_id = 1;
static cublasLtHandle_t g_handle = nullptr;

static cublasLtHandle_t get_handle() {
    if (!g_handle) {
        if (cublasLtCreate(&g_handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("[StyleBlock] cublasLtCreate failed");
    }
    return g_handle;
}

// -- Public API ---------------------------------------------------------------
int64_t create_hgemm_op(int64_t M, int64_t K, int64_t N, const void* bias_ptr = nullptr) {
    cublasLtHandle_t h = get_handle();
    GemmOp* op = new GemmOp();
    op->M = M; op->K = K; op->N = N;
    op->ws_sz = 4 * 1024 * 1024;

    cublasLtMatmulDescCreate(&op->desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    if (bias_ptr) {
        cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(op->desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
        cudaDataType_t bdt = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(op->desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt));
        cublasLtMatmulDescSetAttribute(op->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
    }
    cublasLtMatrixLayoutCreate(&op->lA, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&op->lB, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&op->lC, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &op->ws_sz, sizeof(op->ws_sz));
    cublasLtMatmulHeuristicResult_t res[4];
    int n_res = 0;
    cublasLtMatmulAlgoGetHeuristic(h, op->desc, op->lA, op->lB, op->lC, op->lC, pref, 4, res, &n_res);
    cublasLtMatmulPreferenceDestroy(pref);
    if (n_res > 0) { op->algo = res[0].algo; op->has_algo = true; }

    std::lock_guard<std::mutex> lk(g_reg_mx);
    int64_t id = g_next_id++;
    g_ops[id] = op;
    return id;
}

int64_t create_hgemm_bias_op(int64_t M, int64_t K, int64_t N, torch::Tensor bias) {
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kHalf);
    return create_hgemm_op(M, K, N, bias.data_ptr());
}

torch::Tensor run_hgemm_op(int64_t op_id, torch::Tensor A, torch::Tensor B) {
    GemmOp* op_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_reg_mx);
        op_ptr = g_ops.at(op_id);
    }
    GemmOp& op = *op_ptr;
    auto C = torch::empty({op.M, op.N}, A.options());
    auto workspace = torch::empty({(int64_t)op.ws_sz}, torch::TensorOptions().device(A.device()).dtype(torch::kByte));
    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    cublasLtMatmul(get_handle(), op.desc, &alpha, B.data_ptr(), op.lA, A.data_ptr(), op.lB, &beta,
                   C.data_ptr(), op.lC, C.data_ptr(), op.lC, op.has_algo ? &op.algo : nullptr,
                   workspace.data_ptr(), op.ws_sz, at::cuda::getCurrentCUDAStream());
    return C;
}

void destroy_hgemm_op(int64_t op_id) {
    std::lock_guard<std::mutex> lk(g_reg_mx);
    auto it = g_ops.find(op_id);
    if (it != g_ops.end()) {
        GemmOp* op = it->second;
        if (op->desc) cublasLtMatmulDescDestroy(op->desc);
        if (op->lA)   cublasLtMatrixLayoutDestroy(op->lA);
        if (op->lB)   cublasLtMatrixLayoutDestroy(op->lB);
        if (op->lC)   cublasLtMatrixLayoutDestroy(op->lC);
        delete op;
        g_ops.erase(it);
    }
}
