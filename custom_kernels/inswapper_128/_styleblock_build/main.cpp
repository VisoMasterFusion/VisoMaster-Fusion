#include <torch/extension.h>

int64_t create_hgemm_op(int64_t M, int64_t K, int64_t N);
int64_t create_hgemm_bias_op(int64_t M, int64_t K, int64_t N, torch::Tensor bias);
torch::Tensor run_hgemm_op(int64_t op_id, torch::Tensor A, torch::Tensor B);
void destroy_hgemm_op(int64_t op_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuBLASLt HGEMM extension for InSwapper-128 style-block convolutions";
    m.def("create_hgemm_op",      &create_hgemm_op,
          "Create plain GEMM op: C[M,N] = A[M,K] * B[K,N]");
    m.def("create_hgemm_bias_op", &create_hgemm_bias_op,
          "Create GEMM+bias op with cuBLASLt BIAS epilogue: "
          "C[M,N] = A[M,K] * B[K,N] + bias[M]");
    m.def("run_hgemm_op",         &run_hgemm_op,
          "Run op (bias is fused in op.desc if _bias variant was used)");
    m.def("destroy_hgemm_op",     &destroy_hgemm_op,
          "Release cuBLASLt resources for this op");
}
