#ifndef _GPU_SKMGEN_H
#define _GPU_SKMGEN_H

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    std::cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << std::endl; \
    exit(1); \
}

#define CUDA_CHECK_RETRY(call) \
while((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    std::cerr << "CUDA error calling \""#call"\", code is " << err << std::endl; \
    std::this_thread::sleep_for(100ms); \
}

#include "utilities.hpp"
#include "types.h"
#include "skmstore.hpp"
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
// using namespace std;

// hint: better to sort the reads by their lengths so that cuda will work faster

class PinnedCSR { // for one-time use only
public:
    char* reads_CSR;
    T_CSR_cap* reads_offs;
    T_read_cnt n_reads;
    T_CSR_cap size_capacity; // bytes = char unit
    PinnedCSR(std::vector<ReadPtr> &reads);
    ~PinnedCSR();
    char* get_reads_CSR() {return reads_CSR;}
    T_CSR_cap* get_reads_offs() {return reads_offs;}
    T_read_cnt get_n_reads() {return n_reads;}
    T_CSR_cap size() {return size_capacity;}
};


// host/global functions

size_t GPUReset (int did);

void GenSuperkmerGPU (PinnedCSR &pinned_reads, 
    const T_kvalue K_kmer, const T_kvalue P_minimizer, bool HPC, CUDAParams &gpars,
    const int SKM_partitions, std::vector<SKMStoreNoncon*> skm_partition_stores
    );
#endif