#ifndef _GPU_SKMGEN_H
#define _GPU_SKMGEN_H

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    std::cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << std::endl; \
    size_t avail, total; \
    cudaMemGetInfo(&avail, &total); \
    cerr << "Available memory: " << avail/1048576 << " Total memory: " << total/1048576 << endl; \
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
    // PinnedCSR(vector<string> &reads);
    PinnedCSR(std::vector<ReadPtr> &reads, bool keep_original=true);
    ~PinnedCSR();
    char* get_reads_CSR() {return reads_CSR;}
    T_CSR_cap* get_reads_offs() {return reads_offs;}
    T_read_cnt get_n_reads() {return n_reads;}
    T_CSR_cap size() {return size_capacity;}
};


// host/global functions
// enum CountTask {SKMPartition, SKMPartWithPos, StoreMinimizerPos}; // 正常kmc, wtdbg2的kmc, minimap2的minimizer查找

size_t GPUReset (int did);

void GenSuperkmerGPU (PinnedCSR &pinned_reads, 
    const T_kvalue K_kmer, const T_kvalue P_minimizer, bool HPC, CUDAParams &gpars,
    const int SKM_partitions, std::vector<SKMStoreNoncon*> skm_partition_stores, //std::function<void(T_h_data)> process_func /*must be thread-safe*/,
    int tid, int gpuid
    /*atomic<size_t> skm_part_sizes[]*/);

#endif