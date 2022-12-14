#ifndef _GKC_CUDA_H
#define _GKC_CUDA_H

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    exit(1); \
}

#define CUDA_CHECK_RETRY(call) \
while((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << endl; \
    this_thread::sleep_for(100ms); \
}

#include "types.h"
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
using namespace std;

struct CUDAParams {
    int NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK;
    int n_streams;//, items_per_stream;
    int device_id;
};

// hint: better to sort the reads by their lengths so that cuda will work faster

class PinnedCSR { // for one-time use only
public:
    char* reads_CSR; // TODO: make it int32
    T_CSR_cap* reads_offs;
    T_read_cnt n_reads;
    T_CSR_cap size_capacity; // bytes = char unit
    PinnedCSR(vector<string> &reads);
    PinnedCSR(vector<ReadPtr> &reads, bool keep_original=true);
    ~PinnedCSR();
    char* get_reads_CSR() {return reads_CSR;}
    T_CSR_cap* get_reads_offs() {return reads_offs;}
    T_read_cnt get_n_reads() {return n_reads;}
    T_CSR_cap size() {return size_capacity;}
};


// host/global functions
enum CountTask {SKMPartition, SKMPartWithPos, StoreMinimizerPos}; // 正常kmc, wtdbg2的kmc, minimap2的minimizer查找

// void CalcSKMPartSize_instream (T_read_cnt reads_cnt, T_read_len *superkmer_offs, 
//     T_CSR_cap *reads_offs, T_minimizer *minimizers, 
//     int n_partitions, int k, atomic<size_t> part_sizes[]);

void GPUReset (int did);

void GenSuperkmerGPU (PinnedCSR &pinned_reads, 
    const T_kvalue K_kmer, const T_kvalue P_minimizer, bool HPC, CUDAParams gpars, CountTask task,
    const int SKM_partitions, std::function<void(T_h_data)> process_func
    /*atomic<size_t> skm_part_sizes[]*/);

#endif