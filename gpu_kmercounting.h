#ifndef _GPU_KMERCOUNTING_H
#define _GPU_KMERCOUNTING_H

#include "types.h"
#include <vector>
#include "skmstore.hpp"
#include "utilities.hpp"

// size_t kmc_counting_GPU (T_kvalue k,
//     SKMStoreNoncon &skms_store, CUDAParams &gpars,
//     unsigned short kmer_min_freq, unsigned short kmer_max_freq,
//     _out_ vector<T_kmc> &kmc_result_curthread);

size_t kmc_counting_GPU_streams (T_kvalue k,
    std::vector<SKMStoreNoncon*> skms_stores, CUDAParams &gpars,
    T_kmer_cnt kmer_min_freq, T_kmer_cnt kmer_max_freq,
    string result_file_prefix, size_t avg_kmer_cnt, int gpuid, int tid);

// u_char* load_SKM_from_file (SKMStoreNoncon &skms_store);

size_t GPUVram(int did);

#endif
