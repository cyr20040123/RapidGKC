#ifndef _KMER_COUNTING_HPP
#define _KMER_COUNTING_HPP

#include "types.h"
#include <vector>
#include "V2_superkmers.hpp"
using namespace std;

size_t kmc_counting_GPU (T_kvalue k,
    SKMStoreNoncon &skms_store, CUDAParams &gpars,
    unsigned short kmer_min_freq, unsigned short kmer_max_freq,
    _out_ vector<T_kmc> &kmc_result_curthread,
    bool GPU_compression);

size_t kmc_counting_GPU_streams (T_kvalue k,
    vector<SKMStoreNoncon*> skms_stores, CUDAParams &gpars,
    unsigned short kmer_min_freq, unsigned short kmer_max_freq,
    _out_ vector<T_kmc> kmc_result_curthread [], int gpuid, 
    bool GPU_compression);

byte* load_SKM_from_file (SKMStoreNoncon &skms_store);

#endif
