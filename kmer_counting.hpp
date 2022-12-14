#ifndef _KMER_COUNTING_HPP
#define _KMER_COUNTING_HPP

#include "types.h"
#include <vector>
#include "V2_superkmers.hpp"
using namespace std;

size_t kmc_counting_GPU (T_kvalue k,
    SKMStoreNoncon &skms_store, int gpuid,
    unsigned short kmer_min_freq, unsigned short kmer_max_freq,
    _out_ vector<T_kmc> &kmc_result_curthread);

#endif
