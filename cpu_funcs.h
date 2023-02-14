#ifndef _CPU_FUNCS_H
#define _CPU_FUNCS_H

#include <vector>
#include "types.h"
#include "skmstore.hpp"

void GenSuperkmerCPU (std::vector<ReadPtr> &reads,
    const T_kvalue K_kmer, const T_kvalue P_minimizer, bool HPC, 
    const int SKM_partitions, std::vector<SKMStoreNoncon*> skm_partition_stores);

size_t KmerCountingCPU(T_kvalue k,
    SKMStoreNoncon *skms_store,
    T_kmer_cnt kmer_min_freq, T_kmer_cnt kmer_max_freq,
    _out_ vector<T_kmc> &kmc_result_curpart, int tid);
#endif