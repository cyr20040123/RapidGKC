#ifndef _CPU_FUNCS_H
#define _CPU_FUNCS_H

#include <vector>
#include "types.h"
#include "V2_superkmers.hpp"

void GenSuperkmerCPU (std::vector<ReadPtr> &reads,
    const T_kvalue K_kmer, const T_kvalue P_minimizer, bool HPC, 
    const int SKM_partitions, std::vector<SKMStoreNoncon*> skm_partition_stores);

#endif