#ifndef _TYPES_H
#define _TYPES_H

#define _in_
#define _out_
#define _tmp_

#include <cstdlib> // for size_t

typedef size_t T_CSR_cap;
typedef unsigned long long T_skm_partsize;
typedef int T_CSR_count;

typedef unsigned short T_skm_len;

typedef int T_read_len;
typedef int T_read_cnt;
// typedef size_t T_CSR_cap;
typedef unsigned int T_minimizer; // support minimizer with max length = 16
const T_minimizer T_MM_MAX = (T_minimizer)(0xffffffffffffffff);
typedef unsigned char T_kvalue;
// typedef unsigned short T_spvalue;

struct ReadPtr {
    const char* read;
    T_read_len len;
};

#ifndef LONGERKMER
typedef unsigned long long T_kmer;
#endif
#ifdef LONGERKMER
typedef unsigned __int128 T_kmer;
#endif

typedef unsigned char byte;
const int BYTE_BASES = 3;
// typedef unsigned int QByte;

// host data
struct T_h_data {
    _in_ T_read_cnt reads_cnt;
    
    // Raw reads
    _in_ _out_ char *reads;             // will be only used to store HPC reads (if HPC is enabled)
    _in_ T_CSR_cap *reads_offs;         // reads are in CSR format so offset array is required
    _in_ _out_ T_read_len *read_len;    // len == len(d_read_offs) int
    
    // HPC reads info
    T_read_len *hpc_orig_pos;  // len == len(d_reads)      size_t  base original pos **in a read** (not in CSR)
    
    // SKMs
    _in_ T_skm_partsize *skm_part_bytes;
    _out_ T_skm_partsize *skm_cnt;
    _out_ T_skm_partsize *kmer_cnt;
    _in_ _out_ T_CSR_cap *skmpart_offs;
    size_t tot_skm_bytes;
    _out_ byte *skm_store_csr;

    // Compressed SKMs
    byte **skm_data;
    size_t *skmpart_compressed_bytes;
};

// device data
struct T_d_data {
    _in_ T_read_cnt reads_cnt;
    
    // Raw reads
    _in_ _out_ unsigned char *d_reads; // will be also used to store HPC reads (if HPC is enabled)
    _in_ T_CSR_cap *d_read_offs; // reads are in CSR format so offset array is required
    _in_ _out_ T_read_len *d_read_len;  // len == len(d_read_offs)  int
    
    // HPC reads info
    T_read_len *d_hpc_orig_pos;         // len == len(d_reads)      size_t  base original pos **in a read** (not in CSR)
    
    // Minimizers
    _out_ T_minimizer *d_minimizers;    // len == len(d_reads)      size_t
    _out_ T_read_len *d_superkmer_offs; // len == len(d_reads)      int     supermer_offs **in a read**
    // (for minimizer counting, not used now):
    _out_ T_kvalue *d_mm_pos;           // len == len(d_reads)      u_char  minimizer position in each window
    _out_ char *d_mm_strand;            // len == len(d_reads)      char    0 for forward, 1 for reverse complement, -1 for f==rc
    
    // SKMs
    T_skm_partsize *d_skm_part_bytes;  // len == SKM_part          ull     size of each skm partitions for the current stream batch (2-bit compression size, calc in GPU_GenSKM)
    T_skm_partsize *d_skm_cnt;          // len == SKM_part          ull     skm count of each part, for storing count (calc in GPU_GenSKM)
    T_skm_partsize *d_kmer_cnt;         // len == SKM_part          ull     kmer count of each partition, for kmer extraction memory allocation
    T_skm_partsize *d_store_pos;        // len == SKM_part          size_t  offs to store SKM in a partition of d_skm_store_csr
    byte *d_skm_store_csr;             // len == * part_size[p]    
    T_CSR_cap *d_skmpart_offs;          // len == * skm_cnt[p]      
};


typedef unsigned int T_kmer_cnt;
const T_kmer_cnt MAX_KMER_CNT = T_kmer_cnt(0xffffffffffffffff);
struct T_kmc{
    T_kmer kmer;
    T_kmer_cnt cnt;
};

#endif