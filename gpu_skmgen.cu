#define _in_
#define _out_

// #define KERNEL_TIME_MEASUREMENT

#define FILTER_KERNEL new_filter2 // modify this to change filter: mm_filter, sign_filter, new_filter, new_filter2
#define STR1(R)  #R
#define STR(R) STR1(R)

#include "gpu_skmgen.h"
#include "types.h"
#include "utilities.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
// #include "nvcomp/gdeflate.hpp"
// #include "nvcomp.hpp"

#include <vector>
#include <string>
#include <thread>
#include <future>
#include <functional>
#include <iostream>

using namespace std;
// using namespace nvcomp;

__device__ __constant__ static const unsigned char d_basemap[256] = {
    255, 255, 255, 255, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 0, 255, 1, 255, 255, 255, 2, // 64..71
    255, 255, 255, 255, 255, 255, 255, 255, // 72..79
    255, 255, 255, 255, 3, 0, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 0, 255, 1, 255, 255, 255, 2, // 96..103
    255, 255, 255, 255, 255, 255, 255, 255, // 104..111
    255, 255, 255, 255, 3, 0, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

__device__ __constant__ static const unsigned char d_basemap_compl[256] = { // complement base
    255, 255, 255, 255, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 3, 255, 2, 255, 255, 255, 1, // 64..71
    255, 255, 255, 255, 255, 255, 255, 255, // 72..79
    255, 255, 255, 255, 0, 3, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 3, 255, 2, 255, 255, 255, 1, // 96..103
    255, 255, 255, 255, 255, 255, 255, 255, // 104..111
    255, 255, 255, 255, 0, 3, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

// raw read is not a significant VRAM usage, no need for 2-bit encoding
// the majority VRAM usage is caused by minimizer (positions) etc...

extern Logger *logger;

// =================================================
// ================ CLASS PinnedCSR ================
// =================================================
    PinnedCSR::PinnedCSR(vector<ReadPtr> &reads) { // for sorting CSR (order the pointers as the sorting result)
        this->n_reads = reads.size();
        size_capacity = 0;
        for (const ReadPtr &read_ptr: reads) {
            size_capacity += read_ptr.len;
        } // about cudaHostAlloc https://zhuanlan.zhihu.com/p/188246455
        CUDA_CHECK(cudaHostAlloc((void**)(&reads_offs), (this->n_reads+1)*sizeof(T_CSR_cap), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)(&reads_CSR), size_capacity+1, cudaHostAllocDefault));
        char *cur_ptr = reads_CSR;
        T_CSR_cap *offs_ptr = reads_offs;
        *offs_ptr = 0;
        for (const ReadPtr &read_ptr: reads) {
            memcpy(cur_ptr, read_ptr.read, read_ptr.len);
            cur_ptr += read_ptr.len;
            offs_ptr++;
            *offs_ptr = *(offs_ptr-1) + read_ptr.len;
        }
    }
    PinnedCSR::~PinnedCSR() {
        CUDA_CHECK(cudaFreeHost(reads_offs));
        CUDA_CHECK(cudaFreeHost(reads_CSR));
    }

__global__ void GPU_HPCEncoding (
    _in_ T_read_cnt d_reads_cnt, _out_ T_read_len *d_read_len, 
    _in_ _out_ unsigned char *d_reads, _in_ T_CSR_cap *d_read_offs, 
    bool HPC, _out_ T_read_len *d_hpc_orig_pos = nullptr) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;
    if (!HPC) { // only calculate read_len in global memory (optional but essential for HPC=true)
        for (T_read_cnt rid = tid; rid < d_reads_cnt; rid += n_t) {
            d_read_len[rid] = d_read_offs[rid+1] - d_read_offs[rid];
        }
        __syncthreads();
        return;
    }
    
    for (T_read_cnt rid = tid; rid < d_reads_cnt; rid += n_t) {
        T_read_len read_len = d_read_offs[rid+1] - d_read_offs[rid];
        T_read_len last_idx = 0, hpc_arr_idx = d_read_offs[tid], j;
        d_hpc_orig_pos[hpc_arr_idx] = 0;
        for (T_read_len i = 1; i < read_len; i++) {
            j = i + d_read_offs[rid];
            last_idx += (i-last_idx) * (d_reads[j] != d_reads[j-1]);
            hpc_arr_idx += (d_reads[j] != d_reads[j-1]);
            d_hpc_orig_pos[hpc_arr_idx] = last_idx;
            d_reads[hpc_arr_idx] = d_reads[j];
        }
        d_read_len[rid] = hpc_arr_idx + 1 - d_read_offs[rid];
    }
    return;
}

// ======== Minimizer Functions ========
// traditional minimizer
__device__ __forceinline__ bool mm_filter(T_minimizer mm, int p) {
    // return mm%101>80; // 20.36
    // return ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100) /*ACA*/; // 19.94
    // return ((mm >> (p-2)*2) & 0b11) + ((mm >> (p-3)*2) & 0b11) + ((mm >> (p-1)*2) & 0b11); // 20.03
    // return (mm >> (p-3)*2) * ((mm >> (p-5)*2) & 0b111111); // 20.02
    // return ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100) /*ACA*/ & (mm >> ((p-3)*2) != 0b001000); // 19.92
    // int i=0;
    // int s=0;
    // for (i=1; i<3; i++) {
    //     s += (mm >> ((p-2)*2)) > (mm>>((p-2-i))&0b1111);
    // }
    // return s==0;
    return true;
}
// new design: 2nd/3rd不都为a
__device__ __forceinline__ bool new_filter(T_minimizer mm, int p) {
    return ((mm >> (p-2)*2) & 0b11) + ((mm >> (p-3)*2) & 0b11);
}
__device__ __forceinline__ bool new_filter2(T_minimizer mm, int p) {
    // return ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100) /*ACA*/; //& (mm >> ((p-3)*2) != 0b001000) /*AGA*/;
    return ((((mm >> ((p-3)*2)) & 0b111011) != 0/*no AAA ACA*/) & ((mm & 0b111111) != 0/*no AAA at last*/));
}
// KMC2 signature
__device__ bool sign_filter(T_minimizer mm, int p) {
    T_minimizer t = mm;
    bool flag = true;
    for (int ii = 0; ii < p-2; ii ++) {
        flag *= ((t & 0b1111) != 0);
        t = t >> 2;
    }
    // printf("%d Minimizer: %x\n", flag & ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100), mm);
    return flag & (((mm >> ((p-3)*2)) & 0b111011) != 0); /*AAA ACA*/;
}
/*
 * [INPUT]  d_reads in [(Read#0)['A','C','T','G',...], (Read#1)['A','C','T','G',...]]
 * [OUTPUT] d_minimizers in [(Read#0)[mm1, mm?, mm?, ...], (Read#1)...]
 * all thread do one read at the same time with coalesced global memory access
 */
__global__ void GPU_GenMinimizer(
    _in_ T_read_cnt d_reads_cnt, _in_ T_read_len *d_read_len, 
    _in_ unsigned char *d_reads, _in_ T_CSR_cap *d_read_offs, 
    _out_ T_minimizer *d_minimizers, 
    const T_kvalue K_kmer, const T_kvalue P_minimizer) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;
    int i, j, cur_kmer_i;
    T_minimizer mm_mask = T_MM_MAX >> (sizeof(T_minimizer)*8 - 2*P_minimizer);
    T_minimizer mm_set; // selected minimizer
    T_minimizer mm, new_mm, mm_rc, new_mm_rc; // rc for reverse complement
    
    bool mm_check; // whether is a legal minimizer/signature (filtered by mm_filter)

    for (i=0; i<d_reads_cnt; i++) {
        unsigned char *read = &d_reads[d_read_offs[i]]; // current read
        T_minimizer *minimizer_saving = &(d_minimizers[d_read_offs[i]]);
        T_read_len len = d_read_len[i];
        for (cur_kmer_i=tid; cur_kmer_i <= len-K_kmer; cur_kmer_i+=n_t) { // Coalesced Access
            // gen the first p-mer:
            new_mm = 0;
            for (j = cur_kmer_i; j < cur_kmer_i + P_minimizer; j++) {
                new_mm = (new_mm << 2) | d_basemap[*(read+j)];
            }
            mm_check = FILTER_KERNEL(new_mm, P_minimizer);
            mm = new_mm * mm_check + mm_mask * (!mm_check); // if not a minimizer, let it be maximal (no minimizer can be maximal because canonical)
            
            // gen the first RC p-mer:
            new_mm_rc = 0;
            for (j = cur_kmer_i + P_minimizer - 1; j >= cur_kmer_i; j--) {
                new_mm_rc = (new_mm_rc << 2) | d_basemap_compl[*(read+j)];
            }
            mm_check = FILTER_KERNEL(new_mm_rc, P_minimizer);
            mm_rc = new_mm_rc * mm_check + mm_mask * (!mm_check);

            mm_set = (mm_rc < mm) * mm_rc + (mm_rc >= mm) * mm;////////////
            
            // gen the next p-mers:
            for (j = cur_kmer_i + P_minimizer; j < cur_kmer_i + K_kmer; j++) {
                // gen new minimizers
                new_mm = ((new_mm << 2) | d_basemap[*(read+j)]) & mm_mask;
                new_mm_rc = (new_mm_rc >> 2) | (d_basemap_compl[*(read+j)] << (P_minimizer*2-2));
                // check new minimizers
                mm_check = FILTER_KERNEL(new_mm, P_minimizer);
                mm = new_mm * mm_check + mm * (!mm_check);
                mm_check = FILTER_KERNEL(new_mm_rc, P_minimizer);
                mm_rc = new_mm_rc * mm_check + mm_rc * (!mm_check);
                // set the best minimizer
                mm_set = (mm_set < mm) * mm_set + (mm_set >= mm) * mm;
                mm_set = (mm_set < mm_rc) * mm_set + (mm_set >= mm_rc) * mm_rc;//////////
            }
            minimizer_saving[cur_kmer_i] = mm_set;
        }
    }
    return;
}

__device__ __forceinline__ int _hash_partition (T_minimizer mm, int SKM_partitions) {
    return (~mm) % SKM_partitions;
}
__device__ inline T_skm_len _skm_bytes_required (T_read_len beg, T_read_len end, int k) {
    return sizeof(T_skm_len) + ((beg%4) + end+(k-1)-beg + 3) / 4;
    // return ((beg%3) + end+(k-1)-beg + 3) / 3; // +3 because skm_3x requires an extra empty byte
}
/* [INPUT]  data.minimizers in [[mm1, mm1, mm2, mm3, ...], ...]
 * [OUTPUT] data.superkmer_offs in [[0, 2, 3, ...], ...]
 * [OUTPUT] data.d_skm_part_bytes (size in bytes of each partition)
 * [OUTPUT] data.d_skm_cnt (skm count of each partition)
*/
__global__ void GPU_GenSKMOffs(
    _in_ T_read_cnt d_reads_cnt, _in_ T_read_len *d_read_len, 
    _in_ T_CSR_cap *d_read_offs, 
    _in_ T_minimizer *d_minimizers,
    _out_ T_read_len *d_superkmer_offs,
    _out_ T_skm_partsize *d_skm_part_bytes,
    _out_ T_skm_partsize *d_skm_cnt,
    _out_ T_skm_partsize *d_kmer_cnt,
    const T_kvalue K_kmer, const T_kvalue P_minimizer, const int SKM_partitions) {
        
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;

    int p_i;
    bool new_skm;
    T_read_len i;

    // reset shared arrays for skm and kmer counters
    extern __shared__ unsigned int shared_arr[];
    unsigned int *p_skm_cnt = &shared_arr[0];
    unsigned int *p_kmer_cnt = &shared_arr[SKM_partitions];
    for (i=threadIdx.x; i<SKM_partitions; i+=blockDim.x) p_skm_cnt[i] = 0, p_kmer_cnt[i] = 0;
    __syncthreads();
    
    // each thread processes one read at a time
    for (T_read_cnt rid = tid; rid < d_reads_cnt; rid += n_t) {
        T_read_len len = d_read_len[rid];                               // current read length
        T_minimizer *minimizers = &(d_minimizers[d_read_offs[rid]]);    // minimizer list pointer
        T_read_len *skm = &d_superkmer_offs[d_read_offs[rid]];          // superkmer list pointer
        T_read_len last_skm_pos = 0, skm_count = 0;                     // position of the last minimizer; superkmer count
        skm[0] = 0;
        for (i = 1; i <= len-K_kmer; i++) {
            new_skm = (minimizers[i] != minimizers[i-1]/*||i-last_skm_pos+K_kmer >= MAX_SKM_LEN*/);
            skm_count += new_skm; // current minimizer != last minimizer, new skm generated
            last_skm_pos = (!new_skm) * last_skm_pos + (new_skm) * i;
            skm[skm_count] = last_skm_pos; // skm #skm_count (begins from 1) ends at last_skm_pos
            // count skm part sizes
            if (new_skm) {
                p_i = _hash_partition(minimizers[i-1], SKM_partitions);
                atomicAdd(&d_skm_part_bytes[p_i], _skm_bytes_required(skm[skm_count-1], skm[skm_count], K_kmer));
                atomicAdd(&p_skm_cnt[p_i], 1);
                atomicAdd(&p_kmer_cnt[p_i], skm[skm_count] - skm[skm_count-1]);
            }
        }
        // process the last skm
        skm_count += 1;
        skm[skm_count] = len-K_kmer+1;
        p_i = _hash_partition(minimizers[i-1], SKM_partitions);
        atomicAdd(&d_skm_part_bytes[p_i], _skm_bytes_required(skm[skm_count-1], skm[skm_count], K_kmer));
        atomicAdd(&p_skm_cnt[p_i], 1);
        atomicAdd(&p_kmer_cnt[p_i], skm[skm_count] - skm[skm_count-1]);
        
        // set the ending 0 and store skm_count at skm[len-1]
        skm[skm_count+1] = 0;
        skm[len-1] = skm_count;
    }

    __syncthreads();
    for (i=threadIdx.x; i<SKM_partitions; i+=blockDim.x) {
        atomicAdd(&d_skm_cnt[i], p_skm_cnt[i]);
        atomicAdd(&d_kmer_cnt[i], p_kmer_cnt[i]);
    }
    return;
}

__global__ void GPU_ReadCompression(_in_ _out_ unsigned char *d_reads, _in_ T_CSR_cap *d_read_offs, _in_ T_read_len *d_read_len, _in_ T_read_cnt d_reads_cnt) {
    
    unsigned char* cur_read;
    T_read_len len;
    // uchar3 c4; // 3 = BYTE_BASES
    // uchar4 c4;
    u_char tmp;
    T_read_len i, j, last_byte_bases;
    
    // each block process one read:
    for (T_read_cnt i_read = blockIdx.x; i_read < d_reads_cnt; i_read += gridDim.x) {
        len = d_read_len[i_read];
        cur_read = (&(d_reads[d_read_offs[i_read]]));
        // one thread process 4 bases at a time:
        for (i = threadIdx.x * BYTE_BASES; i - threadIdx.x*BYTE_BASES <= len; i += blockDim.x * BYTE_BASES) { // Coalesced Access
            // why "- threadIdx.x" in ending condition? - To ensure each thread will run the same time for __syncthreads().
            if (i + BYTE_BASES <= len) {
                // TO-DO: [experiment] compare with below (check if uchar3 is faster than three single vars)
                // tmp = (d_basemap[cur_read[i]] << 4) | (d_basemap[cur_read[i+1]] << 2) | (d_basemap[cur_read[i+2]]) | 0b11000000;
                // c4 = *(reinterpret_cast<uchar4*>(&cur_read[i])); // load 3 bases at a time
                // c4.x = d_basemap[c4.x]; c4.y = d_basemap[c4.y]; c4.z = d_basemap[c4.z]; c4.w = d_basemap[c4.w]; // convert 4 bases to 2-bit
                // tmp = (c4.x << 6) | (c4.y << 4) | (c4.z << 2) | (c4.w); // generate byte
                tmp = (d_basemap[cur_read[i]] << 6) | (d_basemap[cur_read[i+1]] << 4) | (d_basemap[cur_read[i+2]] << 2) | (d_basemap[cur_read[i+3]]);
            }
            __syncthreads(); // avoid overwriting before raw read base is loaded
            if (i + BYTE_BASES <= len) {
                cur_read[i/BYTE_BASES] = tmp;
            }
        }
        i -= blockDim.x*BYTE_BASES;
        if ((i < len) & (i > len-BYTE_BASES)) { // process the last byte, only 1 thread should be available here
            last_byte_bases = len-i;
            tmp = 0;
            for (j=0; j<last_byte_bases; j++) {
                tmp |= d_basemap[cur_read[i+j]] << (6-j*2);
            }
            cur_read[i/BYTE_BASES] = tmp;
        }
    }
    return;
}

/// @brief Each block process the skms of one read so block size should not be too large.
__global__ void GPU_ExtractSKM (
    _in_ T_read_cnt d_reads_cnt, _in_ T_read_len *d_read_len, _in_ T_CSR_cap *d_read_offs, _in_ unsigned char *d_reads,
    _in_ T_minimizer *d_minimizers,
    _in_ T_read_len *d_skm_offs_inread,
    _in_ T_skm_partsize *d_store_pos, /*_in_ T_skm_partsize *d_skm_cnt, */_out_ u_char *d_skm_store_csr, _in_ T_CSR_cap *d_skmpart_offs, 
    // _in_ T_skm_partsize *d_len_store_pos, _out_ T_skm_len *d_skm_lengths, _in_ T_CSR_cap *d_skmlen_offs, 
    const T_kvalue K_kmer, const T_kvalue P_minimizer, const int SKM_partitions
) {
    T_read_len *cur_read_skm_offs;      // skm offs pointer of current read
    u_char *cur_read;
    T_read_len cur_read_len;            // length in bases
    
    int partition;                      // the partition of the current skm
    T_skm_len skm_size_bytes;           // bytes of current skm
    T_skm_partsize cur_skm_store_pos;   // where to store the current skm (partition's own offs)

    // each block process one read
    for (T_read_cnt rid = blockIdx.x; rid < d_reads_cnt; rid += gridDim.x) {
        cur_read_len = d_read_len[rid];
        cur_read_skm_offs = &(d_skm_offs_inread[d_read_offs[rid]]);
        cur_read = &(d_reads[d_read_offs[rid]]);
        // each thread process one skm
        for (T_read_len i_skm = threadIdx.x+1; i_skm <= cur_read_skm_offs[cur_read_len-1]; i_skm += blockDim.x) { // cur_read_skm_offs[cur_read_len-1]: number of skms of this read
            // printf("%d|%d\n",rid,i_skm);
            // for each skm of the current read, cur_read_skm_offs[0] == 0, loop begins from 1
            // -- store skm --
            partition = _hash_partition (d_minimizers[d_read_offs[rid] + cur_read_skm_offs[i_skm-1]], SKM_partitions);
            skm_size_bytes = _skm_bytes_required(cur_read_skm_offs[i_skm-1], cur_read_skm_offs[i_skm], K_kmer); // beg, end, k
            cur_skm_store_pos = atomicAdd(&d_store_pos[partition], skm_size_bytes); // assign space to store current skm
            d_skm_store_csr[d_skmpart_offs[partition] + cur_skm_store_pos]
             = (T_skm_len)((cur_read_skm_offs[i_skm] - cur_read_skm_offs[i_skm-1] + K_kmer - 1) | ((cur_read_skm_offs[i_skm-1] % BYTE_BASES) << 14));
            memcpy(&d_skm_store_csr[d_skmpart_offs[partition] + cur_skm_store_pos + sizeof(T_skm_len)], &cur_read[cur_read_skm_offs[i_skm-1]/BYTE_BASES], skm_size_bytes);
            if (cur_read_skm_offs[i_skm] - cur_read_skm_offs[i_skm-1] + K_kmer - 1 > 256) printf(" = %d\n", cur_read_skm_offs[i_skm] - cur_read_skm_offs[i_skm-1] + K_kmer - 1);
            assert((cur_read_skm_offs[i_skm] - cur_read_skm_offs[i_skm-1] + K_kmer - 1) < 8192);
            // printf("%u\n", d_skm_lengths[d_skmlen_offs[partition] + cur_skmlen_store_pos] & 0b0011111111111111);
        }
    }
    return;
}

/// @brief Set device CSR offsets begin from 0.
/// @param d_reads_cnt 
/// @param d_read_offs 
/// @param add [0] for setting to zero, [positive] value for adding back
/// @return
__global__ void MoveOffset(_in_ T_read_cnt d_reads_cnt, _in_ _out_ T_CSR_cap *d_read_offs, long long add=0) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;
    add = add - (add==0) * d_read_offs[0];
    for (T_read_cnt rid = tid; rid <= d_reads_cnt; rid += n_t) {
        d_read_offs[rid] += add;
    }
    return;
}

__host__ size_t GPUReset(int did) {
    cerr<<"now reset GPU "<<did<<endl;
    // do not call it after host malloc
    CUDA_CHECK(cudaSetDevice(did));
    CUDA_CHECK(cudaDeviceReset());
    // CUDA_CHECK(cudaInitDevice(did, ));
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    CUDA_CHECK(cudaDeviceSynchronize());
    return avail;
}

// provide pinned_reads from the shortest to the longest read
__host__ void GenSuperkmerGPU (PinnedCSR &pinned_reads, 
    const T_kvalue K_kmer, const T_kvalue P_minimizer, bool HPC, CUDAParams &gpars,
    const int SKM_partitions, vector<SKMStoreNoncon*> skm_partition_stores
    ) {
    
    int time_all=0, time_filter=0;

    int gpuid = (gpars.device_id++) % gpars.n_devices;
    CUDA_CHECK(cudaSetDevice(gpuid));
    
    cudaStream_t streams[gpars.n_streams];
    T_d_data gpu_data[gpars.n_streams];
    T_h_data host_data[gpars.n_streams];
    T_CSR_cap batch_size[gpars.n_streams];      // raw reads size in bytes of the current batch
    T_read_cnt bat_beg_read[gpars.n_streams];

    int i, started_streams;
    for (i=0; i<gpars.n_streams; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    
    T_read_cnt items_per_stream = gpars.BpG1 * gpars.TpB1 * gpars.items_stream_mul;
    T_read_cnt cur_read = 0, end_read;
    i = 0; // i for which stream
    string logs = "   GPU "+to_string(gpuid)+":";
    while (cur_read < pinned_reads.n_reads) {

        for (i = 0; i < gpars.n_streams && cur_read < pinned_reads.n_reads; i++, cur_read += items_per_stream) {
            // i: which stream

            bat_beg_read[i] = cur_read;
            end_read = min(cur_read + items_per_stream, pinned_reads.n_reads); // the last read in this stream batch
            host_data[i].reads_cnt = gpu_data[i].reads_cnt = end_read-cur_read;
            batch_size[i] = pinned_reads.reads_offs[end_read] - pinned_reads.reads_offs[cur_read]; // read size in bytes
            // logger->log("GPU "+to_string(gpuid)+" Stream "+to_string(i)+":\tread count = "+to_string(gpu_data[i].reads_cnt));
            logs += "\tS "+to_string(i)+"  #Reads "+to_string(gpu_data[i].reads_cnt);

            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            // ---- cudaMalloc ----
            // reads (data, offs, len, hpc), minmers, skms (offs, part_byte, cnt)
            // ~ 5000 reads / GB
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_reads), sizeof(u_char) * (batch_size[i]+8), streams[i]));// +8 for uchar4 access overflow // 8192 threads(reads) * 20 KB/read     = 160MB VRAM
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_read_offs), sizeof(T_CSR_cap) * (gpu_data[i].reads_cnt+1), streams[i]));    // 8192 threads(reads) * 8 B/read       =  64MB VRAM
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_read_len), sizeof(T_read_len) * (gpu_data[i].reads_cnt), streams[i]));      // 8192 threads(reads) * 4 B/read       =  32MB VRAM
            if (HPC) {// cost a lot VRAM
                CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_hpc_orig_pos), sizeof(T_read_len) * (batch_size[i]), streams[i]));      // 8192 threads(reads) * 20K * 4B/read  = 640MB VRAM
            } else {
                gpu_data[i].d_hpc_orig_pos = nullptr;
            }
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_minimizers), sizeof(T_minimizer) * (batch_size[i]), streams[i]));           // 8192 threads(reads) * 20K * 4B/read  = 640MB VRAM
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_superkmer_offs), sizeof(T_read_len) * (batch_size[i]), streams[i]));        // 8192 threads(reads) * 20K * 4B/read  = 640MB VRAM
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_skm_part_bytes), sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
            CUDA_CHECK(cudaMemsetAsync(gpu_data[i].d_skm_part_bytes, 0, sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_skm_cnt), sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_kmer_cnt), sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
            CUDA_CHECK(cudaMemsetAsync(gpu_data[i].d_skm_cnt, 0, sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
            CUDA_CHECK(cudaMemsetAsync(gpu_data[i].d_kmer_cnt, 0, sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
            
            // ---- copy raw reads to device ----
            CUDA_CHECK(cudaMemcpyAsync(gpu_data[i].d_reads, &(pinned_reads.reads_CSR[pinned_reads.reads_offs[cur_read]]), batch_size[i], cudaMemcpyHostToDevice, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(gpu_data[i].d_read_offs, &(pinned_reads.reads_offs[cur_read]), sizeof(T_CSR_cap) * (gpu_data[i].reads_cnt+1), cudaMemcpyHostToDevice, streams[i]));
            
            // ---- GPU gen skm ----
            #ifdef KERNEL_TIME_MEASUREMENT
            WallClockTimer wct;
            #endif
            MoveOffset<<<gpars.BpG1, gpars.TpB1, 0, streams[i]>>>(
                gpu_data[i].reads_cnt, gpu_data[i].d_read_offs, 0
            );
            #ifdef KERNEL_TIME_MEASUREMENT
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            #endif
            
            GPU_HPCEncoding<<<gpars.BpG1, gpars.TpB1, 0, streams[i]>>>(
                gpu_data[i].reads_cnt,  gpu_data[i].d_read_len, 
                gpu_data[i].d_reads,    gpu_data[i].d_read_offs, 
                HPC,                    gpu_data[i].d_hpc_orig_pos
            );
            #ifdef KERNEL_TIME_MEASUREMENT
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            
            WallClockTimer wct2;
            #endif
            GPU_GenMinimizer<<<gpars.BpG1, gpars.TpB1, 0, streams[i]>>>(
                gpu_data[i].reads_cnt,  gpu_data[i].d_read_len,
                gpu_data[i].d_reads,    gpu_data[i].d_read_offs,
                gpu_data[i].d_minimizers, 
                K_kmer, P_minimizer
            );
            #ifdef KERNEL_TIME_MEASUREMENT
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            time_filter += wct2.stop(true);
            #endif

            GPU_GenSKMOffs<<<gpars.BpG1, gpars.TpB1, 2*SKM_partitions*sizeof(unsigned int), streams[i]>>>(
                gpu_data[i].reads_cnt, gpu_data[i].d_read_len, gpu_data[i].d_read_offs, 
                gpu_data[i].d_minimizers,
                gpu_data[i].d_superkmer_offs,
                gpu_data[i].d_skm_part_bytes,
                gpu_data[i].d_skm_cnt,
                gpu_data[i].d_kmer_cnt,
                K_kmer, P_minimizer, SKM_partitions
            );
            #ifdef KERNEL_TIME_MEASUREMENT
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            time_all += wct.stop(true);
            #endif
            
            GPU_ReadCompression<<<gpars.BpG1, gpars.TpB1, 0, streams[i]>>>(
                gpu_data[i].d_reads, gpu_data[i].d_read_offs, gpu_data[i].d_read_len, gpu_data[i].reads_cnt
            );
            
            // ---- copy skm partition sizes to host ----
            host_data[i].skm_part_bytes = new T_skm_partsize[SKM_partitions];//1
            host_data[i].skm_cnt = new T_skm_partsize[SKM_partitions];//2
            host_data[i].kmer_cnt = new T_skm_partsize[SKM_partitions];//3
            // CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(host_data[i].skm_part_bytes,  gpu_data[i].d_skm_part_bytes,    sizeof(T_skm_partsize) * SKM_partitions, cudaMemcpyDeviceToHost, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(host_data[i].skm_cnt,         gpu_data[i].d_skm_cnt,           sizeof(T_skm_partsize) * SKM_partitions, cudaMemcpyDeviceToHost, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(host_data[i].kmer_cnt,        gpu_data[i].d_kmer_cnt,          sizeof(T_skm_partsize) * SKM_partitions, cudaMemcpyDeviceToHost, streams[i]));
            
            // ---- cudaMalloc skm store positions ----
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_store_pos), sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
            CUDA_CHECK(cudaMemsetAsync(gpu_data[i].d_store_pos, 0, sizeof(T_skm_partsize) * SKM_partitions, streams[i]));
        }
        started_streams = i;

        // ==== Calc SKM Partition Sizes and Extract SKMs ====
        for (i = 0; i < started_streams; i++) {
            
            CUDA_CHECK(cudaStreamSynchronize(streams[i])); // for host skm_part_bytes and skm_cnt
            
            // ---- CPU calc bytes of total skm partition and offsets ----
            host_data[i].tot_skm_bytes = 0;
            host_data[i].skmpart_offs = new T_CSR_cap[SKM_partitions+1];//
            host_data[i].skmpart_offs[0] = 0;
            host_data[i].tot_skm_cnt = 0;
            for (int j = 0; j < SKM_partitions; j++) {
                // assert(host_data[i].skm_part_bytes[j] < 0xffffffffu);
                host_data[i].skmpart_offs[j+1] = host_data[i].skmpart_offs[j] + host_data[i].skm_part_bytes[j];
                host_data[i].tot_skm_bytes += host_data[i].skm_part_bytes[j];
                host_data[i].tot_skm_cnt += host_data[i].skm_cnt[j];
            }
            // ---- cudaMalloc skm store ----
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_skm_store_csr), host_data[i].tot_skm_bytes, streams[i]));
            
            // ---- memcpy skm part sizes and offsets to gpu ----
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_skmpart_offs), sizeof(T_CSR_cap) * (SKM_partitions+1), streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(gpu_data[i].d_skmpart_offs, host_data[i].skmpart_offs, sizeof(T_CSR_cap) * (SKM_partitions+1), cudaMemcpyHostToDevice, streams[i]));

            // ---- GPU extract skms ----
            GPU_ExtractSKM<<<gpars.BpG1, gpars.TpB1, 0, streams[i]>>> (
                gpu_data[i].reads_cnt, gpu_data[i].d_read_len, gpu_data[i].d_read_offs, gpu_data[i].d_reads,
                gpu_data[i].d_minimizers, gpu_data[i].d_superkmer_offs, 
                gpu_data[i].d_store_pos, /*gpu_data[i].d_skm_cnt, */gpu_data[i].d_skm_store_csr, gpu_data[i].d_skmpart_offs,
                K_kmer, P_minimizer, SKM_partitions
            );
            // -- Malloc on host for SKM storage --
            host_data[i].skm_store_csr = new u_char[host_data[i].tot_skm_bytes]; // will not be deleted until program ends
        }

        // ==== Copy SKMs Back to CPU ====
        for (i = 0; i < started_streams; i++) {
            // -- Non-compressed SKM collection (D2H) -- 
            CUDA_CHECK(cudaMemcpyAsync(host_data[i].skm_store_csr, gpu_data[i].d_skm_store_csr, host_data[i].tot_skm_bytes, cudaMemcpyDeviceToHost, streams[i]));
            
            // TO-DO: add if on task to indicate whether to new and D2H
            if (HPC) {
                host_data[i].hpc_orig_pos = new T_read_len[batch_size[i]];//
                host_data[i].read_len = new T_read_len[gpu_data[i].reads_cnt];//
                CUDA_CHECK(cudaMemcpyAsync(host_data[i].hpc_orig_pos, gpu_data[i].d_hpc_orig_pos, sizeof(T_read_len) * batch_size[i], cudaMemcpyDeviceToHost, streams[i]));
                CUDA_CHECK(cudaMemcpyAsync(host_data[i].read_len, gpu_data[i].d_read_len, sizeof(T_read_len) * host_data[i].reads_cnt, cudaMemcpyDeviceToHost, streams[i]));
                // TOxDO: add new reads and new reads_offs
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
                // TOxDO: D2H copy reads and calculate reads_offs
            }
            
            // -- Free device memory --
            if (HPC) CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_hpc_orig_pos, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_reads, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_read_offs, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_read_len, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_minimizers, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_superkmer_offs, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_skm_part_bytes, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_skm_cnt, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_kmer_cnt, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_store_pos, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_skm_store_csr, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_skmpart_offs, streams[i]));
        }
        // ==== CPU Store SKMs ====
        for (i = 0; i < started_streams; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            // process_func(host_data[i]);
            SKMStoreNoncon::save_batch_skms (skm_partition_stores, host_data[i].skm_cnt, host_data[i].kmer_cnt, host_data[i].skmpart_offs, host_data[i].skm_store_csr);
            
            // -- clean host variables --
            if (HPC) {
                delete [] host_data[i].hpc_orig_pos;//
                delete [] host_data[i].read_len;//
            }
            
            delete [] host_data[i].skm_part_bytes;//1
            delete [] host_data[i].skm_cnt;//2
            delete [] host_data[i].kmer_cnt;//3
            delete [] host_data[i].skmpart_offs;//
        }
    }
    logger->log(logs);
    if (time_all!=0)
        logger->log("FILTER: " STR(FILTER_KERNEL) " Kernel Functions Time: ALL = "+to_string(time_all)+"ms FILTER = "+to_string(time_filter)+"ms");
}