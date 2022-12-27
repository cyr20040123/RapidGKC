// #define TIMER

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    exit(1); \
}

#define NULL_POS 0xFFFFFFFFFFFFFFFFUL

#include "nvcomp/gdeflate.hpp"
#include "nvcomp.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
// #include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "types.h"
#include "V2_superkmers.hpp"
#ifdef TIMER
#include "utilities.hpp" // timer
#endif
#include <vector>
#include <fstream>
using namespace std;
using namespace nvcomp;

struct sameasprev {
    sameasprev() {}
    __host__ __device__
        bool operator()(const T_kmer& x, const T_kmer& y) const { 
            return x==y;
        }
};
struct canonicalkmer {
    canonicalkmer() {}
    __host__ __device__
        T_kmer operator()(const T_kmer& x, const T_kvalue k) const {
            T_kmer x1 = ~x, res=0;
            for (T_kvalue i=0; i<k; i++) {
                res = (res << 2) | (x1 & 0b11);
                x1 = x1 >> 2;
            }
            return res < x ? res : x;
        }
};
struct replaceidx {
    replaceidx() {}
    __host__ __device__
        T_read_len operator()(const T_read_len& x, const T_read_len& y) const {
            return x*y;
        }
};
struct is_zero {
    __host__ __device__
        bool operator()(const T_read_len x)
        {
            return x==0;
        }
};

__device__ void _process_bytes (size_t beg, size_t end, byte* d_skms, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
    // if called, stop until at least one skm is processed whatever end is exceeded
    T_kmer kmer_mask = T_kmer(0xffffffffffffffff>>(64-k*2));
    size_t i;
    T_kmer kmer;
    T_kvalue kmer_bases; // effective bases
    unsigned long long store_pos;
    byte indicator, ii;
    byte beg_selector[4] = {0, 0b00000011, 0b00001111, 0b00111111};
    byte end_selector[4] = {0, 0b00110000, 0b00111100, 0b00111111};
    // Optimization: use ulonglong4 to store 4 kmers at a time
    for (i = beg; i < end; i++) { // i: byte
        // generate the first k-mer
        kmer = 0;
        kmer_bases = 0;
        while (kmer_bases <= k-3) {
            indicator = (d_skms[i]>>6) & 0b11;
            kmer <<= indicator * 2; // why >>5: >>6*2 (2 bits per base)
            kmer |= d_skms[i] & beg_selector[indicator];
            kmer_bases += indicator;
            i++;
        }
        ii = 0; 
        if (kmer_bases < k) { // process the last byte of the first kmer if necessary
            kmer <<= (k-kmer_bases)*2;
            kmer |= (d_skms[i] & end_selector[k-kmer_bases]) >> ((BYTE_BASES-(k-kmer_bases))*2);
            ii = k-kmer_bases; // ii: bases used of the current byte
        }
        store_pos = atomicAdd(d_kmer_store_pos, 1);
        d_kmers[store_pos] = kmer;
        // printf("%llu\n", kmer);
        
        // generate and store the next kmers
        indicator = (d_skms[i]>>6) & 0b11;
        while ((indicator == BYTE_BASES) | (ii < indicator)) { // full block or ii not end
            kmer = ((kmer << 2) | ((d_skms[i] >> ((BYTE_BASES-ii-1)*2)) & 0b11)) & kmer_mask;
            store_pos = atomicAdd(d_kmer_store_pos, 1);
            d_kmers[store_pos] = kmer;
            // printf("%llu\n", kmer);
            ii = (ii+1) % BYTE_BASES;
            i += (ii == 0);
            indicator = (d_skms[i]>>6) & 0b11;
        }
    }
}
__device__ size_t _find_full_nonfull_pos (size_t beg, size_t end, byte* d_skms) {
    byte FN_pos_found = 0; // 0: not found, 1: find full byte, 2: find non-full block after a full
    size_t i;
    for (i = beg; (FN_pos_found<2) & (i < end); i++) {
        FN_pos_found |= ((d_skms[i] & 0b11000000) == 0b11000000); // if full block found, beg_pos_found=1
        FN_pos_found <<= ((d_skms[i] & 0b11000000) < 0b11000000); // if non-full block found, beg_pos_found*=2
    }
    return (FN_pos_found>=2) * i + (FN_pos_found<2) * NULL_POS; // return the next position after a full and nonfull
}
__global__ void GPU_Extract_Kmers (byte* d_skms, size_t tot_bytes, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
    int n_t = blockDim.x * gridDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t bytes_per_thread = (tot_bytes + n_t - 1) / n_t; // min: 1
    size_t i, search_ending; // which byte to process
    size_t beg_byte_pos, end_byte_pos;
    for (i = tid*bytes_per_thread; i/*+bytes_per_thread*/ < tot_bytes; i += n_t*bytes_per_thread) {
        // printf("i: %llu %llu\n",i,bytes_per_thread);
        // find begin byte:
        beg_byte_pos = i==0 ? 0 : _find_full_nonfull_pos(i, i+bytes_per_thread+1, d_skms); // if i==0 begin position is ULL_MAX+1=0, begins from 0
        // find end byte: (make sure the last full byte is in the area of at least the next thread)
        search_ending = i+2*bytes_per_thread < tot_bytes ? i+2*bytes_per_thread : tot_bytes;
        end_byte_pos = _find_full_nonfull_pos (i+bytes_per_thread, search_ending, d_skms);
        end_byte_pos = (end_byte_pos < tot_bytes) * end_byte_pos + (end_byte_pos >= tot_bytes) * tot_bytes;
        if (beg_byte_pos < tot_bytes) {
            // printf("%llu process %llu %llu (%d %llu)\n",tot_bytes, beg_byte_pos, end_byte_pos, tid, i);
            // printf("%llu %llu\n",beg_byte_pos,end_byte_pos);
            _process_bytes(beg_byte_pos, end_byte_pos, d_skms, d_kmers, d_kmer_store_pos, k);
        }
    }
    return;
}

#ifdef DEBUG
__global__ void GPU_Extract_Kmers_test (byte* d_skms, size_t tot_bytes, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
    if (blockDim.x * blockIdx.x + threadIdx.x == 0) {
        bool beg = true;
        for (size_t i = 0; i < tot_bytes; i++) {
            if (beg) {
                printf("\n");
                for (size_t j = 3-(d_skms[i]>>6); j < 3; j++) {
                    printf("%u", (unsigned char)((d_skms[i]>>((2-j)*2)))&0b11);
                } printf(" ");
                beg = false;
            } else {
                for (size_t j = 0; j < (d_skms[i]>>6); j++) {
                    printf("%u", (unsigned char)((d_skms[i]>>((2-j)*2)))&0b11);
                } printf(" ");
                beg = (d_skms[i]>>6)!=3;
            }
        }
    }
}
#endif

void Extract_Kmers (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* &d_kmers, cudaStream_t &stream, int BpG=8, int TpB=256) {
    // cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));
    
    byte* d_skms;
    CUDA_CHECK(cudaMallocAsync((void**) &(d_skms), skms_store.tot_size_bytes, stream));
    
    unsigned long long *d_kmer_store_pos;
    CUDA_CHECK(cudaMallocAsync((void**) &(d_kmer_store_pos), sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_kmer_store_pos, 0, sizeof(unsigned long long), stream));
    
    // ---- copy skm chunks H2D ----
    if (skms_store.to_file) {
        // todo: cuFile...
    }
    else {
        int i;
        byte *d_store_pos = d_skms;
        for (i=0; i<skms_store.skm_chunk_bytes.size(); i++) {
            CUDA_CHECK(cudaMemcpyAsync(d_store_pos, skms_store.skm_chunks[i], skms_store.skm_chunk_bytes[i], cudaMemcpyHostToDevice, stream));
            d_store_pos += skms_store.skm_chunk_bytes[i];
        }
    }
    // ---- GPU work ----
    GPU_Extract_Kmers<<<BpG, TpB, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    // GPU_Extract_Kmers_test<<<BpG, TpB, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    
    CUDA_CHECK(cudaFreeAsync(d_skms, stream));
    // CUDA_CHECK(cudaFreeAsync(d_kmers, stream));
    CUDA_CHECK(cudaFreeAsync(d_kmer_store_pos, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
}

void Extract_Kmers_Compressed (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* &d_kmers, cudaStream_t &stream, int BpG=8, int TpB=256, int gpuid = 0) {
    cerr<<"Uncompressed: "<<skms_store.tot_size_bytes<<"\tCompressed: "<<skms_store.tot_size_compressed<<endl;
    cerr<<"Ratio: "<<(double)skms_store.tot_size_compressed / (double)skms_store.tot_size_bytes<<endl;
    // cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));
    
    byte* d_skms; // uncompressed skms
    CUDA_CHECK(cudaMalloc((void**) &(d_skms), skms_store.tot_size_bytes));
    
    unsigned long long *d_kmer_store_pos;
    CUDA_CHECK(cudaMallocAsync((void**) &(d_kmer_store_pos), sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_kmer_store_pos, 0, sizeof(unsigned long long), stream));
    
    // decompression
    const int chunk_size = 1 << 16;
    GdeflateManager nvcomp_manager(chunk_size, 0, stream, gpuid);
    size_t num_buffers = skms_store.skm_chunk_bytes.size();
    size_t i;
    vector<byte*> decomp_result_buffers(num_buffers);
    byte* decompress_pos = d_skms;
    const byte* comp_data[num_buffers];
    for(i = 0; i < num_buffers; i++) {
        CUDA_CHECK(cudaMallocAsync((void **)&comp_data[i], skms_store.skm_chunk_compressed_bytes[i], stream));
        CUDA_CHECK(cudaMemcpyAsync((void*)comp_data[i], skms_store.skm_chunks[i], skms_store.skm_chunk_compressed_bytes[i], cudaMemcpyHostToDevice, stream));
        auto decomp_config = nvcomp_manager.configure_decompression(comp_data[i]);
        nvcomp_manager.decompress(decompress_pos, comp_data[i], decomp_config);
        decompress_pos += skms_store.skm_chunk_bytes[i];
    }
    for(i = 0; i < num_buffers; i++)
        cudaFreeAsync((void*)comp_data[i], stream);

    // // ---- copy skm chunks H2D ----
    // byte *d_store_pos = d_skms;
    // for (i=0; i<skms_store.skm_chunk_bytes.size(); i++) {
    //     CUDA_CHECK(cudaMemcpyAsync(d_store_pos, skms_store.skm_chunks[i], skms_store.skm_chunk_bytes[i], cudaMemcpyHostToDevice, stream));
    //     d_store_pos += skms_store.skm_chunk_bytes[i];
    // }
    // ---- GPU work ----
    GPU_Extract_Kmers<<<BpG, TpB, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    // GPU_Extract_Kmers_test<<<BpG, TpB, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    
    CUDA_CHECK(cudaFreeAsync(d_skms, stream));
    // CUDA_CHECK(cudaFreeAsync(d_kmers, stream));
    CUDA_CHECK(cudaFreeAsync(d_kmer_store_pos, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
}

__host__ size_t kmc_counting_GPU (T_kvalue k,
                               SKMStoreNoncon &skms_store, int gpuid,
                               unsigned short kmer_min_freq, unsigned short kmer_max_freq,
                               _out_ vector<T_kmc> &kmc_result_curthread,
                               bool GPU_compression = false) {
    // using CUDA Thrust
    // size_t est_kmer = skms_store.tot_size - skms_store.skms.size_approx() * (k-1);
    // size_t db_skm = skms_store.skms.size_approx();
    // size_t est_skm = 0;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // 0. Extract kmers from SKMStore: (~65-85% time)
    #ifdef TIMER
    double wcts[10];
    WallClockTimer wct0;
    #endif
    
    if (skms_store.tot_size_bytes == 0) return 0;
    // thrust::host_vector<T_kmer> kmers_h;
    size_t tot_kmers;
    // T_skm_len skm_len;
    // T_kmer kmer_mask = T_kmer(0xffffffffffffffff>>(64-k*2));
    // T_kmer kmer;

    // V2 - bulk: 2.5-2.7s
    // thrust::device_vector<T_kmer> kmers_d(kmers_h);
    // cout<<"Begin extract kmer"<<endl;
    thrust::device_vector<T_kmer> kmers_d(skms_store.kmer_cnt);
    T_kmer *d_kmers_data = thrust::raw_pointer_cast(kmers_d.data());
    if (GPU_compression) Extract_Kmers_Compressed(skms_store, k, d_kmers_data, stream, 8, 256, gpuid);
    else Extract_Kmers(skms_store, k, d_kmers_data, stream, 8, 256);
    tot_kmers = kmers_d.size();
    // cerr<<"End extract kmer "<<tot_kmers<<" "<<skms_store.kmer_cnt<<endl;
    // cerr<<est_kmer<<"|"<<db_skm<<"|"<<est_skm<<"|"<<tot_kmers<<endl;
    
    // 1. convert to canonical kmers (~3-8% time)
    #ifdef TIMER
    wcts[0] = wct0.stop();
    // cerr<<wcts[0]<<endl;
    // exit(1);
    WallClockTimer wct1;
    #endif
    thrust::constant_iterator<T_kvalue> ik(k);
    thrust::transform(thrust::device.on(stream), kmers_d.begin(), kmers_d.end(), ik, kmers_d.begin(), canonicalkmer());
    // 2. sort: [ABCBBAC] -> [AABBBCC] (kmers_d) (~5-15% time)
    #ifdef TIMER
    wcts[1] = wct1.stop();
    WallClockTimer wct2;
    #endif
    thrust::sort(thrust::device.on(stream), kmers_d.begin(), kmers_d.end()/*, thrust::greater<T_kmer>()*/);
    skms_store.clear_skm_data();
    // thrust::host_vector<T_kmer> sorted_kmers_h = kmers_d;
    
    // 3. find changes: [AABBBCC] -> [0,0,1,0,0,1,0] (comp_vec_d)
    // thrust::device_vector<bool> comp_vec_d(kmers_d.size());
    // thrust::transform(thrust::device, kmers_d.begin()+1 /*x beg*/, kmers_d.end() /*x end*/, kmers_d.begin()/*y beg*/, comp_vec_d.begin()+1/*res beg*/, differentfromprev());
    // comp_vec_d[0] = 1; //
    // int distinct_kmer_cnt = thrust::reduce(thrust::device, comp_vec_d.begin(), comp_vec_d.end()) + 1;
    
    // 3. find changes: [AABBBCC] -> [0,1,0,1,1,0,1] (same_flag_d)
    #ifdef TIMER
    wcts[2] = wct2.stop();
    WallClockTimer wct3;
    #endif
    thrust::device_vector<bool> same_flag_d(kmers_d.size());
    thrust::transform(thrust::device.on(stream), kmers_d.begin()+1 /*x beg*/, kmers_d.end() /*x end*/, kmers_d.begin()/*y beg*/, same_flag_d.begin()+1/*res beg*/, sameasprev());
    same_flag_d[0] = 0; //
    
    // 4. remove same idx: [0123456] [0101101] -> [0,2,5] (idx_d)
    #ifdef TIMER
    wcts[3] = wct3.stop();
    WallClockTimer wct4;
    #endif
    thrust::device_vector<T_read_len> idx_d(kmers_d.size());
    thrust::sequence(thrust::device.on(stream), idx_d.begin(), idx_d.end());
    auto new_end_idx_d = thrust::remove_if(thrust::device.on(stream), idx_d.begin(), idx_d.end(), same_flag_d.begin(), thrust::identity<bool>()); // new_end_idx_d is an iterator
    
    // 4+
    // thrust::host_vector<T_kmer> sorted_kmers_h = kmers_d;
    auto new_end_sorted_cleared_kmers_d = thrust::remove_if(thrust::device.on(stream), kmers_d.begin(), kmers_d.end(), same_flag_d.begin(), thrust::identity<bool>()); // new_end_idx_d is an iterator
    thrust::host_vector<T_kmer> sorted_kmers_h(kmers_d.begin(), new_end_sorted_cleared_kmers_d);
    volatile T_kmer tmp_kmer = sorted_kmers_h[0];
    
    // 4. replace with index: [0,0,1,0,0,1,0] -> [0,0,2,0,0,5,0] (comp_vec_d)
    // thrust::device_vector<T_read_len> seq_d(kmers_d.size());
    // thrust::sequence(thrust::device, seq_d.begin(), seq_d.end());
    // thrust::transform(thrust::device, comp_vec_d.begin() /*x*/, comp_vec_d.end(), seq_d.begin()/*y*/, comp_vec_d.begin()/*res*/, replaceidx());

    // // 5. skip repeats: [0,0,2,0,0,5,0] -> [0,2,5] (comp_vec_d)
    // auto new_end_d = thrust::remove_if(thrust::device, comp_vec_d.begin(), comp_vec_d.end(), is_zero());

    // 5. copy device_vector back to host_vector
    #ifdef TIMER
    wcts[4] = wct4.stop();
    WallClockTimer wct5;
    #endif
    thrust::host_vector<T_read_len> idx_h(idx_d.begin(), new_end_idx_d);
    idx_h.push_back(tot_kmers); // [0,2,5] -> [0,2,5,7] A2 B3 C2
    
    #ifdef TIMER
    wcts[5] = wct5.stop();
    WallClockTimer wct6;
    #endif
    size_t total_kmer_cnt = 0;
    int i;
    T_kmer_cnt cnt;
    for(i=0; i<idx_h.size()-1; i++) {
        cnt = idx_h[i+1]-idx_h[i] > MAX_KMER_CNT ? MAX_KMER_CNT : idx_h[i+1]-idx_h[i];
        total_kmer_cnt += idx_h[i+1]-idx_h[i];
        // Add kmer-cnt to result vector:
        // if (cnt >= kmer_min_freq && cnt <= kmer_max_freq) {
        //     kmc_result_curthread.push_back({sorted_kmers_h[idx_h[i]], cnt});
        // }
    }
    assert(total_kmer_cnt == skms_store.kmer_cnt);
    #ifdef TIMER
    wcts[6] = wct6.stop();
    cout<<wcts[0]<<"\t"<<wcts[1]<<"\t"<<wcts[2]<<"\t"<<wcts[3]<<"\t"<<wcts[4]<<"\t"<<wcts[5]<<"\t"<<wcts[6]<<endl;
    #endif
    // return total_kmer_cnt; // total kmer
    delete &skms_store;//
    return idx_h.size()-1; // total distinct kmer
}
