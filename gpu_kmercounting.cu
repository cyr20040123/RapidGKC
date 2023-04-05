// #define GPU_EXTRACT_TIMING
// #define TIMING_CUDAMEMCPY

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    exit(1); \
}
#define CUDA_MALLOC_CHECK(call) \
while((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "Malloc error: \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    this_thread::sleep_for(100ms); \
}
#define CUFILE_STATUS_CHECK(cuerr, lineno) \
if (cuerr.err != CU_FILE_SUCCESS) { \
    cerr << "cuFile error calling line #" << lineno << ", code is " << cuerr.err << endl; \
    exit(1); \
} \
if (cuerr.cu_err != CUDA_SUCCESS) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "cuFile error calling line #" << lineno << ", code is " << cuerr.cu_err <<"|"<< err << ": " << cudaGetErrorString(err) << endl; \
    exit(1); \
}

#define NULL_POS 0xFFFFFFFFFFFFFFFFUL

// #include "nvcomp/gdeflate.hpp"
// #include "nvcomp.hpp"

// #include <fcntl.h> // open
// #include <unistd.h> // close
// #include "cufile.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
// #include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

// #include "types.h"
// #include "skmstore.hpp"
// #include <vector>
#include "gpu_kmercounting.h"
#include "utilities.hpp"
#include <fstream>
using namespace std;
// using namespace nvcomp;

extern Logger *logger;

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

__device__ void _process_bytes (size_t beg, size_t end, u_char* d_skms, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
    // if called, stop until at least one skm is processed whatever end is exceeded
    // T_kmer kmer_mask = T_kmer(0xffffffffffffffff>>(64-k*2));
    T_kmer kmer_mask = (T_kmer)(((T_kmer)(-1)) >> (sizeof(T_kmer)*8-k*2));
    size_t i;
    T_kmer kmer;
    T_kvalue kmer_bases; // effective bases
    unsigned long long store_pos;

    // const unsigned long long KMER_BATCH_SIZE = 256;
    // unsigned long long  res_size = (k/3+4)*3/*max bytes for a SKM with only 1 kmer*/ * KMER_BATCH_SIZE;
    // unsigned long long  in_batch_cnt = KMER_BATCH_SIZE;

    u_char indicator, ii;
    u_char beg_selector[4] = {0, 0b00000011, 0b00001111, 0b00111111};
    u_char end_selector[4] = {0, 0b00110000, 0b00111100, 0b00111111};
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
        // // assign space to store kmers (replace line above)
        // if (in_batch_cnt == KMER_BATCH_SIZE && i + res_size >= end)
        //     store_pos = atomicAdd(d_kmer_store_pos, 1);
        // else {
        //     store_pos++;
        //     if (in_batch_cnt == KMER_BATCH_SIZE) {
        //         in_batch_cnt = 0;
        //         store_pos = atomicAdd(d_kmer_store_pos, KMER_BATCH_SIZE);
        //     }
        //     in_batch_cnt++;
        // }
        // //

        d_kmers[store_pos] = kmer;
        // printf(" %llu\n", kmer);
        
        // generate and store the next kmers
        indicator = (d_skms[i]>>6) & 0b11;
        while ((indicator == BYTE_BASES) | (ii < indicator)) { // full block or ii not end
            kmer = ((kmer << 2) | ((d_skms[i] >> ((BYTE_BASES-ii-1)*2)) & 0b11)) & kmer_mask;
            store_pos = atomicAdd(d_kmer_store_pos, 1);
            // // assign space to store kmers (replace line above)
            // if (in_batch_cnt == KMER_BATCH_SIZE && i + res_size >= end)
            //     store_pos = atomicAdd(d_kmer_store_pos, 1);
            // else {
            //     store_pos++;
            //     if (in_batch_cnt == KMER_BATCH_SIZE) {
            //         in_batch_cnt = 0;
            //         store_pos = atomicAdd(d_kmer_store_pos, KMER_BATCH_SIZE);
            //     }
            //     in_batch_cnt++;
            // }
            // //
            d_kmers[store_pos] = kmer;
            // printf(" %llu\n", kmer);
            ii = (ii+1) % BYTE_BASES;
            i += (ii == 0);
            indicator = (d_skms[i]>>6) & 0b11;
        }
    }
}
__device__ size_t _find_full_nonfull_pos (size_t beg, size_t end, u_char* d_skms) {
    u_char FN_pos_found = 0; // 0: not found, 1: find full byte, 2: find non-full block after a full
    size_t i;
    for (i = beg; (FN_pos_found<2) & (i < end); i++) {
        // FN_pos_found |= ((d_skms[i] & 0b11000000) == 0b11000000); // if full block found, beg_pos_found=1
        // FN_pos_found <<= ((d_skms[i] & 0b11000000) < 0b11000000); // if non-full block found, beg_pos_found*=2
        FN_pos_found |= (d_skms[i] >= 0b11000000); // if full block found, beg_pos_found=1
        FN_pos_found <<= (d_skms[i] < 0b11000000); // if non-full block found, beg_pos_found*=2
    }
    return (FN_pos_found>=2) * i + (FN_pos_found<2) * NULL_POS; // return the next position after a full and nonfull
}
// __global__ void GPU_Extract_Kmers (u_char* d_skms, size_t tot_bytes, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
//     int n_t = blockDim.x * gridDim.x;
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     size_t bytes_per_thread = (tot_bytes + n_t - 1) / n_t; // min: 1
//     size_t i, search_ending; // which byte to process
//     size_t beg_byte_pos, end_byte_pos;
//     for (i = tid*bytes_per_thread; i/*+bytes_per_thread*/ < tot_bytes; i += n_t*bytes_per_thread) {
//         // printf("i: %llu %llu\n",i,bytes_per_thread);
//         // find begin byte:
//         beg_byte_pos = i==0 ? 0 : _find_full_nonfull_pos(i, i+bytes_per_thread+1, d_skms); // if i==0 begin position is ULL_MAX+1=0, begins from 0
//         // find end byte: (make sure the last full byte is in the area of at least the next thread)
//         search_ending = i+2*bytes_per_thread < tot_bytes ? i+2*bytes_per_thread : tot_bytes;
//         end_byte_pos = _find_full_nonfull_pos (i+bytes_per_thread, search_ending, d_skms);
//         end_byte_pos = (end_byte_pos < search_ending) * end_byte_pos + (end_byte_pos >= search_ending) * search_ending;
//         if (beg_byte_pos < tot_bytes) {
//             // printf("%llu process %llu %llu (%d %llu)\n",tot_bytes, beg_byte_pos, end_byte_pos, tid, i);
//             // printf("%llu %llu\n",beg_byte_pos,end_byte_pos);
//             _process_bytes(beg_byte_pos, end_byte_pos, d_skms, d_kmers, d_kmer_store_pos, k);
//         }
//     }
//     return;
// }

__global__ void GPU_Extract_Kmers (u_char* d_skms, size_t tot_bytes, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
    int n_t = blockDim.x * gridDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t bytes_per_thread = (tot_bytes + n_t - 1) / n_t; // min: 1
    size_t i, search_ending1, search_ending2; // which byte to process
    size_t beg_byte_pos, end_byte_pos;
    for (i = tid*bytes_per_thread; i/*+bytes_per_thread*/ < tot_bytes; i += n_t*bytes_per_thread) {
        // printf("i: %llu %llu\n",i,bytes_per_thread);
        // find begin byte:
        search_ending1 = i+bytes_per_thread-1 < tot_bytes ? i+bytes_per_thread-1 : tot_bytes;
        beg_byte_pos = i<2 ? 0 : _find_full_nonfull_pos(i-2, search_ending1, d_skms); // if i==0 begin position is ULL_MAX+1=0, begins from 0
        // find end byte: (make sure the last full byte is in the area of at least the next thread)
        search_ending2 = i+2*bytes_per_thread < tot_bytes ? i+2*bytes_per_thread : tot_bytes;
        end_byte_pos = _find_full_nonfull_pos (search_ending1-1, search_ending2, d_skms);
        end_byte_pos = (end_byte_pos < search_ending2) * end_byte_pos + (end_byte_pos >= search_ending2) * search_ending2; // DEBUGGED DONE
        // end_byte_pos = (end_byte_pos < tot_bytes) * end_byte_pos + (end_byte_pos >= tot_bytes) * tot_bytes;
        if (beg_byte_pos < end_byte_pos) {
            // printf("%llu process %llu %llu (%d %llu)\n",tot_bytes, beg_byte_pos, end_byte_pos, tid, i);
            // printf("%llu %llu\n",beg_byte_pos,end_byte_pos);
            _process_bytes(beg_byte_pos, end_byte_pos, d_skms, d_kmers, d_kmer_store_pos, k);
        }
    }
    return;
}

#ifdef DEBUG
__global__ void GPU_Extract_Kmers_test (u_char* d_skms, size_t tot_bytes, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
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

__host__ u_char* load_SKM_from_file (SKMStoreNoncon &skms_store) {
    u_char* d_skms;
    CUDA_CHECK(cudaMalloc((void**) &(d_skms), skms_store.tot_size_bytes));
    skms_store.load_from_file();
    CUDA_CHECK(cudaMemcpy(d_skms, skms_store.skms_from_file, skms_store.tot_size_bytes, cudaMemcpyHostToDevice));
    delete skms_store.skms_from_file;
    return d_skms;
}

#ifdef TIMING_CUDAMEMCPY
float Extract_Kmers (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* d_kmers, cudaStream_t &stream, int BpG2=8, int TpB2=256) {
#else
void Extract_Kmers (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* d_kmers, cudaStream_t &stream, int BpG2=8, int TpB2=256) {
#endif
    // cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));
    
    #ifdef TIMING_CUDAMEMCPY
    cudaEvent_t memcpy_start, memcpy_end;
    cudaEventCreate(&memcpy_start); cudaEventCreate(&memcpy_end);
    #endif

    u_char* d_skms;
    
    unsigned long long *d_kmer_store_pos;
    CUDA_MALLOC_CHECK(cudaMallocAsync((void**) &(d_kmer_store_pos), sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_kmer_store_pos, 0, sizeof(unsigned long long), stream));

    // ---- copy skm chunks H2D ----
    if (skms_store.to_file) d_skms = load_SKM_from_file(skms_store);
    else {
        CUDA_MALLOC_CHECK(cudaMallocAsync((void**) &(d_skms), skms_store.tot_size_bytes+1, stream));
        // CUDA_CHECK(cudaMalloc((void**) &(d_skms), skms_store.tot_size_bytes+1));
        // u_char *d_store_pos = d_skms;
        
        #ifdef TIMING_CUDAMEMCPY
        cudaEventRecord(memcpy_start, stream);
        #endif
    
        // #ifdef SKMSTOREV1
        // for (i=0; i<skms_store.skm_chunk_bytes.size(); i++) {
        //     CUDA_CHECK(cudaMemcpyAsync(d_skms+d_store_pos, skms_store.skm_chunks[i], skms_store.skm_chunk_bytes[i], cudaMemcpyHostToDevice, stream));
        //     // CUDA_CHECK(cudaStreamSynchronize(stream)); // 不加这行用CPU step 1会卡死 7.60s(+) vs 7.40s(-) // TODO: check this
        //     d_store_pos += skms_store.skm_chunk_bytes[i];
        // }
        #ifdef _SKMSTORE2_HPP
        CUDA_CHECK(cudaMemcpyAsync(d_skms, skms_store.skms.c_str(), skms_store.tot_size_bytes, cudaMemcpyHostToDevice, stream));
        #else
        int i;
        size_t d_store_pos = 0;
        SKM skm_bulk[1024];
        size_t count;
        do {
            count = skms_store.skms.try_dequeue_bulk(skm_bulk, 1024);
            for (i=0; i<count; i++) {
                CUDA_CHECK(cudaMemcpyAsync(d_skms+d_store_pos, skm_bulk[i].skm_chunk, skm_bulk[i].chunk_bytes, cudaMemcpyHostToDevice, stream));
                d_store_pos += skm_bulk[i].chunk_bytes;
                // memcpy(skms+skm_store_pos, skm_bulk[i].skm_chunk, skm_bulk[i].chunk_bytes);
                // skm_store_pos += skm_bulk[i].chunk_bytes;
            }
        } while (count);
        #endif

        #ifdef TIMING_CUDAMEMCPY
        cudaEventRecord(memcpy_end, stream);
        #endif
    }
    // cerr<<"debug2"<<endl;
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // ---- GPU work ----
    if (skms_store.tot_size_bytes / 4 <= BpG2 * TpB2) GPU_Extract_Kmers<<<1, skms_store.tot_size_bytes/64+1, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k); // 强行debug // mountain of shit, do not touch
    else GPU_Extract_Kmers<<<BpG2, TpB2, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    // GPU_Extract_Kmers_test<<<BpG2, TpB2, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    
    CUDA_CHECK(cudaFreeAsync(d_skms, stream));
    CUDA_CHECK(cudaFreeAsync(d_kmer_store_pos, stream));
    
    #ifdef TIMING_CUDAMEMCPY
    float cudamemcpy_time;
    cudaEventSynchronize(memcpy_end);
    cudaEventElapsedTime(&cudamemcpy_time, memcpy_start, memcpy_end);
    return cudamemcpy_time;
    #endif

    return;
}

#ifdef TIMING_CUDAMEMCPY
extern atomic<int> debug_cudacp_timing;
#endif

__host__ size_t kmc_counting_GPU_streams (T_kvalue k,
                               vector<SKMStoreNoncon*> skms_stores, CUDAParams &gpars,
                               T_kmer_cnt kmer_min_freq, T_kmer_cnt kmer_max_freq,
                               _out_ vector<T_kmc> kmc_result_curthread [], int gpuid, int tid) {
    // using CUDA Thrust
    // int gpuid = (gpars.device_id++)%gpars.n_devices;
    CUDA_CHECK(cudaSetDevice(gpuid));
    // V2:
    // if (gpars.gpuid_thread[tid] == -1) {
    //     CUDA_CHECK(cudaSetDevice(tid%gpars.n_devices));
    //     gpars.gpuid_thread[tid] = tid%gpars.n_devices;
    // }
    // int gpuid = gpars.gpuid_thread[tid];
    
    size_t return_value = 0;
    int i, n_streams = skms_stores.size();
    cudaStream_t streams[n_streams];
    #ifdef GPU_EXTRACT_TIMING
    cudaEvent_t start[n_streams], mid[n_streams], stop[n_streams];
    #endif

    vector<thrust::device_vector<T_kmer>> kmers_d_vec(n_streams); // for 0
    vector<size_t> tot_kmers(n_streams);
    string logs = "GPU "+to_string(gpuid)+"\t(T"+to_string(tid)+"):";
    for (i=0; i<n_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        #ifdef GPU_EXTRACT_TIMING
        cudaEventCreate(&start[i]); cudaEventCreate(&mid[i]); cudaEventCreate(&stop[i]);
        cudaEventRecord(start[i], streams[i]);
        #endif
        // logger->log("GPU "+to_string(gpuid)+" Stream "+to_string(i)+" counting Partition "+to_string(skms_stores[i]->id), Logger::LV_INFO);
        logs += "\tS "+to_string(i)+" Part "+to_string(skms_stores[i]->id)+" "+to_string(skms_stores[i]->tot_size_bytes)+"|"+to_string(skms_stores[i]->kmer_cnt);
        // logger->log(logs, Logger::LV_INFO);
        if (skms_stores[i]->tot_size_bytes != 0) {
            // ---- 0. Extract kmers from SKMStore: ---- 
            int retry_cnt = 32;
            beg_alloc_kmer:;
            try {
                kmers_d_vec[i] = thrust::device_vector<T_kmer>(skms_stores[i]->kmer_cnt);
            // } catch(thrust::system_error e) {
            } catch(thrust::system::detail::bad_alloc e) {
                cerr<<"Out of VRAM for kmers, now retry ... "<<retry_cnt<<endl;
                this_thread::sleep_for(500ms);
                if (retry_cnt-- <= 0) {logger->log(e.what(), Logger::LV_FATAL); exit(1);}
                goto beg_alloc_kmer;
            }
            T_kmer *d_kmers_data = thrust::raw_pointer_cast(kmers_d_vec[i].data());
            // if (GPU_compression) Extract_Kmers_Compressed(*skms_stores[i], k, d_kmers_data, streams[i], gpars.BpG2, gpars.TpB2, gpuid);
            #ifdef TIMING_CUDAMEMCPY
            float timing = Extract_Kmers(*skms_stores[i], k, d_kmers_data, streams[i], gpars.BpG2, gpars.TpB2);
            debug_cudacp_timing += (int)timing;
            timing = skms_stores[i]->tot_size_bytes / (timing) / 1e6;
            logs += " [Eff.BW "+to_string(timing)+" GB/s]";
            #else
            /*else*/ Extract_Kmers(*skms_stores[i], k, d_kmers_data, streams[i], gpars.BpG2, gpars.TpB2);
            #endif
            tot_kmers[i] = kmers_d_vec[i].size();
        }
        #ifdef GPU_EXTRACT_TIMING
        cudaEventRecord(mid[i], streams[i]);
        #endif
    }

    thrust::constant_iterator<T_kvalue> ik(k);
    vector<thrust::device_vector<bool>> same_flag_d_vec(n_streams); // for 3
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes != 0) {
            // CUDA_CHECK(cudaStreamSynchronize(streams[i])); // maybe don't need this?
            // ---- 1. convert to canonical kmers ---- 
            thrust::transform(thrust::device.on(streams[i]), kmers_d_vec[i].begin(), kmers_d_vec[i].end(), ik, kmers_d_vec[i].begin(), canonicalkmer());
            // ---- 2. sort: [ABCBBAC] -> [AABBBCC] (kmers_d) ---- 
            thrust::sort(thrust::device.on(streams[i]), kmers_d_vec[i].begin(), kmers_d_vec[i].end()/*, thrust::greater<T_kmer>()*/);
            skms_stores[i]->clear_skm_data(); // TODO: 在sort之前加streamsync
            // ---- 3. find changes: [AABBBCC] -> [0,1,0,1,1,0,1] (same_flag_d) ---- 
            int retry_cnt = 32;
            beg_alloc_flag:;
            try {
                same_flag_d_vec[i] = thrust::device_vector<bool>(kmers_d_vec[i].size());
            } catch(thrust::system::detail::bad_alloc e) {
                cerr<<"Out of VRAM for same_flag_d_vec, now retry..."<<endl;
                this_thread::sleep_for(250ms);
                if (retry_cnt-- <= 0) {logger->log(e.what(), Logger::LV_FATAL); exit(1);}
                goto beg_alloc_flag;
            }
            thrust::transform(thrust::device.on(streams[i]), kmers_d_vec[i].begin()+1 /*x beg*/, kmers_d_vec[i].end() /*x end*/, kmers_d_vec[i].begin()/*y beg*/, same_flag_d_vec[i].begin()+1/*res beg*/, sameasprev());
        }
    }

    vector<thrust::device_vector<T_read_len>> idx_d_vec(n_streams); // for 4
    vector<thrust::host_vector<T_kmer>> sorted_kmers_h_vec(n_streams); // for 4+
    vector<thrust::host_vector<T_read_len>> idx_h_vec(n_streams); // for 5
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes != 0) {
            // ---- 3. find changes (cont'd)
            same_flag_d_vec[i][0] = 0; // will it call stream sync?
            // ---- 4. remove same idx: [0123456] [0101101] -> [0,2,5] (idx_d) ----
            int retry_cnt = 32;
            beg_alloc_idx:;
            try {
                idx_d_vec[i] = thrust::device_vector<T_read_len>(kmers_d_vec[i].size());
            } catch(thrust::system::detail::bad_alloc e) {
                cerr<<"Out of VRAM for idx_d_vec, now retry..."<<endl;
                this_thread::sleep_for(250ms);
                if (retry_cnt-- <= 0) {logger->log(e.what(), Logger::LV_FATAL); exit(1);}
                goto beg_alloc_idx;
            }
            thrust::sequence(thrust::device.on(streams[i]), idx_d_vec[i].begin(), idx_d_vec[i].end());
            auto newend_idx_d = thrust::remove_if(thrust::device.on(streams[i]), idx_d_vec[i].begin(), idx_d_vec[i].end(), same_flag_d_vec[i].begin(), thrust::identity<bool>()); // new_end_idx_d is an iterator
            // 4+. copy sorted kmers back to host
            auto newend_sorted_cleared_kmers_d = thrust::remove_if(thrust::device.on(streams[i]), kmers_d_vec[i].begin(), kmers_d_vec[i].end(), same_flag_d_vec[i].begin(), thrust::identity<bool>()); // new_end_idx_d is an iterator
            sorted_kmers_h_vec[i] = thrust::host_vector<T_kmer>(kmers_d_vec[i].begin(), newend_sorted_cleared_kmers_d);
            volatile T_kmer tmp_kmer = sorted_kmers_h_vec[i][0];
            // ---- 5. copy device_vector back to host_vector ----
            idx_h_vec[i] = thrust::host_vector<T_read_len>(idx_d_vec[i].begin(), newend_idx_d);
            idx_h_vec[i].push_back(tot_kmers[i]); // [0,2,5] -> [0,2,5,7] A2 B3 C2
        }
        #ifdef GPU_EXTRACT_TIMING
        cudaEventRecord(stop[i], streams[i]);
        #endif
    }
    
    #ifdef RESULT_VALIDATION
    // validation and export result:
    for (i=0; i<n_streams; i++) {
        // string outfile = "/mnt/f/study/bio_dataset/tmp/" + to_string(skms_stores[i]->id) + ".txt";
        // FILE *fp = fopen(outfile.c_str(), "w");

        if (skms_stores[i]->tot_size_bytes == 0) continue;
        size_t total_kmer_cnt = 0;
        T_kmer_cnt cnt;
        for(int j=0; j<idx_h_vec[i].size()-1; j++) {
            cnt = idx_h_vec[i][j+1]-idx_h_vec[i][j] > MAX_KMER_CNT ? MAX_KMER_CNT : idx_h_vec[i][j+1]-idx_h_vec[i][j];
            total_kmer_cnt += idx_h_vec[i][j+1]-idx_h_vec[i][j];
            // Add kmer-cnt to result vector:
            // if (cnt >= kmer_min_freq && cnt <= kmer_max_freq) {
            //     kmc_result_curthread[skms_stores[i]->id].push_back({sorted_kmers_h[idx_h[j]], cnt});
            //     kmc_result_curthread[skms_stores[i]->id].push_back({sorted_kmers_h[j], cnt});
            // }
            // cerr<<sorted_kmers_h_vec[i][j]<<": "<<idx_h_vec[i][j+1]-idx_h_vec[i][j]<<endl;
            // if (idx_h_vec[i][j+1]-idx_h_vec[i][j] > 1)
            //     fprintf(fp, "%llu %llu\n", sorted_kmers_h_vec[i][j], idx_h_vec[i][j+1]-idx_h_vec[i][j]);
        }
        assert(total_kmer_cnt == skms_stores[i]->kmer_cnt);
        // fclose(fp);
    }
    #endif

    #ifdef GPU_EXTRACT_TIMING
    float time1, time2, time_ratio = 0;
    #endif
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        return_value += idx_h_vec[i].size()-1;
        #ifdef GPU_EXTRACT_TIMING
        cudaEventElapsedTime(&time1, start[i], mid[i]); cudaEventElapsedTime(&time2, mid[i], stop[i]);
        time_ratio += time1 / (time1+time2);
        #endif
    }
    #ifdef GPU_EXTRACT_TIMING
    time_ratio /= (float)n_streams;
    #endif
    for (i=0; i<n_streams; i++) {
        // skms_stores[i]->clear_skm_data();
        delete skms_stores[i];//
    }
    logger->log(logs+" "+to_string(return_value)
    #ifdef GPU_EXTRACT_TIMING
    +" "+to_string(time_ratio)
    #endif
    , Logger::LV_DEBUG);
    return return_value; // total distinct kmer
}
