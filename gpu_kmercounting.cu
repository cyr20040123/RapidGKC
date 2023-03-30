// #define GPU_EXTRACT_TIMING
#define TIMING_CUDAMEMCPY

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    exit(1); \
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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h> // prefix-sum exclusive_scan
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "gpu_kmercounting.h"
#include "utilities.hpp"
#include <fstream>
using namespace std;

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

__device__ void process_skm(T_skm_len skm_len, unsigned char start_offs, u_char *skm, 
    T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) 
{
    T_kmer kmer_mask = (T_kmer)(((T_kmer)(-1)) >> (sizeof(T_kmer)*8-k*2));
    T_kmer kmer = 0;
    unsigned long long store_pos;

    unsigned char which_base = start_offs;
    unsigned int i;
    for (i=0; i<k-BYTE_BASES;) {
        kmer = (kmer << ((BYTE_BASES-which_base)*2)) | *skm;
        i += BYTE_BASES-which_base;
        which_base = 0;
        skm++;
    }
    kmer = (kmer << ((k-i)*2)) | (*skm >> ((BYTE_BASES - k+i)*2));
    kmer &= kmer_mask;
    which_base = k-i;
    for (i = k; i < skm_len; i++, which_base++) {
        store_pos = atomicAdd(d_kmer_store_pos, 1);
        d_kmers[store_pos] = kmer;
        skm += (which_base == 4);
        which_base -= 4 * (which_base == 4);
        kmer = ((kmer<<2) | ((*skm >> ((3-which_base)*2)) & 0b11)) & kmer_mask;
    }
    store_pos = atomicAdd(d_kmer_store_pos, 1);
    d_kmers[store_pos] = kmer;
    return;
}
__global__ void GPU_Extract_Kmers (u_char* d_skms, 
    T_skm_len *d_skm_lengths, unsigned int *d_skm_byte_offs, unsigned char *d_start_offs, unsigned int skm_cnt,
    T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k)
{
    int n_t = blockDim.x * gridDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t skms_per_thread = (skm_cnt + n_t - 1) / n_t; // min: 1
    unsigned int beg_skm = tid * skms_per_thread;
    unsigned int end_skm = (tid+1) * skms_per_thread;
    if (end_skm > skm_cnt) end_skm = skm_cnt;
    unsigned int i;
    for (i = beg_skm; i < end_skm; i++) {
        process_skm(d_skm_lengths[i], d_start_offs[i], &d_skms[d_skm_byte_offs[i]], d_kmers, d_kmer_store_pos, k);
    }
    __syncthreads();
    if (tid==0) printf("kmer: %llu skm: %u\n", *d_kmer_store_pos, skm_cnt);
    return;
}

__global__ void resolve_skm_len (T_skm_len *d_skm_lengths_ushort, unsigned int *d_skm_byte_offs, unsigned char *d_start_offs, unsigned int cnt) {
    int n_t = blockDim.x * gridDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i=tid; i<cnt; i+=n_t) {
        d_start_offs[i] = d_skm_lengths_ushort[i] >> 14;
        d_skm_lengths_ushort[i] &= 0b0011111111111111;
        d_skm_byte_offs[i] = (d_skm_lengths_ushort[i] + d_start_offs[i] + BYTE_BASES - 1) / BYTE_BASES;
    }
}

__host__ void load_SKM_from_file (SKMStoreNoncon &skms_store, _out_ u_char* &d_skms, _out_ T_skm_len* &d_skm_lengths) {
    CUDA_CHECK(cudaMalloc((void**) &(d_skms), skms_store.tot_size_bytes));
    CUDA_CHECK(cudaMalloc((void**) &(d_skm_lengths), skms_store.skm_cnt * sizeof(T_skm_len)));
    skms_store.load_from_file();
    CUDA_CHECK(cudaMemcpy(d_skms, skms_store.skms_from_file, skms_store.tot_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_skms, skms_store.skm_lengths_from_file, skms_store.skm_cnt * sizeof(T_skm_len), cudaMemcpyHostToDevice));
    delete [] skms_store.skms_from_file;
    delete [] skms_store.skm_lengths_from_file;
    return;
}

float Extract_Kmers (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* d_kmers, cudaStream_t &stream, int BpG2=8, int TpB2=256) {
    
    #ifdef TIMING_CUDAMEMCPY
    cudaEvent_t memcpy_start, memcpy_end;
    cudaEventCreate(&memcpy_start); cudaEventCreate(&memcpy_end);
    #endif

    u_char *d_skms;
    T_skm_len *d_skm_lengths;
    u_char *d_start_offs;
    unsigned int* d_skm_byte_offs;
    
    unsigned long long *d_kmer_store_pos;
    CUDA_CHECK(cudaMallocAsync((void**) &(d_kmer_store_pos), sizeof(size_t), stream));//
    CUDA_CHECK(cudaMemsetAsync(d_kmer_store_pos, 0, sizeof(unsigned long long), stream));

    // ---- copy skm chunks H2D ----
    // load d_skms, d_skm_lengths
    if (skms_store.to_file) load_SKM_from_file(skms_store, d_skms, d_skm_lengths);
    else {
        CUDA_CHECK(cudaMallocAsync((void**) &(d_skms), skms_store.tot_size_bytes+1, stream));//
        CUDA_CHECK(cudaMallocAsync((void**) &(d_skm_lengths), skms_store.skm_cnt * sizeof(T_skm_len), stream));//
        
        #ifdef TIMING_CUDAMEMCPY
        cudaEventRecord(memcpy_start, stream);
        #endif
    
        int i;
        size_t d_store_pos = 0;
        size_t d_len_store_pos = 0;
        SkmChunk skm_bulk[1024];
        size_t count;
        do {
            count = skms_store.skms.try_dequeue_bulk(skm_bulk, 1024);
            for (i=0; i<count; i++) {
                // if (i==0) {
                //     int len = skm_bulk[i].skm_lengths[0] & 0b0011111111111111;
                //     int start = skm_bulk[i].skm_lengths[0] >> 14;
                //     int idx = 0;
                //     char c[4] = {'A','C','G','T'};
                //     cerr<<"SKM LEN:"+to_string(len)<<endl;
                //     for (int j=0; j<len; j++) {
                //         idx += (start==4);
                //         start %= 4;
                //         cerr<<c[(skm_bulk[i].skm_chunk[idx] >> ((3-start)*2)) & 0b11];
                //         start ++;
                //     }cerr<<endl;
                // }
                CUDA_CHECK(cudaMemcpyAsync(d_skms+d_store_pos, skm_bulk[i].skm_chunk, skm_bulk[i].chunk_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_skm_lengths+d_len_store_pos, skm_bulk[i].skm_lengths, skm_bulk[i].chunk_cnt * sizeof(T_skm_len), cudaMemcpyHostToDevice, stream));
                d_store_pos += skm_bulk[i].chunk_bytes;
                d_len_store_pos += skm_bulk[i].chunk_cnt; // * sizeof(T_skm_len);
            }
        } while (count);

        #ifdef TIMING_CUDAMEMCPY
        cudaEventRecord(memcpy_end, stream);
        #endif
    }
    
    CUDA_CHECK(cudaMallocAsync((void**) &(d_start_offs), skms_store.skm_cnt, stream));//
    CUDA_CHECK(cudaMallocAsync((void**) &(d_skm_byte_offs), (skms_store.skm_cnt+1) * sizeof(unsigned int), stream));//
    
    // thrust::device_vector<unsigned int> d_skm_byte_offs(skms_store.skm_cnt);
    // transfrom ushort skm lengths to uint skm byte offs, move out d_start_offs
    resolve_skm_len<<<BpG2, TpB2, 0, stream>>>(d_skm_lengths, d_skm_byte_offs, d_start_offs, skms_store.skm_cnt);
    thrust::exclusive_scan(thrust::device.on(stream), d_skm_byte_offs, d_skm_byte_offs + (size_t)skms_store.skm_cnt, d_skm_byte_offs, 0); // prefix-sum and right-shift-one on d_skm_byte_offs

    // ---- GPU work ----
    // if (skms_store.tot_size_bytes / 4 <= BpG2 * TpB2) GPU_Extract_Kmers<<<1, skms_store.tot_size_bytes/64+1, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k); // 强行debug // mountain of shit, do not touch
    // else GPU_Extract_Kmers<<<BpG2, TpB2, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    GPU_Extract_Kmers<<<BpG2, TpB2, 0, stream>>>(d_skms, d_skm_lengths, d_skm_byte_offs, d_start_offs, skms_store.skm_cnt, d_kmers, d_kmer_store_pos, k);
    
    CUDA_CHECK(cudaFreeAsync(d_skms, stream));//
    CUDA_CHECK(cudaFreeAsync(d_kmer_store_pos, stream));//
    CUDA_CHECK(cudaFreeAsync(d_skm_lengths, stream));//
    CUDA_CHECK(cudaFreeAsync(d_start_offs, stream));//
    CUDA_CHECK(cudaFreeAsync(d_skm_byte_offs, stream));//
    
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
        logs += "\tS "+to_string(i)+" Part "+to_string(skms_stores[i]->id)+" "+to_string(skms_stores[i]->tot_size_bytes)+"|"+to_string(skms_stores[i]->kmer_cnt)+"|"+to_string(skms_stores[i]->skm_cnt);
        // logger->log(logs, Logger::LV_INFO);
        if (skms_stores[i]->tot_size_bytes != 0) {
            // ---- 0. Extract kmers from SKMStore: ---- 
            kmers_d_vec[i] = thrust::device_vector<T_kmer>(skms_stores[i]->kmer_cnt);
            T_kmer *d_kmers_data = thrust::raw_pointer_cast(kmers_d_vec[i].data());
            #ifdef TIMING_CUDAMEMCPY
            float timing = Extract_Kmers(*skms_stores[i], k, d_kmers_data, streams[i], gpars.BpG2, gpars.TpB2);
            debug_cudacp_timing += (int)timing;
            timing = skms_stores[i]->tot_size_bytes / (timing) / 1e6;
            logs += " [Eff.BW "+to_string(timing)+" GB/s]";
            #else
            Extract_Kmers(*skms_stores[i], k, d_kmers_data, streams[i], gpars.BpG2, gpars.TpB2);
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
            same_flag_d_vec[i] = thrust::device_vector<bool>(kmers_d_vec[i].size());
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
            idx_d_vec[i] = thrust::device_vector<T_read_len>(kmers_d_vec[i].size());
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
