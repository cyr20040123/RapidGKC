// #define LONGERKMER

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    size_t avail, total; \
    cudaMemGetInfo(&avail, &total); \
    cerr << "Available memory: " << avail/1048576 << " Total memory: " << total/1048576 << endl; \
    exit(1); \
}
#define CUDA_MALLOC_CHECK(call) \
for (int _i_cuRetry=0; (call) != cudaSuccess; _i_cuRetry++) { \
    cudaError_t err = cudaGetLastError(); \
    if (_i_cuRetry%50 == 0) cerr << "("<<_i_cuRetry<<") Malloc error: \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
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

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/sort.h>
// #include <thrust/sequence.h>
// #include <thrust/remove.h>
// #include <thrust/execution_policy.h>
// #include <thrust/iterator/constant_iterator.h> // Required by CUDA12+

#include <cub/cub.cuh>

#include "gpu_kmercounting.h"
#include "utilities.hpp"
#include <fstream>
#include <sstream>
#include "cuda_utils.cuh"
using namespace std;

extern Logger *logger;

__device__ void _process_bytes (size_t beg, size_t end, u_char* d_skms, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
    // if called, stop until at least one skm is processed whatever end is exceeded
    T_kmer kmer_mask = (T_kmer)(((T_kmer)(-1)) >> (sizeof(T_kmer)*8-k*2));
    size_t i;
    T_kmer kmer;
    T_kvalue kmer_bases; // effective bases
    unsigned long long store_pos;

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

        d_kmers[store_pos] = kmer;
        
        // generate and store the next kmers
        indicator = (d_skms[i]>>6) & 0b11;
        while ((indicator == BYTE_BASES) | (ii < indicator)) { // full block or ii not end
            kmer = ((kmer << 2) | ((d_skms[i] >> ((BYTE_BASES-ii-1)*2)) & 0b11)) & kmer_mask;
            store_pos = atomicAdd(d_kmer_store_pos, 1);
            d_kmers[store_pos] = kmer;
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
        FN_pos_found |= (d_skms[i] >= 0b11000000); // if full block found, beg_pos_found=1
        FN_pos_found <<= (d_skms[i] < 0b11000000); // if non-full block found, beg_pos_found*=2
    }
    return (FN_pos_found>=2) * i + (FN_pos_found<2) * NULL_POS; // return the next position after a full and nonfull
}

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
        
        if (beg_byte_pos < end_byte_pos) {
            _process_bytes(beg_byte_pos, end_byte_pos, d_skms, d_kmers, d_kmer_store_pos, k);
        }
    }
    return;
}

void Extract_Kmers (SKMStoreNoncon &skms_store, T_kvalue k, u_char* d_skms, _out_ T_kmer* d_kmers, cudaStream_t &stream, int BpG2=8, int TpB2=256) {

    unsigned long long *d_kmer_store_pos; // for device atomic add, cannot use size_t
    CUDA_MALLOC_CHECK(cudaMalloc((void**) &(d_kmer_store_pos), sizeof(unsigned long long)));//
    CUDA_CHECK(cudaMemsetAsync(d_kmer_store_pos, 0, sizeof(unsigned long long), stream));

    // ---- copy skm chunks H2D ----
    if (skms_store.to_file) {
        skms_store.load_from_file();
        CUDA_CHECK(cudaMemcpy(d_skms, skms_store.skms_from_file, skms_store.tot_size_bytes, cudaMemcpyHostToDevice));
        delete skms_store.skms_from_file;
    }
    else {
        assert(skms_store.tot_size_bytes <= skms_store.kmer_cnt * sizeof(T_kmer));
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
            }
        } while (count);
        #endif
    }
    // ---- GPU work ----
    CUDA_CHECK(cudaStreamSynchronize(stream));// // wait for d_kmer_store_pos
    if (skms_store.tot_size_bytes / 4 <= BpG2 * TpB2) 
        GPU_Extract_Kmers<<<1, skms_store.tot_size_bytes/64+1, 0, stream>>> // 强行debug // mountain of shit, do not touch
            (d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    else 
        GPU_Extract_Kmers<<<BpG2, TpB2, 0, stream>>>
            (d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    
    // aaaaaaaaaaaa debug passed
    // #ifndef LONGERKMER
    // #ifndef SHORTERKMER
    // cudaStreamSynchronize(stream);
    // T_kmer *tmp = new T_kmer[skms_store.kmer_cnt];
    // cudaMemcpy(tmp, d_kmers, sizeof(T_kmer)*skms_store.kmer_cnt, cudaMemcpyDeviceToHost);
    // T_kmer t_min = 0xFFFFFFFFFFFFFFFFu, t_max = 0;
    // for (int t=0; t<skms_store.kmer_cnt; t++) {
    //     if (tmp[t]<t_min) t_min=tmp[t];
    //     if (tmp[t]>t_max) t_max=tmp[t];
    // }
    // cerr<<"PART"<<skms_store.id<<" min "<<t_min<<" max "<<t_max<<endl;
    // delete tmp;
    // #endif
    // #endif

    return;
}

__device__  T_kmer _cano_kmer(const T_kmer& x, const T_kvalue k) {
    T_kmer x1 = ~x, res=0;
    for (T_kvalue i=0; i<k; i++) {
        res = (res << 2) | (x1 & 0b11);
        x1 = x1 >> 2;
    }
    return res < x ? res : x;
}
__global__ void canonical_kmer(T_kmer *d_kmers, const T_kvalue k, const size_t n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_t = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += n_t) {
        d_kmers[i] = _cano_kmer(d_kmers[i], k);
    }
}
template<typename T_data> // inferred typename
__host__ void CubSort_db(cub::DoubleBuffer<T_data> &d_db, MyDevicePtr &d_temp_storage, T_kvalue k, T_skm_partsize N, cudaStream_t &stream) {
    size_t   temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_db, N, 0, k*2));
    d_temp_storage.use(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp_storage.ptr, temp_storage_bytes, d_db, N, 0, k*2, stream));
}
#ifdef LONGERKMER
struct CompareOpT {
    template <typename T>
    __device__ __forceinline__ bool operator()(const T &lhs, const T &rhs) const {
        return lhs < rhs;
    }
};
__host__ void CubMergeSort(T_kmer *d_in, MyDevicePtr &d_temp, T_skm_partsize N, cudaStream_t &stream) {
    size_t   temp_storage_bytes = 0;
    // Check the required buffer temp size and allocate the buffer:
    CompareOpT custom_op;
    CUDA_CHECK(cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, d_in, N, custom_op));
    d_temp.use(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceMergeSort::SortKeys(d_temp.ptr, temp_storage_bytes, d_in, N, custom_op, stream));
}
#endif

__host__ void CubRLE(T_skm_partsize N, T_kmer *d_in_unique_out, T_kmer_cnt *d_counts_out, T_skm_partsize *d_num_runs_out, MyDevicePtr &d_temp_storage, cudaStream_t stream) {

    T_kmer *d_in = d_in_unique_out;
    T_kmer *d_unique_out = d_in_unique_out;

    // Determine requirements and allocate temporary device storage
    size_t   temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, N));
    d_temp_storage.use(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_temp_storage.ptr, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, N, stream));
}

/// @brief Assign corresponding position in d_out with 1 if the value in d_in pass the filter, otherwise 0.
/// @param d_in     [2, 4, 3, 5, 1, 8]
/// @param d_out    [0, 1, 1, 1, 0, 0]
/// @param N        6
/// @param low_thr  3
/// @param max_thr  5
/// @return 
__global__ void ClearLowHighAbundance(T_kmer_cnt *d_counts, T_skm_partsize N, T_kmer_cnt low_thr, T_kmer_cnt max_thr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int n_t = blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += n_t) {
        if (d_counts[i] < low_thr || d_counts[i] > max_thr) 
            d_counts[i] = 0;
    }
}

struct IsNotZero {
    CUB_RUNTIME_FUNCTION __forceinline__ __host__ __device__
    bool operator() (const T_kmer_cnt &a) const { return a != 0; }
};
__host__ void FilterOutKmers(T_kmer_cnt *d_cnt, T_kmer *d_kmer, T_skm_partsize N, T_skm_partsize *d_counts_out, MyDevicePtr &d_temp_storage, cudaStream_t stream) {
    size_t   temp_storage_bytes = 0;
    // Filter out low abundance k-mers
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, d_kmer, d_cnt, d_counts_out, N));
    d_temp_storage.use(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage.ptr, temp_storage_bytes, d_kmer, d_cnt, d_counts_out, N, stream));
    // Filter out the counts of low abundance k-mers
    IsNotZero op;
    CUDA_CHECK(cub::DeviceSelect::If(nullptr, temp_storage_bytes, d_cnt, d_counts_out, N, op));
    d_temp_storage.use(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::If(d_temp_storage.ptr, temp_storage_bytes, d_cnt, d_counts_out, N, op, stream));
}

thread_local vector<MyDevicePtr> kmers_d_vec(PAR.n_streams_p2); // save k-mer and count result (large xGB)
thread_local vector<MyDevicePtr> temp_d_vec(PAR.n_streams_p2);  // as CUB temp space (small x0MB)
thread_local vector<MyDevicePtr> d_num_runs_out(PAR.n_streams_p2);  // number of unique kmers

/// @brief 
/// @param k K-value of kmer.
/// @param skms_stores SKM partitions to be processed.
/// @param gpars GPU parameters.
/// @param kmer_min_freq 
/// @param kmer_max_freq 
/// @param kmc_result_curthread 
/// @param avg_kmer_cnt Average number of kmers per partion, this value is for VRAM allocation.
/// @param gpuid 
/// @param tid 
/// @return 
__host__ size_t kmc_counting_GPU_streams (T_kvalue k,
                               vector<SKMStoreNoncon*> skms_stores, CUDAParams &gpars,
                               T_kmer_cnt kmer_min_freq, T_kmer_cnt kmer_max_freq,
                               string result_file_prefix, size_t avg_kmer_cnt, int gpuid, int tid) {
    
    CUDA_CHECK(cudaSetDevice(gpuid));
    
    int i, n_streams = skms_stores.size();
    cudaStream_t streams[n_streams];

    // ---- 0. Extract kmers from SKMStore: ---- 
    for (i=0; i<n_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        
        // VRAM allocation
        size_t vram_alloc_kmer_cnt = max(skms_stores[i]->kmer_cnt, avg_kmer_cnt*3/2);
        #ifndef LONGERKMER
        // For CUB radix sort, saving kmer counts into kmer space (allocate d_kmer 2*N*T_kmer)
        size_t vram_required = max(max(
            vram_alloc_kmer_cnt * sizeof(T_kmer)*2, 
            vram_alloc_kmer_cnt * (sizeof(T_kmer) + sizeof(T_kmer_cnt))), 
            vram_alloc_kmer_cnt * sizeof(T_kmer) + skms_stores[i]->tot_size_bytes
        );
        #else
        // For CUB merge sort, saving kmer counts into temp space (allocate d_kmer only N*T_kmer)
        size_t vram_required = vram_alloc_kmer_cnt * (sizeof(T_kmer) + sizeof(T_kmer_cnt)); // +T_kmer_cnt for RLE count result
        size_t temp_required = max(max(
            size_t(vram_required * 1.1), 
            vram_alloc_kmer_cnt * (sizeof(T_kmer) + sizeof(T_kmer_cnt))),
            skms_stores[i]->tot_size_bytes
        );
        temp_d_vec[i].use(temp_required, streams[i]);
        #endif
        // *2 for cub sort (cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);)
        // *1 size(kmer)+ size(T_kmer_cnt) for ResultFilter
        kmers_d_vec[i].use(vram_required, streams[i]);
        d_num_runs_out[i].use(sizeof(T_skm_partsize), streams[i]);
    }
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        cudaStreamSynchronize(streams[i]); // to wait MyDevicePtr.use
        // cerr<<"S"<<i<<" allocated:"<<kmers_d_vec[i].size<<" require:"<<sizeof(T_kmer)*skms_stores[i]->kmer_cnt+skms_stores[i]->tot_size_bytes <<endl;
        #ifndef LONGERKMER
        Extract_Kmers(*skms_stores[i], k, 
            reinterpret_cast<u_char*>((static_cast<T_kmer*>(kmers_d_vec[i].ptr))+skms_stores[i]->kmer_cnt), 
            static_cast<T_kmer*>(kmers_d_vec[i].ptr), 
            streams[i], gpars.BpG2, gpars.TpB2);
        #else
        Extract_Kmers(*skms_stores[i], k, 
            static_cast<u_char*>(temp_d_vec[i].ptr), 
            static_cast<T_kmer*>(kmers_d_vec[i].ptr), 
            streams[i], gpars.BpG2, gpars.TpB2);
        #endif
    }

    // ---- Count K-mers ----
    #ifndef LONGERKMER
    vector<cub::DoubleBuffer<T_kmer>> d_db_vec(n_streams);
    #endif
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        
        // CUDA_CHECK(cudaStreamSynchronize(streams[i])); // maybe don't need this?
        // ---- 1. convert to canonical kmers ---- 
        canonical_kmer<<<gpars.BpG2, gpars.TpB2, 0, streams[i]>>>(static_cast<T_kmer *>(kmers_d_vec[i].ptr), k, skms_stores[i]->kmer_cnt);
        // thrust::transform(thrust::device.on(streams[i]), kmers_d_vec[i].begin(), kmers_d_vec[i].end(), ik, kmers_d_vec[i].begin(), canonicalkmer());

        // ---- 2. sort: [ABCBBAC] -> [AABBBCC] (kmers_d) ---- 
        #ifndef LONGERKMER
        d_db_vec[i] = cub::DoubleBuffer<T_kmer>(static_cast<T_kmer*>(kmers_d_vec[i].ptr), static_cast<T_kmer*>(kmers_d_vec[i].ptr)+skms_stores[i]->kmer_cnt);
        CubSort_db(d_db_vec[i], temp_d_vec[i], k, skms_stores[i]->kmer_cnt, streams[i]);
        // after sorting, kmers_d_vec = (N*T_kmer) {sorted kmers...}/{temp}, (N*T_kmer) {temp}/{sorted kmers...}
        #else
        CubMergeSort(static_cast<T_kmer*>(kmers_d_vec[i].ptr), temp_d_vec[i], skms_stores[i]->kmer_cnt, streams[i]);
        // after sorting, kmers_d_vec = (N*T_kmer) {sorted kmers...}
        #endif
    }

    vector<T_skm_partsize> N_kmer_cleared(n_streams);
    vector<T_kmer*> d_unique_kmers(n_streams);
    vector<T_kmer_cnt*> d_counts(n_streams);
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        CUDA_CHECK(cudaStreamSynchronize(streams[i])); // to wait CubSort_db (d_db)

        // ---- 3. CUB RLE: [AABBBCC] -> [ABC] [232] ----
        #ifndef LONGERKMER
        d_unique_kmers[i] = d_db_vec[i].Current();
        d_counts[i] = reinterpret_cast<T_kmer_cnt*>(d_db_vec[i].Alternate());
        
        // aaaaaaaaaaaa
        // print first and last kmer to check if sorting is corrects
        // T_kmer first_kmer, last_kmer;
        // cudaMemcpyAsync(&first_kmer, d_unique_kmers[i], sizeof(T_kmer), cudaMemcpyDeviceToHost, streams[i]);
        // cudaMemcpyAsync(&last_kmer, d_unique_kmers[i]+skms_stores[i]->kmer_cnt-1, sizeof(T_kmer), cudaMemcpyDeviceToHost, streams[i]);
        // cudaStreamSynchronize(streams[i]);
        // logger->log("S"+to_string(i)+" Part "+to_string(skms_stores[i]->id)+" KMER "+to_string(first_kmer)+", "+to_string(last_kmer));

        #else
        d_unique_kmers[i] = static_cast<T_kmer*>(kmers_d_vec[i].ptr);
        d_counts[i] = reinterpret_cast<T_kmer_cnt*>(static_cast<T_kmer*>(kmers_d_vec[i].ptr) + skms_stores[i]->kmer_cnt);
        #endif
        CubRLE(skms_stores[i]->kmer_cnt, 
            d_unique_kmers[i], 
            d_counts[i], 
            static_cast<T_skm_partsize*>(d_num_runs_out[i].ptr), 
            temp_d_vec[i], 
            streams[i]
        );

        CUDA_CHECK(cudaMemcpyAsync(&N_kmer_cleared[i], d_num_runs_out[i].ptr, sizeof(T_skm_partsize), cudaMemcpyDeviceToHost, streams[i]));
    }

    // ---- Filter out low/high abundance kmers ----
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        CUDA_CHECK(cudaStreamSynchronize(streams[i])); // for N_kmer_cleared[i]
        
        // ---- 4. Filter out low/high abundance kmers ----
        skms_stores[i]->clear_skm_data();
        logger->log("S"+to_string(i)+" Part "+to_string(skms_stores[i]->id)+" "+to_string(N_kmer_cleared[i])+"/"+to_string(skms_stores[i]->kmer_cnt));
        ClearLowHighAbundance<<<gpars.BpG2, gpars.TpB2, 0, streams[i]>>>(d_counts[i], N_kmer_cleared[i], kmer_min_freq, kmer_max_freq);
        FilterOutKmers(d_counts[i], d_unique_kmers[i], N_kmer_cleared[i], static_cast<T_skm_partsize*>(d_num_runs_out[i].ptr), temp_d_vec[i], streams[i]);
        CUDA_CHECK(cudaMemcpyAsync(&N_kmer_cleared[i], d_num_runs_out[i].ptr, sizeof(T_skm_partsize), cudaMemcpyDeviceToHost, streams[i]));
    }

    // ---- Copy results to host ----
    size_t return_value = 0;
    T_kmer *h_distinct_kmer;
    T_kmer_cnt *h_counts;
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        CUDA_CHECK(cudaStreamSynchronize(streams[i])); // for N_kmer_cleared[i]
        
        // ---- 5. Copy results to host ----
        return_value += N_kmer_cleared[i];
        if (result_file_prefix.length() > 1) { // output to file, can be optimized with better pipelining
            h_distinct_kmer = new T_kmer[N_kmer_cleared[i]];
            h_counts = new T_kmer_cnt[N_kmer_cleared[i]];
            CUDA_CHECK(cudaMemcpy(h_distinct_kmer, d_unique_kmers[i], N_kmer_cleared[i]*sizeof(T_kmer), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_counts, d_counts[i], N_kmer_cleared[i]*sizeof(T_kmer_cnt), cudaMemcpyDeviceToHost));
            int part_idx = skms_stores[i]->id;
            T_skm_partsize distinct_kmer_cnt = N_kmer_cleared[i];
            std::async([h_distinct_kmer, h_counts, part_idx, result_file_prefix, distinct_kmer_cnt](){
                FILE *fp = fopen((result_file_prefix + std::to_string(part_idx) + ".gkc").c_str(), "w");
                assert(fp != NULL);
                fwrite(&distinct_kmer_cnt, sizeof(T_skm_partsize), 1, fp);
                fwrite(h_distinct_kmer, sizeof(T_kmer), distinct_kmer_cnt, fp);
                fwrite(h_counts, sizeof(T_kmer_cnt), distinct_kmer_cnt, fp);
                fclose(fp);
                delete[] h_distinct_kmer;
                delete[] h_counts;
            }); // std::launch::async,
        }
        // for comparison:
        // kmers_d_vec[i].free(streams[i]);
        // temp_d_vec[i].free(streams[i]);
        // d_num_runs_out[i].free(streams[i]);
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return return_value; // total distinct kmer
}

__host__ size_t GPUVram(int did) {
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    return avail;
}