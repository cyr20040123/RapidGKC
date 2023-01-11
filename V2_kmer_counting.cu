// #define TIMER

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

#include "nvcomp/gdeflate.hpp"
#include "nvcomp.hpp"

#include <fcntl.h> // open
#include <unistd.h> // close
// #include "cufile.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
// #include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

// #include "types.h"
// #include "V2_superkmers.hpp"
// #include <vector>
#include "kmer_counting.hpp"
#include "utilities.hpp"
#include <fstream>
using namespace std;
using namespace nvcomp;

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
// __global__ void GPU_Extract_Kmers (byte* d_skms, size_t tot_bytes, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
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
//         end_byte_pos = (end_byte_pos < tot_bytes) * end_byte_pos + (end_byte_pos >= tot_bytes) * tot_bytes;
//         if (beg_byte_pos < tot_bytes) {
//             // printf("%llu process %llu %llu (%d %llu)\n",tot_bytes, beg_byte_pos, end_byte_pos, tid, i);
//             // printf("%llu %llu\n",beg_byte_pos,end_byte_pos);
//             _process_bytes(beg_byte_pos, end_byte_pos, d_skms, d_kmers, d_kmer_store_pos, k);
//         }
//     }
//     return;
// }

__global__ void GPU_Extract_Kmers (byte* d_skms, size_t tot_bytes, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
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
            if (tot_bytes == 25109) printf("[%d] %llu-%llu\n",tid,beg_byte_pos,end_byte_pos);
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

// TODO: use cufile to load file?
// __host__ byte* load_SKM_from_file_GDS (SKMStoreNoncon &skms_store) {
//     byte* d_skms;
//     CUDA_CHECK(cudaMalloc((void**) &(d_skms), skms_store.tot_size_bytes));
    
//     int fd;
//     CUfileError_t cf_status;
//     CUfileDescr_t cf_descr;
//     CUfileHandle_t cf_handl;
//     ssize_t bytes_read;

//     // cf_status = cuFileDriverOpen();
//     // CUFILE_STATUS_CHECK(cf_status, __LINE__);
//     fd = open(skms_store.filename.c_str(), O_RDONLY|O_DIRECT);
//     cerr<<fd<<endl;
//     cf_descr.handle.fd = fd;
//     cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
//     cf_status = cuFileHandleRegister(&cf_handl, &cf_descr); // 5030
//     CUFILE_STATUS_CHECK(cf_status, __LINE__);
//     cf_status = cuFileBufRegister(d_skms, skms_store.tot_size_bytes, 0);
//     CUFILE_STATUS_CHECK(cf_status, __LINE__);
//     WallClockTimer wct_file;
//     bytes_read = cuFileRead(cf_handl, (void*) d_skms, skms_store.tot_size_bytes, 0, 0); // TODO: where is cuFileReadAsync???
//     int speed = skms_store.tot_size_bytes*1000/1048576/wct_file.stop(true);
//     logger->log(skms_store.filename+" File loading speed (MB/s): "+to_string(speed)+" "+to_string(bytes_read)+" "+to_string(skms_store.tot_size_bytes));
//     assert(bytes_read == skms_store.tot_size_bytes);
//     // cf_status = cuFileReadAsync(cf_handl, (void*) d_skms, &(skms_store.tot_size_bytes), 0, 0, &bytes_read, stream); // must enable NVreg_EnableStreamMemOPs=1
//     cf_status = cuFileBufDeregister(d_skms);
//     CUFILE_STATUS_CHECK(cf_status, __LINE__);
//     cuFileHandleDeregister(cf_handl);
//     close(fd);
//     return d_skms;
// }
__host__ byte* load_SKM_from_file (SKMStoreNoncon &skms_store) {
    byte* d_skms;
    CUDA_CHECK(cudaMalloc((void**) &(d_skms), skms_store.tot_size_bytes));
    FILE* fp;
    fp = fopen(skms_store.filename.c_str(), "rb");
    assert(fp);
    byte* tmp;
    tmp = new byte[skms_store.tot_size_bytes];
    assert(fread(tmp, 1, skms_store.tot_size_bytes, fp)==skms_store.tot_size_bytes);
    CUDA_CHECK(cudaMemcpy(d_skms, tmp, skms_store.tot_size_bytes, cudaMemcpyHostToDevice));
    delete tmp;
    fclose(fp);
    return d_skms;
}

void Extract_Kmers (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* &d_kmers, cudaStream_t &stream, int BpG=8, int TpB=256) {
    // cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));
    
    byte* d_skms;
    
    unsigned long long *d_kmer_store_pos;
    CUDA_CHECK(cudaMallocAsync((void**) &(d_kmer_store_pos), sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_kmer_store_pos, 0, sizeof(unsigned long long), stream));

    // ---- copy skm chunks H2D ----
    if (skms_store.to_file) d_skms = load_SKM_from_file(skms_store);
    else {
        CUDA_CHECK(cudaMallocAsync((void**) &(d_skms), skms_store.tot_size_bytes, stream));
        int i;
        byte *d_store_pos = d_skms;
        for (i=0; i<skms_store.skm_chunk_bytes.size(); i++) {
            CUDA_CHECK(cudaMemcpyAsync(d_store_pos, skms_store.skm_chunks[i], skms_store.skm_chunk_bytes[i], cudaMemcpyHostToDevice, stream));
            d_store_pos += skms_store.skm_chunk_bytes[i];
        }
    }
    // cerr<<"debug2"<<endl;
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // ---- GPU work ----
    GPU_Extract_Kmers<<<BpG, TpB, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    // GPU_Extract_Kmers_test<<<BpG, TpB, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    
    CUDA_CHECK(cudaFreeAsync(d_skms, stream));
    CUDA_CHECK(cudaFreeAsync(d_kmer_store_pos, stream));
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

// deprecated
__host__ size_t kmc_counting_GPU (T_kvalue k,
                               SKMStoreNoncon &skms_store, CUDAParams &gpars,
                               unsigned short kmer_min_freq, unsigned short kmer_max_freq,
                               _out_ vector<T_kmc> &kmc_result_curthread,
                               bool GPU_compression = false) {
    // using CUDA Thrust
    int gpuid = (gpars.device_id++)%gpars.n_devices;
    CUDA_CHECK(cudaSetDevice(gpuid));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    logger->log("GPU "+to_string(gpuid)+" Counting partition "+to_string(skms_store.id), Logger::LV_INFO);
    
    if (skms_store.tot_size_bytes == 0) return 0;
    
    // 0. Extract kmers from SKMStore: (~65-85% time)
    #ifdef TIMER
    double wcts[10];
    WallClockTimer wct0;
    #endif
    
    // V2 - bulk: 2.5-2.7s
    // thrust::device_vector<T_kmer> kmers_d(kmers_h);
    // cout<<"Begin extract kmer"<<endl;
    thrust::device_vector<T_kmer> kmers_d(skms_store.kmer_cnt);
    T_kmer *d_kmers_data = thrust::raw_pointer_cast(kmers_d.data());
    if (GPU_compression) Extract_Kmers_Compressed(skms_store, k, d_kmers_data, stream, gpars.NUM_BLOCKS_PER_GRID, gpars.NUM_THREADS_PER_BLOCK, gpuid);
    else {Extract_Kmers(skms_store, k, d_kmers_data, stream, 8, 256); CUDA_CHECK(cudaStreamSynchronize(stream));}
    size_t tot_kmers = kmers_d.size();
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
    
    // 4+. copy sorted kmers back to host
    // thrust::host_vector<T_kmer> sorted_kmers_h = kmers_d;
    auto new_end_sorted_cleared_kmers_d = thrust::remove_if(thrust::device.on(stream), kmers_d.begin(), kmers_d.end(), same_flag_d.begin(), thrust::identity<bool>()); // new_end_idx_d is an iterator
    thrust::host_vector<T_kmer> sorted_kmers_h(kmers_d.begin(), new_end_sorted_cleared_kmers_d);
    volatile T_kmer tmp_kmer = sorted_kmers_h[0];
    
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
        //     kmc_result_curthread.push_back({sorted_kmers_h[i], cnt});
        // }
    }
    assert(total_kmer_cnt == skms_store.kmer_cnt);
    #ifdef TIMER
    wcts[6] = wct6.stop();
    cout<<wcts[0]<<"\t"<<wcts[1]<<"\t"<<wcts[2]<<"\t"<<wcts[3]<<"\t"<<wcts[4]<<"\t"<<wcts[5]<<"\t"<<wcts[6]<<endl;
    #endif
    delete &skms_store;//
    return idx_h.size()-1; // total distinct kmer
    // return total_kmer_cnt; // total kmer
}

__host__ size_t kmc_counting_GPU_streams (T_kvalue k,
                               vector<SKMStoreNoncon*> skms_stores, CUDAParams &gpars,
                               unsigned short kmer_min_freq, unsigned short kmer_max_freq,
                               _out_ vector<T_kmc> kmc_result_curthread [],
                               bool GPU_compression = false) {
    // using CUDA Thrust
    int gpuid = (gpars.device_id++)%gpars.n_devices;
    CUDA_CHECK(cudaSetDevice(gpuid));
    
    size_t return_value = 0;
    int i, n_streams = skms_stores.size();
    cudaStream_t streams[n_streams];

    vector<thrust::device_vector<T_kmer>> kmers_d_vec(n_streams); // for 0
    vector<size_t> tot_kmers(n_streams);
    string logs = "GPU "+to_string(gpuid)+":";
    for (i=0; i<n_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        // logger->log("GPU "+to_string(gpuid)+" Stream "+to_string(i)+" counting Partition "+to_string(skms_stores[i]->id), Logger::LV_INFO);
        logs += "\tS "+to_string(i)+" Part "+to_string(skms_stores[i]->id)+" "+to_string(skms_stores[i]->tot_size_bytes)+"|"+to_string(skms_stores[i]->kmer_cnt);
        if (skms_stores[i]->tot_size_bytes != 0) {
            // ---- 0. Extract kmers from SKMStore: ---- 
            kmers_d_vec[i] = thrust::device_vector<T_kmer>(skms_stores[i]->kmer_cnt);
            T_kmer *d_kmers_data = thrust::raw_pointer_cast(kmers_d_vec[i].data());
            if (GPU_compression) Extract_Kmers_Compressed(*skms_stores[i], k, d_kmers_data, streams[i], gpars.NUM_BLOCKS_PER_GRID, gpars.NUM_THREADS_PER_BLOCK, gpuid);
            else Extract_Kmers(*skms_stores[i], k, d_kmers_data, streams[i], gpars.NUM_BLOCKS_PER_GRID, gpars.NUM_THREADS_PER_BLOCK);
            tot_kmers[i] = kmers_d_vec[i].size();
        }
    }
    logger->log(logs, Logger::LV_INFO);

    thrust::constant_iterator<T_kvalue> ik(k);
    vector<thrust::device_vector<bool>> same_flag_d_vec(n_streams); // for 3
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes != 0) {
            // CUDA_CHECK(cudaStreamSynchronize(streams[i])); // maybe don't need this?
            // ---- 1. convert to canonical kmers ---- 
            thrust::transform(thrust::device.on(streams[i]), kmers_d_vec[i].begin(), kmers_d_vec[i].end(), ik, kmers_d_vec[i].begin(), canonicalkmer());
            // ---- 2. sort: [ABCBBAC] -> [AABBBCC] (kmers_d) ---- 
            thrust::sort(thrust::device.on(streams[i]), kmers_d_vec[i].begin(), kmers_d_vec[i].end()/*, thrust::greater<T_kmer>()*/);
            skms_stores[i]->clear_skm_data(); // only when gpu compression and in-mem
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
    }
    
    // validation:
    for (i=0; i<n_streams; i++) {
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
        }
        assert(total_kmer_cnt == skms_stores[i]->kmer_cnt);
    }
    for (i=0; i<n_streams; i++) {
        if (skms_stores[i]->tot_size_bytes == 0) continue;
        return_value += idx_h_vec[i].size()-1;
    }
    for (i=0; i<n_streams; i++) {
        delete skms_stores[i];//
    }
    return return_value; // total distinct kmer
}
