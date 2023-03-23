#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
// #include <atomic>
#include "utilities.hpp"
#include "types.h"
#include "gpu_kmercounting.h"
using namespace std;

#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    exit(1); \
}
#define CHECK_PTR_RETURN(ptr, ...) { \
if (ptr == NULL) { \
    printf (__VA_ARGS__); \
    printf ("Error in returned value: NULL\n"); \
    exit (1); \
} }

/* Prime table on the GPU */
__constant__ static const unsigned int prime_tab[30] = {
            7,
           13,
           31,
           61,
          127,
          251,
          509,
         1021,
         2039,
         4093,
         8191,
        16381,
        32749,
        65521,
       131071,
       262139,
       524287,
      1048573,
      2097143,
      4194301,
      8388593,
     16777213,
     33554393,
     67108859,
    134217689,
    268435399,
    536870909,
   1073741789,
   2147483647,
  /* Avoid "decimal constant so large it is unsigned" for 4294967291.  */
   0xfffffffb
};

/* Prime table on the host */
static const unsigned int hprime_tab[30] = {
            7,
           13,
           31,
           61,
          127,
          251,
          509,
         1021,
         2039,
         4093,
         8191,
        16381,
        32749,
        65521,
       131071,
       262139,
       524287,
      1048573,
      2097143,
      4194301,
      8388593,
     16777213,
     33554393,
     67108859,
    134217689,
    268435399,
    536870909,
   1073741789,
   2147483647,
  /* Avoid "decimal constant so large it is unsigned" for 4294967291.  */
   0xfffffffb
};

//* Compute the secondary hash for HASH given HASHTAB's current size.  *
__device__ static inline unsigned int
hashtab_mod_m2 (unsigned int hash, uint size_prime_index) {
    return 1 + hash % (prime_tab[size_prime_index] - 2);
}

// MURMURHASH:
#define DEFAULT_SEED 3735928559

#define ONE32   0xFFFFFFFFUL
#define LL(v)   (v##ULL)
#define ONE64   LL(0xFFFFFFFFFFFFFFFF)

#define T32(x)  ((x) & ONE32)
#define T64(x)  ((x) & ONE64)

#define ROTL32(v, n)   \
	(T32((v) << ((n)&0x1F)) | T32((v) >> (32 - ((n)&0x1F))))

#define ROTL64(v, n)   \
	(T64((v) << ((n)&0x3F)) | T64((v) >> (64 - ((n)&0x3F))))

#define UNIT_LENGTH 2 // ULL: 2 int128: 4
#define UNIT_BYTES 4


/*FNV HASH FUNCTION*/

#define FNV_PRIME_32 16777619UL
#define OFFSET_BASIS_32 2166136261UL

__device__ __host__ static uint fnv_hash_32 (const char * key)
{
	uint h = OFFSET_BASIS_32;
	for (int i=0; i<UNIT_BYTES; i++)
	{
		h ^= *key++;
		h *= FNV_PRIME_32;
	}

	return h;
}

__device__ __host__ static uint
murmur_hash3_32 (const uint * key, uint seed)
{
    int i;

    uint h = seed;
    uint k;

    uint c1 = 0xcc9e2d51;
    uint c2 = 0x1b873593;

    for (i = 0; i < UNIT_LENGTH; i++)
    {
        k = *key++;
        k *= c1;
        k = ROTL32(k,15);
        k *= c2;
        h ^= k;
        h = ROTL32(h,13);
        h = h*5+0xe6546b64;
    }

    h ^= (UNIT_LENGTH * UNIT_BYTES);

    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;

}


#define SLOT_WIDTH 32

typedef T_kmer T_key;
typedef unsigned int T_value;
struct T_entry {
    T_key key;
    T_value value;
    int state_flag; // 0: empty; 1: locked; 2: occupied
};
// struct T_entry_cpu {
//     T_key key;
//     T_value value;
//     atomic<int> state_flag; // 0: empty; 1: locked; 2: occupied
// };

__device__ inline T_key insert_int_key (volatile T_key *key, T_key new_key) {
    if (*key != 0) printf("error!!!!!!!!!!\n");
    *key = new_key; // volatile T
    return (*key);
}

__device__ __host__ inline int compare_keys (T_key *key1, T_key *key2) {
    return (*key1) == (*key2);
}

__device__ inline T_value increment_cnt (T_value *value) {
    return atomicAdd(value, 1);
}

//* Replace-value based hash table with one thread *
__device__ bool insert_key (T_key * key, unsigned long long *distinct_cnt, T_entry * dtab, unsigned int dsize, unsigned int d_size_prime_index) {
    unsigned int hash = murmur_hash3_32 ((const uint *)key, DEFAULT_SEED);
    unsigned int index1, index2;
    T_entry * entry;
    T_entry * entries = dtab;
    unsigned int size_prime_index = d_size_prime_index;
    unsigned int table_size = dsize;

    int flag = -1;
    // index1 = hashtab_mod (hash, size_prime_index); // primary index
    index1 = hash % prime_tab[size_prime_index];
    if (index1 >= table_size) {
        printf ("Error: index greater than hash table size!\n");
        return false;
    }
    index2 = fnv_hash_32((char *)key) % SLOT_WIDTH; // secondary index

    //* first try of insertion: *
    entry = entries + index1 * SLOT_WIDTH + index2;
    flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
    if (flag == 0) { // empty
        insert_int_key (&(entry->key), *key);
        assert(atomicCAS (&(entry->state_flag), 1, 2) == 1); // locked -> occupied
    }
    while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {} // occupied validation

    if (compare_keys (key, &(entry->key))) {
        if (increment_cnt (&(entry->value)) == 1) atomicAdd(distinct_cnt, 1);
        return true;
    }
    //* otherwise, probe secondary hash slots *
    for (int s=0; s<SLOT_WIDTH; s++) {
        if (s==index2) continue;
        entry = entries + index1*SLOT_WIDTH + s;
        flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
        if (flag == 0) { // empty
            insert_int_key (&(entry->key), *key);
            assert(atomicCAS (&(entry->state_flag), 1, 2)==1); // locked -> occupied
        }
        while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {} // occupied validation
        
        if (compare_keys(key, &(entry->key))) {
            if (increment_cnt (&(entry->value)) == 1) atomicAdd(distinct_cnt, 1);
            return true;
        }
    }

    //* first try fails, continue with secondary hash2 *
    unsigned int hash2 = hashtab_mod_m2 (hash, size_prime_index);
    for (size_t i=0; i<table_size; i++) {
        index1 += hash2;
        if (index1 >= table_size) index1 -= table_size;

        T_entry * entry = entries + index1*SLOT_WIDTH + index2;
        flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
        if (flag == 0) { // empty
            insert_int_key (&(entry->key), *key);
            assert(atomicCAS (&(entry->state_flag), 1, 2) == 1); // locked -> occupied
        }
        while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {} // occupied validation
        if (compare_keys(key, &(entry->key))) {
            if (increment_cnt (&(entry->value)) == 1) atomicAdd(distinct_cnt, 1);
            return true;
        }
        //* otherwise, probe secondary hash slots *
        for (int s=0; s<SLOT_WIDTH; s++) {
            if (s==index2) continue;
            entry = entries + index1*SLOT_WIDTH + s;
            flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
            if (flag == 0) { // empty
                insert_int_key (&(entry->key), *key);
                assert(atomicCAS (&(entry->state_flag), 1, 2) == 1); // locked -> occupied
            }
            while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {}
            if (compare_keys(key, &(entry->key))) {
                if (increment_cnt (&(entry->value)) == 1) atomicAdd(distinct_cnt, 1);
                return true;
            }
        }
    }
    return false;
}

// // TODO here
// bool insert_key_cpu (T_key *key, atomic<size_t> *distinct_cnt, T_entry_cpu *tab, unsigned int size, unsigned int size_prime_index) {
//     unsigned int hash = murmur_hash3_32 ((const uint *)key, DEFAULT_SEED);
//     unsigned int index1, index2;
//     T_entry_cpu *entry;
//     T_entry_cpu *entries = tab;
//     unsigned int size_prime_index = size_prime_index;
//     unsigned int table_size = size;

//     int flag = -1;
//     // index1 = hashtab_mod (hash, size_prime_index); // primary index
//     index1 = hash % prime_tab[size_prime_index];
//     if (index1 >= table_size) {
//         printf ("Error: index greater than hash table size!\n");
//         return false;
//     }
//     index2 = fnv_hash_32((char *)key) % SLOT_WIDTH; // secondary index

//     //* first try of insertion: *
//     entry = entries + index1 * SLOT_WIDTH + index2;
//     // flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
//     flag = entry->state_flag.compare_exchange_strong(0, 1);
//     if (flag == 0) { // empty
//         insert_int_key (&(entry->key), *key);
//         // assert(atomicCAS (&(entry->state_flag), 1, 2) == 1); // locked -> occupied
//         assert(entry->state_flag.compare_exchange_strong(1, 2));
//     }
//     // while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {} // occupied validation
//     while (!entry->state_flag.compare_exchange_strong(2, 2));

//     if (compare_keys (key, &(entry->key))) {
//         if (increment_cnt (&(entry->value)) == 1) (*distinct_cnt)++;
//         return true;
//     }
//     //* otherwise, probe secondary hash slots *
//     for (int s=0; s<SLOT_WIDTH; s++) {
//         if (s==index2) continue;
//         entry = entries + index1*SLOT_WIDTH + s;
//         flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
//         if (flag == 0) { // empty
//             insert_int_key (&(entry->key), *key);
//             assert(atomicCAS (&(entry->state_flag), 1, 2)==1); // locked -> occupied
//         }
//         while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {} // occupied validation
        
//         if (compare_keys(key, &(entry->key))) {
//             if (increment_cnt (&(entry->value)) == 1) (*distinct_cnt)++;
//             return true;
//         }
//     }

//     //* first try fails, continue with secondary hash2 *
//     unsigned int hash2 = hashtab_mod_m2 (hash, size_prime_index);
//     for (size_t i=0; i<table_size; i++) {
//         index1 += hash2;
//         if (index1 >= table_size) index1 -= table_size;

//         T_entry * entry = entries + index1*SLOT_WIDTH + index2;
//         flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
//         if (flag == 0) { // empty
//             insert_int_key (&(entry->key), *key);
//             assert(atomicCAS (&(entry->state_flag), 1, 2) == 1); // locked -> occupied
//         }
//         while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {} // occupied validation
//         if (compare_keys(key, &(entry->key))) {
//             if (increment_cnt (&(entry->value)) == 1) (*distinct_cnt)++;
//             return true;
//         }
//         //* otherwise, probe secondary hash slots *
//         for (int s=0; s<SLOT_WIDTH; s++) {
//             if (s==index2) continue;
//             entry = entries + index1*SLOT_WIDTH + s;
//             flag = atomicCAS(&(entry->state_flag), 0, 1); // empty -> locked
//             if (flag == 0) { // empty
//                 insert_int_key (&(entry->key), *key);
//                 assert(atomicCAS (&(entry->state_flag), 1, 2) == 1); // locked -> occupied
//             }
//             while ((flag = atomicCAS(&(entry->state_flag), 2, 2)) != 2) {}
//             if (compare_keys(key, &(entry->key))) {
//                 if (increment_cnt (&(entry->value)) == 1) (*distinct_cnt)++;
//                 return true;
//             }
//         }
//     }
//     return false;
// }


// #define THREADS_PER_BLOCK 1024
// #define MAX_NUM_BLOCKS 16 // maximum number of blocks
// #define MAX_NUM_THREADS (MAX_NUM_BLOCKS*THREADS_PER_BLOCK)

__global__ void insert_batch (T_key *d_keys, size_t num_of_keys, unsigned long long *d_distinct_cnt, T_entry * dtab,  unsigned int size, unsigned int size_prime_index) {
    uint gid = blockDim.x * blockIdx.x + threadIdx.x;

    size_t keys_per_thread = (num_of_keys + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x); // keys per thread
    if (gid==0) printf ("number of keys per thread: %llu\n", keys_per_thread);
    
    for (size_t i = 0; i < keys_per_thread; i++) {
        if (gid * keys_per_thread + i >= num_of_keys) return;
        if (insert_key (&d_keys[gid*keys_per_thread + i], d_distinct_cnt, dtab, size, size_prime_index) == false)
            printf ("CAREFUL!!!!!!! INSERTION FAILS!\n");
    }
}

/* The following function returns an index into the above table of the
   nearest prime number which is greater than N, and near a power of two. */
__host__ static uint higher_prime_index (unsigned long n) {
    unsigned int low = 0;
    unsigned int high = sizeof(hprime_tab) / sizeof(unsigned int);

    while (low != high) {
        unsigned int mid = low + (high - low) / 2;
        if (n > hprime_tab[mid]) low = mid + 1;
        else high = mid;
    }

    /* If we've run out of primes, abort.  */
    if (n > hprime_tab[low]) printf ("Cannot find prime bigger than %lu\n", n);

    return low;
}

void createHashtab (size_t num_of_elems, T_entry* &dtab, unsigned int &size, unsigned int &size_prime_index, cudaStream_t &stream) {
    uint h_size_prime_index;
    unsigned int table_size; //primary table size
    num_of_elems = (num_of_elems + SLOT_WIDTH - 1) / SLOT_WIDTH;
    h_size_prime_index = higher_prime_index (num_of_elems);
    table_size = hprime_tab[h_size_prime_index];

    // cout<<"required vram: "<<sizeof(T_entry) * SLOT_WIDTH * table_size / 1048576 <<endl;
    CUDA_CHECK (cudaMallocAsync ((void**)&dtab, sizeof(T_entry) * SLOT_WIDTH * table_size, stream));
    CUDA_CHECK (cudaMemsetAsync (dtab, 0, sizeof(T_entry) * SLOT_WIDTH * table_size, stream));
    size = table_size;
    size_prime_index = h_size_prime_index;
}

void destoryHashtab (T_entry *dtab, T_key *dkeys, cudaStream_t &stream) {
    CUDA_CHECK(cudaFreeAsync (dtab, stream));
    CUDA_CHECK(cudaFreeAsync (dkeys, stream));
}

// void createHashtab_cpu (size_t num_of_elems, T_entry* &tab, unsigned int &size, unsigned int &size_prime_index) {
//     uint h_size_prime_index;
//     unsigned int table_size; //primary table size
//     num_of_elems = (num_of_elems + SLOT_WIDTH - 1) / SLOT_WIDTH;
//     h_size_prime_index = higher_prime_index (num_of_elems);
//     table_size = hprime_tab[h_size_prime_index];

//     tab = new T_entry[SLOT_WIDTH * table_size];
//     size = table_size; size_prime_index = h_size_prime_index;
// }

// void destoryHashtab_cpu (T_entry *tab) {
//     delete tab;
// }



void check_hash_results (T_entry * tab, size_t size) {
    cout << "HASH TABLE SIZE:\t" << size << endl;
    size_t count=0;
    for (size_t i=0; i<size; i++)
        if (tab[i].state_flag == 2) count++;
    printf ("NUMBER OF ELEMENTS IN HASH TABLE: %lu\n", count);
    cout << "NUMBER OF ELEMENTS IN HASH TABLE:\t" << count << endl;
}

size_t* construct_hashtab_gpu_stream (T_key *keys, size_t num_of_keys, bool keys_from_gpu, cudaStream_t &stream, int grid_size, int block_size)
{
    T_entry * dtab;
    unsigned int size, size_prime_index;

    // WallClockTimer wct0;
    T_key * dkeys = NULL;
    if (keys_from_gpu) {
        dkeys = keys;
    } else {
        CUDA_CHECK (cudaMallocAsync ((void**)&dkeys, sizeof(T_key) * num_of_keys, stream));
        CUDA_CHECK (cudaMemcpyAsync (dkeys, keys, sizeof(T_key) * num_of_keys, cudaMemcpyHostToDevice, stream));
    }

    unsigned long long *d_distinct_cnt;
    CUDA_CHECK (cudaMallocAsync ((void**)&d_distinct_cnt, sizeof(unsigned long long), stream));
    CUDA_CHECK (cudaMemsetAsync (d_distinct_cnt, 0, sizeof(unsigned long long), stream));
    // printf ("Malloc  time: %f\n", (float)wct0.stop());
    
    // WallClockTimer wct;
    // CUDA_CHECK (cudaDeviceSynchronize());
    createHashtab(num_of_keys, dtab, size, size_prime_index, stream);//, device_type::gpu);
    insert_batch<<<grid_size, block_size, 0, stream>>>(dkeys, num_of_keys, d_distinct_cnt, dtab, size, size_prime_index);
    // CUDA_CHECK (cudaDeviceSynchronize());
    // printf ("Hash insert  time: %f\n", (float)wct.stop());
    // above: 30% time

    // printf ("primary table size::::: %u\n", size);
    // // CPU validation:
    // size_t _size = size * SLOT_WIDTH;
    // T_entry * table = (T_entry *) malloc (sizeof(T_entry) * _size);
    // CUDA_CHECK(cudaMemcpy(table, htab, sizeof(T_entry) * _size, cudaMemcpyDeviceToHost));
    // check_hash_results(table, _size);
    // free (table);

    size_t *distinct_cnt = new size_t;//
    CUDA_CHECK (cudaMemcpyAsync(distinct_cnt, d_distinct_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK (cudaFreeAsync(d_distinct_cnt, stream));
    // printf("%llu\n", distinct_cnt);
    destoryHashtab(dtab, dkeys, stream);
    return distinct_cnt;
}


// void construct_hashtab_cpu (T_key * keys, size_t num_of_keys)
// {
//     T_entry *tab;
//     unsigned int size, size_prime_index;
    
//     WallClockTimer wct0;
//     atomic<size_t> distinct_cnt {0};
    
//     WallClockTimer wct;
//     createHashtab_cpu (num_of_keys, tab, size, size_prime_index);
//     for (size_t i=0; i<num_of_keys; i++) {
//         insert_key (&d_keys[gid*keys_per_thread + i], distinct_cnt, dtab, size, size_prime_index);
//     }
//     printf ("CPU hash insert time: %f\n", (float)wct.stop());
//     // above: 30% time

//     printf ("primary table size::::: %u\n", size);
//     // // CPU validation:
//     // size_t _size = size * SLOT_WIDTH;
//     // T_entry * table = (T_entry *) malloc (sizeof(T_entry) * _size);
//     // CUDA_CHECK(cudaMemcpy(table, htab, sizeof(T_entry) * _size, cudaMemcpyDeviceToHost));
//     // check_hash_results(table, _size);
//     // free (table);

//     destoryHashtab(dtab);
// }


#define NULL_POS 0xFFFFFFFFFFFFFFFFUL

__device__ void _process_bytes (size_t beg, size_t end, u_char* d_skms, T_kmer *d_kmers, unsigned long long *d_kmer_store_pos, T_kvalue k) {
    // if called, stop until at least one skm is processed whatever end is exceeded
    // T_kmer kmer_mask = T_kmer(0xffffffffffffffff>>(64-k*2));
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
        // end_byte_pos = (end_byte_pos < tot_bytes) * end_byte_pos + (end_byte_pos >= tot_bytes) * tot_bytes;
        if (beg_byte_pos < end_byte_pos) {
            _process_bytes(beg_byte_pos, end_byte_pos, d_skms, d_kmers, d_kmer_store_pos, k);
        }
    }
    return;
}

__host__ u_char* load_SKM_from_file (SKMStoreNoncon &skms_store) {
    u_char* d_skms;
    CUDA_CHECK(cudaMalloc((void**) &(d_skms), skms_store.tot_size_bytes));
    FILE* fp;
    fp = fopen(skms_store.filename.c_str(), "rb");
    assert(fp);
    u_char* tmp;
    tmp = new u_char[skms_store.tot_size_bytes];
    assert(fread(tmp, 1, skms_store.tot_size_bytes, fp)==skms_store.tot_size_bytes);
    CUDA_CHECK(cudaMemcpy(d_skms, tmp, skms_store.tot_size_bytes, cudaMemcpyHostToDevice));
    delete tmp;
    fclose(fp);
    return d_skms;
}

void Extract_Kmers (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* &d_kmers, cudaStream_t &stream, int BpG2=8, int TpB2=256) {
    
    u_char* d_skms;
    
    unsigned long long *d_kmer_store_pos;
    CUDA_CHECK(cudaMallocAsync((void**) &(d_kmer_store_pos), sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(d_kmer_store_pos, 0, sizeof(unsigned long long), stream));

    // ---- copy skm chunks H2D ----
    if (skms_store.to_file) d_skms = load_SKM_from_file(skms_store);
    else {
        CUDA_CHECK(cudaMallocAsync((void**) &(d_skms), skms_store.tot_size_bytes+1, stream));
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
    }
    // ---- GPU work ----
    if (skms_store.tot_size_bytes / 4 <= BpG2 * TpB2) GPU_Extract_Kmers<<<1, skms_store.tot_size_bytes/64+1, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k); // 强行debug // mountain of shit, do not touch
    else GPU_Extract_Kmers<<<BpG2, TpB2, 0, stream>>>(d_skms, skms_store.tot_size_bytes, d_kmers, d_kmer_store_pos, k);
    
    // unsigned long long kmer_cnt;
    // CUDA_CHECK(cudaMemcpyAsync(&kmer_cnt, d_kmer_store_pos, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // assert(kmer_cnt == skms_store.kmer_cnt);
    
    CUDA_CHECK(cudaFreeAsync(d_skms, stream));
    CUDA_CHECK(cudaFreeAsync(d_kmer_store_pos, stream));
    return;
}

size_t kmc_counting_GPU_streams (T_kvalue k,
    vector<SKMStoreNoncon*> skms_stores, CUDAParams &gpars,
    T_kmer_cnt kmer_min_freq, T_kmer_cnt kmer_max_freq,
    _out_ vector<T_kmc> kmc_result_curthread [], int gpuid, int tid)
{
    CUDA_CHECK(cudaSetDevice(gpuid));

    size_t return_value = 0;
    int i, n_streams = skms_stores.size();
    cudaStream_t streams[n_streams];
    T_kmer *d_kmers[n_streams];
    size_t *distinct_cnt[n_streams];
    
    string logs = "GPU "+to_string(gpuid)+"\t(T"+to_string(tid)+"):";
    
    for (i=0; i<n_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        logs += "\tS "+to_string(i)+" Part "+to_string(skms_stores[i]->id)+" "+to_string(skms_stores[i]->tot_size_bytes)+"|"+to_string(skms_stores[i]->kmer_cnt);
        if (skms_stores[i]->tot_size_bytes != 0) {
            // ---- 0. Extract kmers from SKMStore: ---- 
            CUDA_CHECK(cudaMallocAsync((void**)&d_kmers[i], sizeof(T_kmer) * skms_stores[i]->kmer_cnt, streams[i]));
            Extract_Kmers(*skms_stores[i], k, d_kmers[i], streams[i], gpars.BpG2, gpars.TpB2);
            distinct_cnt[i] = construct_hashtab_gpu_stream (d_kmers[i], skms_stores[i]->kmer_cnt, true, streams[i], gpars.BpG2, gpars.TpB2);
        }
    }
    for (i=0; i<n_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        return_value += *(distinct_cnt[i]);
        cout<<*(distinct_cnt[i])<<endl;
        delete distinct_cnt[i];//
        delete skms_stores[i];//
    }
    cout<<logs<<endl;
    // logger->log(logs+" "+to_string(return_value), Logger::LV_DEBUG);
    return return_value;
}
// int main() {
//     WallClockTimer wct;
//     CUDA_CHECK(cudaSetDevice(0));
//     CUDA_CHECK(cudaDeviceReset());
//     cudaStream_t stream;
//     CUDA_CHECK(cudaStreamCreate(&stream));
//     unsigned long long *keys = new unsigned long long [40000000];
//     int i;
//     for (i=0; i<40000000; i++) keys[i] = (unsigned long long)i%1234567 + (unsigned long long)(0xffffffffu);
//     size_t *res = construct_hashtab_gpu_stream(keys, 40000000, false, stream, 16, 1024);
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     cout<<wct.stop()<<endl;
//     cout<<*res<<endl;
//     delete res;
//     return 0;
// }
