#include <mutex>
#include <vector>
#include <cstring>
#include "types.h"
using namespace std;

class SKMStoreNoncon {
public:
    // no compression
    vector<size_t> skm_chunk_bytes;
    vector<const byte*> skm_chunks;
    
    // compression
    vector<size_t> skm_chunk_uncompressed_bytes; // uncompressed bytes of batches
    vector<size_t> skm_chunk_compressed_bytes;
    vector<const void*> skm_chunks_compressed;
    
    mutex vec_mtx;
    size_t tot_size_bytes = 0;
    size_t skm_cnt = 0;
    size_t kmer_cnt = 0;
    
    SKMStoreNoncon () {}
    /// @brief add skms directly from a chunk
    /// @param skms_chunk the chunk which stores multiple skms
    /// @param data_bytes the uncompressed size in bytes of this chunk
    /// @param compressed_bytes the compressed size in bytes of this chunk
    /// @param emplace true: store the pointer directly, false: allocate new data and store the new pointer
    void add_skms (const byte* skms_chunk, size_t data_bytes, size_t b_skm_cnt, size_t b_kmer_cnt, size_t compressed_bytes = 0, bool emplace = true) {
        vec_mtx.lock();
        if (compressed_bytes) {
            if (emplace) skm_chunks_compressed.push_back(skms_chunk);
            else {
                byte* new_cskm = new byte [compressed_bytes];
                memcpy(new_cskm, skms_chunk, compressed_bytes * sizeof(byte));
                skm_chunks_compressed.push_back(skms_chunk);
            }
            skm_chunk_uncompressed_bytes.push_back(data_bytes);
            skm_chunk_compressed_bytes.push_back(compressed_bytes);
        } else {
            if (emplace) skm_chunks.push_back(skms_chunk);
            else {
                byte* new_skm = new byte [data_bytes];
                memcpy(new_skm, skms_chunk, data_bytes);
                skm_chunks.push_back(new_skm);
            }
            skm_chunk_bytes.push_back(data_bytes);
        }
        tot_size_bytes += compressed_bytes == 0 ? data_bytes : compressed_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
        vec_mtx.unlock();
    }
    bool is_compressed() {return skm_chunk_compressed_bytes.size();}

    static void save_batch_skms (vector<SKMStoreNoncon*> &skms_stores, T_skm_partsize *skm_cnt, T_skm_partsize *kmer_cnt, T_CSR_cap *skmpart_offs, byte *skm_store_csr, size_t *skmpart_compressed_bytes = nullptr, bool emplace = true) {
        // memory layout of skm_store_csr:
        // [<part0><part1><part2><...>]
        int i;
        int SKM_partitions = skms_stores.size();
        if (skmpart_compressed_bytes == nullptr) {
            for (i=0; i<SKM_partitions; i++)
                skms_stores[i]->add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i], 0, emplace);
        } else {// for GPU-compressed data, use new size (skmpart_compressed_sizes)
            for (i=0; i<SKM_partitions; i++)
                skms_stores[i]->add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i], skmpart_compressed_bytes[i], emplace);
        }
    }
};