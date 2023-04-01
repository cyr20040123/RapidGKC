#ifndef _SKMSTORE_HPP
#define _SKMSTORE_HPP

#define SKM_ALIGN 2048

#include <mutex>
#include <vector>
#include <cstring>
#include "types.h"
#include "utilities.hpp"
#include "concurrent_queue.h"
#include <iostream>
// #define DEBUG
#include <cassert>
using namespace std;

struct SkmChunk {
    size_t chunk_bytes;
    u_char* skm_chunk;
};

class SKMStoreNoncon {
private:
    FILE *fp;
    string filename;
    int flush_cnt = 0;
    
    /// @brief add skms directly from a chunk
    /// @param skms_chunk the chunk which stores multiple skms
    /// @param data_bytes the uncompressed size in bytes of this chunk
    /// @param compressed_bytes the compressed size in bytes of this chunk
    void _add_skms (u_char* skms_chunk, size_t data_bytes, unsigned int b_skm_cnt, size_t b_kmer_cnt, bool flush = false) {
        if (to_file) {
            data_mtx.lock();
            fwrite(skms_chunk, 1, data_bytes, fp);
            if (flush && flush_cnt++ > 8192) {fflush(fp); flush_cnt=0;}
            data_mtx.unlock();
        } else {
            SkmChunk data{data_bytes, skms_chunk};
            skms.enqueue(data);
        }
        tot_size_bytes += data_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
    }

public:
    // file
    bool to_file = false;
    int id = -1;
    
    u_char* skms_from_file = nullptr;

    // no compression
    moodycamel::ConcurrentQueue<SkmChunk> skms;
    
    atomic<size_t> tot_size_bytes{0};
    atomic<size_t> skm_cnt{0};
    atomic<size_t> kmer_cnt{0};
    
    mutex data_mtx;
    
    moodycamel::ConcurrentQueue<u_char*> delete_list;

    // Load skms from file, can be called by P2 partition file loader and counter.
    void load_from_file() {
        data_mtx.lock();
        if (skms_from_file == nullptr) {
            fp = fopen(filename.c_str(), "rb");
            assert(fp);
            skms_from_file = new u_char[tot_size_bytes];
            assert(fread(skms_from_file, 1, tot_size_bytes, fp) == tot_size_bytes);
            fclose(fp);
        }
        data_mtx.unlock();
        return;
    }

    void close_file() {
        fclose(fp);
    }

    SKMStoreNoncon (int id = -1, bool to_file = false) {
        this->to_file = to_file;
        this->id = id;
        if (to_file) {
            filename = PAR.tmp_file_folder+to_string(id)+".skm";
            fp = fopen(filename.c_str(), "wb");
        }
    }

    void clear_skm_data () {
        size_t count;
        u_char* items[1024];
        int i;
        do {
            count = delete_list.try_dequeue_bulk(items, 1024);
            for (i = 0; i < count; i++) delete items[i];
        } while (count);
    }

    /// @brief for uncompressed skms saving, do not support delete in advance. Called by GPU function only.
    /// @param skms_stores SKMStoreNoncon * N_Partitions
    /// @param skm_cnt number of skms of each partition
    /// @param kmer_cnt number of kmers of each partition
    /// @param skmpart_offs offset of each skm partition
    /// @param skm_store_csr skm partitions in csr format
    static void save_batch_skms (vector<SKMStoreNoncon*> &skms_stores, T_skm_partsize *skm_cnt, T_skm_partsize *kmer_cnt, T_CSR_cap *skmpart_offs, u_char *skm_store_csr) {
        // memory layout of skm_store_csr:
        // [<part0><part1><part2><...>]
        int SKM_partitions = skms_stores.size();
        bool all_to_file = true;
        for (int i=0; i<SKM_partitions; i++) {
            all_to_file *= skms_stores[i]->to_file;
            skms_stores[i]->_add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i], false);
        }
        if (all_to_file) delete skm_store_csr;//
    }

    /**
     * Will take over the control of skm_data (no need to delete outside). Called by CPU function only.
     * @param  {SKMStoreNoncon*} skms_store : 
     * @param  {T_skm_partsize} skm_cnt     : 
     * @param  {T_skm_partsize} kmer_cnt    : 
     * @param  {u_char*} skm_data             : Will be deleted if <skm_to_file> is enabled.
     * @param  {size_t} data_bytes          : 
     * @param  {int} buf_size               : 
     */
    static void save_skms (SKMStoreNoncon* skms_store, T_skm_partsize skm_cnt, T_skm_partsize kmer_cnt, u_char *skm_data, size_t data_bytes, const int buf_size = 0, bool flush = false) {
        if (data_bytes == 0) {
            delete [] skm_data;
            return;
        }
        if (data_bytes < buf_size * 0.9) {
            u_char *tmp = new u_char [data_bytes];
            memcpy(tmp, skm_data, data_bytes);
            delete [] skm_data;
            skm_data = tmp;
        }
        skms_store->_add_skms(skm_data, data_bytes, skm_cnt, kmer_cnt, flush);
        if (skms_store->to_file) delete [] skm_data;
        else skms_store->delete_list.enqueue(skm_data);
    }
};

#endif
