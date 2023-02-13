#ifndef _SUPERKMERS_HPP
#define _SUPERKMERS_HPP

#include <mutex>
#include <vector>
#include <cstring>
#include "types.h"
#include "utilities.hpp"
#include <iostream>
// #define DEBUG
#include <cassert>
using namespace std;

class SKMStoreNoncon {
public:
    // file
    bool to_file = false;
    FILE *fp;
    int id = -1;
    bool file_closed;
    string filename;

    // no compression
    vector<size_t> skm_chunk_bytes;
    vector<const byte*> skm_chunks;
    
    // compression
    vector<size_t> skm_chunk_compressed_bytes;
    size_t tot_size_compressed = 0;
    
    mutex data_mtx;
    size_t tot_size_bytes = 0;
    size_t skm_cnt = 0;
    size_t kmer_cnt = 0;

    mutex dl_mtx;
    vector<const byte*> delete_list;
    
    void _write_to_file(const byte* data, size_t size) {
        #ifdef DEBUG
        assert(file_closed == false);
        #endif
        // this->data_mtx.lock();
        assert(fwrite(data, 1, size, fp) == size);
        // this->data_mtx.unlock();
    }
    void close_file() {
        fclose(fp);
        file_closed = true;
    }

    SKMStoreNoncon (int id = -1, bool to_file = false) {
        this->to_file = to_file;
        this->id = id;
        if (to_file) {
            filename = PAR.tmp_file_folder+to_string(id)+".skm";
            fp = fopen(filename.c_str(), "wb");
            #ifdef DEBUG
            assert(fp != NULL);
            #endif
            file_closed = false;
        }
    }
    /// @brief add skms directly from a chunk
    /// @param skms_chunk the chunk which stores multiple skms
    /// @param data_bytes the uncompressed size in bytes of this chunk
    /// @param compressed_bytes the compressed size in bytes of this chunk
    void add_skms (const byte* skms_chunk, size_t data_bytes, size_t b_skm_cnt, size_t b_kmer_cnt, size_t compressed_bytes = 0) {
        data_mtx.lock();
        if (to_file) {
            #ifdef DEBUG
            assert (compressed_bytes==0);
            #endif
            _write_to_file (skms_chunk, data_bytes);
        } else {
            if (compressed_bytes) {
                skm_chunks.push_back(skms_chunk);
                // else {
                //     byte* new_cskm = new byte [compressed_bytes];
                //     memcpy(new_cskm, skms_chunk, compressed_bytes * sizeof(byte));
                //     skm_chunks.push_back(skms_chunk);
                // }
                skm_chunk_bytes.push_back(data_bytes);
                skm_chunk_compressed_bytes.push_back(compressed_bytes);
            } else {
                skm_chunks.push_back(skms_chunk);
                // else {
                //     byte* new_skm = new byte [data_bytes];
                //     memcpy(new_skm, skms_chunk, data_bytes);
                //     skm_chunks.push_back(new_skm);
                // }
                skm_chunk_bytes.push_back(data_bytes);
            }
        }
        tot_size_bytes += data_bytes;
        tot_size_compressed += compressed_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
        data_mtx.unlock();
    }
    bool is_compressed() {return skm_chunk_compressed_bytes.size();}
    void clear_skm_data () {
        for (auto i:delete_list) delete i;
    }

    /// @brief for uncompressed skms saving, do not support delete in advance
    /// @param skms_stores SKMStoreNoncon * N_Partitions
    /// @param skm_cnt number of skms of each partition
    /// @param kmer_cnt number of kmers of each partition
    /// @param skmpart_offs offset of each skm partition
    /// @param skm_store_csr skm partitions in csr format
    static void save_batch_skms (vector<SKMStoreNoncon*> &skms_stores, T_skm_partsize *skm_cnt, T_skm_partsize *kmer_cnt, T_CSR_cap *skmpart_offs, byte *skm_store_csr) {
        // memory layout of skm_store_csr:
        // [<part0><part1><part2><...>]
        int i;
        int SKM_partitions = skms_stores.size();
        for (i=0; i<SKM_partitions; i++)
            skms_stores[i]->add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i], 0);
        if (skms_stores[0]->to_file) delete skm_store_csr;//
    }

    /**
     * Will take over the control of skm_data (no need to delete outside).
     * @param  {SKMStoreNoncon*} skms_store : 
     * @param  {T_skm_partsize} skm_cnt     : 
     * @param  {T_skm_partsize} kmer_cnt    : 
     * @param  {byte*} skm_data             : Will be deleted if <skm_to_file> is enabled.
     * @param  {size_t} data_bytes          : 
     * @param  {int} buf_size               : 
     */
    static void save_skms (SKMStoreNoncon* skms_store, T_skm_partsize skm_cnt, T_skm_partsize kmer_cnt, byte *skm_data, size_t data_bytes, const int buf_size = 0) {
        if (data_bytes == 0) {
            delete [] skm_data;
            return;
        }
        byte *tmp;
        if (data_bytes < buf_size / 4 * 3) { // TODO: space or time
            tmp = new byte [data_bytes];
            memcpy(tmp, skm_data, data_bytes);
            skms_store->add_skms(tmp, data_bytes, skm_cnt, kmer_cnt);
            delete [] skm_data;
            if (skms_store->to_file) delete tmp;
            else {
                skms_store->dl_mtx.lock();
                skms_store->delete_list.push_back(tmp);
                skms_store->dl_mtx.unlock();
            }
        } else {
            skms_store->add_skms(skm_data, data_bytes, skm_cnt, kmer_cnt);
            if (skms_store->to_file) delete [] skm_data;
            else {
                skms_store->dl_mtx.lock();
                skms_store->delete_list.push_back(skm_data);
                skms_store->dl_mtx.unlock();
            }
        }
    }
};

#endif
