#ifndef _SKMSTORE2_HPP
#define _SKMSTORE2_HPP

#include <mutex>
#include <string>
// #include <cstring>
// #include "types.h"
#include "utilities.hpp"
// #include <iostream>
// #define DEBUG
#include <cassert>
// using namespace std;

class SKMStoreNoncon {
private:
    std::mutex _string_mtx;
    std::mutex _data_mtx;
    
    void _write_to_file(const u_char* data, size_t size) {
        assert(fwrite(data, 1, size, fp) == size);
    }

    void _add_skms_to_file (u_char* skms_chunk, size_t data_bytes, size_t b_skm_cnt, size_t b_kmer_cnt, bool flush = false) {
        _data_mtx.lock();
        _write_to_file (skms_chunk, data_bytes);
        if (flush && flush_cnt++ > 8192) {fflush(fp); flush_cnt=0;}
        _data_mtx.unlock();
        this->tot_size_bytes += data_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
    }
    
    /// @brief add skms directly from a chunk
    /// @param skms_chunk the chunk which stores multiple skms
    /// @param data_bytes the uncompressed size in bytes of this chunk
    /// @param compressed_bytes the compressed size in bytes of this chunk
    void _add_skms (u_char* skms_chunk, size_t data_bytes, size_t b_skm_cnt, size_t b_kmer_cnt) {
        if (to_file) { // keep to_file for add_batch_skms
            _data_mtx.lock();
            _write_to_file (skms_chunk, data_bytes);
            _data_mtx.unlock();
        } else {
            _string_mtx.lock();
            skms.append((char*)skms_chunk, data_bytes);
            _string_mtx.unlock();
        }
        tot_size_bytes += data_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
    }

public:
    // file
    bool to_file = false;
    FILE *fp;
    int id = -1;
    bool file_closed;
    std::string filename;
    int flush_cnt = 0;

    // no compression
    std::string skms;
    std::atomic<size_t> tot_size_bytes{0};
    std::atomic<size_t> skm_cnt{0};
    std::atomic<size_t> kmer_cnt{0};
    
    void close_file() {
        fclose(fp);
        file_closed = true;
    }

    SKMStoreNoncon (int id = -1, bool to_file = false) {
        this->to_file = to_file;
        this->id = id;
        if (to_file) {
            filename = PAR.tmp_file_folder+std::to_string(id)+".skm";
            fp = fopen(filename.c_str(), "wb");
            file_closed = false;
        }
    }
    
    void clear_skm_data () {
        std::string().swap(skms);
        return;
    }

    /// @brief for uncompressed skms saving, do not support delete in advance
    /// @param skms_stores SKMStoreNoncon * N_Partitions
    /// @param skm_cnt number of skms of each partition
    /// @param kmer_cnt number of kmers of each partition
    /// @param skmpart_offs offset of each skm partition
    /// @param skm_store_csr skm partitions in csr format
    static void save_batch_skms (std::vector<SKMStoreNoncon*> &skms_stores, T_skm_partsize *skm_cnt, T_skm_partsize *kmer_cnt, T_CSR_cap *skmpart_offs, u_char *skm_store_csr) {
        // memory layout of skm_store_csr:
        // [<part0><part1><part2><...>]
        int i;
        int SKM_partitions = skms_stores.size();
        for (i=0; i<SKM_partitions; i++)
            skms_stores[i]->_add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i]);
        if (skms_stores[0]->to_file) delete skm_store_csr;//
    }

    /**
     * Will take over the control of skm_data (no need to delete outside).
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
        if (skms_store->to_file) {
            skms_store->_add_skms_to_file(skm_data, data_bytes, skm_cnt, kmer_cnt, flush);
        } else {
            skms_store->_add_skms (skm_data, data_bytes, skm_cnt, kmer_cnt);
        }
        delete [] skm_data;
    }
};

#endif
