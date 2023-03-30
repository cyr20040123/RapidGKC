#ifndef _SKMSTORE_HPP
#define _SKMSTORE_HPP

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
    T_skm_len* skm_lengths;
    unsigned int chunk_cnt;
    bool delete_avail;
};

class SKMStoreNoncon {
private:
    FILE *fp;
    FILE *fp_lengths;
    string filename, lenfilename;
    int flush_cnt = 0;
    void _write_to_file(const u_char* data, size_t skm_size, const T_skm_len* skm_lengths, int skm_cnt) {
        fwrite(data, 1, skm_size, fp);
        fwrite(skm_lengths, 1, skm_cnt, fp_lengths);
    }
    void _add_skms_to_file (u_char* skms_chunk, size_t data_bytes, T_skm_len* skm_lengths, size_t b_skm_cnt, size_t b_kmer_cnt, bool flush = false) {
        data_mtx.lock();
        _write_to_file (skms_chunk, data_bytes, skm_lengths, b_skm_cnt);
        if (flush && flush_cnt++ > 8192) {fflush(fp); flush_cnt=0;}
        data_mtx.unlock();
        this->tot_size_bytes += data_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
    }
    /// @brief add skms directly from a chunk
    /// @param skms_chunk the chunk which stores multiple skms
    /// @param data_bytes the uncompressed size in bytes of this chunk
    /// @param compressed_bytes the compressed size in bytes of this chunk
    void _add_skms (u_char* skms_chunk, size_t data_bytes, T_skm_len* skm_lengths, unsigned int b_skm_cnt, size_t b_kmer_cnt, bool delete_chunk = false) {
        // size_t debug_kmer_cnt=0;
        // for (unsigned int i=0; i<b_skm_cnt; i++) debug_kmer_cnt += (skm_lengths[i]&0b0011111111111111)-27;
        // std::cout<<debug_kmer_cnt<<" "<<b_kmer_cnt<<std::endl;
        // assert(debug_kmer_cnt == b_kmer_cnt);

        if (to_file) { // gpu to file, don't flush
            data_mtx.lock();
            _write_to_file (skms_chunk, data_bytes, skm_lengths, b_skm_cnt);
            data_mtx.unlock();
        } else {
            SkmChunk data{data_bytes, skms_chunk, skm_lengths, b_skm_cnt, delete_chunk};
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
    T_skm_len* skm_lengths_from_file = nullptr;

    // no compression
    moodycamel::ConcurrentQueue<SkmChunk> skms;
    
    atomic<size_t> tot_size_bytes{0};
    atomic<size_t> skm_cnt{0};
    atomic<size_t> kmer_cnt{0};
    
    mutex data_mtx;
    
    moodycamel::ConcurrentQueue<u_char*> delete_list_uc;
    moodycamel::ConcurrentQueue<T_skm_len*> delete_list_len;

    // Load skms from file, can be called by P2 partition file loader and counter.
    void load_from_file() {
        data_mtx.lock();
        if (skms_from_file == nullptr) {
            fp = fopen(filename.c_str(), "rb");
            assert(fp);
            skms_from_file = new u_char[tot_size_bytes];
            assert(fread(skms_from_file, 1, tot_size_bytes, fp) == tot_size_bytes);
            fclose(fp);

            fp_lengths = fopen(lenfilename.c_str(), "rb");
            assert(fp_lengths);
            skm_lengths_from_file = new T_skm_len[skm_cnt];
            assert(fread(skm_lengths_from_file, sizeof(T_skm_len), skm_cnt, fp_lengths) == skm_cnt/sizeof(T_skm_len));
            fclose(fp_lengths);
        }
        data_mtx.unlock();
        return;
    }

    void close_file() {
        fclose(fp);
        fclose(fp_lengths);
    }

    SKMStoreNoncon (int id = -1, bool to_file = false) {
        this->to_file = to_file;
        this->id = id;
        if (to_file) {
            filename = PAR.tmp_file_folder+to_string(id)+".skm";
            lenfilename = filename + ".len";
            fp = fopen(filename.c_str(), "wb");
            fp_lengths = fopen(lenfilename.c_str(), "wb");
        }
    }
    

    void clear_skm_data () {
        size_t count;
        u_char* items[1024];
        int i;
        do {
            count = delete_list_uc.try_dequeue_bulk(items, 1024);
            for (i = 0; i < count; i++) delete items[i];
        } while (count);
        
        T_skm_len* items_len[1024];
        do {
            count = delete_list_len.try_dequeue_bulk(items_len, 1024);
            for (i = 0; i < count; i++) delete items_len[i];
        } while (count);
    }

    /// @brief for uncompressed skms saving, do not support delete in advance. Called by GPU function only.
    /// @param skms_stores SKMStoreNoncon * N_Partitions
    /// @param skm_cnt number of skms of each partition
    /// @param kmer_cnt number of kmers of each partition
    /// @param skmpart_offs offset of each skm partition
    /// @param skm_store_csr skm partitions in csr format
    static void save_batch_skms (vector<SKMStoreNoncon*> &skms_stores, T_skm_len *skm_lengths, T_skm_partsize *skm_cnt, T_skm_partsize *kmer_cnt, T_CSR_cap *skmpart_offs, u_char *skm_store_csr) {
        // memory layout of skm_store_csr:
        // [<part0><part1><part2><...>]
        int SKM_partitions = skms_stores.size();
        size_t skm_len_offs = 0;
        for (int i=0; i<SKM_partitions; i++) {
            if (i==0) {
                int len = skm_lengths[skm_len_offs] & 0b0011111111111111;
                int start = skm_lengths[skm_len_offs] >> 14;
                int idx = 0;
                char c[4] = {'A','C','G','T'};
                cerr<<"SKM LEN:"+to_string(len)<<endl;
                for (int j=0; j<len; j++) {
                    idx += (start==4);
                    start %= 4;
                    cerr<<c[(skm_store_csr[skmpart_offs[i]+idx] >> ((3-start)*2)) & 0b11];
                    start ++;
                }cerr<<endl;
            }
            skms_stores[i]->_add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], &skm_lengths[skm_len_offs], skm_cnt[i], kmer_cnt[i], false);
            skm_len_offs += skm_cnt[i];
        }
        if (skms_stores[0]->to_file) {
            delete skm_store_csr;//
            delete skm_lengths;
        }
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
    static void save_skms (SKMStoreNoncon* skms_store, T_skm_len *skm_lengths, T_skm_partsize skm_cnt, T_skm_partsize kmer_cnt, u_char *skm_data, size_t data_bytes, const int buf_size = 0, bool flush = false) {
        if (data_bytes == 0) {
            delete [] skm_data;
            delete [] skm_lengths;
            return;
        }
        if (skms_store->to_file) {
            skms_store->_add_skms_to_file(skm_data, data_bytes, skm_lengths, skm_cnt, kmer_cnt, flush);
            delete [] skm_data;
            delete [] skm_lengths;
            return;
        } 
        if (data_bytes < buf_size * 0.9) {
            u_char *tmp = new u_char [data_bytes];
            memcpy(tmp, skm_data, data_bytes);
            delete [] skm_data;
            skm_data = tmp;
        }
        if (sizeof(T_skm_len) * skm_cnt < sizeof(skm_lengths) * 0.9) {
            T_skm_len *tmp = new T_skm_len [skm_cnt];
            memcpy(tmp, skm_lengths, sizeof(T_skm_len) * skm_cnt);
            delete [] skm_lengths;
            skm_lengths = tmp;
        }
        skms_store->_add_skms(skm_data, data_bytes, skm_lengths, skm_cnt, kmer_cnt, true);
        skms_store->delete_list_uc.enqueue(skm_data);
        skms_store->delete_list_len.enqueue(skm_lengths);
    }
};

#endif
