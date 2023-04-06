#ifndef _SKMSTORE_HPP
#define _SKMSTORE_HPP

// #define SKMSTOREV1

#include <mutex>
#include <vector>
#include <cstring>
#include "types.h"
#include "utilities.hpp"
#include "concurrent_queue.h"
#include "concqueue.hpp"
#include <future>
#include <iostream>
// #define DEBUG
#include <cassert>
using namespace std;

struct SKM {
    size_t chunk_bytes;
    u_char* skm_chunk;
};

struct write_task{
    u_char* data;
    size_t size;
    FILE* fp;
    u_char* del;
};
class FileWriter {
private:
    ConcQueue<write_task> _tasks;
    ConcQueue<u_char*> _delete;
    std::future<void> _fu;
    std::future<void> _fudel;
public:
    FileWriter() {
        _fu = std::async([this](){
            write_task t;
            while (this->_tasks.pop(t)) {
                fwrite(t.data, 1, t.size, t.fp);
                if (t.del != nullptr) this->_delete.push(t.del);
            }
            this->_delete.finish();
        });
        _fudel = std::async([this](){
            u_char* t;
            while(this->_delete.pop(t)) delete t;
        });
    }
    void write(u_char* data, size_t size, FILE* fp, u_char* delete_data = nullptr) {
        write_task t{data, size, fp, delete_data};
        _tasks.wait_push(t, 65536);
    }
    void finish() {
        _tasks.finish();
        _fu.get();
        _fudel.get();
    }
};

class SKMStoreNoncon {
public:
    // file
    FileWriter *fw;
    bool to_file = false;
    FILE *fp;
    int id = -1;
    bool file_closed;
    string filename;
    int flush_cnt = 0;

    u_char* skms_from_file = nullptr;

    // no compression
    #ifdef SKMSTOREV1
    vector<size_t> skm_chunk_bytes;
    vector<const u_char*> skm_chunks;
    
    size_t tot_size_bytes = 0;
    size_t skm_cnt = 0;
    size_t kmer_cnt = 0;
    #else
    moodycamel::ConcurrentQueue<SKM> skms;
    atomic<size_t> tot_size_bytes{0};
    atomic<size_t> skm_cnt{0};
    atomic<size_t> kmer_cnt{0};
    // u_char* skm_all_concat;
    #endif
    
    mutex data_mtx;
    
    // compression
    // vector<size_t> skm_chunk_compressed_bytes;
    // size_t tot_size_compressed = 0;
    
    // mutex dl_mtx;
    // vector<const u_char*> delete_list;
    moodycamel::ConcurrentQueue<u_char*> delete_list;

    // Load skms from file, can be called by P2 partition file loader and counter.
    void load_from_file() {
        data_mtx.lock();
        if (skms_from_file == nullptr) {
            FILE* myfp;
            myfp = fopen(filename.c_str(), "rb");
            assert(myfp);
            skms_from_file = new u_char[tot_size_bytes];
            assert(fread(skms_from_file, 1, tot_size_bytes, myfp) == tot_size_bytes);
            fclose(myfp);
        }
        data_mtx.unlock();
        return;
    }

    void _write_to_file(const u_char* data, size_t size) {
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

    SKMStoreNoncon (int id = -1, bool to_file = false, FileWriter *fw = nullptr) {
        this->to_file = to_file;
        this->id = id;
        if (to_file) {
            this->fw = fw;
            filename = PAR.tmp_file_folder+to_string(id)+".skm";
            fp = fopen(filename.c_str(), "wb");
            #ifdef DEBUG
            assert(fp != NULL);
            #endif
            file_closed = false;
        }
    }
    void _add_skms_to_file (u_char* skms_chunk, size_t data_bytes, size_t b_skm_cnt, size_t b_kmer_cnt, bool flush = false) {
        // data_mtx.lock();
        // // _write_to_file (skms_chunk, data_bytes);
        // // if (flush && flush_cnt++ > 8192) {fflush(fp); flush_cnt=0;}
        // data_mtx.unlock();
        fw->write(skms_chunk, data_bytes, this->fp, skms_chunk);
        this->tot_size_bytes += data_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
    }
    /// @brief add skms directly from a chunk (FOR GPU GENERATED SKMS, NO DELETE AFTER)
    /// @param skms_chunk the chunk which stores multiple skms
    /// @param data_bytes the uncompressed size in bytes of this chunk
    /// @param compressed_bytes the compressed size in bytes of this chunk
    void add_skms (u_char* skms_chunk, size_t data_bytes, size_t b_skm_cnt, size_t b_kmer_cnt/*, size_t compressed_bytes = 0*/, u_char* delete_data = nullptr) {
        #ifdef SKMSTOREV1
        data_mtx.lock();
        #endif
        if (to_file) {
            #ifdef DEBUG
            assert (compressed_bytes==0);
            #endif
            #ifndef SKMSTOREV1
            // data_mtx.lock();
            #endif
            // _write_to_file (skms_chunk, data_bytes);
            fw->write(skms_chunk, data_bytes, this->fp, delete_data);
            #ifndef SKMSTOREV1
            // data_mtx.unlock();
            #endif
        } else {
            #ifdef SKMSTOREV1
            skm_chunks.push_back(skms_chunk);
            skm_chunk_bytes.push_back(data_bytes);
            #else
            SKM data{data_bytes, skms_chunk};
            skms.enqueue(data);
            #endif
        }
        tot_size_bytes += data_bytes;
        this->skm_cnt += b_skm_cnt;
        this->kmer_cnt += b_kmer_cnt;
        #ifdef SKMSTOREV1
        data_mtx.unlock();
        #endif
    }
    // bool is_compressed() {return skm_chunk_compressed_bytes.size();}
    void clear_skm_data () {
        size_t count;
        u_char* items[1024];
        int i;
        do {
            count = delete_list.try_dequeue_bulk(items, 1024);
            for (i = 0; i < count; i++) delete [] items[i];
        } while (count);
        // for (auto i:delete_list) delete i;
    }

    // #ifndef SKMSTOREV1
    // void concatenate_skm_chunks () { // TODO HERE, efficient for short read dataset
    //     skm_all_concat = new u_char[tot_size_bytes];
    //     SKM data[1024];
    //     size_t count;
    //     do {
    //         count = skms.try_dequeue_bulk(data, 1024);
    //         for (int i=0; i<count; i++) {

    //             skm_chunk_bytes.push_back(data[i].chunk_bytes);
    //             skm_chunks.push_back(data[i].skm_chunk);
    //         }
    //     } while (count);
    // }
    // #endif

    /// @brief for uncompressed skms saving, do not support delete in advance, CALLED BY GPU WORKER
    /// @param skms_stores SKMStoreNoncon * N_Partitions
    /// @param skm_cnt number of skms of each partition
    /// @param kmer_cnt number of kmers of each partition
    /// @param skmpart_offs offset of each skm partition
    /// @param skm_store_csr skm partitions in csr format
    static void save_batch_skms (vector<SKMStoreNoncon*> &skms_stores, T_skm_partsize *skm_cnt, T_skm_partsize *kmer_cnt, T_CSR_cap *skmpart_offs, u_char *skm_store_csr) {
        // memory layout of skm_store_csr:
        // [<part0><part1><part2><...>]
        int i;
        int SKM_partitions = skms_stores.size();
        for (i=0; i<SKM_partitions-1; i++) {
            if (skmpart_offs[i+1]-skmpart_offs[i] > 0)
                skms_stores[i]->add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i]);
        }
        if (skms_stores[0]->to_file) 
            // delete skm_store_csr;//
            skms_stores[i]->add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i], skm_store_csr);
        else
            skms_stores[i]->add_skms(&skm_store_csr[skmpart_offs[i]], skmpart_offs[i+1]-skmpart_offs[i], skm_cnt[i], kmer_cnt[i], nullptr);
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
        u_char *tmp;
        if (skms_store->to_file) {
            skms_store->_add_skms_to_file(skm_data, data_bytes, skm_cnt, kmer_cnt, flush);
            // delete [] skm_data;
        } else if (data_bytes < buf_size * 0.9) { // TODO: space or time
            tmp = new u_char [data_bytes];
            memcpy(tmp, skm_data, data_bytes);
            skms_store->add_skms(tmp, data_bytes, skm_cnt, kmer_cnt);
            delete [] skm_data;
            // if (skms_store->to_file) delete tmp;
            // else {
                skms_store->delete_list.enqueue(tmp);
                // skms_store->dl_mtx.lock();
                // skms_store->delete_list.push_back(tmp);
                // skms_store->dl_mtx.unlock();
            // }
        } else {
            skms_store->add_skms(skm_data, data_bytes, skm_cnt, kmer_cnt);
            // if (skms_store->to_file) delete [] skm_data;
            // else {
                skms_store->delete_list.enqueue(skm_data);
                // skms_store->dl_mtx.lock();
                // skms_store->delete_list.push_back(skm_data);
                // skms_store->dl_mtx.unlock();
            // }
        }
    }
};

#endif
