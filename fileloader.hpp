#ifndef _FILELOADER_HPP
#define _FILELOADER_HPP

// #define DEBUG

#include <algorithm>
#include <future>
#include <thread>
#include <functional>
#include <vector>
#include <cstring>
#include "concqueue.hpp"
#include "types.h"
#include "thread_pool.hpp"

#ifdef DEBUG
#include <cassert>
#endif

struct DataBuffer {
    char *buf;
    size_t size;
};
struct LineBuffer {
    DataBuffer data;
    std::vector<size_t> newline_vec;
    /* std::vector<size_t> newline_vec2;*/
};

class ReadLoader {
private:
    ConcQueue<DataBuffer> _DBQ;             // data buffer queue
    ConcQueue<LineBuffer> _LBQ;             // line buffer queue
    ConcQueue<std::vector<ReadPtr>> _RBQ;    // read batch queue
    std::vector<std::string> _filenames;
    bool _is_fasta;
    
    size_t _read_batch_size;
    T_read_len _min_read_len;
    
    size_t _buffer_size;
    size_t _max_queue_size;
    
    int _n_threads_consumer;

    std::vector<std::thread> _started_threads;
    
    void _STEP1_load_from_file () {
        // size_t push_cnt = 0;

        char *buf = new char [_buffer_size];//
        size_t cur_size;
        for (std::string &filename: _filenames) {
            FILE *fp = fopen(filename.c_str(), "rb");
            while ((cur_size = fread(buf, 1, _buffer_size, fp)) > 0) {
                DataBuffer t;
                t.buf = buf;
                t.size = cur_size;
                _DBQ.wait_push(t, _max_queue_size);
                // push_cnt++;
                buf = new char [_buffer_size]; // deleted in _STEP3_extract_read
            }
            std::cout<<filename<<" closed: "<<fclose(fp)<<std::endl;
            DataBuffer t;
            t.size = 0;
            #ifdef WAITMEASURE
            output_wait();
            #endif
            _DBQ.push(t); // push a null block when file ends
            // push_cnt++;
        }
        _DBQ.finish();
        std::cout<<"Loader finish 1 "/*<<push_cnt*/<<std::endl;
        delete buf;//
    }
    void _STEP2_find_newline () {
        // size_t push_cnt = 0;
        // size_t line_cnt = 0;
        DataBuffer t;
        while (_DBQ.pop(t)) {
            // size_t move_offs = 0;
            LineBuffer x;
            x.data = std::move(t);
            x.newline_vec.reserve((t.size>>11)+8);
            x.newline_vec.push_back(-1);
            /*
            x.newline_vec2.reserve((t.size>>11)+8);
            // x.newline_vec = std::vector<size_t>();
            std::future<void> fu = std::async(std::launch::async, [&x](){
                for (int i=x.data.size/2; i<x.data.size; i++)
                    if (x.data.buf[i] == '\n') x.newline_vec2.push_back(i);
            });
            for (size_t i=0; i<x.data.size/2; i++) {
                // clean '\r':
                // if (move_offs != 0) x.data.buf[i-move_offs] = x.data.buf[i];
                // check '\n':
                // if (x.data.buf[i-move_offs] == '\n') x.newline_vec.push_back(i-move_offs); //, line_cnt++;
                if (x.data.buf[i] == '\n') x.newline_vec.push_back(i);
                // check '\r':
                // if (x.data.buf[i] == '\r') move_offs++;
            }
            fu.get();
            */
            char *find_beg = x.data.buf;
            while (true) {
                find_beg = std::find(find_beg, x.data.buf+x.data.size, '\n');
                if (find_beg == x.data.buf+x.data.size) break;
                x.newline_vec.push_back(find_beg - x.data.buf);
                find_beg++;
            }
            _LBQ.wait_push(x, _max_queue_size);
            // push_cnt++;
        }
        _LBQ.finish();
        std::cout<<"Loader finish 2 "<</*push_cnt<<" "<<line_cnt<<*/std::endl;
    }
    void _STEP3_extract_read () {
        // size_t push_cnt = 0, pop_cnt = 0, line_cnt = 0;;
        
        std::vector<ReadPtr> batch_reads;
        batch_reads.reserve(_read_batch_size);

        unsigned char line_flag = 0; // indicates a read line when it equals to 1
        unsigned char flag_mask = _is_fasta ? 0b1 : 0b11; // line_flag = (line_flag+1) & flag_mask;
        bool start_from_buffer = false;
        std::string last_line_buffer="";
        ReadPtr read;

        LineBuffer t;
        size_t i;
        while (_LBQ.pop(t)) {
            // pop_cnt++;
            // line_cnt+=t.newline_vec.size();
            if (t.data.size==0) { // new file
                line_flag = 0;
                continue;
            }
            /* t.newline_vec.reserve(t.newline_vec.size() + t.newline_vec2.size());
            t.newline_vec.insert(t.newline_vec.end(), t.newline_vec2.begin(), t.newline_vec2.end()); */
            for (i = 1; i < t.newline_vec.size(); i++, line_flag=(line_flag+1) & flag_mask) { // begins from 1 because q[0]=-1
                if (line_flag == 1) {
                    if (start_from_buffer) {
                        read.len = last_line_buffer.size() + t.newline_vec[i];
                        if (t.newline_vec[i]-1>0 && t.data.buf[t.newline_vec[i]-1]=='\t') read.len--;
                        read.read = new char [read.len];
                        memcpy(read.read, last_line_buffer.c_str(), last_line_buffer.size());
                        memcpy(read.read + last_line_buffer.size(), t.data.buf, t.newline_vec[i]);
                        last_line_buffer = "";
                        start_from_buffer = false;
                    } else {
                        read.len = t.newline_vec[i] - (t.newline_vec[i-1] + 1);
                        if (t.newline_vec[i]-1>0 && t.data.buf[t.newline_vec[i]-1]=='\t') read.len--;
                        read.read = new char [read.len];
                        memcpy(read.read, &(t.data.buf[t.newline_vec[i-1] + 1]), read.len);
                    }
                    if (read.len >= _min_read_len) batch_reads.push_back(read);
                    if (batch_reads.size() >= _read_batch_size) {
                        _RBQ.wait_push(batch_reads, _n_threads_consumer + 2);
                        batch_reads = std::vector<ReadPtr>();
                        batch_reads.reserve(_read_batch_size);
                    }
                    // push_cnt++;
                }
            }
            if (line_flag == 1 && *t.newline_vec.rbegin() != t.data.size) { // prepare last line buffer
                start_from_buffer = true;
                last_line_buffer = std::string(&t.data.buf[*t.newline_vec.rbegin()+1], t.data.size - *t.newline_vec.rbegin() - 1);
            }
            delete t.data.buf; // malloc in _STEP1_load_from_file
        }
        if (batch_reads.size()) _RBQ.push(batch_reads); // add last batch of reads
        _RBQ.finish();
        std::cout<<"Loader finish 3 "/*<<pop_cnt<<" "<<push_cnt<<" "<<line_cnt*/<<std::endl;
    }
public:
    static const size_t MB = 1048576;
    static const size_t KB = 1024;
    ReadLoader (std::vector<std::string> filenames, T_read_len min_read_len = 0, size_t read_batch_size = 8192, int buffer_size_MB = 16, int max_buffer_size_MB = 1024, int n_threads_consumer = 16) {
        _filenames = filenames;
        _is_fasta = *(filenames[0].rbegin()) == 'a' || *(filenames[0].rbegin()) == 'A';
        if (_is_fasta) max_buffer_size_MB /= 2; // for fasta, the buffer size required should be 1/2 smaller.
        _min_read_len = min_read_len;
        _read_batch_size = read_batch_size;
        _max_queue_size = max_buffer_size_MB / buffer_size_MB / 2;
        _buffer_size = buffer_size_MB * MB;
        _n_threads_consumer = n_threads_consumer;
        std::cout<<(_is_fasta?"Fasta format.":"Fastq format.")<<std::endl;
    }
    void start_load_reads() {
        _started_threads.push_back(std::thread(&ReadLoader::_STEP1_load_from_file, this));
        _started_threads.push_back(std::thread(&ReadLoader::_STEP2_find_newline, this));
        _started_threads.push_back(std::thread(&ReadLoader::_STEP3_extract_read, this));
    }
    void join_threads() {
        for (std::thread &t: _started_threads)
            if (t.joinable()) t.join();
    }
    #ifdef DEBUG
    void debug_load_reads_with_worker (std::function<void(std::vector<ReadPtr>&, int)> work_func) {
        std::thread t1(&ReadLoader::_STEP1_load_from_file, this);
        std::thread t2(&ReadLoader::_STEP2_find_newline, this);
        std::thread t3(&ReadLoader::_STEP3_extract_read, this);
        std::vector<ReadPtr> batch_reads;
        while (_RBQ.pop(batch_reads)) {
            work_func(batch_reads, 0);
            batch_reads.clear();
        }
        if (t1.joinable()) t1.join();
        if (t2.joinable()) t2.join();
        if (t3.joinable()) t3.join();
    }
    #endif
    #ifdef WAITMEASURE
    void output_wait () {
        std::cout<<"1 Load    push wait: "<<_DBQ.debug_push_wait<<"\tpop wait: "<<_DBQ.debug_pop_wait<<std::endl;
        std::cout<<"2 Newline push wait: "<<_LBQ.debug_push_wait<<"\tpop wait: "<<_LBQ.debug_pop_wait<<std::endl;
        std::cout<<"3 Extract push wait: "<<_RBQ.debug_push_wait<<"\tpop wait: "<<_RBQ.debug_pop_wait<<std::endl;
    }
    #endif
    static void work_while_loading (T_kvalue K_kmer, std::function<void(std::vector<ReadPtr>&, int)> work_func, int worker_threads, std::vector<std::string> &filenames, 
    T_read_cnt batch_size, size_t max_buffer_size_MB = 1024, size_t buffer_size_MB = 16) {
        
        T_read_len n_read_loaded = 0;

        ThreadPool<void> tp(worker_threads, worker_threads + worker_threads/2 > 2 ? worker_threads/4 : 2);
        ReadLoader rl(filenames, K_kmer, batch_size, buffer_size_MB, max_buffer_size_MB, worker_threads);
        rl.start_load_reads();

        std::vector<ReadPtr> read_batch;
        while (rl._RBQ.pop(read_batch)) {
            tp.hold_when_busy();
            //phase1(vector<ReadPtr> &reads, CUDAParams &gpars, vector<SKMStoreNoncon*> &skm_partition_stores, int tid)
            n_read_loaded += read_batch.size();
            std::vector<ReadPtr> *my_read_batch = new std::vector<ReadPtr>();//
            *my_read_batch = std::move(read_batch);
            tp.commit_task([my_read_batch, &work_func] (int tid) {
                // std::vector<ReadPtr> my_read_batch = my_read_batch;
                work_func(*my_read_batch, tid);
                for (ReadPtr &i: *my_read_batch) delete i.read;
                delete my_read_batch;//
            });
        }
        std::cerr<<"Total reads loaded: "<<n_read_loaded<<std::endl;
        rl.join_threads();
        tp.finish();
        // #ifdef WAITMEASURE
        // rl.output_wait();
        // #endif
    }
};

#ifdef DEBUG
int main() {
    std::vector<std::string> filenames {"/mnt/f/study/bio_dataset/hg002pacbio/man_files/SRR8858432.man.fasta"};
    ReadLoader rl(filenames, 0, 8192, 16, 512, 1);
    size_t s = 0;
    rl.debug_load_reads_with_worker([&s](std::vector<ReadPtr>& batch_reads, int tid){
        for(auto &i:batch_reads) {
            if (!(i.read[0]>='A'&&i.read[0]<='T')) std::cout<<std::string(i.read, 20)<<std::endl;
            assert(i.read[0]>='A'&&i.read[0]<='T');
            delete i.read;
        }
        s+=batch_reads.size();
    });
    rl.output_wait();
    std::cout<<s<<std::endl;
    return 0;
}
#endif

#endif