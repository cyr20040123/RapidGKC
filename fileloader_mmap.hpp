#ifndef _FILELOADER_HPP
#define _FILELOADER_HPP

// #define STEP3P

#include <atomic>
#include <algorithm>
#include <future>
#include <thread>
#include <functional>
#include <vector>
#include <cstring>
#include "concqueue.hpp"
#include "types.h"
#include "thread_pool.hpp"

#include "gpu_skmgen.h"
#include "cpu_funcs.h"

#include "zlib.h"

#ifdef DEBUG
#include "utilities.hpp"
#endif

// #define _GNU_SOURCE             /* See feature_test_macros(7) */
#include <fcntl.h>      // open
#include <sys/mman.h>   // mmap
#include <sys/stat.h>   // fstat
#include <unistd.h>     // close

#ifdef DEBUG
#include <cassert>
#endif

#include <immintrin.h>

struct DataBuffer {
    char *buf;
    size_t size;
};
struct ReadLine {
    char *ptr;
    int len; // +: read length, -: read length (not ended)
};
struct LineBuffer {
    DataBuffer data;
    std::vector<ReadLine> readline_vec;
};

void GenMMDict() {
    // calc mm dict from mm histogram
    extern DevMMHisto *dev_mm_histo[];
    extern std::atomic<unsigned long long> mm_histo[];
    unsigned long long *histo_gpu[PAR.n_devices];
    for (int igpu = 0; igpu < PAR.n_devices; ++igpu) {
        if (dev_mm_histo[igpu] == NULL) {
            histo_gpu[igpu] = new unsigned long long [1<<(2*PAR.P_minimizer)];
        } else {
            histo_gpu[igpu] = dev_mm_histo[igpu]->CopyToHost(PAR.P_minimizer, igpu);
        }
    }
    unsigned int *dict = GenDict(PAR.SKM_partitions, PAR.P_minimizer, PAR.MM_histo, mm_histo, histo_gpu, PAR.n_devices);
    extern unsigned int *mm_dict;
    // extern unsigned int *d_mm_dict[];
    mm_dict = dict;
    for (int igpu = 0; igpu < PAR.n_devices; ++igpu) {
        // d_mm_dict[igpu] = 
        CopyDictToDevice(mm_dict, PAR.P_minimizer, igpu);
    }
}

class ReadLoader {
private:
    ConcQueue<DataBuffer> _DBQ;             // data buffer queue
    ConcQueue<LineBuffer> _LBQ;             // line buffer queue
    #ifdef STEP3P
    ConcQueue<LineBuffer> _LBQ2;            // line buffer queue
    int _n_step3p = 4;                      // number of STEP3P threads
    std::atomic<int> _started_step3p{0};
    #endif
    ConcQueue<std::vector<ReadPtr>> _RBQ;   // read batch queue
    std::vector<std::string> _filenames;
    bool _is_fasta;
    
    size_t _read_batch_size;
    T_read_len _min_read_len;
    
    size_t _buffer_size;
    size_t _max_queue_size;
    
    int _n_threads_consumer;

    std::vector<std::thread> _started_threads;
    
    void _STEP1_load_from_file () {
        size_t cur_size;
        char tmp;
        // set _buffer_size aligning to the page size of 8KB
        _buffer_size = (_buffer_size>>13)<<14;
        assert(_buffer_size > 0);
        std::cerr<<"IO buffer size: "<<_buffer_size/1024<<" KB"<<std::endl;
        for (int i=0; i<_filenames.size(); i++) {
            std::string filename = _filenames[i];
            
            // open read file and get file size
            int fd = open(filename.c_str(), O_RDONLY);
            struct stat statue;
            fstat(fd, &statue);
            // std::cerr<<"Open file ["<<filename<<"]: "<<fd<<" "<<"filesize = "<<statue.st_size<<std::endl;

            size_t cur_buffer_size = statue.st_size < _buffer_size ? statue.st_size : _buffer_size;
            for (size_t i=0; i<statue.st_size; i+=_buffer_size) {
                char *fileptr = (char *) mmap(NULL, cur_buffer_size, PROT_READ, MAP_SHARED, fd, i); // MAP_SHARED
                DataBuffer t;
                t.buf = fileptr;
                t.size = cur_buffer_size;
                // readahead for the next loop
                if (i+_buffer_size < statue.st_size) {
                    cur_buffer_size = statue.st_size - (i + cur_buffer_size) < _buffer_size ? statue.st_size - (i + cur_buffer_size) : _buffer_size;
                    readahead(fd, i+_buffer_size, cur_buffer_size);
                }
                _DBQ.wait_push(t, _max_queue_size);
            }
            DataBuffer t;
            t.size = 0;
            t.buf = nullptr;
            #ifdef WAITMEASURE
            output_wait();
            #endif
            _DBQ.push(t); // push a null block when file ends
            std::cerr<<"  ("<<i+1<<"/"<<_filenames.size()<<")\t["<<filename<<"] closed: "<<close(fd)<<std::endl;
        }
        _DBQ.finish();
        std::cout<<"Loader finish 1 "/*<<push_cnt*/<<std::endl;
    }
    
    char *find_avx2(char *b, char *e, char c)
    {
        char *i = b;
        __m256i q = _mm256_set1_epi8(c);

        for (; i+32 < e; i+=32) {
            __m256i x = _mm256_lddqu_si256(
                reinterpret_cast<const __m256i*>(i));
            __m256i r = _mm256_cmpeq_epi8(x, q);
            int z     = _mm256_movemask_epi8(r);
            if (z) return i + __builtin_ffs(z) - 1;
        }

        for (; i < e; ++i)
            if (*i == c) return i;

        return e;
    }

    void _STEP2_find_newline () { // NOT MMAP
        DataBuffer t;
        unsigned char flag_mask = _is_fasta ? 0b1 : 0b11;
        unsigned char is_read_line = 0;
        char *line_beg, *line_end;
        while (_DBQ.pop(t)) {
            LineBuffer x;
            x.data = std::move(t);
            if (x.data.size == 0) {
                is_read_line = 0; // new file
                continue;
            }
            x.readline_vec.reserve((t.size>>10)+8);
            line_end = x.data.buf-1;
            while (true) {
                line_beg = line_end+1;
                // line_end = std::find(line_beg, x.data.buf+x.data.size, '\n');
                // line_end = (char*) memchr(line_beg, '\n', x.data.buf+x.data.size-line_beg);
                // if (line_end == NULL) line_end = x.data.buf+x.data.size;
                line_end = find_avx2(line_beg, x.data.buf+x.data.size, '\n');
                if (is_read_line == 1) {
                    if (line_end == x.data.buf+x.data.size) {
                        x.readline_vec.push_back({line_beg, (int)(-(line_end-line_beg))});
                        break;
                    } else {
                        x.readline_vec.push_back({line_beg, (int)(line_end-line_beg)});
                    }
                }
                if (line_end == x.data.buf+x.data.size) break;
                is_read_line = (is_read_line+1)&flag_mask;
            }
            _LBQ.wait_push(x, _max_queue_size);
        }
        _LBQ.finish();
        std::cout<<"Loader finish 2 "<</*push_cnt<<" "<<line_cnt<<*/std::endl;
    }
    
    #ifdef STEP3P
    void _STEP3PLUS () {
        std::vector<ReadPtr> batch_reads;
        batch_reads.reserve(_read_batch_size);
        ReadPtr read;

        LineBuffer t;
        int i_end;

        while (_LBQ2.pop(t)) {
            size_t skip = t.readline_vec.size() * _n_step3p / (_n_step3p+1);
            if (skip >= t.readline_vec.size()-1) skip = max(0ul, t.readline_vec.size()-2);
            i_end = skip+1;
            for (int i = 1; i < i_end; i++) {
                if (t.readline_vec[i].len >= _min_read_len) {
                    read.len = t.readline_vec[i].len;
                    read.read = new char[read.len];
                    memcpy(read.read, t.readline_vec[i].ptr, read.len);
                    batch_reads.push_back(read);
                }
                if (batch_reads.size() >= _read_batch_size) {
                    _RBQ.wait_push(batch_reads, _n_threads_consumer + 1);
                    batch_reads = std::vector<ReadPtr>();
                    batch_reads.reserve(_read_batch_size);
                }
            }
            // if (t.flag->test_and_set()) {delete t.data.buf; delete t.flag;}
            // delete t.data.buf;
            munmap(t.data.buf, t.data.size);
        }
        if (batch_reads.size()) _RBQ.push(batch_reads); // add last batch of reads
        if (_started_step3p.fetch_sub(1) == 1) _RBQ.finish();
    }
    #endif
    void _STEP3_extract_read () {
        WallClockTimer wct_readloader;
        
        std::vector<ReadPtr> batch_reads;
        batch_reads.reserve(_read_batch_size);

        bool start_from_buffer = false;
        // std::string last_line_buffer;
        char last_line_buffer[MB]; int last_line_buffer_len = 0;
        ReadPtr read;

        LineBuffer t;

        while (_LBQ.pop(t)) {
            #ifdef STEP3P
            size_t skip = t.readline_vec.size() * _n_step3p / (_n_step3p+1); //  > 1 ? t.readline_vec.size()*RATIO_STEP3P : 1; // move first 60% read lines (except the first) to LBQ2 for parallel processing
            if (skip >= t.readline_vec.size()-1) skip = max(0ul, t.readline_vec.size()-2);
            #endif
            // for (ReadLine i: t.readline_vec) {
            // for (std::vector<ReadLine>::iterator i = t.readline_vec.begin(); i != t.readline_vec.end();) {
            for (int i=0; i<t.readline_vec.size();) {
                if (start_from_buffer) {
                    // read.len = last_line_buffer.length() + i->len;
                    read.len = last_line_buffer_len + t.readline_vec[i].len;
                    read.read = new char[read.len];
                    // memcpy(read.read, last_line_buffer.c_str(), last_line_buffer.length());
                    memcpy(read.read, last_line_buffer, last_line_buffer_len);
                    // memcpy(read.read + last_line_buffer.length(), i->ptr, i->len);
                    memcpy(read.read + last_line_buffer_len, t.readline_vec[i].ptr, t.readline_vec[i].len);
                    start_from_buffer = false;
                // } else if (i->len > 0) {
                } else if (t.readline_vec[i].len > 0) {
                    // read.len = i->len;
                    read.len = t.readline_vec[i].len;
                    read.read = new char[read.len];
                    // memcpy(read.read, i->ptr, i->len);
                    memcpy(read.read, t.readline_vec[i].ptr, t.readline_vec[i].len);
                // } else if (i->len < 0) {
                } else if (t.readline_vec[i].len < 0) {
                    start_from_buffer = true;
                    // last_line_buffer = std::string(i->ptr, -(i->len));
                    memcpy(last_line_buffer, t.readline_vec[i].ptr, -(t.readline_vec[i].len));
                    last_line_buffer_len = -(t.readline_vec[i].len);
                    i++; continue;
                } else {
                    i++; continue;
                }
                if (read.len >= _min_read_len) batch_reads.push_back(read);
                else delete read.read;
                if (batch_reads.size() >= _read_batch_size) {
                    _RBQ.wait_push(batch_reads, _n_threads_consumer + 1);
                    batch_reads = std::vector<ReadPtr>();
                    batch_reads.reserve(_read_batch_size);
                }
                #ifdef STEP3P
                i += skip+1;
                skip = 0;
                #else
                i++;
                #endif
            }
            #ifdef STEP3P
            _LBQ2.wait_push(t, _max_queue_size);
            // if (t.flag->test_and_set()) {delete t.data.buf; delete t.flag;}
            #else
            // delete t.data.buf; // malloc in _STEP1_load_from_file // NOT MMAP
            munmap(t.data.buf, t.data.size); // mmap in _STEP1_load_from_file // MMAP
            #endif
        }
        if (start_from_buffer) {
            // read.len = last_line_buffer.length();
            read.len = last_line_buffer_len;
            if (read.len >= _min_read_len) {
                read.read = new char[read.len];
                // memcpy(read.read, last_line_buffer.c_str(), read.len);
                memcpy(read.read, last_line_buffer, read.len);
                batch_reads.push_back(read);
            }
        }
        if (batch_reads.size()) _RBQ.push(batch_reads); // add last batch of reads
        #ifdef STEP3P
        _LBQ2.finish();
        #else
        _RBQ.finish();
        #endif
        std::cout<<"Loader finish 3 in "/*<<pop_cnt<<" "<<push_cnt<<" "<<line_cnt*/<<wct_readloader.stop()<<std::endl;
    }
public:
    ReadLoader(const ReadLoader&) = delete;
    void operator=(const ReadLoader&) = delete;
    ReadLoader () = delete;
    
    static const size_t MB = 1048576;
    static const size_t KB = 1024;
    ReadLoader (std::vector<std::string> filenames, T_read_len min_read_len = 0, size_t read_batch_size = 4096, int buffer_size_MB = 16, int max_buffer_size_MB = 1024, int n_threads_consumer = 16) {
        _filenames = filenames;
        _is_fasta = *(filenames[0].rbegin()) == 'a' || *(filenames[0].rbegin()) == 'A';
        if (_is_fasta) max_buffer_size_MB /= 2; // for fasta, the buffer size required should be 1/2 smaller.
        _min_read_len = min_read_len;
        _read_batch_size = read_batch_size;
        _max_queue_size = max_buffer_size_MB / buffer_size_MB;
        _buffer_size = buffer_size_MB * MB;
        _n_threads_consumer = n_threads_consumer;
        std::cout<<(_is_fasta?"Fasta format.":"Fastq format.")<<std::endl;
    }
    void start_load_reads() {
        _started_threads.push_back(std::thread(&ReadLoader::_STEP1_load_from_file, this));
        _started_threads.push_back(std::thread(&ReadLoader::_STEP2_find_newline, this));
        _started_threads.push_back(std::thread(&ReadLoader::_STEP3_extract_read, this));
        // if (_n_threads_consumer >= 16) {
        //     ThreadPool<void>::set_thread_affinity(_started_threads[0], 1, _n_threads_consumer);
        //     ThreadPool<void>::set_thread_affinity(_started_threads[1], 1, _n_threads_consumer);
        //     ThreadPool<void>::set_thread_affinity(_started_threads[2], 0);
        // }
        #ifdef STEP3P
        for (int i=0; i<_n_step3p; i++)
            _started_threads.push_back(std::thread(&ReadLoader::_STEP3PLUS, this));
        _started_step3p = _n_step3p;
        #endif
    }
    void join_threads() {
        for (std::thread &t: _started_threads)
            if (t.joinable()) t.join();
    }
    #ifdef DEBUG
    void ss() {
        WallClockTimer wct0;
        _STEP1_load_from_file();
        std::cerr<<wct0.stop(true)<<std::endl;
        WallClockTimer wct1;
        _STEP2_find_newline();
        std::cerr<<wct1.stop(true)<<std::endl;
        WallClockTimer wct2;
        _STEP3_extract_read();
        std::cerr<<wct2.stop(true)<<std::endl;
    }
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
        std::cout<<"1 Load    push wait: "<<_DBQ.debug_push_wait<<"\tpop wait: "<<_DBQ.debug_pop_wait<<"\tsize: "<<_DBQ.size_est()<<std::endl;
        std::cout<<"2 Newline push wait: "<<_LBQ.debug_push_wait<<"\tpop wait: "<<_LBQ.debug_pop_wait<<"\tsize: "<<_LBQ.size_est()<<std::endl;
        #ifdef STEP3P
        std::cout<<"    _LBQ2 push wait: "<<_LBQ2.debug_push_wait<<"\tpop wait: "<<_LBQ2.debug_pop_wait<<std::endl;
        #endif
        std::cout<<"3 Extract push wait: "<<_RBQ.debug_push_wait<<"\tpop wait: "<<_RBQ.debug_pop_wait<<"\tsize: "<<_RBQ.size_est()<<std::endl;
    }
    #endif

    static void work_while_loading (T_kvalue K_kmer, 
    std::function<void(std::vector<ReadPtr>&, int)> work_func, 
    std::function<void(std::vector<ReadPtr>&, int)> p0_func, 
    int worker_threads, std::vector<std::string> &filenames, 
    T_read_cnt batch_size, size_t max_buffer_size_MB = 1024, size_t buffer_size_MB = 16, T_read_cnt reads_for_mmhisto = 100000) {
        
        T_read_cnt n_read_loaded = 0;

        // ThreadAffinity ta;
        // if (worker_threads >= 16) ta={1, worker_threads, 0, 2}; // TODO: pass GPU threads
        ThreadPool<void> tp0(worker_threads, worker_threads);
        ThreadPool<void> tp(worker_threads, worker_threads); // +(worker_threads >= 8 ? worker_threads / 4 : 2));
        ReadLoader rl(filenames, K_kmer, batch_size, buffer_size_MB, max_buffer_size_MB, worker_threads);
        rl.start_load_reads();

        std::vector<ReadPtr> read_batch;
        WallClockTimer wct0;
        ConcQueue<std::vector<ReadPtr>> unprocessed_reads;
        T_read_cnt n_reads_p0 = 0;
        atomic<int> timer_p0{0};
        while (reads_for_mmhisto && rl._RBQ.pop(read_batch)) {
            std::vector<ReadPtr> *my_read_batch = new std::vector<ReadPtr>();//
            n_read_loaded += read_batch.size();
            n_reads_p0 += read_batch.size();
            *my_read_batch = std::move(read_batch);
            tp0.commit_task([my_read_batch, &p0_func, &unprocessed_reads, &timer_p0] (int tid) {
                WallClockTimer wct;
                p0_func(*my_read_batch, tid);
                unprocessed_reads.push(*my_read_batch);
                timer_p0.fetch_add(wct.stop(true));
            });
            if (n_reads_p0 >= reads_for_mmhisto) break;
        }
        tp0.finish();
        unprocessed_reads.finish();
        GenMMDict();
        cerr<<"timer_p0_threads: "<<timer_p0<<endl;
        cerr<<"wct0 all: "<<wct0.stop(true)<<endl;

        while (rl._RBQ.pop(read_batch)) {
            // std::cerr<<rl._DBQ.size_est()<<" | "<<rl._LBQ.size_est()<<" | "<<rl._RBQ.size_est()<<std::endl;
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
        // process the reads for MM histo calculation
        while (reads_for_mmhisto && unprocessed_reads.pop(read_batch)) {
            tp.hold_when_busy();
            std::vector<ReadPtr> *my_read_batch = new std::vector<ReadPtr>();//
            *my_read_batch = std::move(read_batch);
            tp.commit_task([my_read_batch, &work_func] (int tid) {
                work_func(*my_read_batch, tid);
                for (ReadPtr &i: *my_read_batch) delete i.read;
                delete my_read_batch;//
            });
        }
        // std::cerr<<"Total reads loaded: "<<n_read_loaded<<std::endl;
        std::cerr<<"Total reads loaded: "<<n_read_loaded<<"\tReads for minimizer histogram: "<<n_reads_p0<<std::endl;
        // logger->log("Total reads loaded: "+std::to_string(n_read_loaded)+"\tReads for minimizer histogram: "+std::to_string(n_reads_p0));
        
        rl.join_threads();
        tp.finish();
        #ifdef WAITMEASURE
        rl.output_wait();
        #endif
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

class GZReadLoader {
private:
    ConcQueue<std::string> *_FNQ;           // filenames queue
    ConcQueue<DataBuffer> _DBQ;             // data buffer queue
    ConcQueue<LineBuffer> _LBQ;             // line buffer queue
    ConcQueue<std::vector<ReadPtr>> *_RBQ;  // read batch queue
    std::atomic<int> *_RBQ_FINISH_CNT;
    bool _is_fasta;
    
    size_t _read_batch_size;
    T_read_len _min_read_len;
    
    size_t _buffer_size;
    size_t _max_queue_size;
    
    int _n_threads_consumer;

    std::vector<std::thread> _started_threads;
    
    void _STEP1_load_from_file () { // NOT MMAP
        char *buf = new char [_buffer_size];//
        size_t cur_size;
        char tmp;
        std::string filename;
        while (_FNQ->pop(filename)) {
            gzFile fp = gzopen(filename.c_str(), "rb");
            // std::cerr<<"Open gz file ["<<filename<<"]: "<<fp<<std::endl;
            
            while (true) {
                DataBuffer t;
                t.size = gzread(fp, buf, _buffer_size);
                if (t.size == 0) break;
                t.buf = buf;
                buf = new char[_buffer_size];
                _DBQ.wait_push(t, _max_queue_size);
            }
            DataBuffer t;
            t.size = 0;
            t.buf = nullptr;
            #ifdef WAITMEASURE
            output_wait();
            #endif
            _DBQ.push(t); // push a null block when file ends
            std::cerr<<"          ["<<filename<<"] closed: "<<gzclose(fp)<<std::endl;
        }
        _DBQ.finish();
        std::cout<<"Loader finish 1 "/*<<push_cnt*/<<std::endl;
    }
    void _STEP2_find_newline () { // NOT MMAP
        DataBuffer t;
        unsigned char flag_mask = _is_fasta ? 0b1 : 0b11;
        unsigned char is_read_line = 0;
        char *line_beg, *line_end;
        while (_DBQ.pop(t)) {
            LineBuffer x;
            x.data = std::move(t);
            if (x.data.size == 0) {
                is_read_line = 0; // new file
                continue;
            }
            x.readline_vec.reserve((t.size>>10)+8);
            line_end = x.data.buf-1;
            while (true) {
                line_beg = line_end+1;
                line_end = std::find(line_beg, x.data.buf+x.data.size, '\n');
                if (is_read_line == 1) {
                    if (line_end == x.data.buf+x.data.size) {
                        x.readline_vec.push_back({line_beg, (int)(-(line_end-line_beg))});
                        break;
                    } else {
                        x.readline_vec.push_back({line_beg, (int)(line_end-line_beg)});
                    }
                }
                if (line_end == x.data.buf+x.data.size) break;
                is_read_line = (is_read_line+1)&flag_mask;
            }
            _LBQ.wait_push(x, _max_queue_size);
        }
        _LBQ.finish();
        std::cout<<"Loader finish 2 "<</*push_cnt<<" "<<line_cnt<<*/std::endl;
    }
    void _STEP3_extract_read () {
        WallClockTimer wct_readloader;
        
        std::vector<ReadPtr> batch_reads;
        batch_reads.reserve(_read_batch_size);

        bool start_from_buffer = false;
        std::string last_line_buffer;
        ReadPtr read;

        LineBuffer t;
        while (_LBQ.pop(t)) {
            for (std::vector<ReadLine>::iterator i = t.readline_vec.begin(); i != t.readline_vec.end();) {
                if (start_from_buffer) {
                    read.len = last_line_buffer.length() + i->len;
                    read.read = new char[read.len];
                    memcpy(read.read, last_line_buffer.c_str(), last_line_buffer.length());
                    memcpy(read.read + last_line_buffer.length(), i->ptr, i->len);
                    start_from_buffer = false;
                } else if (i->len > 0) {
                    read.len = i->len;
                    read.read = new char[read.len];
                    memcpy(read.read, i->ptr, i->len);
                } else if (i->len < 0) {
                    start_from_buffer = true;
                    last_line_buffer = std::string(i->ptr, -(i->len));
                    i++; continue;
                } else {
                    i++; continue;
                }
                if (read.len >= _min_read_len) batch_reads.push_back(read);
                else delete read.read;
                if (batch_reads.size() >= _read_batch_size) {
                    _RBQ->wait_push(batch_reads, _n_threads_consumer + 1);
                    batch_reads = std::vector<ReadPtr>();
                    batch_reads.reserve(_read_batch_size);
                }
                i++;
            }
            delete t.data.buf; // malloc in _STEP1_load_from_file // NOT MMAP
        }
        if (start_from_buffer) {
            read.len = last_line_buffer.length();
            if (read.len >= _min_read_len) {
                read.read = new char[read.len];
                memcpy(read.read, last_line_buffer.c_str(), read.len);
                batch_reads.push_back(read);
            }
        }
        if (batch_reads.size()) _RBQ->push(batch_reads); // add last batch of reads
        if (_RBQ_FINISH_CNT->fetch_sub(1) == 1) _RBQ->finish();
        std::cout<<"Loader finish 3 in "/*<<pop_cnt<<" "<<push_cnt<<" "<<line_cnt*/<<wct_readloader.stop()<<std::endl;
    }

public:
    GZReadLoader (const GZReadLoader&) = delete;
    void operator=(const GZReadLoader&) = delete;
    GZReadLoader () = delete;
    
    static const size_t MB = 1048576;
    static const size_t KB = 1024;
    GZReadLoader (ConcQueue<std::string> *filenames, bool is_fasta, ConcQueue<std::vector<ReadPtr>> *outputbatches, std::atomic<int> *rbq_finish_cnt, T_read_len min_read_len = 0, size_t read_batch_size = 4096, int buffer_size_MB = 16, int max_buffer_size_MB = 1024, int n_threads_consumer = 16) {
        _FNQ = filenames;
        _RBQ = outputbatches;
        _RBQ_FINISH_CNT = rbq_finish_cnt;
        _is_fasta = is_fasta;
        if (_is_fasta) max_buffer_size_MB /= 2; // for fasta, the buffer size required should be 1/2 smaller.
        _min_read_len = min_read_len;
        _read_batch_size = read_batch_size;
        _max_queue_size = max_buffer_size_MB / buffer_size_MB / 2;
        _buffer_size = buffer_size_MB * MB;
        _n_threads_consumer = n_threads_consumer;
        std::cout<<(_is_fasta?"GZ Fasta format.":"GZ Fastq format.")<<std::endl;
    }
    void start_load_reads() {
        _started_threads.push_back(std::thread(&GZReadLoader::_STEP1_load_from_file, this));
        _started_threads.push_back(std::thread(&GZReadLoader::_STEP2_find_newline, this));
        _started_threads.push_back(std::thread(&GZReadLoader::_STEP3_extract_read, this));
    }
    void join_threads() {
        for (std::thread &t: _started_threads)
            if (t.joinable()) t.join();
    }
    #ifdef WAITMEASURE
    void output_wait () {
        std::cout<<"1 Load    push wait: "<<_DBQ.debug_push_wait<<"\tpop wait: "<<_DBQ.debug_pop_wait<<"\tsize: "<<_DBQ.size_est()<<std::endl;
        std::cout<<"2 Newline push wait: "<<_LBQ.debug_push_wait<<"\tpop wait: "<<_LBQ.debug_pop_wait<<"\tsize: "<<_LBQ.size_est()<<std::endl;
        std::cout<<"3 Extract push wait: "<<_RBQ->debug_push_wait<<"\tpop wait: "<<_RBQ->debug_pop_wait<<"\tsize: "<<_RBQ->size_est()<<std::endl;
    }
    #endif

    static void work_while_loading_gz (T_kvalue K_kmer, 
    std::function<void(std::vector<ReadPtr>&, int)> work_func, 
    std::function<void(std::vector<ReadPtr>&, int)> p0_func, 
    int worker_threads, 
    std::vector<std::string> &filenames, int loader_threads,
    T_read_cnt batch_size, size_t max_buffer_size_MB = 1024, size_t buffer_size_MB = 16, T_read_cnt reads_for_mmhisto = 100000) {
        
        T_read_cnt n_read_loaded = 0;

        ThreadPool<void> tp0(worker_threads, worker_threads);
        ThreadPool<void> tp(worker_threads, worker_threads); // +(worker_threads >= 8 ? worker_threads / 4 : 2));
        std::vector<GZReadLoader*> rls;

        ConcQueue<std::string> filenames_queue;
        ConcQueue<std::vector<ReadPtr>> outputbatches_queue;
        std::atomic<int> rbq_finish_cnt;
        rbq_finish_cnt = loader_threads;
        bool is_fasta = *(filenames[0].end()-4) == 'a' || *(filenames[0].end()-4) == 'A';
        for (auto i: filenames) filenames_queue.push(i);
        filenames_queue.finish();
        for (int i=0; i<loader_threads; i++) {
            rls.push_back(new GZReadLoader(&filenames_queue, is_fasta, &outputbatches_queue, &rbq_finish_cnt, K_kmer, batch_size, buffer_size_MB, max_buffer_size_MB, worker_threads));
            (*rls.rbegin())->start_load_reads();
        }

        std::vector<ReadPtr> read_batch;
        ConcQueue<std::vector<ReadPtr>> unprocessed_reads;
        T_read_cnt n_reads_p0 = 0;
        while (outputbatches_queue.pop(read_batch)) {
            std::vector<ReadPtr> *my_read_batch = new std::vector<ReadPtr>();//
            n_read_loaded += read_batch.size();
            n_reads_p0 += read_batch.size();
            *my_read_batch = std::move(read_batch);
            tp0.commit_task([my_read_batch, &p0_func, &unprocessed_reads] (int tid) {
                p0_func(*my_read_batch, tid);
                unprocessed_reads.push(*my_read_batch);
            });
            if (n_read_loaded >= reads_for_mmhisto) break;
        }
        tp0.finish();
        unprocessed_reads.finish();
        GenMMDict();

        while (outputbatches_queue.pop(read_batch)) {
            tp.hold_when_busy();
            n_read_loaded += read_batch.size();
            std::vector<ReadPtr> *my_read_batch = new std::vector<ReadPtr>();//
            *my_read_batch = std::move(read_batch);
            tp.commit_task([my_read_batch, &work_func] (int tid) {
                work_func(*my_read_batch, tid);
                for (ReadPtr &i: *my_read_batch) delete i.read;
                delete my_read_batch;//
            });
        }

        // process the reads for MM histo calculation
        while (unprocessed_reads.pop(read_batch)) {
            tp.hold_when_busy();
            std::vector<ReadPtr> *my_read_batch = new std::vector<ReadPtr>();//
            *my_read_batch = std::move(read_batch);
            tp.commit_task([my_read_batch, &work_func] (int tid) {
                work_func(*my_read_batch, tid);
                for (ReadPtr &i: *my_read_batch) delete i.read;
                delete my_read_batch;//
            });
        }
        // std::cerr<<"Total reads loaded: "<<n_read_loaded<<std::endl;
        std::cerr<<"Total reads loaded: "<<n_read_loaded<<"\tReads for minimizer histogram: "<<n_reads_p0<<std::endl;
        // logger->log("Total reads loaded: "+std::to_string(n_read_loaded)+"\tReads for minimizer histogram: "+std::to_string(n_reads_p0));
        
        for (int i=0; i<loader_threads; i++) {
            rls[i]->join_threads();
            delete rls[i];
        }
        tp.finish();
    }
};

#endif