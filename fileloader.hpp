#ifndef _FILELOADER_HPP
#define _FILELOADER_HPP

// #define STEP3P

#include <algorithm>
#include <future>
#include <thread>
#include <functional>
#include <vector>
#include <cstring>
#include "concqueue.hpp"
#include "types.h"
#include "thread_pool.hpp"

#ifdef USEMMAP
#include <fcntl.h>      // open
#include <sys/mman.h>   // mmap
#include <sys/stat.h>   // fstat
#include <unistd.h>     // close
#endif

#ifdef DEBUG
#include <cassert>
#endif

struct DataBuffer {
    char *buf;
    size_t size;
    #ifdef USEMMAP
    int closefd = -1;
    #endif
};
struct ReadLine {
    char *ptr;
    int len; // +: read length, -: read length (not ended)
};
struct LineBuffer {
    DataBuffer data;
    // std::vector<char*> newline_vec;
    std::vector<ReadLine> readline_vec;
    /* std::vector<char*> newline_vec2;*/
};

class ReadLoader {
private:
    ConcQueue<DataBuffer> _DBQ;             // data buffer queue
    ConcQueue<LineBuffer> _LBQ;             // line buffer queue
    #ifdef STEP3P
    ConcQueue<LineBuffer> _LBQ2;             // line buffer queue
    #endif
    ConcQueue<std::vector<ReadPtr>> _RBQ;    // read batch queue
    std::vector<std::string> _filenames;
    bool _is_fasta;
    
    size_t _read_batch_size;
    T_read_len _min_read_len;
    
    size_t _buffer_size;
    size_t _max_queue_size;
    
    int _n_threads_consumer;

    std::vector<std::thread> _started_threads;
    
    #ifdef USEMMAP
    void _STEP1_load_from_file () {
        // char *buf = new char [_buffer_size];//
        size_t cur_size;
        char tmp;
        for (std::string &filename: _filenames) {
            int fd = open(filename.c_str(), O_RDONLY);
            if (fd == -1) {
                std::cerr<<"Error when open "<<filename<<" ["<<errno<<"]: ";
                perror("");
                exit(errno);
            } // assert(fd != -1);
            struct stat statue;
            fstat(fd, &statue);
            std::cerr<<"Open file ["<<filename<<"]: "<<fd<<" "<<"filesize = "<<statue.st_size<<std::endl;
            char *fileptr = (char *) mmap(NULL, statue.st_size, PROT_READ, MAP_SHARED, fd, 0); // MAP_SHARED
            close(fd);

            for (size_t i=0; i<statue.st_size; i+=_buffer_size) {
                DataBuffer t;
                t.buf = fileptr + i;
                t.size = statue.st_size - i < _buffer_size ? statue.st_size - i : _buffer_size;
                _DBQ.wait_push(t, _max_queue_size);
                // tmp = *t.buf; // help to prefetch
            }
            
            DataBuffer t;
            t.size = statue.st_size;
            t.buf = fileptr;
            t.closefd = fd;
            #ifdef WAITMEASURE
            output_wait();
            #endif
            _DBQ.push(t); // push a null block when file ends
        }
        _DBQ.finish();
        std::cout<<"Loader finish 1 "/*<<push_cnt*/<<std::endl;
    }
    #else
    void _STEP1_load_from_file () { // NOT MMAP
        char *buf = new char [_buffer_size];//
        size_t cur_size;
        char tmp;
        for (std::string &filename: _filenames) {
            FILE *fp = fopen(filename.c_str(), "rb");
            std::cerr<<"Open file ["<<filename<<"]: "<<fp<<std::endl;
            while (true) {
                DataBuffer t;
                t.size = fread(buf, 1, _buffer_size, fp);
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
            std::cerr<<"           "<<filename<<"  closed: "<<fclose(fp)<<std::endl;
        }
        _DBQ.finish();
        std::cout<<"Loader finish 1 "/*<<push_cnt*/<<std::endl;
    }
    #endif
    #ifdef USEMMAP
    void _STEP2_find_newline () {
        // size_t push_cnt = 0;
        // size_t line_cnt = 0;
        DataBuffer t;
        while (_DBQ.pop(t)) {
            // size_t move_offs = 0;
            LineBuffer x;
            x.data = std::move(t);
            x.newline_vec.reserve((t.size>>11)+8);
            // x.newline_vec.push_back(-1);
            x.newline_vec.push_back(x.data.buf-1);
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
            char *find_beg = x.data.buf-1;
            while (true) {
                find_beg = std::find(find_beg+1, x.data.buf+x.data.size, '\n');
                if (find_beg == x.data.buf+x.data.size) break;
                // x.newline_vec.push_back(find_beg - x.data.buf);
                x.newline_vec.push_back(find_beg);
            }
            _LBQ.wait_push(x, _max_queue_size);
            // push_cnt++;
        }
        _LBQ.finish();
        std::cout<<"Loader finish 2 "<</*push_cnt<<" "<<line_cnt<<*/std::endl;
    }
    #else
    // void _STEP2_find_newline () { // NOT MMAP
    //     DataBuffer t;
    //     while (_DBQ.pop(t)) {
    //         LineBuffer x;
    //         x.data = std::move(t);
    //         x.newline_vec.reserve((t.size>>10)+8);
    //         x.newline_vec.push_back(x.data.buf-1);
    //         char *find_beg = x.data.buf-1;
    //         while (true) {
    //             find_beg = std::find(find_beg+1, x.data.buf+x.data.size, '\n');
    //             if (find_beg == x.data.buf+x.data.size) break;
    //             x.newline_vec.push_back(find_beg);
    //         }
    //         _LBQ.wait_push(x, _max_queue_size);
    //     }
    //     _LBQ.finish();
    //     std::cout<<"Loader finish 2 "<</*push_cnt<<" "<<line_cnt<<*/std::endl;
    // }
    void _STEP2_find_newline () { // NOT MMAP
        DataBuffer t;
        unsigned char flag_mask = _is_fasta ? 0b1 : 0b11;
        unsigned char is_read_line = 0;
        char *line_beg, *line_end;
        while (_DBQ.pop(t)) {
            LineBuffer x;
            // #ifdef STEP3P
            // x.flag = new std::atomic_flag();
            // x.flag -> clear();
            // #endif
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
    #endif
    #ifdef STEP3P
    void _STEP3PLUS () {
        std::vector<ReadPtr> batch_reads;
        batch_reads.reserve(_read_batch_size);
        ReadPtr read;

        LineBuffer t;
        int i_end;
        while (_LBQ2.pop(t)) {
            i_end = t.readline_vec.size()/5*3;
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
            delete t.data.buf;
        }
        if (batch_reads.size()) _RBQ.push(batch_reads); // add last batch of reads
        _RBQ.finish();
    }
    #endif
    void _STEP3_extract_read () {
        WallClockTimer wct_readloader;
        
        std::vector<ReadPtr> batch_reads;
        batch_reads.reserve(_read_batch_size);

        bool start_from_buffer = false;
        std::string last_line_buffer;
        ReadPtr read;

        LineBuffer t;
        while (_LBQ.pop(t)) {
            #ifdef STEP3P
            int add = t.readline_vec.size()/5*3 > 1 ? t.readline_vec.size()/5*3 : 1;
            #endif
            // for (ReadLine i: t.readline_vec) {
            for (std::vector<ReadLine>::iterator i = t.readline_vec.begin(); i != t.readline_vec.end();) {
                if (start_from_buffer) {
                    read.len = last_line_buffer.length() + i->len;
                    read.read = new char[read.len];
                    memcpy(read.read, last_line_buffer.c_str(), last_line_buffer.length());
                    memcpy(read.read + last_line_buffer.length(), i->ptr, i->len);
                    start_from_buffer = false;
                } else if (i->len > 0) {
                    // std::cerr<<i.len<<"@"<<i.ptr<<std::endl;
                    read.len = i->len;
                    read.read = new char[read.len];
                    memcpy(read.read, i->ptr, i->len);
                } else if (i->len < 0) {
                    start_from_buffer = true;
                    last_line_buffer = std::string(i->ptr, -(i->len));
                    i++; continue;
                }
                if (read.len >= _min_read_len) batch_reads.push_back(read);
                if (batch_reads.size() >= _read_batch_size) {
                    _RBQ.wait_push(batch_reads, _n_threads_consumer + 1);
                    batch_reads = std::vector<ReadPtr>();
                    batch_reads.reserve(_read_batch_size);
                }
                #ifdef STEP3P
                i += add;
                add = 1;
                #else
                i++;
                #endif
            }
            #ifdef STEP3P
            _LBQ2.wait_push(t, _max_queue_size);
            // if (t.flag->test_and_set()) {delete t.data.buf; delete t.flag;}
            #else
            delete t.data.buf; // malloc in _STEP1_load_from_file // NOT MMAP
            #endif
        }
        if (start_from_buffer) {
            read.len = last_line_buffer.length();
            read.read = new char[read.len];
            memcpy(read.read, last_line_buffer.c_str(), read.len);
            batch_reads.push_back(read);
        }
        if (batch_reads.size()) _RBQ.push(batch_reads); // add last batch of reads
        #ifdef STEP3P
        _LBQ2.finish();
        #else
        _RBQ.finish();
        #endif
        std::cout<<"Loader finish 3 in "/*<<pop_cnt<<" "<<push_cnt<<" "<<line_cnt*/<<wct_readloader.stop()<<std::endl;
    }
    // void _STEP3_extract_read () {
    //     WallClockTimer wct_readloader;
    //     // size_t push_cnt = 0, pop_cnt = 0, line_cnt = 0;;
        
    //     std::vector<ReadPtr> batch_reads;
    //     batch_reads.reserve(_read_batch_size);

    //     unsigned char line_flag = 0; // indicates a read line when it equals to 1
    //     unsigned char flag_mask = _is_fasta ? 0b1 : 0b11; // line_flag = (line_flag+1) & flag_mask;
    //     bool start_from_buffer = false;
    //     std::string last_line_buffer="";
    //     ReadPtr read;

    //     LineBuffer t;
    //     size_t i;
    //     while (_LBQ.pop(t)) {
    //         // pop_cnt++;
    //         // line_cnt+=t.newline_vec.size();
    //         #ifdef USEMMAP
    //         if (t.data.closefd!=-1) { // new file // MMAP
    //             std::cerr<<"File closed: "<<munmap(t.data.buf, t.data.size)<</*close(t.data.closefd)<<*/", size: "<<t.data.size<<std::endl;
    //             line_flag = 0;
    //             continue;
    //         }
    //         #else
    //         if (t.data.size==0) { // new file // NOT MMAP
    //             line_flag = 0;
    //             continue;
    //         }
    //         #endif
    //         /* t.newline_vec.reserve(t.newline_vec.size() + t.newline_vec2.size());
    //         t.newline_vec.insert(t.newline_vec.end(), t.newline_vec2.begin(), t.newline_vec2.end()); */
    //         // for (i = 1; i < t.newline_vec.size(); i++, line_flag=(line_flag+1) & flag_mask) { // begins from 1 because q[0]=-1
    //         for(std::vector<char*>::iterator it = t.newline_vec.begin()+1; it != t.newline_vec.end(); it++, line_flag=(line_flag+1) & flag_mask) {
    //             if (line_flag == 1) {
    //                 if (start_from_buffer) {
    //                     read.len = last_line_buffer.size() + *it - *(it-1) - 1;// t.newline_vec[i];
    //                     // if (t.newline_vec[i]-1>0 && t.data.buf[t.newline_vec[i]-1]=='\t') read.len--;
                        
    //                     if ((*it-1) > (*t.newline_vec.begin()) && *(*it-1) == '\t') read.len--;
                        
    //                     read.read = new char [read.len];
    //                     memcpy(read.read, last_line_buffer.c_str(), last_line_buffer.size());
    //                     memcpy(read.read + last_line_buffer.size(), t.data.buf, read.len - last_line_buffer.size());
    //                     last_line_buffer = "";
    //                     start_from_buffer = false;
    //                 } else {
    //                     // read.len = t.newline_vec[i] - (t.newline_vec[i-1] + 1);
    //                     read.len = *it - *(it-1) - 1;
    //                     // if (t.newline_vec[i]-1>0 && t.data.buf[t.newline_vec[i]-1]=='\t') read.len--;
                        
    //                     if ((*it-1) > (*t.newline_vec.begin()) && *(*it-1) == '\t') read.len--;
                        
    //                     read.read = new char [read.len];
    //                     // memcpy(read.read, &(t.data.buf[t.newline_vec[i-1] + 1]), read.len);
    //                     memcpy(read.read, *(it-1)+1, read.len);
    //                 }
    //                 if (read.len >= _min_read_len) batch_reads.push_back(read);
    //                 if (batch_reads.size() >= _read_batch_size) {
    //                     _RBQ.wait_push(batch_reads, _n_threads_consumer + 1);
    //                     batch_reads = std::vector<ReadPtr>();
    //                     batch_reads.reserve(_read_batch_size);
    //                 }
    //                 // push_cnt++;
    //             }
    //         }
    //         // if (line_flag == 1 && *t.newline_vec.rbegin() < t.data.size - 1) { // prepare last line buffer
    //         if (line_flag == 1 && *t.newline_vec.rbegin() - *t.newline_vec.begin() < t.data.size) {
    //             start_from_buffer = true;
    //             // last_line_buffer = std::string(&t.data.buf[*t.newline_vec.rbegin()+1], t.data.size - *t.newline_vec.rbegin() - 1);
    //             last_line_buffer = std::string((*t.newline_vec.rbegin())+1, t.data.size - (*t.newline_vec.rbegin() - *t.newline_vec.begin() - 1) - 1);
    //         }
    //         #ifndef USEMMAP
    //         delete t.data.buf; // malloc in _STEP1_load_from_file // NOT MMAP
    //         #endif
    //     }
    //     if (batch_reads.size()) _RBQ.push(batch_reads); // add last batch of reads
    //     _RBQ.finish();
    //     std::cout<<"Loader finish 3 in "/*<<pop_cnt<<" "<<push_cnt<<" "<<line_cnt*/<<wct_readloader.stop()<<std::endl;
    // }
public:
    static const size_t MB = 1048576;
    static const size_t KB = 1024;
    ReadLoader (std::vector<std::string> filenames, T_read_len min_read_len = 0, size_t read_batch_size = 4096, int buffer_size_MB = 16, int max_buffer_size_MB = 1024, int n_threads_consumer = 16) {
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
        #ifdef STEP3P
        _started_threads.push_back(std::thread(&ReadLoader::_STEP3PLUS, this));
        #endif
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
        #ifdef STEP3P
        std::cout<<"    _LBQ2 push wait: "<<_LBQ2.debug_push_wait<<"\tpop wait: "<<_LBQ2.debug_pop_wait<<std::endl;
        #endif
        std::cout<<"3 Extract push wait: "<<_RBQ.debug_push_wait<<"\tpop wait: "<<_RBQ.debug_pop_wait<<std::endl;
    }
    #endif
    static void work_while_loading (T_kvalue K_kmer, std::function<void(std::vector<ReadPtr>&, int)> work_func, int worker_threads, std::vector<std::string> &filenames, 
    T_read_cnt batch_size, size_t max_buffer_size_MB = 1024, size_t buffer_size_MB = 16) {
        
        T_read_len n_read_loaded = 0;

        ThreadPool<void> tp(worker_threads, worker_threads); // +(worker_threads >= 8 ? worker_threads / 4 : 2));
        ReadLoader rl(filenames, K_kmer, batch_size, buffer_size_MB, max_buffer_size_MB, worker_threads);
        rl.start_load_reads();

        std::vector<ReadPtr> read_batch;
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
        std::cerr<<"Total reads loaded: "<<n_read_loaded<<std::endl;
        
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

#endif