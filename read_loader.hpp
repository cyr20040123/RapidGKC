#define DEBUG

#ifndef _READ_LOADER_HPP
#define _READ_LOADER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sys/stat.h>   // for getting file size
#include "types.h"
#include "thread_pool.hpp"
#ifdef DEBUG
#include <cassert>
#endif
using namespace std;

#ifdef DEBUG
atomic<size_t> checksum{0};
#endif

class ReadLoader {
private:
    size_t CUR_BUF_SIZE = 32 * MB;      // batch buffer size
    size_t LINE_BUF_SIZE = 4 * MB;      // should be larger than 3 * max_read_len (minimum buffer size)
    string _filename;                   // filename
    char** _buf_cur;                    // current thread[i]'s buffer
    int _n_threads;                     // n_threads for file loading
    future<int> *_proc_res;             // the number of reads loaded by this thread
    string* _buf_prev_remain;           // the buffer of the tail data prepared for the next thread[i]
    mutex *_pbuf_mtxs;                  // _buf_prev_remain mutex
    unique_lock<mutex> *_pbuf_locks;    // _buf_prev_remain locks
    condition_variable *_pbuf_cvs;      // _buf_prev_remain condition variable
    bool *_pbuf_set;                    // whether _buf_prev_remain is set by the previous thread
    vector<string*> *_thread_reads;     // each thread store its reads in one vector
    // vector<bool> *_reads_deleted;       // whether corresponding thread read is deleted
    mutex *_thread_reads_mtx;           // for getting _thread_reads while file loading not finished
    vector<T_read_cnt> *_thread_bat_split_pos; // store the split position of each batch to keep the original order of the reads

    T_read_cnt _load_fastq (int tid, bool last_batch = false) {  // 1. process prev, 2. set next prev, 3. process cur (consider 1 process only)
        // string format = '%' + to_string(PREV_BUF_SIZE) + 's';
        string buf;
        if (!last_batch) buf.resize(CUR_BUF_SIZE + LINE_BUF_SIZE);
        buf = "";
        
        // ---- Process _buf_prev_remain ----
        while(!_pbuf_set[tid]) {
            // _pbuf_cvs[tid].wait(_pbuf_locks[tid]);
            _pbuf_cvs[tid].wait_for(_pbuf_locks[tid], 100ms);
        } // xxx busy wait: 当buf_size过小(10m)时，会卡且满cpu占用。去掉也会概率卡死。。。
        buf += _buf_prev_remain[tid];   // handle pbuf
        _pbuf_set[tid] = false;
        _pbuf_cvs[tid].notify_all();
        
        // ---- Set the next _buf_prev_remain ----
        size_t remain_pos;
        bool find_at;
        size_t newline_pos;
        if (!last_batch) {
            int tid_next = (tid+1)%_n_threads;
            
            // set next pbuf
            buf += _buf_cur[tid];
            
            find_at = false;
            remain_pos = buf.rfind('\n');
            while (remain_pos < buf.length()) { // 不能直接反向找@开头的行，因为quality也可能是@开头的.
                if (buf[remain_pos+1] == '@') { // find info line
                    if (remain_pos+1 == 0) {
                        find_at = true;
                        break;
                    }
                    newline_pos = buf.rfind('\n', remain_pos-1);
                    if (newline_pos == string::npos) exit(1);
                    newline_pos = buf.rfind('\n', newline_pos-1);
                    if (newline_pos == string::npos) exit(1);
                    if (buf[newline_pos+1] == '+') {
                        find_at = true;
                        break;
                    }
                }
                remain_pos = buf.rfind('\n', remain_pos-1);
            }
            if (find_at == false) remain_pos = -1; // give all batch data to the next thread if can't find a proper start

            while(_pbuf_set[tid_next]) {
                // _pbuf_cvs[tid_next].wait(_pbuf_locks[tid_next]);
                _pbuf_cvs[tid_next].wait_for(_pbuf_locks[tid_next], 100ms);
            } // not busy wait
            _buf_prev_remain[tid_next] = string(buf.begin()+remain_pos+1, buf.end());
            _pbuf_set[tid_next] = true;
            _pbuf_cvs[tid_next].notify_all();
        } else {
            remain_pos = buf.length();
        }
        // ---- Process _buf_cur ----
        lock_guard<mutex> lg(_thread_reads_mtx[tid]);
        assert(buf[0] == '@'); // TODO: 不用判断.
        // if (buf[0] != '@') { 
        //     cerr << tid << " Error: wrong file format (not fastq). " << string(buf.begin(), buf.begin()+20) << endl;
        //     exit(1);
        // }
        size_t i = 0, j;
        if (remain_pos != string::npos) {////改动
            while (i < remain_pos) {
                // assert(buf[i] == '@');
                i = buf.find('\n', i+1); // line end of read info // TODO: save read info
                j = buf.find('\n', i+1); // line end of read
                if (buf[j-1] == '\r') j--;
                #ifdef DEBUG
                if (j >= remain_pos) {cerr<<"unexpected end"<<endl; break;}
                checksum += *(buf.begin()+i+1) + *(buf.begin()+j) + (j-i+1);
                #endif
                _thread_reads[tid].push_back(new string(buf.begin()+i+1, buf.begin()+j));
                // _reads_deleted[tid].push_back(false);
                if (last_batch) break;  // 防止find==-1又开始一轮新的查找
                if (buf[j] == '\r') j++;
                i = buf.find('\n', j+1); // line end of '+'
                i = buf.find('\n', i+1)+1; // line end of quality
            }
        }
        _thread_bat_split_pos[tid].push_back(_thread_reads[tid].size()); // record split position
        // ...
        return *(_thread_bat_split_pos[tid].rbegin()) - *(_thread_bat_split_pos[tid].rbegin()+1);
    }

    T_read_cnt _load_fasta (int tid, bool last_batch = false) {  // 1. process prev, 2. set next prev, 3. process cur (consider 1 process only)
        const size_t MIN_READ_LEN = 10;
        string buf;
        if (!last_batch) buf.resize(CUR_BUF_SIZE + LINE_BUF_SIZE);
        buf = "";
        
        // ---- Process _buf_prev_remain ----
        while(!_pbuf_set[tid]) {
            _pbuf_cvs[tid].wait_for(_pbuf_locks[tid], 100ms);
        }
        buf += _buf_prev_remain[tid];   // handle pbuf
        _pbuf_set[tid] = false;
        _pbuf_cvs[tid].notify_all();
        
        // ---- Set the next _buf_prev_remain ----
        size_t remain_pos;
        if (!last_batch) {
            int tid_next = (tid+1)%_n_threads;
            // set next pbuf - find remain_pos
            buf += _buf_cur[tid];
            remain_pos = buf.rfind('\n');
            while (remain_pos < buf.length() && buf[remain_pos+1] != '>') {
                remain_pos = buf.rfind('\n', remain_pos-1);
            }
            // set next pbuf - set _buf_prev_remain
            while(_pbuf_set[tid_next])
                _pbuf_cvs[tid_next].wait_for(_pbuf_locks[tid_next], 100ms);
            _buf_prev_remain[tid_next] = string(buf.begin()+remain_pos+1, buf.end());
            _pbuf_set[tid_next] = true;
            _pbuf_cvs[tid_next].notify_all();
        } else {
            remain_pos = buf.length();
        }

        // ---- Process _buf_cur ----
        lock_guard<mutex> lg(_thread_reads_mtx[tid]);
        assert(buf[0] == '>');
        // if (buf[0] != '>') { // TODO: 不用判断.
        //     cerr << tid << " Error: wrong file format (not fasta). " << buf[0] << endl;
        //     exit(1);
        // }
        size_t i = 0, j;
        if (remain_pos != string::npos) {////改动
            while (i < remain_pos - MIN_READ_LEN) { // - MIN_READ_LEN to avoid unnecessary newline at the end of the file
                i = buf.find('\n', i+1); // line end of read info // TODO: save read info
                j = buf.find('\n', i+1); // line end of read
                if (j == string::npos && last_batch) j = buf.length(); // avoid no newline at last
                else if (buf[j-1] == '\r') j--;
                #ifdef DEBUG
                // if (j > remain_pos) {cerr<<"unexpected end"<<last_batch<<endl; break;}
                // if (last_batch) {
                //     cerr<<string(buf.begin(), buf.begin()+5)<<" "<<j<<" "<<string(buf.begin()+i+1, buf.begin()+i+10)<<endl;
                //     cerr<<j<<" "<<remain_pos<<endl;
                // }
                checksum += *(buf.begin()+i+1) + *(buf.begin()+j) + (j-i+1);
                #endif
                _thread_reads[tid].push_back(new string(buf.begin()+i+1, buf.begin()+j));
                // _reads_deleted[tid].push_back(false);
                if (buf[j] == '\r') j++;
                i = j;
            }
        }
        _thread_bat_split_pos[tid].push_back(_thread_reads[tid].size()); // record split position
        // ...
        return *(_thread_bat_split_pos[tid].rbegin()) - *(_thread_bat_split_pos[tid].rbegin()+1);
    }
    void _lock_thread_reads() {
        for (int i=0; i<_n_threads; i++)
            _thread_reads_mtx[i].lock();
    }
    void _unlock_thread_reads() {
        for (int i=0; i<_n_threads; i++)
            _thread_reads_mtx[i].unlock();
    }

public:
    static const size_t KB = 1024;
    static const size_t MB = 1048576;
    
    T_read_cnt read_cnt = 0;
    T_read_cnt reads_consumed = 0;
    T_read_cnt batch_size = 2000;

    ReadLoader (int n_threads, string filename, T_read_cnt batch_size = 8000, size_t buffer_size = 20*MB) {
        size_t file_size = get_file_size(filename.c_str());
        _filename = filename;
        int i;
        this->_n_threads = n_threads;
        this->batch_size = batch_size;
        _proc_res = new future<T_read_cnt> [n_threads];//
        _buf_cur = new char* [n_threads];//
        _buf_prev_remain = new string [n_threads];//
        _pbuf_mtxs = new mutex [n_threads] ();//
        _pbuf_locks = new unique_lock<mutex> [n_threads]();
        _pbuf_cvs = new condition_variable [n_threads]();//
        _pbuf_set = new bool [n_threads];//
        _thread_reads = new vector<string*> [n_threads];//
        // _reads_deleted = new vector<bool> [n_threads];//
        _thread_reads_mtx = new mutex [n_threads];//
        _thread_bat_split_pos = new vector<T_read_cnt> [n_threads];//

        // set buf size:
        if (file_size / n_threads < buffer_size)
            CUR_BUF_SIZE = max (LINE_BUF_SIZE, (file_size + 1*KB) / n_threads);
        else
            CUR_BUF_SIZE = buffer_size; // buf_size 过小4MB(bug) 16MB(no bug) on 8858432也会出bug。。。
        cerr << "CUR_BUF_SIZE = " << CUR_BUF_SIZE / MB << "MB \tloading threads = " << n_threads << endl;

        _pbuf_mtxs[0].unlock(); // unlock only when _buf_prev_remain is prepared
        for (i=0; i<n_threads; i++) {
            _pbuf_set[i] = false;
            _pbuf_locks[i] = unique_lock<mutex>(_pbuf_mtxs[i]);
            _buf_cur[i] = new char [CUR_BUF_SIZE+4];
            _buf_prev_remain[i].resize(LINE_BUF_SIZE);
            _buf_prev_remain[i] = "";
            _thread_bat_split_pos[i].push_back(0);
        }
        _pbuf_set[0] = true;
    }
    ~ReadLoader () {
        for (int i=0; i<_n_threads; i++) {
            delete _buf_cur[i];
        }
        delete _buf_cur;//
        delete [] _proc_res;//
        delete [] _buf_prev_remain;//
        delete [] _pbuf_locks;//
        delete [] _pbuf_cvs;//
        delete [] _pbuf_mtxs;// //!!! delete after related locks and condition_variables were deleted
        delete [] _pbuf_set;//
        for (int i=0; i<_n_threads; i++) {
            for (int j=0; j<_thread_reads[i].size(); j++) {
                // if (!_reads_deleted[i][j]) {
                if (_thread_reads[i][j] != nullptr) {
                    delete _thread_reads[i][j];
                    // _thread_reads[i][j] = nullptr;
                    // _reads_deleted[i][j] = true;
                }
            }
        }
        delete [] _thread_reads;//
        // delete [] _reads_deleted;//
        delete [] _thread_reads_mtx;//
        delete [] _thread_bat_split_pos;//
    }
    void load_file () {
        // Open file:
        FILE *fqfile = fopen(_filename.c_str(), "rb");
        if (fqfile == NULL) {
            cerr << "Unable to open: " << _filename << endl;
            exit(1);
        }

        // Determine the file type:
        bool fastq = *(_filename.rbegin()) == 'q';
        std::function<T_read_cnt(int,bool)> proc_func;
        if (fastq) {
            cerr << "fastq" << endl;
            proc_func = [this](int tid, bool last_bat) -> T_read_cnt {return this->_load_fastq(tid, last_bat);}; // [this] pass by value
            // proc_func = [this](int tid, bool last_bat) -> T_read_cnt {return this->_load_fastq_bw(tid, last_bat);}; // [this] pass by value
            // proc_func = std::bind(&ReadLoader::_load_fastq, this, placeholders::_1);
        } else {
            cerr << "fasta" << endl;
            proc_func = [this](int tid, bool last_bat) -> T_read_cnt {return this->_load_fasta(tid, last_bat);}; // [&this] pass by ref
        }

        // Load and process the file:
        int i = 0, i_break;
        size_t tmp_size; // i_thread
        bool not_1st_loop = false;
        while ((tmp_size = fread(_buf_cur[i], sizeof(char), CUR_BUF_SIZE, fqfile)) > 0) {
            _buf_cur[i][tmp_size] = 0;
            _proc_res[i] = async(std::launch::async, proc_func, i, false);
            i = (i+1) % _n_threads;
            not_1st_loop |= i==0;
            if (not_1st_loop) {
                read_cnt += _proc_res[i].get(); // wait for the previous round
                // cerr<<"LOADED "<<read_cnt<<" CONSUMED "<<reads_consumed<<" BATCH "<<batch_size<<endl;
                while (read_cnt - reads_consumed > 2 * batch_size) this_thread::sleep_for(1ms);
            }
        }
        i_break = i;
        if (not_1st_loop) {
            for (i = (i+1)%_n_threads; i != i_break; i = (i+1)%_n_threads)
                read_cnt += _proc_res[i].get(); // wait for the previous round
        } else {
            for (i = 0; i != i_break; i++)
                read_cnt += _proc_res[i].get(); // wait for the previous round
        }
        read_cnt += proc_func(i_break, true); // process the data in the last pbuf
    }

    size_t get_file_size(const char *filename) {
        struct stat statbuf;
        if (stat(filename, &statbuf) != 0) {
            cerr<<"ERROR "<<errno<<": can't get file size of "<<filename;
            perror("");
            exit(errno);
        }
        return statbuf.st_size;
    }

    T_read_cnt get_read_cnt() {return read_cnt;}
    vector<string*>* get_thread_reads() {return _thread_reads;}
    vector<T_read_len>* get_thread_bat_split_pos() {return _thread_bat_split_pos;}
    
    /// @brief Load reads from thread vector to an external vector.
    /// @param reads [out] The external vector storing the reads in ReadPtr{const char*, T_read_len}.
    /// @param beg The begin position of reads to load.
    /// @param max_n Max number of reads to load by this calling.
    /// @return The number of reads loaded out
    T_read_cnt get_reads(vector<ReadPtr> &reads, T_read_cnt beg=0, T_read_cnt max_n=-1) {
        T_read_cnt n = 0;
        int bat, i, j;
        bool no_read_left = false;
        #ifdef DEBUG
        bool reads_loaded_flag = false;
        #endif
        _lock_thread_reads();
        if (max_n == -1) max_n = read_cnt;
        for (bat=1; n-beg<max_n && (!no_read_left); bat++) {
            for (i=0; i<_n_threads; i++) {
                if (bat >= _thread_bat_split_pos[i].size()) {no_read_left = true; break;}
                if (n >= beg) {
                    #ifdef DEBUG
                    reads_loaded_flag = true;
                    #endif
                    for (j=_thread_bat_split_pos[i][bat-1]; j<_thread_bat_split_pos[i][bat]; j++) {
                        assert(_thread_reads[i][j] != nullptr);
                        reads.push_back({_thread_reads[i][j]->c_str(), T_read_len(_thread_reads[i][j]->length())});
                    }
                }
                n += _thread_bat_split_pos[i][bat] - _thread_bat_split_pos[i][bat-1];
                // cerr<<"load tid="<<i<<" pos="<<_thread_bat_split_pos[i][bat-1]<<"~"<<_thread_bat_split_pos[i][bat]<<endl;
            }
        }
        reads_consumed += n-beg;
        _unlock_thread_reads();
        #ifdef DEBUG
        assert(reads_loaded_flag);
        #endif
        return n-beg; // returns the number of loaded reads
    }
    /// @brief Delete reads in the thread buffers.
    /// @param beg The beginning index of read to be deleted.
    /// @param n The number of reads to be deleted (from beg).
    /// @return Return false if number of deleted reads equals to the given value, otherwise true.
    bool delete_read_buffers (T_read_cnt beg=0, T_read_cnt n_reads=-1) {
        T_read_cnt n = 0;
        int bat, i, j;
        bool no_read_left = false;
        _lock_thread_reads();
        if (n_reads == -1) n_reads = read_cnt;
        for (bat=1; n-beg<n_reads && (!no_read_left); bat++) {
            for (i=0; i<_n_threads && n-beg<n_reads; i++) {
                if (bat >= _thread_bat_split_pos[i].size()) {no_read_left = true; break;}
                if (n >= beg) {
                    for (j=_thread_bat_split_pos[i][bat-1]; j<_thread_bat_split_pos[i][bat]; j++)
                        // if (!_reads_deleted[i][j]) {
                        if (_thread_reads[i][j] != nullptr) {
                            delete _thread_reads[i][j];
                            _thread_reads[i][j] = nullptr;
                            // _reads_deleted[i][j] = true;
                        }
                }
                n += _thread_bat_split_pos[i][bat] - _thread_bat_split_pos[i][bat-1];
                // cerr<<n_reads<<" delete tid="<<i<<" pos="<<_thread_bat_split_pos[i][bat-1]<<"~"<<_thread_bat_split_pos[i][bat]<<endl;
            }
        }
        _unlock_thread_reads();
        return !(n-beg==n_reads);
    }
    

    static void work_while_loading (std::function<void(vector<ReadPtr>&)> work_func, int loader_threads, string filename, 
        T_read_cnt batch_size=5000, bool delete_after_proc=false, size_t buffer_size = 20 * ReadLoader::MB)
    {
        vector<ReadPtr> reads;
        ReadLoader rl(loader_threads, filename, batch_size, buffer_size);
        future<void> file_loading_res = async(std::launch::async, [&rl](){return rl.load_file();});
        
        T_read_cnt n_read_loaded = 0, reads_loaded;
        bool loading_not_finished = true;
        future_status status;
        while (loading_not_finished) {
            status = file_loading_res.wait_for(1ms); // TODO: set smaller when storage is fast
            switch (status) {
                case future_status::deferred:
                case future_status::timeout:
                    if (rl.get_read_cnt() - n_read_loaded >= batch_size) {
                        reads.clear();
                        reads_loaded = rl.get_reads(reads, n_read_loaded, batch_size);
                        // ... process reads
                        work_func(reads);
                        if (delete_after_proc) assert(!rl.delete_read_buffers(n_read_loaded, reads_loaded));
                        n_read_loaded += reads_loaded;
                    }
                    break;
                case future_status::ready:  // all reads are loaded
                    reads.clear();
                    reads_loaded = rl.get_reads(reads, n_read_loaded, -1);
                    // ... process reads
                    work_func(reads);
                    if (delete_after_proc) rl.delete_read_buffers(n_read_loaded, reads_loaded);
                    n_read_loaded += reads_loaded;
                    loading_not_finished = false;
                    break;
            }
        }
        cerr<<"Total reads loaded: "<<n_read_loaded<<endl;
        return;
    }

    static void work_while_loading_V2 (std::function<void(vector<ReadPtr>&)> work_func, int loader_threads, string filename, 
        T_read_cnt batch_size=5000, bool delete_after_proc=false, size_t buffer_size = 20 * ReadLoader::MB)
    {
        ThreadPool<void> tp(PAR.N_threads);

        vector<ReadPtr> *reads;
        ReadLoader rl(loader_threads, filename, batch_size, buffer_size);
        // promise<void> file_loading_prom;
        // thread file_loading_t([&rl, &file_loading_prom](){return rl.load_file_V2(file_loading_prom);});
        future<void> file_loading_res = async(std::launch::async, [&rl](){return rl.load_file();});
        // future<void> file_loading_res = file_loading_prom.get_future();

        T_read_cnt n_read_loaded = 0, reads_loaded;
        bool loading_not_finished = true;
        future_status status;
        while (loading_not_finished) {
            status = file_loading_res.wait_for(1ms); // TODO: set smaller when storage is fast
            switch (status) {
                case future_status::deferred:
                case future_status::timeout:
                    if (rl.get_read_cnt() - n_read_loaded >= batch_size) {
                        reads = new vector<ReadPtr>();//
                        reads_loaded = rl.get_reads(*reads, n_read_loaded, batch_size);
                        // ... process reads
                        tp.commit_task([reads, &work_func, delete_after_proc, &rl, n_read_loaded, reads_loaded](){
                            work_func(*reads); delete reads;//
                            if (delete_after_proc) assert(!rl.delete_read_buffers(n_read_loaded, reads_loaded));
                        });
                        // if (delete_after_proc) assert(!rl.delete_read_buffers(n_read_loaded, reads_loaded));
                        n_read_loaded += reads_loaded;
                    }
                    break;
                case future_status::ready:  // all reads are loaded
                    reads = new vector<ReadPtr>();//
                    reads_loaded = rl.get_reads(*reads, n_read_loaded, -1);
                    // ... process reads
                    tp.commit_task([reads, &work_func, delete_after_proc, &rl, n_read_loaded, reads_loaded](){
                        work_func(*reads); delete reads;
                        if (delete_after_proc) assert(!rl.delete_read_buffers(n_read_loaded, reads_loaded));
                    });
                    // if (delete_after_proc) rl.delete_read_buffers(n_read_loaded, reads_loaded);
                    n_read_loaded += reads_loaded;
                    loading_not_finished = false;
                    break;
            }
        }
        // if (file_loading_t.joinable()) file_loading_t.join();
        tp.finish();
        cerr<<"Total reads loaded: "<<n_read_loaded<<endl;
        return;
    }
};

#endif
