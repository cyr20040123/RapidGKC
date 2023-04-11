#ifndef _THREAD_POOL_HPP
#define _THREAD_POOL_HPP

#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <vector>
#include <future>
#include <cerrno>
#include <iostream>
#include "concqueue.hpp"
// using namespace std;

// for thread affinity
#include <pthread.h>
#include <sched.h>
#include <cassert>

struct ThreadAffinity {
    int avail_logical_cores_beg = 0;
    int avail_logical_cores_end = 1;
    int vip_tid_beg = -1; // -1 means TA is not set
    int vip_tid_end = -1; // -1 means TA is not set
};

template</*typename T_IN=void, */typename T_OUT=void>
class ThreadPool {
private:
    int _n_threads;
    
    struct Task {
        std::function<T_OUT(int/*T_IN*/)> func;
        std::promise<T_OUT>* prom;
    };
    ConcQueue<Task> _CQ;
    
    std::condition_variable _holder_cv;
    std::mutex _holder_mtx;
    
    std::vector<std::thread> _threads;
    std::atomic<bool> _exit_flag{false};          // must be atomic (SWMR)

    int _busy_thr = 2;
    inline bool _not_busy () {
        return _CQ.size_est() <= _busy_thr;
    }
    
    template<typename NONVOID>
    inline void _set_promise(std::promise<NONVOID> & prom, std::function<T_OUT(int/*T_IN*/)> & func, int tid=-1) {
        prom.set_value(func(tid)); // non-void promise
    }
    inline void _set_promise(std::promise<void> & prom, std::function<T_OUT(int/*T_IN*/)> & func, int tid=-1) {
        func(tid);
        prom.set_value(); // void promise
    }
    void _worker (int tid = -1) {
        Task t;
        while (_CQ.pop(t)) {
            if (_not_busy()) _holder_cv.notify_all();
            if (t.prom == nullptr) t.func(tid); // TODO remove?
            else _set_promise(*t.prom, t.func, tid);
        }
    }
    inline void _check_finished() {
        if (_exit_flag) {
            errno = ECANCELED;
            perror("The thread pool is already finished and no operation is allowed");
            exit(errno);
        }
    }
    /* Example quad-core: ta={0, 4, 1, 3}
    T0 = Core2, TVIP1 = C0, TVIP2 = C1, T3=C3, T4=C2C3, T5=C2C3
    */
    void _set_thread_affinity(ThreadAffinity ta) {
        assert(ta.avail_logical_cores_beg < ta.avail_logical_cores_end);
        assert(ta.vip_tid_end - ta.vip_tid_beg <= ta.avail_logical_cores_end - ta.avail_logical_cores_beg);
        if (ta.avail_logical_cores_end > std::thread::hardware_concurrency()) {
            std::cerr<<"Warning! Wrong argument setting: ta.avail_logical_cores_end = "<<ta.avail_logical_cores_end<<std::endl;
            ta.avail_logical_cores_end = std::thread::hardware_concurrency();
        }
        int vip_core_i = ta.avail_logical_cores_beg;
        int normal_core_i = ta.avail_logical_cores_beg + ta.vip_tid_end - ta.vip_tid_beg;
        
        cpu_set_t cpuset_normal;
        CPU_ZERO(&cpuset_normal);
        for (int i=normal_core_i; i<ta.avail_logical_cores_end; i++) CPU_SET(i, &cpuset_normal); // cores for normal threads
        CPU_SET(ta.avail_logical_cores_end-1, &cpuset_normal); // at least one thread
        
        for (int tid=0; tid<_n_threads; tid++) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            if (tid >= ta.vip_tid_beg && tid < ta.vip_tid_end) { // vip thread
                CPU_SET(vip_core_i++, &cpuset);
                assert(!pthread_setaffinity_np(_threads[tid].native_handle(), sizeof(cpu_set_t), &cpuset));
            } else { // normal thread
                if (normal_core_i < ta.avail_logical_cores_end) {
                    CPU_SET(normal_core_i++, &cpuset);
                    assert(!pthread_setaffinity_np(_threads[tid].native_handle(), sizeof(cpu_set_t), &cpuset));
                } else { // no cores available
                    assert(!pthread_setaffinity_np(_threads[tid].native_handle(), sizeof(cpu_set_t), &cpuset_normal));
                }
            }
        }
    }
public:
    ThreadPool(int n_threads, int busy_thr = -1, ThreadAffinity ta = {0,1,-1,-1}) {
        if (busy_thr < 0) _busy_thr = 2*n_threads;
        else _busy_thr = busy_thr;
        _n_threads = n_threads;
        int tid;
        for (tid=0; tid<_n_threads; tid++) {
            _threads.emplace_back(&ThreadPool::_worker, this, tid);
        }
        if (ta.vip_tid_end>=0) this->_set_thread_affinity(ta);
    }
    ~ThreadPool() {
        for (auto &t: _threads) {
            if (t.joinable()) t.join();
        }
    }
    std::future<T_OUT> commit_task(std::function<T_OUT(int/*T_IN*/)> task_func) {
        _check_finished();
        Task t;
        t.func = std::move(task_func);
        std::promise<T_OUT> *prom = t.prom = new std::promise<T_OUT>();
        _CQ.push(t);
        _holder_cv.notify_all();
        return prom->get_future();
    }
    void commit_task_no_return(std::function<T_OUT(int/*T_IN*/)> task_func) { // not used
        _check_finished();
        Task t;
        t.func = std::move(task_func);
        t.prom = nullptr;
        _CQ.push(t);
        _holder_cv.notify_all();
        return;
    }
    void finish() {
        _check_finished();
        _exit_flag = true;
        hold_when_busy();
        _CQ.finish();
        _holder_cv.notify_all();
        for (auto &t: _threads) if (t.joinable()) t.join();
        #ifdef WAITMEASURE
        std::cerr<<"  Thread Pool wait: "<<_CQ.debug_pop_wait<<std::endl;
        #endif
    }
    void hold_when_busy () {
        std::unique_lock<std::mutex> tmp_lck(_holder_mtx);
        for (int i=0; i<8 && !_holder_cv.wait_for(tmp_lck, 500ms, [=](){return this->_not_busy();}); i++);
    }
    // static void set_thread_affinity(int core = 0) {
    //     assert(core < std::thread::hardware_concurrency());
    //     cpu_set_t cpuset;
    //     CPU_ZERO(&cpuset);
    //     CPU_SET(core, &cpuset);
    //     assert(!pthread_setaffinity_np(std::this_thread::get_id()._M_thread, sizeof(cpu_set_t), &cpuset));
    //     std::cerr<<"bind thread to core "<<core<<std::endl;
    // }
    static void set_thread_affinity(std::thread &t, int core_beg = 0, int core_end = -1) {
        assert(core_beg < std::thread::hardware_concurrency());
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (core_end == -1) CPU_SET(core_beg, &cpuset);
        else for (int i=core_beg; i<core_end; i++) CPU_SET(i, &cpuset);
        assert(!pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset));
        std::cerr<<"bind thread to core "<<core_beg<<"-"<<core_end<<std::endl;
    }
};

#endif