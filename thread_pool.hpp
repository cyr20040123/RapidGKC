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
public:
    ThreadPool(int n_threads, int busy_thr = -1) {
        if (busy_thr < 0) _busy_thr = 2*n_threads;
        else _busy_thr = busy_thr;
        _n_threads = n_threads;
        int tid;
        for (tid=0; tid<_n_threads; tid++) {
            _threads.emplace_back(&ThreadPool::_worker, this, tid);
        }
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
    }
    void hold_when_busy () {
        std::unique_lock<std::mutex> tmp_lck(_holder_mtx);
        _holder_cv.wait(tmp_lck, [=](){return this->_not_busy();});
    }
};

#endif