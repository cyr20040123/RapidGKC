#ifndef _THREAD_POOL_HPP
#define _THREAD_POOL_HPP

#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <queue>
#include <vector>
#include <future>
#include <cerrno>
#include <iostream>
// using namespace std;

template</*typename T_IN=void, */typename T_OUT=void>
class ThreadPool {
private:
    int _n_threads;
    
    std::mutex _wake_mtx;
    std::condition_variable _wake_cv;

    std::condition_variable _holder_cv;
    std::mutex _holder_mtx;
    
    std::mutex _queue_mtx; // for locking two queues below
    
    std::queue<std::function<T_OUT(int/*T_IN*/)>> _tasks;
    std::queue<std::promise<T_OUT>*> _task_promises;   // OUTPUTS
    std::atomic<bool> _task_queue_empty{true};    // must be atomic (SWMR)
    
    std::vector<std::thread> _threads;
    std::atomic<bool> _exit_flag{false};          // must be atomic (SWMR)

    int _busy_thr = 2;
    
    inline bool _not_busy () {
        return _tasks.size() <= _busy_thr;
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
        while ((!_exit_flag) || (!_task_queue_empty)) {
            {
                std::unique_lock<std::mutex> tmp_lck(_wake_mtx);
                _wake_cv.wait(tmp_lck, [&](){return _task_queue_empty==false || _exit_flag==true;});
            }
            
            // -- fetch task (mutex zone) --
            // std::unique_lock<std::mutex> ul(_queue_mtx, defer_lock);
            _queue_mtx.lock();//
            if (_task_queue_empty) {
                _queue_mtx.unlock();//
                continue;
            }
            std::function<T_OUT(int/*T_IN*/)> func = _tasks.front();
            _tasks.pop();
            std::promise<T_OUT> *prom = _task_promises.front();
            _task_promises.pop();
            _task_queue_empty = _tasks.empty();
            if (_not_busy()) _holder_cv.notify_all();
            _queue_mtx.unlock();//
            // -- (mutex zone ends) --

            if (prom == nullptr) func(tid); // TODO remove
            // else prom->set_value(func());
            else _set_promise(*prom, func, tid);
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
        std::unique_lock<std::mutex> tmp_lck(_queue_mtx);
        _tasks.push(task_func);
        _task_promises.push(new std::promise<T_OUT>());
        if (_task_queue_empty) _task_queue_empty = false;
        _wake_cv.notify_one();
        _holder_cv.notify_all();
        return _task_promises.back()->get_future();
    }
    void commit_task_no_return(std::function<T_OUT(int/*T_IN*/)> task_func) { // not used
        _check_finished();
        std::unique_lock<std::mutex> tmp_lck(_queue_mtx);
        _tasks.push(task_func);
        _task_promises.push(nullptr);
        if (_task_queue_empty) _task_queue_empty = false;
        _wake_cv.notify_one();
        _holder_cv.notify_all();
        return;
    }
    void finish() {
        _check_finished();
        hold_when_busy();
        _exit_flag = true;
        _wake_cv.notify_all();
        _holder_cv.notify_all();
        for (auto &t: _threads) t.join();
    }
    void hold_when_busy () {
        std::unique_lock<std::mutex> tmp_lck(_holder_mtx);
        _holder_cv.wait(tmp_lck, [=](){return this->_not_busy();});
    }
};

/* an example: g++ -std=c++11 thread_pool.cpp -o test -lpthread -g
int main() {
    int n_threads = 4;
    size_t total = 0;
    ThreadPool<size_t> tp(n_threads);
    std::vector<future<size_t>> fu;
    for (int i=1; i<=100; i++) {
        fu.push_back(tp.commit_task([i](){return size_t(i*2);}));
    }
    tp.finish();
    for (int i=0; i<100; i++) {
        total += fu[i].get();
    }
    cout<<total<<endl;
    return 0;
}
*/
#endif