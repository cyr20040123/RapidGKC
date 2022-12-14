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
using namespace std;

template</*typename T_IN=void, */typename T_OUT=void>
class ThreadPool {
private:
    int _n_threads;
    
    mutex _wake_mtx;
    condition_variable _wake_cv;

    // condition_variable _waiter_cv;
    // mutex _waiter_mtx;
    // atomic_flag _in_waiting;
    
    mutex _queue_mtx; // for locking two queues below
    
    queue<function<T_OUT(/*T_IN*/)>> _tasks;
    queue<promise<T_OUT>*> _task_promises;   // OUTPUTS
    atomic<bool> _task_queue_empty{true};    // must be atomic (SWMR)
    
    vector<thread> _threads;
    atomic<bool> _exit_flag{false};          // must be atomic (SWMR)
    
    template<typename NONVOID>
    inline void _set_promise(std::promise<NONVOID> & prom, function<T_OUT()> & func) {
        prom.set_value(func()); // non-void promise
    }
    inline void _set_promise(std::promise<void> & prom, function<T_OUT()> & func) {
        func();
        prom.set_value(); // void promise
    }
    void _worker (int tid = -1) {
        while ((!_exit_flag) || (!_task_queue_empty)) {
            {
                unique_lock<mutex> tmp_lck(_wake_mtx);
                _wake_cv.wait(tmp_lck, [&](){return _task_queue_empty==false || _exit_flag==true;});
            }
            
            // -- fetch task (mutex zone) --
            // unique_lock<mutex> ul(_queue_mtx, defer_lock);
            _queue_mtx.lock();//
            if (_task_queue_empty) {
                _queue_mtx.unlock();//
                continue;
            }
            function<T_OUT(/*T_IN*/)> func = _tasks.front();
            _tasks.pop();
            promise<T_OUT> *prom = _task_promises.front();
            _task_promises.pop();
            _task_queue_empty = _tasks.empty();
            _queue_mtx.unlock();//
            // -- (mutex zone ends) --

            if (prom == nullptr) func(); // TODO remove
            // else prom->set_value(func());
            else _set_promise(*prom, func);
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
    ThreadPool(int n_threads) {
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
    future<T_OUT> commit_task(function<T_OUT(/*T_IN*/)> task_func) {
        _check_finished();
        unique_lock<mutex> tmp_lck(_queue_mtx);
        _tasks.push(task_func);
        _task_promises.push(new promise<T_OUT>());
        if (_task_queue_empty) _task_queue_empty = false;
        _wake_cv.notify_one();
        return _task_promises.back()->get_future();
    }
    void commit_task_no_return(function<T_OUT(/*T_IN*/)> task_func) {
        _check_finished();
        unique_lock<mutex> tmp_lck(_queue_mtx);
        _tasks.push(task_func);
        _task_promises.push(nullptr);
        if (_task_queue_empty) _task_queue_empty = false;
        _wake_cv.notify_one();
        return;
    }
    void finish() {
        _check_finished();
        _exit_flag = true;
        _wake_cv.notify_all();
        for (auto &t: _threads) t.join();
    }
    // void wait () {
    //     _in_waiting.test_and_set();
    //     unique_lock<mutex> tmp_lck(_waiter_mtx);
    //     _waiter_cv.wait(tmp_lck, [&](){return _task_queue_empty==true});
    // }
};

/* an example: g++ -std=c++11 thread_pool.cpp -o test -lpthread -g
int main() {
    int n_threads = 4;
    size_t total = 0;
    ThreadPool<size_t> tp(n_threads);
    vector<future<size_t>> fu;
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