#ifndef _CONCQUEUE_HPP
#define _CONCQUEUE_HPP

#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <chrono>

using namespace std::chrono_literals;

template <typename T>
class ConcQueue {
private:
    std::deque<T> _Q;
    std::mutex _mtx;
    std::condition_variable _cv;
    bool _finished = false;
    size_t _size = 0;
    bool _external_wait;
public:
    #ifdef WAITMEASURE
    std::atomic<size_t> debug_push_wait{0};
    std::atomic<size_t> debug_pop_wait{0};
    #endif
    ConcQueue& operator=(const ConcQueue&) = delete;
    ConcQueue (bool external_wait_support = false) {
        _external_wait = external_wait_support;
        // _Q.resize(1024);
        // _Q.resize(0);
    }
    void finish () {
        _finished = true;
        _cv.notify_all();
    }
    void push (T &item) {
        std::lock_guard<std::mutex> lg(_mtx);
        _Q.push_back(std::move(item));
        if (_external_wait) _cv.notify_all(); // all for both pop and newlinepos consumer
        else _cv.notify_one();
        _size = _Q.size();
    }
    void wait_push (T &item, size_t size_thr) {
        std::unique_lock<std::mutex> lck(_mtx);
        #ifdef WAITMEASURE
        // if (!_cv.wait_for(lck, 200ms, [this, size_thr](){return this->_size < size_thr;})) // true: finish waiting, false: timeout
        //     debug_push_wait++;
        _cv.wait_for(lck, 500ms, [this, size_thr]() {
            if (this->_size >= size_thr) {this->debug_push_wait++; return false;}
            else return true;
        });
        // for (int i=0; i<8 && !_cv.wait_for(lck, 250ms, [this, size_thr]() {
        //     if (this->_size >= size_thr) {this->debug_push_wait++; return false;}
        //     else return true;
        // }); i++);
        #else
        _cv.wait_for(lck, 500ms, [this, size_thr](){return this->_size < size_thr;});
        // for (int i=0; i<8 && !_cv.wait_for(lck, 250ms, [this, size_thr](){return this->_size < size_thr;}); i++);
        #endif
        _Q.push_back(std::move(item));
        if (_external_wait) _cv.notify_all(); // all for both pop and newlinepos consumer
        else _cv.notify_one();
        _size = _Q.size();
    }
    std::deque<T>& native_handle() { // remember to lock before using
        return _Q;
    }
    std::mutex& get_mtx() {
        return _mtx;
    }
    std::condition_variable& get_cv() {
        return _cv;
    }
    bool is_finished () {
        return _finished;
    }
    bool pop (T &item) {
        std::unique_lock<std::mutex> lck(_mtx);
        #ifdef WAITMEASURE
        _cv.wait(lck, [this](){
            if ((!this->_Q.empty()) || this->_finished) return true;
            else {debug_pop_wait++; return false;}
        });
        #else
        _cv.wait(lck, [this](){return (!this->_Q.empty()) || this->_finished;});
        #endif
        if (_Q.empty()) return false;
        item = _Q.front();
        _Q.pop_front();
        _size = _Q.size();
        return true;
    }
    bool consumer_finish_check () { // return if producer finished and all items are poped
        if (!_finished) return false;
        else {
            std::lock_guard<std::mutex> lg(_mtx);
            _size = _Q.size();
            return _finished && _size == 0;
        }
    }
    size_t size_est () { // return the estimated size of queue without lock
        return _size;
    }
};

#endif