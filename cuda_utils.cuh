#ifndef _CUDA_UTILS_CUH
#define _CUDA_UTILS_CUH

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    cerr << "CUDA error calling \""#call"\", code is " << err << ": " << cudaGetErrorString(err) << endl; \
    size_t avail, total; \
    cudaMemGetInfo(&avail, &total); \
    cerr << "Available memory: " << avail/1048576 << " Total memory: " << total/1048576 << endl; \
    exit(1); \
}
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sstream>
#include <thread>
#include "utilities.hpp"

extern Logger *logger;

/// @brief For thread_local use only.
class MyDevicePtr{
public:
    void *ptr = nullptr;
    size_t size = 0;
    int resize_count = 0;
    MyDevicePtr(){
        this->ptr = nullptr; this->size = 0;
        #ifdef NOREUSE
        logger->log("<MyDevicePtr> NOREUSE mode.", Logger::LV_WARNING);
        #endif
    }
    // MyDevicePtr(void *_ptr, size_t _size){this->ptr = _ptr; this->size = _size;}
    ~MyDevicePtr() {
        if (this->ptr != nullptr) { // get rid of CPU threads
            auto myid = std::this_thread::get_id();
            std::stringstream ss;
            ss << myid;
            logger->log("<MyDevicePtr> @"+ss.str()+" final_size = "+to_string(size/MB1)+"MB\tresize_count = "+to_string(resize_count/MB1), Logger::LV_INFO);
            this->free();
        } else {
            // logger->log("<MyDevicePtr> @CPU pass");
        }
    }
    void* get(size_t _byte_offs=0) {
        return (void *)(static_cast<char*>(this->ptr) + _byte_offs);
    }
    void use(size_t _size, bool more=false) {
        if (_size == 0) {
            logger->log("<MyDevicePtr> use size = 0.", Logger::LV_WARNING);
            return;
        }
        #ifdef NOREUSE
        this->free();
        #endif
        if (this->ptr == nullptr) {
            _size = more ? (_size + _size/8) : _size;
            _size = (_size+7)/8*8; // align to 8 bytes
            CUDA_CHECK(cudaMalloc(&this->ptr, _size));
            this->size = _size;
        }
        else if (this->size < _size) {
            _size = more ? (_size + _size/8) : _size;
            _size = (_size+7)/8*8; // align to 8 bytes
            this->free();
            CUDA_CHECK(cudaMalloc(&this->ptr, _size));
            this->resize_count++;
            this->size = _size;
        }
    }
    void use(size_t _size, cudaStream_t stream, bool more=false) {
        if (_size == 0) {
            logger->log("<MyDevicePtr> use size = 0.", Logger::LV_WARNING);
            return;
        }
        #ifdef NOREUSE
        this->free(stream);
        #endif
        if (this->ptr == nullptr) {
            _size = more ? (_size + _size/8) : _size;
            _size = (_size+7)/8*8; // align to 8 bytes
            CUDA_CHECK(cudaMallocAsync(&this->ptr, _size, stream));
            this->size = _size;
        }
        else if (this->size < _size) {
            _size = more ? (_size + _size/8) : _size;
            _size = (_size+7)/8*8; // align to 8 bytes
            this->free(stream);
            CUDA_CHECK(cudaMallocAsync(&this->ptr, _size, stream));
            this->resize_count++;
            this->size = _size;
        }
    }
    void free(){
        if (this->ptr != nullptr) CUDA_CHECK(cudaFree(this->ptr));
        this->ptr = nullptr;
        this->size = 0;
    }
    void free(cudaStream_t stream){
        if (this->ptr != nullptr) CUDA_CHECK(cudaFreeAsync(this->ptr, stream));
        this->ptr = nullptr;
        this->size = 0;
    }
};

class MyPinnedPtr{
public:
    void *ptr = nullptr;
    size_t size = 0;
    int resize_count = 0;
    MyPinnedPtr() {
        this->ptr = nullptr;
        this->size = 0;
    }
    ~MyPinnedPtr() {
        this->free();
    }
    void* get(size_t _byte_offs=0) {
        return (void *)(static_cast<char*>(this->ptr) + _byte_offs);
    }
    void use(size_t _size, bool more=false) {
        if (_size == 0) {
            logger->log("<MyDevicePtr> use size = 0.", Logger::LV_WARNING);
            return;
        }
        if (this->ptr == nullptr) {
            _size = more ? (_size + _size/4) : _size;
            _size = (_size+7)/8*8; // align to 8 bytes
            CUDA_CHECK(cudaHostAlloc(&this->ptr, _size, cudaHostAllocDefault));
            this->size = _size;
        }
        else if (this->size < _size) {
            _size = more ? (_size + _size/4) : _size;
            _size = (_size+7)/8*8; // align to 8 bytes
            this->free();
            CUDA_CHECK(cudaHostAlloc(&this->ptr, _size, cudaHostAllocDefault));
            this->resize_count++;
            this->size = _size;
        }
    }
    void free(){
        if (this->ptr != nullptr) CUDA_CHECK(cudaFreeHost(this->ptr));
        this->ptr = nullptr;
        this->size = 0;
    }
};

class MyStreams{
public:
    cudaStream_t *streams = nullptr;
    int size = 0;
    MyStreams() {
        streams = new cudaStream_t[GlobalParams::MAX_STREAMS];
        size = 0;
    }
    void use(int _size) {
        if (this->size < _size) {
            for (int i=size; i<_size; i++) {
                CUDA_CHECK(cudaStreamCreate(&this->streams[i]));
            }
            this->size = _size;
        }
    }
    ~MyStreams() {
        for (int i=0; i<size; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        delete [] streams;
    }
};

#endif