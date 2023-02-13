#pragma once

#ifndef _UTILITIES_HPP
#define _UTILITIES_HPP

#include "types.h"
#include <atomic>
#include <chrono>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
// #include <sstream>
#include <iostream>
#include <mutex>
// using namespace std;

// ======================================================
// ================ CLASS WallClockTimer ================
// ======================================================
class WallClockTimer {  // evaluating wall clock time
private:
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
public:
    // Initialize a wall-clock timer and start.
    WallClockTimer() {
        start = std::chrono::high_resolution_clock::now();
    }
    // Return end-start duration in sec from init or restart till now.
    double stop(bool millisecond = false) {
        end = std::chrono::high_resolution_clock::now();
        if (millisecond) return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        else return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    }
    // Restart the clock start time.
    void restart() {
        start = std::chrono::high_resolution_clock::now();
    }
};


// ==============================================
// ================ CLASS Logger ================
// ==============================================
class Logger { // for CLI logging and file logging
public:
    enum LogLevel {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG};
    std::string loglabel[6] = {"[FATAL]", "[ERROR]", "[NOTICE]", "[WARNING]", "[INFO]", "[DEBUG]"};
private:
    LogLevel log_lv = LV_WARNING; // output level only equal or smaller than this value
    bool to_file = false;
    int my_rank = 0;
    std::ofstream flog;
    std::string path, logfilename;
    std::string timestring(bool short_format = false) {
        time_t rawtime;    struct tm *info;    char buffer[80];
        time(&rawtime);    info = localtime(&rawtime);
        if (short_format) strftime(buffer, 80, "%m%d-%H%M%S", info);
        else strftime(buffer, 80, "(%m-%d %H:%M:%S) ", info);
        return std::string(buffer);
    }
public:
    /**
     * @brief Logger initialization.
     * @param  process_id       : (optional) For multithreading process identification.
     * @param  log_level        : (optional) Only record with no greater level logs. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  to_log_file      : (optional) Whether to write logs into file.
     * @param  logfile_folder   : (optional) Save log file to which folder.
     */
    Logger (int process_id = 0, LogLevel log_level = LV_WARNING, bool to_log_file = false, std::string logfile_folder="./")
    : log_lv(log_level), to_file(to_log_file), my_rank(process_id), path(logfile_folder) {
        if (path.back() != '/') path += '/';
        if (to_file) {
            logfilename = path + "log_P" + std::to_string(my_rank) + ".log";
            // logfilename = path + "log_P" + std::to_string(my_rank) + "_" + timestring(true) + ".log";
            flog = std::ofstream(logfilename, std::ios::app|std::ios::out);
            if (!flog) std::cerr<<"Unable to open and write to log file!"<<std::endl;
            flog<<"================================"<<std::endl;
        }
    }
    
    /**
     * @brief Call this function to log a string.
     * @param  info         : Log text.
     * @param  lv           : (optional) Log level. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  with_time    : (optional) Whether to log with current wall clock time.
     */
    // void log(const char *info, LogLevel lv = LV_DEBUG, bool with_time = false);
    void log(std::string info, LogLevel lv = LV_DEBUG, bool with_time = false) {
        if (lv <= log_lv) {
            if (with_time) std::cerr<<timestring()<<"P"<<my_rank<<" "<<loglabel[lv]<<"\t"<<info<<std::endl;
            else std::cerr<<"P"<<my_rank<<" "<<loglabel[lv]<<"\t"<<info<<std::endl;
            if (to_file && flog.is_open()) flog<<timestring()<<"P"<<my_rank<<" "<<loglabel[lv]<<"\t"<<info<<std::endl;
        }
        if (lv == LV_FATAL) exit(1);
    }
    
    /**
     * @brief Call this function to log a vector. If len<5 items will be output with TAB between else one item per line.
     * @param  vec_data     : The vector to be logged. Must be printable.
     * @param  info         : The log text.
     * @param  lv           : (optional) Log level. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  with_time    : (optional) Whether to log with current wall clock time.
     */
    template<typename T_item>
    void logvec(std::vector<T_item> &vec_data, std::string info, LogLevel lv = LV_DEBUG, bool with_time = false) {
        if (lv <= log_lv) {
            this->log(info, lv, with_time);
            if (vec_data.size()<5) { // use \t
                std::cerr<<"    ";
                for (T_item &item: vec_data) std::cerr<<std::to_string(item)<<"\t";
                std::cerr<<std::endl;
                if (to_file && flog.is_open()) {
                    flog<<"    ";
                    for (T_item &item: vec_data) flog<<std::to_string(item)<<"\t";
                    flog<<std::endl;
                }
            } else {
                int i=0;
                for (T_item &item: vec_data) {
                    if (i++ % 10 == 0) std::cerr<<std::endl;
                    std::cerr<<"\t"<<std::to_string(item);
                }
                std::cerr<<std::endl;
                i=0;
                if (to_file && flog.is_open()) {
                    for (T_item &item: vec_data) {
                        flog<<"\t"<<std::to_string(item);
                        if (i++ % 10 == 0) flog<<std::endl;
                    }
                    flog<<std::endl;
                }
            }
        }
    }

    // Deconstructor
    ~Logger() {if (to_file && flog.is_open()) flog.close();}
};

    // ** Implementation **
    // std::string Logger::timestring(bool short_format/* = false*/) 
    // Logger::Logger (int process_id/* = 0*/, LogLevel log_level/* = LV_WARNING*/, bool to_log_file/* = false*/, std::string logfile_folder/*="./"*/)
    // void Logger::log(std::string info, LogLevel lv/* = LV_DEBUG*/, bool with_time/* = false*/)
    // Logger::~Logger()
    // template<typename T_item>
    // void Logger::logvec(std::vector<T_item> &vec_data, std::string info, LogLevel lv/* = LV_DEBUG*/, bool with_time/* = false*/)
    

// ====================================================
// ================ CLASS GlobalParams ================
// ====================================================
static class GlobalParams {
public:
    T_kvalue K_kmer = 21;   // length of kmer
    T_kvalue P_minimizer = 7;
    int SKM_partitions = 31; // minimizer length and superkmer partitions
    T_kmer_cnt kmer_min_freq = 1, kmer_max_freq = 1000; // count kmer cnt in [min,max] included
    bool HPC = false;           // homopolymer compression assembly
    bool CPU_only = false;
    int Kmer_filter = 25;       // percentage
    int N_threads = 4;          // threads per process
    int RD_threads_min = 2;
    // string tmp_file_folder = "./tmp/";
    std::string tmp_file_folder = "/home/cyr/tmp/";
    bool to_file = true;
    std::string log_file_folder = "./log/";
    int log_lv = 5;
    std::vector<std::string> read_files;
    
    T_read_cnt Batch_read_loading = 2000;
    size_t Buffer_fread_size_MB = 12;

    int grid_size = 8;
    int block_size = 256;
    int grid_size2 = 16;
    int block_size2 = 512;
    int n_devices = 1;
    int n_streams = 6;
    int n_streams_phase2 = 2;
    int reads_per_stream_mul = 1;
    int max_threads_per_gpu = 2;

    bool GPU_compression = false;

    void ArgParser(int argc, char* argvs[]) {
        // exe <k> <p> <tmp_file_folder> <_Kmer_filter>
        for (int i=1; i<argc-1; i++) {
            if (!strcmp(argvs[i], "-t")) N_threads = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-rdt")) RD_threads_min = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-k")) K_kmer = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-p")) P_minimizer = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-skm")) SKM_partitions = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-hpc")) HPC = bool(atoi(argvs[++i]));
            else if (!strcmp(argvs[i], "-cpuonly")) CPU_only = bool(atoi(argvs[++i]));
            else if (!strcmp(argvs[i], "-kf")) Kmer_filter = atof(argvs[++i])*100;
            else if (!strcmp(argvs[i], "-tmp")) tmp_file_folder = std::string(argvs[++i]);
            else if (!strcmp(argvs[i], "-im")) to_file = false;
            else if (!strcmp(argvs[i], "-log")) log_file_folder = std::string(argvs[++i]);
            else if (!strcmp(argvs[i], "-lv")) log_lv = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-fb")) Buffer_fread_size_MB = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-rb")) Batch_read_loading = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-grid")) grid_size = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-block")) block_size = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-grid2")) grid_size2 = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-block2")) block_size2 = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-ngpu")) n_devices = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-ns1")) n_streams = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-ns2")) n_streams_phase2 = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-rps")) reads_per_stream_mul = atoi(argvs[++i]); // if short reads, set it to 2 or 4 to fully utilize gpu
            else if (!strcmp(argvs[i], "-tgpu")) max_threads_per_gpu = atoi(argvs[++i]);
            else if (!strcmp(argvs[i], "-read")) {
                int j;
                for (j=i+1; j<argc; j++, i++) {
                    if (argvs[j][0] == '-') {i=j-1; break;}
                    read_files.push_back(std::string(argvs[j]));
                }
            }
            else {
                std::cerr<<"Wrong argument: "<<argvs[i]<<std::endl;
                std::cerr<<"Usage: "<< argvs[0] <<" -t <threads_per_process> -k <K_kmer> -p <P_minimizer> -skm <SKM_partitions> -hpc <HPC> -tmp <tmp_file_folder> -read <read_files...>"<<std::endl;
                exit(1);
            }
        }
        if (read_files.size() == 0) {
            std::cerr<<"Usage: "<< argvs[0] <<" -t <threads_per_process> -k <K_kmer> -p <P_minimizer> -skm <SKM_partitions> -hpc <HPC> -tmp <tmp_file_folder> -read <read_files...>"<<std::endl;
            exit(1);
        }
        if (*tmp_file_folder.rbegin() != '/')
            tmp_file_folder.push_back('/');
        if (*log_file_folder.rbegin() != '/')
            log_file_folder.push_back('/');
    }
} PAR;
// ** Implementation **
// void GlobalParams::ArgParser(int argc, char* argvs[])


// ===================================================
// ================ struct CUDAParams ================
// ===================================================
struct CUDAParams {
    int NUM_BLOCKS_PER_GRID, NUM_THREADS_PER_BLOCK;
    int BpG, TpB; // for step 2
    int n_streams, items_stream_mul;
    int n_streams_phase2;
    int n_devices;
    std::atomic<int> device_id;
    std::vector<size_t> vram;
    std::vector<size_t> vram_used;
    std::vector<std::mutex*> vram_mtx;
    // std::vector<int> gpuid_thread;
    std::atomic<int> running_threads_for_gpu;
    int max_threads_per_gpu;
};

// ================ Read Sorting ================
static bool sort_comp (const ReadPtr x, const ReadPtr y) {
    return x.len < y.len;
}
#endif