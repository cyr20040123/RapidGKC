#ifndef _UTILITIES_HPP
#define _UTILITIES_HPP

#include "types.h"
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

// ================ CLASS WallClockTimer ================
class WallClockTimer {  // evaluating wall clock time
private:
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
public:
    WallClockTimer();                           // Initialize a wall-clock timer and start.
    double stop(bool millisecond = false);      // Return end-start duration in sec from init or restart till now.
    void restart();                             // Restart the clock start time.
};


// ================ CLASS Logger ================
class Logger { // for CLI logging and file logging
public:
    enum LogLevel {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG};
    string loglabel[6] = {"[FATAL]", "[ERROR]", "[NOTICE]", "[WARNING]", "[INFO]", "[DEBUG]"};
private:
    LogLevel log_lv = LV_WARNING; // output level only equal or smaller than this value
    bool to_file = false;
    int my_rank = 0;
    ofstream flog;
    string path, logfilename;
    string timestring(bool short_format = false);
public:
    /**
     * @brief Logger initialization.
     * @param  process_id       : (optional) For multithreading process identification.
     * @param  log_level        : (optional) Only record with no greater level logs. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  to_log_file      : (optional) Whether to write logs into file.
     * @param  logfile_folder   : (optional) Save log file to which folder.
     */
    Logger (int process_id = 0, LogLevel log_level = LV_WARNING, bool to_log_file = false, string logfile_folder="./");
    
    /**
     * @brief Call this function to log a string.
     * @param  info         : Log text.
     * @param  lv           : (optional) Log level. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  with_time    : (optional) Whether to log with current wall clock time.
     */
    // void log(const char *info, LogLevel lv = LV_DEBUG, bool with_time = false);
    void log(string info, LogLevel lv = LV_DEBUG, bool with_time = false);
    
    /**
     * @brief Call this function to log a vector. If len<5 items will be output with TAB between else one item per line.
     * @param  vec_data     : The vector to be logged. Must be printable.
     * @param  info         : The log text.
     * @param  lv           : (optional) Log level. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  with_time    : (optional) Whether to log with current wall clock time.
     */
    template<typename T_item>
    void logvec(vector<T_item> &vec_data, string info, LogLevel lv = LV_DEBUG, bool with_time = false);

    // Deconstructor
    ~Logger();
};

    template<typename T_item>
    void Logger::logvec(vector<T_item> &vec_data, string info, LogLevel lv/* = LV_DEBUG*/, bool with_time/* = false*/) {
        if (lv <= log_lv) {
            this->log(info, lv, with_time);
            if (vec_data.size()<5) { // use \t
                cerr<<"    ";
                for (T_item &item: vec_data) cerr<<to_string(item)<<"\t";
                cerr<<endl;
                if (to_file && flog.is_open()) {
                    flog<<"    ";
                    for (T_item &item: vec_data) flog<<to_string(item)<<"\t";
                    flog<<endl;
                }
            } else {
                int i=0;
                for (T_item &item: vec_data) {
                    if (i++ % 10 == 0) cerr<<endl;
                    cerr<<"\t"<<to_string(item);
                }
                cerr<<endl;
                i=0;
                if (to_file && flog.is_open()) {
                    for (T_item &item: vec_data) {
                        flog<<"\t"<<to_string(item);
                        if (i++ % 10 == 0) flog<<endl;
                    }
                    flog<<endl;
                }
            }
        }
    }
    

// ================ CLASS GlobalParams ================
static class GlobalParams {
public:
    T_kvalue K_kmer = 21;   // length of kmer
    T_kvalue P_minimizer = 7;
    int SKM_partitions = 31; // minimizer length and superkmer partitions
    unsigned short kmer_min_freq = 1, kmer_max_freq = 1000; // count kmer cnt in [min,max] included
    bool HPC = false;           // homopolymer compression assembly
    bool CPU_only = false;
    int Kmer_filter = 25;       // percentage
    int N_threads = 4;          // threads per process
    string tmp_file_folder = "./tmp/";
    string log_file_folder = "./log/";
    vector<string> read_files;
    vector<string> local_read_files;
    
    T_read_cnt Batch_read_loading = 2000;
    size_t Buffer_fread_size_MB = 20;

    void ArgParser(int argc, char* argvs[]);
} PAR;

// ================ Read Sorting ================
static bool sort_comp (const ReadPtr x, const ReadPtr y) {
    return x.len < y.len;
}
#endif