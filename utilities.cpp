#include "utilities.hpp"
#include <iostream>
#include <cstring>
using namespace std;

// ======================================================
// ================ CLASS WallClockTimer ================
// ======================================================
    WallClockTimer::WallClockTimer() {
        start = chrono::high_resolution_clock::now();
    }
    double WallClockTimer::stop(bool millisecond /* = false*/) {
        end = chrono::high_resolution_clock::now();
        if (millisecond) return chrono::duration_cast<chrono::milliseconds>(end - start).count();
        else return chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0;
    }
    void WallClockTimer::restart() {
        start = chrono::high_resolution_clock::now();
    }

// ==============================================
// ================ CLASS Logger ================
// ==============================================
    string Logger::timestring(bool short_format/* = false*/) {
        time_t rawtime;    struct tm *info;    char buffer[80];
        time(&rawtime);    info = localtime(&rawtime);
        if (short_format) strftime(buffer, 80, "%m%d-%H%M%S", info);
        else strftime(buffer, 80, "(%m-%d %H:%M:%S) ", info);
        return string(buffer);
    }
    Logger::Logger (int process_id/* = 0*/, LogLevel log_level/* = LV_WARNING*/, bool to_log_file/* = false*/, string logfile_folder/*="./"*/)
     : log_lv(log_level), to_file(to_log_file), my_rank(process_id), path(logfile_folder) {
        if (path.back() != '/') path += '/';
        if (to_file) {
            logfilename = path + "log_P" + to_string(my_rank) + ".log";
            // logfilename = path + "log_P" + to_string(my_rank) + "_" + timestring(true) + ".log";
            flog = ofstream(logfilename, ios::app|ios::out);
            if (!flog) cerr<<"Unable to open and write to log file!"<<endl;
            flog<<"================================"<<endl;
        }
    }
    void Logger::log(string info, LogLevel lv/* = LV_DEBUG*/, bool with_time/* = false*/) {
        if (lv <= log_lv) {
            if (with_time) cerr<<timestring()<<"P"<<my_rank<<" "<<loglabel[lv]<<"\t"<<info<<endl;
            else cerr<<"P"<<my_rank<<" "<<loglabel[lv]<<"\t"<<info<<endl;
            if (to_file && flog.is_open()) flog<<timestring()<<"P"<<my_rank<<" "<<loglabel[lv]<<"\t"<<info<<endl;
        }
        if (lv == LV_FATAL) exit(1);
    }
    // void Logger::log(string info, LogLevel lv = LV_DEBUG, bool with_time = false) {
    //     Logger::log(info.c_str(), lv, with_time);
    // }
    
    Logger::~Logger() {if (to_file && flog.is_open()) flog.close();}



// ====================================================
// ================ CLASS GlobalParams ================
// ====================================================
void GlobalParams::ArgParser(int argc, char* argvs[]) {
    // exe <k> <p> <tmp_file_folder> <_Kmer_filter>
    for (int i=1; i<argc-1; i++) {
        if (!strcmp(argvs[i], "-t")) N_threads = atoi(argvs[++i]);
        else if (!strcmp(argvs[i], "-k")) K_kmer = atoi(argvs[++i]);
        else if (!strcmp(argvs[i], "-p")) P_minimizer = atoi(argvs[++i]);
        else if (!strcmp(argvs[i], "-skm")) SKM_partitions = atoi(argvs[++i]);
        else if (!strcmp(argvs[i], "-hpc")) HPC = bool(atoi(argvs[++i]));
        else if (!strcmp(argvs[i], "-cpuonly")) CPU_only = bool(atoi(argvs[++i]));
        else if (!strcmp(argvs[i], "-kf")) Kmer_filter = atof(argvs[++i])*100;
        else if (!strcmp(argvs[i], "-tmp")) tmp_file_folder = string(argvs[++i]);
        else if (!strcmp(argvs[i], "-log")) log_file_folder = string(argvs[++i]);
        else if (!strcmp(argvs[i], "-fb")) Buffer_fread_size_MB = atoi(argvs[++i]);
        else if (!strcmp(argvs[i], "-rb")) Batch_read_loading = atoi(argvs[++i]);
        else if (!strcmp(argvs[i], "-read")) {
            int j;
            for (j=i+1; j<argc; j++, i++) {
                if (argvs[j][0] == '-') {i=j-1; break;}
                read_files.push_back(string(argvs[j]));
            }
        }
        else {
            cerr<<"Wrong argument: "<<argvs[i]<<endl;
            cerr<<"Usage: "<< argvs[0] <<" -t <threads_per_process> -k <K_kmer> -p <P_minimizer> -skm <SKM_partitions> -hpc <HPC> -kf <Kmer_filter> -tmp <tmp_file_folder> -read <read_files...>"<<endl;
            exit(1);
        }
    }
    if (read_files.size() == 0) {
        cerr<<"Usage: "<< argvs[0] <<" -t <threads_per_process> -k <K_kmer> -p <P_minimizer> -skm <SKM_partitions> -hpc <HPC> -kf <Kmer_filter> -tmp <tmp_file_folder> -read <read_files...>"<<endl;
        exit(1);
    }
}

