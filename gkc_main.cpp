#include "utilities.hpp"
#include <atomic>
#include <future>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
// #include "csr.hpp"
#include "read_loader.hpp"
// #include "superkmers.hpp"
#include "gkc_cuda.hpp"
#include "kmer_counting.hpp"
#include "thread_pool.hpp"
using namespace std;

Logger *logger;

void calVarStdev(vector<size_t> &vecNums) // calc avg max min var std cv (Coefficient of variation)
{
    size_t max_val = 0, min_val = 0xffffffffffffffff;
	size_t sumNum = accumulate(vecNums.begin(), vecNums.end(), 0);
	size_t mean = sumNum / vecNums.size();
	double accum = 0.0;
	for_each(vecNums.begin(), vecNums.end(), [&](const size_t d) {
		accum += (d - mean)*(d - mean);
        if (d>max_val) max_val = d;
        if (d<min_val) min_val = d;
	});
	double variance = accum / vecNums.size();
	double stdev = sqrt(variance);
    
    logger->log("SKM TOT_LEN="+to_string(sumNum));
    // logger->log("SKM TOT_LEN="+sumNum); // seg fault ???
    stringstream ss;
	ss << "AVG=" << mean << "\tMAX=" << max_val << "\tmin=" << min_val << "\tvar=" << variance << "\tSTD=" << stdev << "\tCV=" << stdev/double(mean) << endl;
    logger->log(ss.str());
}

/// @brief Will be called in file loading
/// @param reads 
/// @param gpars 
/// @param skm_partition_stores 
void process_reads_count(vector<ReadPtr> &reads, CUDAParams &gpars, vector<SKMStoreNoncon*> &skm_partition_stores) {
    sort(reads.begin(), reads.end(), sort_comp); // TODO: remove and compare the performance
    PinnedCSR pinned_reads(reads);
    stringstream ss;
    ss << "---- BATCH\tn_reads = " << reads.size() << "\tmin_len = " << reads.begin()->len << "\tmax_len = " << reads.rbegin()->len <<"\tsize = " << pinned_reads.size_capacity << "\t----";
    logger->log(ss.str());
    assert(pinned_reads.get_n_reads() == reads.size());
    // logger->log("Pinned: "+to_string(pinned_reads.get_n_reads())+" size = "+to_string(pinned_reads.size_capacity));
    
    // function<void(T_h_data)> process_func = [&skm_partition_stores](T_h_data hd) {
    //     SKMStoreNoncon::save_batch_skms (skm_partition_stores, hd.skm_cnt, hd.kmer_cnt, hd.skmpart_offs, hd.skm_store_csr, nullptr, true);
    // };
    GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, CountTask::SKMPartition, PAR.SKM_partitions, skm_partition_stores, PAR.GPU_compression);
}

void GPUKmerCounting_TP(CUDAParams &gpars) {
    // GPUReset(gpars.device_id); // must before not after pinned memory allocation

    vector<SKMStoreNoncon*> skm_part_vec;
    int i, tid;
    for (i=0; i<PAR.SKM_partitions; i++) skm_part_vec.push_back(new SKMStoreNoncon(i, PAR.to_file));// deleted in kmc_counting_GPU
    
    // ==== 1st phase: loading and generate superkmers ====
    logger->log("**** Phase 1: Loading and generate superkmers ****", Logger::LV_NOTICE);
    WallClockTimer wct1;
    
    for (auto readfile:PAR.read_files) {
        ReadLoader::work_while_loading_V2(
            [&gpars, &skm_part_vec](vector<ReadPtr> &reads){process_reads_count(reads, gpars, skm_part_vec);},
            2, PAR.read_files[0], PAR.Batch_read_loading, true, PAR.Buffer_fread_size_MB*ReadLoader::MB
        );
        logger->log("-- ["+readfile+"] processed --", Logger::LV_NOTICE);
    }
    
    double p1_time = wct1.stop();
    logger->log("**** All reads loaded and SKMs generated (Phase 1 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 1 Time: " + to_string(p1_time) + " sec", Logger::LV_INFO);

    size_t skm_tot_cnt = 0, skm_tot_bytes = 0, kmer_tot_cnt = 0;
    for(i=0; i<PAR.SKM_partitions; i++) {
        kmer_tot_cnt += skm_part_vec[i]->kmer_cnt;
        skm_tot_cnt += skm_part_vec[i]->skm_cnt;
        skm_tot_bytes += skm_part_vec[i]->tot_size_bytes;
        if (PAR.to_file) skm_part_vec[i]->close_file();
    }
    logger->log("SKM TOT CNT = " + to_string(skm_tot_cnt) + " BYTES = " + to_string(skm_tot_bytes));
    logger->log("KMER TOT CNT = " + to_string(kmer_tot_cnt));

    // cout<<"Continue? ..."; char tmp; cin>>tmp;
    // GPUReset(gpars.device_id);
    // if (PAR.to_file) exit(0);
    
    // ==== 2nd phase: superkmer extraction and kmer counting ====
    logger->log("**** Phase 2: Superkmer extraction and kmer counting ****", Logger::LV_NOTICE);
    logger->log("(with "+to_string(PAR.N_threads)+" threads)");
    
    WallClockTimer wct2;
    int n_threads = PAR.N_threads;
    ThreadPool<size_t> tp(n_threads);
    
    vector<T_kmc> kmc_result[PAR.SKM_partitions];
    future<size_t> distinct_kmer_cnt[PAR.SKM_partitions];

    // for (i=0; i<PAR.SKM_partitions; i++) {
    //     distinct_kmer_cnt[i] = tp.commit_task([&skm_part_vec, &gpars, &kmc_result, i] () {
    //         return kmc_counting_GPU (PAR.K_kmer, *(skm_part_vec[i]), gpars, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[i], PAR.GPU_compression);
    //     });
    // }
    // multi-stream phase 2
    for (i=0; i<PAR.SKM_partitions; i+=gpars.n_streams_phase2) {
        vector<SKMStoreNoncon*> store_vec;
        for (int j=i; j<min(PAR.SKM_partitions, i+gpars.n_streams_phase2); j++) store_vec.push_back(skm_part_vec[j]);
        distinct_kmer_cnt[i] = tp.commit_task([store_vec, &gpars, &kmc_result] () {
            return kmc_counting_GPU_streams (PAR.K_kmer, store_vec, gpars, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result, PAR.GPU_compression);
        });
    }
    tp.finish();

    size_t distinct_kmer_cnt_tot = 0;
    for (i=0; i<PAR.SKM_partitions; i++) {
        if (distinct_kmer_cnt[i].valid()) {
            distinct_kmer_cnt_tot += distinct_kmer_cnt[i].get();
        }
    }
    cerr<<endl;
    
    logger->log("Total number of distinct kmers: "+to_string(distinct_kmer_cnt_tot));
    
    double p2_time = wct2.stop();
    logger->log("**** Kmer counting finished (Phase 2 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 2 Time: " + to_string(p2_time) + " sec", Logger::LV_INFO);

    // for (i=0; i<PAR.SKM_partitions; i++) delete skm_part_vec[i];// deleted in kmc_counting_GPU
    return;
}

int main (int argc, char** argvs) {
    cerr<<"================ PROGRAM BEGINS ================"<<endl;
    Logger _logger(0, Logger::LV_DEBUG, true, "./");
    logger = &_logger;
    PAR.ArgParser(argc, argvs);
    
    stringstream ss;
    for(int i=0; i<argc; i++) ss<<argvs[i]<<" ";
    logger->log(ss.str());

    CUDAParams gpars;
    gpars.device_id = 0;
    gpars.n_devices = PAR.n_devices;
    gpars.n_streams = PAR.n_streams;
    gpars.n_streams_phase2 = PAR.n_streams;
    gpars.NUM_BLOCKS_PER_GRID = PAR.grid_size;
    gpars.NUM_THREADS_PER_BLOCK = PAR.block_size;
    gpars.items_stream_mul = PAR.reads_per_stream_mul;

    for (int i=0; i<gpars.n_devices; i++) GPUReset(i); // must before not after pinned memory allocation

    WallClockTimer wct_oa;
    cerr<<"----------------------------------------------"<<endl;
    GPUKmerCounting_TP(gpars);
    cerr<<"================ PROGRAM ENDS ================"<<endl;
    cout<<wct_oa.stop()<<endl;
    return 0;
}