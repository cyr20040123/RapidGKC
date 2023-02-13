#include "utilities.hpp"
#include <atomic>
#include <future>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
#include "read_loader.hpp"
#include "cpu_funcs.h"
#include "gkc_cuda.h"
#include "kmer_counting.hpp"
#include "thread_pool.hpp"
using namespace std;

Logger *logger;

size_t calVarStdev(vector<size_t> &vecNums) // calc avg max min var std cv (Coefficient of variation)
{
    size_t max_val = 0, min_val = 0xffffffffffffffff;
    int max_id = -1;
    size_t sumNum = 0;//accumulate(vecNums.begin(), vecNums.end(), 0);
    for (auto &item: vecNums) sumNum += item;
	size_t mean = sumNum / vecNums.size();
	double accum = 0.0;
    int i=0;
	for_each(vecNums.begin(), vecNums.end(), [&](const size_t d) {
        accum += (d - mean)*(d - mean);
        if (d>max_val) {max_val = d; max_id = i;}
        if (d<min_val) min_val = d;
        i++;
	});
	double variance = accum / vecNums.size();
	double stdev = sqrt(variance);
    
    // logger->log("SKM TOT_LEN="+to_string(sumNum), Logger::LV_INFO);
    // logger->log("SKM TOT_LEN="+sumNum); // seg fault ???
    stringstream ss;
	ss << "SIZE=" << vecNums.size() << " AVG=" << mean << "\tMAX=" << max_val << " @" << max_id << "\tmin=" << min_val << "\tvar=" << variance << "\tSTD=" << stdev << "\tCV=" << stdev/double(mean) << endl;
    logger->log(ss.str(), Logger::LV_INFO);
    return max_val;
}

/// @brief Will be called in file loading
/// @param reads 
/// @param gpars 
/// @param skm_partition_stores 
void process_reads_count(vector<ReadPtr> &reads, CUDAParams &gpars, vector<SKMStoreNoncon*> &skm_partition_stores, int tid) {
    if (true || gpars.running_threads_for_gpu >= gpars.max_threads_per_gpu * gpars.n_devices) {
        // call CPU splitter
        GenSuperkmerCPU (reads, PAR.K_kmer, PAR.P_minimizer, false, PAR.SKM_partitions, skm_partition_stores);
        return;
    }
    gpars.running_threads_for_gpu ++;//
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
    GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, CountTask::SKMPartition, PAR.SKM_partitions, skm_partition_stores, tid);
    gpars.running_threads_for_gpu --;//
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
        WallClockTimer wct_tmp;
        ReadLoader::work_while_loading_V2(
            [&gpars, &skm_part_vec](vector<ReadPtr> &reads, int tid){process_reads_count(reads, gpars, skm_part_vec, tid);},
            PAR.RD_threads_min, PAR.N_threads, readfile, PAR.Batch_read_loading, true, PAR.Buffer_fread_size_MB*ReadLoader::MB
        );
        logger->log(to_string(wct_tmp.stop())+"---- ["+readfile+"] processed ----\n", Logger::LV_NOTICE);
    }
    
    double p1_time = wct1.stop();
    logger->log("**** All reads loaded and SKMs generated (Phase 1 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 1 Time: " + to_string(p1_time) + " sec", Logger::LV_NOTICE);

    size_t skm_tot_cnt = 0, skm_tot_bytes = 0, kmer_tot_cnt = 0;
    for(i=0; i<PAR.SKM_partitions; i++) {
        kmer_tot_cnt += skm_part_vec[i]->kmer_cnt;
        skm_tot_cnt += skm_part_vec[i]->skm_cnt;
        skm_tot_bytes += skm_part_vec[i]->tot_size_bytes;
        if (PAR.to_file) skm_part_vec[i]->close_file();
    }
    logger->log("SKM TOT CNT = " + to_string(skm_tot_cnt) + " BYTES = " + to_string(skm_tot_bytes), Logger::LV_INFO);
    logger->log("KMER TOT CNT = " + to_string(kmer_tot_cnt), Logger::LV_INFO);

    vector<size_t> partition_sizes;
    for(i=0; i<PAR.SKM_partitions; i++) partition_sizes.push_back(skm_part_vec[i]->tot_size_bytes);
    calVarStdev(partition_sizes);
    
    // std::cout<<"Continue? ..."; char tmp; cin>>tmp;
    // GPUReset(gpars.device_id);
    // if (PAR.to_file) exit(0);
    
    // ==== 2nd phase: superkmer extraction and kmer counting ====
    logger->log("**** Phase 2: Superkmer extraction and kmer counting ****", Logger::LV_NOTICE);
    logger->log("(with "+to_string(PAR.N_threads)+" threads)");
    
    WallClockTimer wct2;
    int n_threads = PAR.N_threads;
    ThreadPool<size_t> tp(n_threads, 1);
    
    vector<T_kmc> kmc_result[PAR.SKM_partitions];
    future<size_t> distinct_kmer_cnt[PAR.SKM_partitions];

    // for (i=0; i<PAR.SKM_partitions; i++) {
    //     distinct_kmer_cnt[i] = tp.commit_task([&skm_part_vec, &gpars, &kmc_result, i] () {
    //         return kmc_counting_GPU (PAR.K_kmer, *(skm_part_vec[i]), gpars, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[i], PAR.GPU_compression);
    //     });
    // }
    // multi-stream phase 2
    // V1:
    // for (i=0; i<PAR.SKM_partitions; i+=gpars.n_streams_phase2) {
    //     vector<SKMStoreNoncon*> store_vec;
    //     for (int j=i; j<min(PAR.SKM_partitions, i+gpars.n_streams_phase2); j++) store_vec.push_back(skm_part_vec[j]);
    //     distinct_kmer_cnt[i] = tp.commit_task([store_vec, &gpars, &kmc_result] () {
    //         return kmc_counting_GPU_streams (PAR.K_kmer, store_vec, gpars, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result, (gpars.device_id++)%gpars.n_devices, PAR.GPU_compression);
    //     });
    // }
    // V2: auto streams
    // todo: available gpu mem = GPU_VRAM // N_THREADS * N_GPUS, required: kmer_cnt * [8|16] * 3;
    int max_streams = min((PAR.SKM_partitions+max(PAR.n_devices, PAR.N_threads)) / max(PAR.n_devices, PAR.N_threads), PAR.n_streams_phase2);
    bool force_cpu = false;
    for (i=0; i<PAR.SKM_partitions; ) {
        tp.hold_when_busy(); // to avoid such: working=[G G C C] queue=<C C C C C ...>
        if (true || (force_cpu || gpars.running_threads_for_gpu >= gpars.max_threads_per_gpu * gpars.n_devices)) {
            distinct_kmer_cnt[i] = tp.commit_task([&skm_part_vec, &kmc_result, i] (int tid) {
                return KmerCountingCPU(PAR.K_kmer, skm_part_vec[i], PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[i]);
            });
            i++;
            force_cpu=false;
            continue;
        }
        // int gpuid = (gpars.device_id++)%gpars.n_devices;
        int gpuid = (gpars.device_id++)%gpars.n_devices;
        int t = i;
        long long vram_avail = gpars.vram[gpuid];
        vector<SKMStoreNoncon*> store_vec;
        for (; i<PAR.SKM_partitions; i++) {
            if (vram_avail - skm_part_vec[i]->kmer_cnt * sizeof(T_kmer) * 3 > 0.1 * gpars.vram[gpuid] && store_vec.size() < max_streams) {
                store_vec.push_back(skm_part_vec[i]);
                vram_avail -= skm_part_vec[i]->kmer_cnt * sizeof(T_kmer) * 3;
            }
            else break;
        }
        if (store_vec.size() == 0) {
            logger->log("VRAM NOT ENOUGH @ PART"+to_string(i)+": REQUIRED="+to_string(skm_part_vec[i]->kmer_cnt * sizeof(T_kmer) * 3)+" AVAILABLE="+to_string(gpars.vram[gpuid]), Logger::LV_WARNING);
            force_cpu = true; // force the current skm part counted by cpu
            continue;
        }
        else {
            distinct_kmer_cnt[t] = tp.commit_task([store_vec, &gpars, &kmc_result, gpuid, vram_avail] (int tid) {
                gpars.running_threads_for_gpu ++;//
                size_t vram_required = gpars.vram[gpuid] - vram_avail;
                int vram_check = 1;
                while (vram_check) {
                    gpars.vram_mtx[gpuid]->lock();//
                    if (gpars.vram_used[gpuid] + vram_required < gpars.vram[gpuid]) {
                        gpars.vram_used[gpuid] += vram_required;//
                        vram_check = 0;
                    }
                    gpars.vram_mtx[gpuid]->unlock();//
                    if (vram_check) {
                        this_thread::sleep_for(100ms);
                        vram_check++;
                        if (vram_check >= 100 && vram_check % 50 == 0)
                            logger->log("Out of VRAM, now waiting... GPU:"+to_string(gpuid)+", REQ:"+to_string(vram_required)+" AVAL:"+to_string(gpars.vram[gpuid]-gpars.vram_used[gpuid]), Logger::LV_WARNING);
                    }
                }
                size_t res = kmc_counting_GPU_streams (PAR.K_kmer, store_vec, gpars, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result, gpuid, PAR.GPU_compression);
                gpars.vram_mtx[gpuid]->lock();//
                gpars.vram_used[gpuid] -= vram_required;//
                gpars.vram_mtx[gpuid]->unlock();//
                gpars.running_threads_for_gpu --;//
                return res;
            });
        }
    }
    tp.finish();

    size_t distinct_kmer_cnt_tot = 0;
    for (i=0; i<PAR.SKM_partitions; i++) {
        if (distinct_kmer_cnt[i].valid()) {
            distinct_kmer_cnt_tot += distinct_kmer_cnt[i].get();
        }
    }
    std::cerr<<endl;
    
    logger->log("Total number of distinct kmers: "+to_string(distinct_kmer_cnt_tot), Logger::LV_NOTICE);
    
    double p2_time = wct2.stop();
    logger->log("**** Kmer counting finished (Phase 2 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 2 Time: " + to_string(p2_time) + " sec (P1: " + to_string(p1_time) + " s)", Logger::LV_NOTICE);

    // for (i=0; i<PAR.SKM_partitions; i++) delete skm_part_vec[i];// deleted in kmc_counting_GPU
    return;
}
int main (int argc, char** argvs) {
    PAR.ArgParser(argc, argvs);
    
    std::cerr<<"================ PROGRAM BEGINS ================"<<endl;
    Logger _logger(0, static_cast<Logger::LogLevel>(PAR.log_lv), true, "./");
    logger = &_logger;
    
    stringstream ss;
    for(int i=0; i<argc; i++) ss<<argvs[i]<<" ";
    logger->log(ss.str());

    CUDAParams gpars;
    gpars.device_id = 0;
    gpars.n_devices = PAR.n_devices;
    gpars.n_streams = PAR.n_streams;
    gpars.n_streams_phase2 = PAR.n_streams_phase2;
    gpars.NUM_BLOCKS_PER_GRID = PAR.grid_size;
    gpars.NUM_THREADS_PER_BLOCK = PAR.block_size;
    gpars.BpG = PAR.grid_size2;
    gpars.TpB = PAR.block_size2;
    gpars.items_stream_mul = PAR.reads_per_stream_mul;
    // for (int i=0; i<PAR.N_threads; i++) gpars.gpuid_thread.push_back(-1);
    gpars.running_threads_for_gpu = 0;
    gpars.max_threads_per_gpu = PAR.max_threads_per_gpu;

    for (int i=0; i<gpars.n_devices; i++) {
        gpars.vram.push_back(GPUReset(i)); // must before not after pinned memory allocation
        gpars.vram_used.push_back(0);
        gpars.vram_mtx.push_back(new mutex());//
        cerr<<"GPU "<<i<<" VRAM "<<*(gpars.vram.rbegin())/1024/1024<<endl;
    }

    WallClockTimer wct_oa;
    std::cerr<<"----------------------------------------------"<<endl;
    GPUKmerCounting_TP(gpars);
    std::cerr<<"================ PROGRAM ENDS ================"<<endl;
    std::cout<<wct_oa.stop()<<endl;
    
    for (int i=0; i<gpars.n_devices; i++) {
        delete gpars.vram_mtx[i];//
    }
    return 0;
}