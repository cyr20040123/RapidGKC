#include "utilities.hpp"
#include <atomic>
#include <future>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
// #include "read_loader_V2.hpp"
#include "fileloader.hpp"
// #include "fileloader_gz.hpp"
#include "cpu_funcs.h"
#include "gpu_skmgen.h"
#include "gpu_kmercounting.h"
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

void phase1(vector<ReadPtr> &reads, CUDAParams &gpars, vector<SKMStoreNoncon*> &skm_partition_stores, int tid) {
    if ((!PAR.GPU_only) && (PAR.CPU_only || tid >= gpars.n_devices * gpars.max_threads_per_gpu)) {
        // call CPU splitter
        GenSuperkmerCPU (reads, PAR.K_kmer, PAR.P_minimizer, false, PAR.SKM_partitions, skm_partition_stores, tid);
    } else { // use GPU splitter
        sort(reads.begin(), reads.end(), sort_comp); // TODO: remove and compare the performance
        PinnedCSR pinned_reads(reads);
        stringstream ss;
        ss << "-- BATCH  GPU: #reads: " << reads.size() << "\tmin_len = " << reads.begin()->len << "\tmax_len = " << reads.rbegin()->len <<"\tsize = " << pinned_reads.size_capacity << "\t--";
        logger->log(ss.str());
        GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, PAR.SKM_partitions, skm_partition_stores);
    }
}

std::atomic<int> cpu_p2_cnt{0};
std::atomic<int> gpu_p2_cnt{0};

size_t phase2 (int tid, vector<SKMStoreNoncon*> store_vec, CUDAParams &gpars, vector<T_kmc> *kmc_result) {
    if (tid == 0) {
        if (store_vec[0]->id % 7 == 0) cerr<<"\r"<<(int)(store_vec[0]->id*100/PAR.SKM_partitions)<<"%";
        if (store_vec[0]->id > PAR.SKM_partitions - PAR.N_threads - 2) cerr<<"\r100%";
    }
    size_t res = 0;
    // if (tid / gpars.max_threads_per_gpu >= gpars.n_devices) {
    // WallClockTimer wct;
    if ((!PAR.GPU_only) && (PAR.CPU_only || tid >= gpars.n_devices * gpars.max_threads_per_gpu)) {
        for (auto i: store_vec)
            res += KmerCountingCPU(PAR.K_kmer, i, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[i->id], tid, PAR.threads_cpu_sorter);
        // cerr<<"-";
        cpu_p2_cnt++;
        // cerr<<"-"+to_string(wct.stop());
    }
    else {
        int gpuid = tid / gpars.max_threads_per_gpu;
        res += kmc_counting_GPU_streams (PAR.K_kmer, store_vec, gpars, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result, gpuid, tid);
        // cerr<<"*";
        gpu_p2_cnt++;
        // cerr<<"*"+to_string(wct.stop());
    }
    return res;
}
size_t phase2_forceCPU (int tid, SKMStoreNoncon* skm_store, vector<T_kmc> *kmc_result) {
    cerr<<"o";
    return KmerCountingCPU(PAR.K_kmer, skm_store, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[skm_store->id], tid, PAR.threads_cpu_sorter);
}
#define TIMING_CUDAMEMCPY
#ifdef TIMING_CUDAMEMCPY
atomic<int> debug_cudacp_timing;
#endif

void KmerCounting_TP(CUDAParams &gpars) {
    // GPUReset(gpars.device_id); // must before not after pinned memory allocation

    vector<SKMStoreNoncon*> skm_part_vec;
    int i, tid;
    
    // ====================================================
    // ==== 1st phase: loading and generate superkmers ====
    // ====================================================
    logger->log("**** Phase 1: Loading and generate superkmers ****", Logger::LV_NOTICE);
    WallClockTimer wct1;
    
    for (i=0; i<PAR.SKM_partitions; i++) skm_part_vec.push_back(new SKMStoreNoncon(i, PAR.to_file));// deleted in kmc_counting_GPU
    
    // for (auto readfile:PAR.read_files) {
    //     WallClockTimer wct_tmp;
    //     ReadLoader::work_while_loading_V2(
    //         [&gpars, &skm_part_vec](vector<ReadPtr> &reads, int tid){process_reads_count(reads, gpars, skm_part_vec, tid);},
    //         PAR.RD_threads_min, PAR.N_threads, readfile, PAR.Batch_read_loading, true, PAR.Buffer_fread_size_MB*ReadLoader::MB
    //     );
    //     logger->log(to_string(wct_tmp.stop())+"---- ["+readfile+"] processed ----\n", Logger::LV_NOTICE);
    // }

    // ReadLoader::work_while_loading_V2(PAR.K_kmer,
    //     [&gpars, &skm_part_vec](vector<ReadPtr> &reads, int tid){process_reads_count(reads, gpars, skm_part_vec, tid);},
    //     PAR.RD_threads_min, PAR.N_threads, PAR.read_files, PAR.Batch_read_loading, true, PAR.Buffer_fread_size_MB*ReadLoader::MB
    // );
    if (*(PAR.read_files[0].rbegin()) == 'z' || *(PAR.read_files[0].rbegin()) == 'Z') { // gz files
        GZReadLoader::work_while_loading_gz(PAR.K_kmer,
            [&gpars, &skm_part_vec](vector<ReadPtr> &reads, int tid){phase1(reads, gpars, skm_part_vec, tid);},
            PAR.N_threads, PAR.read_files, PAR.threads_gz, PAR.Batch_read_loading, PAR.Buffer_size_MB*PAR.N_threads);
    } else {
        ReadLoader::work_while_loading(PAR.K_kmer,
            [&gpars, &skm_part_vec](vector<ReadPtr> &reads, int tid){phase1(reads, gpars, skm_part_vec, tid);},
            PAR.N_threads, PAR.read_files, PAR.Batch_read_loading, PAR.Buffer_size_MB*PAR.N_threads);
    }

    size_t skm_tot_cnt = 0, skm_tot_bytes = 0, kmer_tot_cnt = 0;
    for(i=0; i<PAR.SKM_partitions; i++) {
        kmer_tot_cnt += skm_part_vec[i]->kmer_cnt;
        skm_tot_cnt += skm_part_vec[i]->skm_cnt;
        skm_tot_bytes += skm_part_vec[i]->tot_size_bytes;
        if (PAR.to_file) skm_part_vec[i]->close_file();
    }
    
    double p1_time = wct1.stop();
    logger->log("**** All reads loaded and SKMs generated (Phase 1 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 1 Time: " + to_string(p1_time) + " sec", Logger::LV_NOTICE);

    logger->log("SKM TOT CNT = " + to_string(skm_tot_cnt) + " BYTES = " + to_string(skm_tot_bytes), Logger::LV_INFO);
    logger->log("KMER TOT CNT = " + to_string(kmer_tot_cnt), Logger::LV_INFO);

    vector<size_t> partition_sizes;
    for(i=0; i<PAR.SKM_partitions; i++) partition_sizes.push_back(skm_part_vec[i]->tot_size_bytes);
    calVarStdev(partition_sizes);
    
    // std::cout<<"Continue? ..."; char tmp; cin>>tmp;
    // GPUReset(gpars.device_id);
    // if (PAR.to_file) exit(0);
    
    // ===========================================================
    // ==== 2nd phase: superkmer extraction and kmer counting ====
    // ===========================================================
    logger->log("**** Phase 2: Superkmer extraction and kmer counting ****", Logger::LV_NOTICE);
    logger->log("-t2 = "+to_string(PAR.threads_p2)+
                "\tGPU threads = "+to_string(PAR.max_threads_per_gpu)+ " * "+to_string(PAR.n_devices)+
                "\tCPU threads = "+to_string(PAR.threads_p2 - PAR.max_threads_per_gpu * PAR.n_devices)+" * "+to_string(PAR.threads_cpu_sorter));
    cerr<<endl;
    WallClockTimer wct2;

    vector<T_kmc> kmc_result[PAR.SKM_partitions];
    int max_streams = min((PAR.SKM_partitions+max(PAR.n_devices, PAR.N_threads)) / max(PAR.n_devices, PAR.N_threads), PAR.n_streams_phase2);
    PAR.N_threads = PAR.threads_p2;
    
    // todo: available gpu mem = GPU_VRAM // N_THREADS * N_GPUS, required: kmer_cnt * [8|16] * 3;
    /*
    // ==== V3: CPU+GPU ====
    vector<size_t> distinct_kmer_cnt(PAR.SKM_partitions, 0);
    ThreadPool<size_t> tp(PAR.N_threads, PAR.N_threads);
    atomic<int> i_part{0};
    vector<T_kmc> *kmc_result_arr = kmc_result;
    // while (i_part < PAR.SKM_partitions) {
    for (int i=0; i<PAR.SKM_partitions; i+=max_streams) {
        tp.commit_task([max_streams, &i_part, &distinct_kmer_cnt, &skm_part_vec, kmc_result_arr, &gpars](int tid){
            return p2_kmc (tid, max_streams, i_part, distinct_kmer_cnt, skm_part_vec, kmc_result_arr, gpars);
        });
    }
    tp.finish();
    assert(i_part >= PAR.SKM_partitions);
     */
    #ifdef TIMING_CUDAMEMCPY
    debug_cudacp_timing = 0;
    #endif
    future<size_t> distinct_kmer_cnt[PAR.SKM_partitions];
    ThreadPool<size_t> tp(PAR.N_threads, PAR.threads_p2 + 2); //,{0,PAR.N_threads,0,PAR.max_threads_per_gpu});
    ThreadPool<void> tp_fileloader(2); //async file loader
    SKMStoreNoncon *tmp_part;
    int j;
    for (i=0; i<PAR.SKM_partitions; i=j) {
        long long vram_avail = gpars.vram[0];
        vector<SKMStoreNoncon*> store_vec;
        // group a batch of skm partitions for GPU:
        for (j = i; j < i+max_streams && j < PAR.SKM_partitions; j++) {
            if (vram_avail - skm_part_vec[j]->kmer_cnt * sizeof(T_kmer) * 3 > 0.1 * gpars.vram[0]) {
                store_vec.push_back(skm_part_vec[j]);
                vram_avail -= skm_part_vec[j]->kmer_cnt * sizeof(T_kmer) * 3;
                if (PAR.to_file) {
                    tp.hold_when_busy();
                    tmp_part = skm_part_vec[j];
                    tp_fileloader.commit_task_no_return([tmp_part](int tid){tmp_part->load_from_file();});
                }
            } else break;
        }
        if (store_vec.size() == 0) { // if VRAM is not enough to handle even one partition, force using CPU
            logger->log("Part "+to_string(j)+" is too large to be handled by GPU, use CPU...", Logger::LV_WARNING);
            SKMStoreNoncon *t = skm_part_vec[j];
            distinct_kmer_cnt[i] = tp.commit_task([t, &kmc_result](int tid){
                return phase2_forceCPU (tid, t, kmc_result);
            });
            j++;
        } else {
            distinct_kmer_cnt[i] = tp.commit_task([store_vec, &gpars, &kmc_result](int tid){
                return phase2 (tid, store_vec, gpars, kmc_result);
            });
        }
    }
    tp_fileloader.finish();
    tp.finish();

    size_t distinct_kmer_cnt_tot = 0;
    /*
    // ==== V3: CPU+GPU ====
    for (i=0; i<PAR.SKM_partitions; i++)
        distinct_kmer_cnt_tot += distinct_kmer_cnt[i];
     */
    for (i=0; i<PAR.SKM_partitions; i++) {
        if (distinct_kmer_cnt[i].valid()) {
            distinct_kmer_cnt_tot += distinct_kmer_cnt[i].get();
        }
    }
    std::cerr<<endl;
    
    #ifdef TIMING_CUDAMEMCPY
    cerr<<"debug_cudacp_timing: "<< debug_cudacp_timing <<endl;
    #endif
    
    logger->log("Phase 2 CPU:GPU = "+to_string(cpu_p2_cnt)+":"+to_string(gpu_p2_cnt), Logger::LV_NOTICE);

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
    gpars.BpG1 = PAR.grid_size;
    gpars.TpB1 = PAR.block_size;
    gpars.BpG2 = PAR.grid_size2;
    gpars.TpB2 = PAR.block_size2;
    gpars.items_stream_mul = PAR.reads_per_stream_mul;
    // for (int i=0; i<PAR.N_threads; i++) gpars.gpuid_thread.push_back(-1);
    // gpars.gpuworker_threads = 0;
    gpars.max_threads_per_gpu = PAR.max_threads_per_gpu;

    for (int i=0; i<gpars.n_devices; i++) {
        gpars.vram.push_back(GPUReset(i)); // must before not after pinned memory allocation
        gpars.vram_used.push_back(0);
        gpars.vram_mtx.push_back(new mutex());//
        cerr<<"GPU "<<i<<" VRAM "<<*(gpars.vram.rbegin())/1024/1024<<endl;
        // gpars.running_threads_of_gpu.push_back(new atomic<int>(0));
    }

    WallClockTimer wct_oa;
    std::cerr<<"----------------------------------------------"<<endl;
    KmerCounting_TP(gpars);
    std::cerr<<"================ PROGRAM ENDS ================"<<endl;
    std::cout<<wct_oa.stop()<<endl;
    
    for (int i=0; i<gpars.n_devices; i++) {
        delete gpars.vram_mtx[i];//
        // delete gpars.running_threads_of_gpu[i];//
    }
    return 0;
}