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

#ifdef MMFILTER_TIMING
#include "minimizer_filter.h"
#endif

using namespace std;

Logger *logger;
GlobalParams PAR;

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
        // sort(reads.begin(), reads.end(), sort_comp); // TODO: remove and compare the performance
        PinnedCSR pinned_reads(reads);
        stringstream ss;
        ss << "-- BATCH  GPU: #reads: " << reads.size() << "\tmin_len = " << reads.begin()->len << "\tmax_len = " << reads.rbegin()->len <<"\tsize = " << pinned_reads.size_capacity << "\t--";
        logger->log(ss.str());
        int gpuid = (tid / gpars.max_threads_per_gpu) % (PAR.n_devices); // no need to mod
        GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, PAR.SKM_partitions, skm_partition_stores, tid, gpuid);
    }
}

std::atomic<int> cpu_p2_cnt{0};
std::atomic<int> gpu_p2_cnt{0};

std::atomic<int> gput_run_cpu_cnt{0};

size_t phase2 (int tid, vector<SKMStoreNoncon*> store_vec, CUDAParams &gpars, vector<T_kmc> *kmc_result, size_t avg_kmer_cnt = 0) {
    if (tid == 0) { // output progress
        if (store_vec[0]->id % 7 == 0) cerr<<"\r"<<(int)(store_vec[0]->id*100/PAR.SKM_partitions)<<"%";
        if (store_vec[0]->id > PAR.SKM_partitions - PAR.N_threads - 2) cerr<<"\r100%";
    }
    size_t res = 0;
    if ((!PAR.GPU_only) && (PAR.CPU_only || tid >= gpars.n_devices * gpars.max_threads_per_gpu_p2 + gput_run_cpu_cnt)) {
        for (auto i: store_vec)
            res += KmerCountingCPU(PAR.K_kmer, i, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[i->id], tid, PAR.threads_cpu_sorter);
        cpu_p2_cnt.fetch_add(store_vec.size());
    }
    else {
        int gpuid = (tid / gpars.max_threads_per_gpu_p2) % (PAR.n_devices);
        res += kmc_counting_GPU_streams (PAR.K_kmer, store_vec, gpars, PAR.kmer_min_freq, PAR.kmer_max_freq, PAR.output_file_prefix, avg_kmer_cnt, gpuid, tid);
        gpu_p2_cnt.fetch_add(store_vec.size());
    }
    return res;
}
size_t phase2_forceCPU (int tid, SKMStoreNoncon* skm_store, vector<T_kmc> *kmc_result) {
    cerr<<"o";
    cpu_p2_cnt++;
    gput_run_cpu_cnt++;
    size_t res = KmerCountingCPU(PAR.K_kmer, skm_store, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[skm_store->id], tid, PAR.threads_cpu_sorter);
    gput_run_cpu_cnt--;
    return res;
}

#ifdef MMFILTER_TIMING
atomic<int> mm_filter_tot_time{0};
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
    #ifdef MMFILTER_TIMING
    logger->log("FILTER: " STR(FILTER_KERNEL) " Kernel Functions Time: "+to_string(mm_filter_tot_time)+" ms", Logger::LV_INFO);
    #endif
    logger->log("**** All reads loaded and SKMs generated (Phase 1 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 1 Time: " + to_string(p1_time) + " sec", Logger::LV_NOTICE);

    logger->log("SKM TOT CNT = " + to_string(skm_tot_cnt) + " BYTES = " + to_string(skm_tot_bytes), Logger::LV_NOTICE);
    logger->log("KMER TOT CNT = " + to_string(kmer_tot_cnt), Logger::LV_NOTICE);

    vector<size_t> partition_sizes;
    for(i=0; i<PAR.SKM_partitions; i++) partition_sizes.push_back(skm_part_vec[i]->tot_size_bytes);
    calVarStdev(partition_sizes);
    size_t avg_part_kmers = kmer_tot_cnt / PAR.SKM_partitions;
    
    // std::cout<<"Continue? ..."; char tmp; cin>>tmp;
    // GPUReset(gpars.device_id);
    // if (PAR.to_file) exit(0);
    
    // ===========================================================
    // ==== 2nd phase: superkmer extraction and kmer counting ====
    // ===========================================================
    logger->log("**** Phase 2: Superkmer extraction and kmer counting ****", Logger::LV_NOTICE);
    logger->log("[GPU counter] "+to_string(PAR.max_threads_per_gpu_p2)+ " threads\t* "+to_string(PAR.n_devices)+" GPU(s)");
    logger->log("[CPU counter] "+to_string(PAR.threads_cpu_sorter)+" threads\t*"      +to_string((PAR.threads_p2 - PAR.max_threads_per_gpu_p2 * PAR.n_devices) / PAR.threads_cpu_sorter)+" CPU worker(s)");
    logger->log("Total CPU threads used: "+to_string(PAR.max_threads_per_gpu_p2 * PAR.n_devices + (PAR.threads_p2 - PAR.max_threads_per_gpu_p2*PAR.n_devices)/PAR.threads_cpu_sorter*PAR.threads_cpu_sorter));
    logger->log("Total GPU workers: "+to_string(PAR.max_threads_per_gpu_p2 * PAR.n_devices)+"\tTotal CPU workers"+to_string((PAR.threads_p2 - PAR.max_threads_per_gpu_p2*PAR.n_devices)/PAR.threads_cpu_sorter));
    cerr<<endl;
    WallClockTimer wct2;
    
    vector<T_kmc> kmc_result[PAR.SKM_partitions];
    int max_streams_p2 = PAR.n_streams_p2;
    PAR.N_threads = PAR.threads_p2;
    int n_workers_tp = PAR.max_threads_per_gpu_p2 * PAR.n_devices + (PAR.threads_p2 - PAR.max_threads_per_gpu_p2*PAR.n_devices)/PAR.threads_cpu_sorter;
    
    future<size_t> distinct_kmer_cnt[PAR.SKM_partitions];
    ThreadPool<size_t> tp(n_workers_tp, n_workers_tp + 2); //,{0,PAR.N_threads,0,PAR.max_threads_per_gpu});
    ThreadPool<void> tp_fileloader(1); //async file loader
    SKMStoreNoncon *tmp_part;
    int j;
    vector<int> toolarge_bins;
    if (PAR.n_devices == 0) gpars.vram.push_back((size_t)MB1*(size_t)32768);
    for (i=0; i<PAR.SKM_partitions; i=j) {
        long long vram_avail = gpars.vram[0] / max(1, PAR.max_threads_per_gpu_p2); // vram available for each GPU worker (1 GPU worker = 1 CPU thread)
        vector<SKMStoreNoncon*> store_vec;
        // group a batch of skm partitions for GPU:
        for (j = i; j < i+max_streams_p2 && j < PAR.SKM_partitions; j++) {
            size_t vram_required = max((size_t)(skm_part_vec[j]->kmer_cnt * sizeof(T_kmer) * 2.1), skm_part_vec[j]->kmer_cnt * (sizeof(T_kmer) + sizeof(T_kmer_cnt)*2)); // sync calc with gpu_kmercounting.cu
            vram_required += 128 * MB1;
            if (vram_avail - vram_required > 0.1 * gpars.vram[0]) {
                store_vec.push_back(skm_part_vec[j]);
                vram_avail -= vram_required;
                if (PAR.to_file) { // load currently assigned partition from file
                    tp.hold_when_busy();
                    tmp_part = skm_part_vec[j];
                    tp_fileloader.commit_task_no_return([tmp_part](int tid){tmp_part->load_from_file();});
                }
            } else break;
        }
        if (store_vec.size() == 0) 
        { // if VRAM is not enough to handle even one partition, force using CPU
            toolarge_bins.push_back(j);
            SKMStoreNoncon *t = skm_part_vec[j];
            distinct_kmer_cnt[i] = tp.commit_task([t, &kmc_result](int tid){
                return phase2_forceCPU (tid, t, kmc_result);
            });
            j++;
        } else { // normal call (determine GPU or CPU automatically by the thread ID in phase2())
            distinct_kmer_cnt[i] = tp.commit_task([store_vec, &gpars, &kmc_result, avg_part_kmers](int tid){
                return phase2 (tid, store_vec, gpars, kmc_result, avg_part_kmers);
            });
        }
    }
    if (toolarge_bins.size() > 0) {
        logger->log(to_string(toolarge_bins.size())+" partition(s) are too large to be handled by GPU, use CPU.", Logger::LV_WARNING);
        if (toolarge_bins.size() <= 16) {
            for (auto i: toolarge_bins) cerr<<"Part-"<<i<<"\t";
            cerr<<endl;
        }
    }
    tp_fileloader.finish();
    if (logger->log_lv == Logger::LV_DEBUG && PAR.n_devices > 0) {
        tp.hold_when_busy();
        logger->log("GPU VRAM LEFT = "+to_string(GPUVram(0)/MB1));
    }
    tp.finish();

    // Gather total distinct k-mer count
    size_t distinct_kmer_cnt_tot = 0;
    for (i=0; i<PAR.SKM_partitions; i++) {
        if (distinct_kmer_cnt[i].valid()) {
            distinct_kmer_cnt_tot += distinct_kmer_cnt[i].get();
        }
    }
    logger->log("Phase 2 processed partitions CPU:GPU = "+to_string(cpu_p2_cnt)+":"+to_string(gpu_p2_cnt), Logger::LV_INFO);
    std::cerr<<endl;
    logger->log("Total number of distinct kmers: "+to_string(distinct_kmer_cnt_tot), Logger::LV_NOTICE);
    
    double p2_time = wct2.stop();
    logger->log("**** Kmer counting finished (Phase 2 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 2 Time: " + to_string(p2_time) + " sec (P1: " + to_string(p1_time) + " s)", Logger::LV_NOTICE);

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
    gpars.n_streams_p2 = PAR.n_streams_p2;
    gpars.BpG1 = PAR.grid_size;
    gpars.TpB1 = PAR.block_size;
    gpars.BpG2 = PAR.grid_size2;
    gpars.TpB2 = PAR.block_size2;
    gpars.items_stream_mul = PAR.reads_per_stream_mul;
    gpars.max_threads_per_gpu = PAR.max_threads_per_gpu;
    gpars.max_threads_per_gpu_p2 = PAR.max_threads_per_gpu_p2;

    for (int i=0; i<gpars.n_devices; i++) {
        gpars.vram.push_back(GPUReset(i)); // must before not after pinned memory allocation
        // gpars.vram_used.push_back(0);
        // gpars.vram_mtx.push_back(new mutex());//
        cerr<<"GPU "<<i<<" VRAM (MB):"<<*(gpars.vram.rbegin())/MB1<<endl;
    }

    WallClockTimer wct_oa;
    std::cerr<<"----------------------------------------------"<<endl;
    KmerCounting_TP(gpars);
    std::cerr<<"================ PROGRAM ENDS ================"<<endl;
    std::cout<<wct_oa.stop()<<endl;

    return 0;
}