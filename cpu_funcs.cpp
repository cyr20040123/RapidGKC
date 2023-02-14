#include "cpu_funcs.h"
#include "utilities.hpp"
#include <cstring>
#include <algorithm>

using namespace std;
// #define TEST

const unsigned char basemap[256] = {
    255, 255, 255, 255, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 0, 255, 1, 255, 255, 255, 2, // 64..71
    255, 255, 255, 255, 255, 255, 255, 255, // 72..79
    255, 255, 255, 255, 3, 0, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 0, 255, 1, 255, 255, 255, 2, // 96..103
    255, 255, 255, 255, 255, 255, 255, 255, // 104..111
    255, 255, 255, 255, 3, 0, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

const unsigned char basemap_compl[256] = { // complement base
    255, 255, 255, 255, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 3, 255, 2, 255, 255, 255, 1, // 64..71
    255, 255, 255, 255, 255, 255, 255, 255, // 72..79
    255, 255, 255, 255, 0, 3, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 3, 255, 2, 255, 255, 255, 1, // 96..103
    255, 255, 255, 255, 255, 255, 255, 255, // 104..111
    255, 255, 255, 255, 0, 3, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

#ifndef TEST
extern Logger *logger;
#endif

T_read_len _hpc_encoding (T_read_len len, const char* raw_read, byte* hpc_read) {
    T_read_len i_raw, i_hpc = 0;
    hpc_read[i_hpc] = raw_read[0];
    for (i_raw = 1; i_raw < len; i_raw++) {
        i_hpc += raw_read[i_raw] != raw_read[i_raw-1];
        hpc_read[i_hpc] = raw_read[i_raw];
    }
    return i_hpc;
}
bool new_filter2(T_minimizer mm, int p) {
    // return ((((mm >> ((p-3)*2)) & 0b111011) != 0/*no AAA ACA*/) & ((mm & 0b111111) != 0/*no AAA at last*/));
    return ((mm >> ((p-3)*2)) & 0b111011) && (mm & 0b111111);
}
/**
 * Filter pmer and pmer_rc, store the minimizer to mm, return if the minimizer was updated.
 */
bool _filter_and_assign_mm (_out_ T_minimizer &mm, T_minimizer pmer, T_minimizer pmer_rc, const T_kvalue P_minimizer, const T_minimizer mm_mask) {
    bool new_mm_flag = false;
    // T_minimizer mm_mask = T_MM_MAX >> (sizeof(T_minimizer)*8 - 2*P_minimizer);
    if (!new_filter2(pmer, P_minimizer)) pmer = mm_mask;
    if (!new_filter2(pmer_rc, P_minimizer)) pmer_rc = mm_mask;
    if (pmer <= mm) {
        mm = pmer;
        new_mm_flag = true;
    }
    if (pmer_rc <= mm) {
        mm = pmer_rc;
        new_mm_flag = true;
    }
    return new_mm_flag;
}
void _get_mm_of_kmer (byte* read, T_read_len i, const T_kvalue P_minimizer, const T_kvalue K_kmer, const T_minimizer mm_mask,
    _out_ T_minimizer &mm, _out_ T_minimizer &pmer, _out_ T_minimizer &pmer_rc, _out_ T_read_len &mm_beg) {

    mm = mm_mask;
    T_read_len j;
    pmer = 0;
    pmer_rc = 0;
    for (j = 0; j < P_minimizer; j++) {
        pmer = (pmer << 2) | basemap[*(read+i+j)];
        pmer_rc |= basemap_compl[*(read+i+j)] << (2*(j));
    }
    _filter_and_assign_mm (mm, pmer, pmer_rc, P_minimizer, mm_mask);
    mm_beg = 0;
    for (j=P_minimizer; j<K_kmer; j++) {
        pmer = ((pmer << 2) | basemap[*(read+i+j)]) & mm_mask;
        pmer_rc = (pmer_rc >> 2) | (basemap_compl[*(read+i+j)] << (P_minimizer*2-2));
        if (_filter_and_assign_mm (mm, pmer, pmer_rc, P_minimizer, mm_mask)) {
            mm_beg = j-P_minimizer+1;
        }
    }
    mm_beg += i;
    return;
}
T_read_len gen_skm_offs(byte* read, T_read_len len, const T_kvalue K_kmer, const T_kvalue P_minimizer, _out_ T_read_len *skm_offs, _out_ T_minimizer *minimizers) {
    T_minimizer cur_mm, mm;
    T_minimizer pmer, pmer_rc;
    T_minimizer mm_mask = T_MM_MAX >> (sizeof(T_minimizer)*8 - 2*P_minimizer);
    int i, j;
    T_read_len skm_begin_pos = 0, skm_cnt = 0;
    int mm_beg = -1;
    
    // -- Generate the first k-mer's minimizer --
    _get_mm_of_kmer (read, 0, P_minimizer, K_kmer, mm_mask, mm, pmer, pmer_rc, mm_beg); // get mm pmer pmer_rc mm_beg
    cur_mm = mm;
    
    for (i=1; i<=len-K_kmer; i++) { // i: k-mer ID
        if (mm_beg < i) {
            _get_mm_of_kmer (read, i, P_minimizer, K_kmer, mm_mask, mm, pmer, pmer_rc, mm_beg);
            if (mm != cur_mm) {
                // find skm, save the previous index
                skm_offs[skm_cnt] = skm_begin_pos;
                minimizers[skm_cnt] = cur_mm;
                skm_cnt++;
                
                skm_begin_pos = i;
                cur_mm = mm;
            }
        } else {
            j = i+K_kmer-1; // the last base of the current k-mer
            pmer = ((pmer << 2) | basemap[*(read+j)]) & mm_mask;
            pmer_rc = (pmer_rc >> 2) | (basemap_compl[*(read+j)] << (P_minimizer*2-2));
            if (_filter_and_assign_mm (mm, pmer, pmer_rc, P_minimizer, mm_mask)) {
                mm_beg = j-P_minimizer+1;
                if (mm < cur_mm) {
                    if (mm_beg + P_minimizer - K_kmer > skm_begin_pos) {
                        // mm can't reach the first k-mer in skm and is different from the previous
                        // <find skm>
                        skm_offs[skm_cnt] = skm_begin_pos;
                        minimizers[skm_cnt] = cur_mm;
                        skm_cnt++;

                        skm_begin_pos = i;
                    }
                    cur_mm = mm;
                }
            }
        }
    }
    skm_offs[skm_cnt] = skm_begin_pos;
    minimizers[skm_cnt] = cur_mm;
    skm_cnt++;
    skm_offs[skm_cnt] = len-K_kmer+1;

    return skm_cnt;
}

void read_compression (byte *read, T_read_len len) {
    T_read_len i;
    for (i=0; i<=len-BYTE_BASES; i+=BYTE_BASES)
        read[i/BYTE_BASES] = (0b11<<6) | (basemap[read[i]]<<4) | (basemap[read[i+1]]<<2) | (basemap[read[i+2]]);
    if (len-i == 0) {
        read[i/BYTE_BASES] = 0b0;
    } else if (len-i == 1) {
        read[i/BYTE_BASES] = (0b1<<6) | (basemap[read[i]]<<4);
    } else if (len-i == 2) {
        read[i/BYTE_BASES] = (0b10<<6) | (basemap[read[i]]<<4) | (basemap[read[i+1]]<<2);
    }
}

inline int _hash_partition (T_minimizer mm, int SKM_partitions) {
    return (~mm) % SKM_partitions;
}
inline T_skm_len _skm_bytes_required (T_read_len beg, T_read_len end, int k) {
    return ((beg%3) + end+(k-1)-beg + 3) / 3; // +3 because skm_3x requires an extra empty byte
}
void gen_skms (byte *read, T_read_len len, T_read_len *skm_offs, T_minimizer *minimizers, T_read_len n_skms, 
    const T_kvalue K_kmer, const int SKM_partitions, const int buffer_size, 
    _out_ T_skm_partsize *skm_cnt, _out_ T_skm_partsize *kmer_cnt, _out_ byte **skm_buffer, _out_ int *skm_buf_pos,
    _out_ vector<SKMStoreNoncon*> skm_partition_stores) {
    
    T_read_len i;
    T_skm_len skm_size_bytes;
    int partition;
    for (i = 0; i < n_skms; i++) {
        partition = _hash_partition (minimizers[i], SKM_partitions);
        skm_size_bytes = _skm_bytes_required(skm_offs[i], skm_offs[i+1], K_kmer);
        if (skm_buf_pos[partition] + skm_size_bytes >= buffer_size) {
            // save skms to SKMStoreNoncon
            SKMStoreNoncon::save_skms(skm_partition_stores[partition], skm_cnt[partition], kmer_cnt[partition], skm_buffer[partition], skm_buf_pos[partition], buffer_size);//
            skm_buffer[partition] = new byte [buffer_size];//
            skm_buf_pos[partition] = 0;
            skm_cnt[partition] = 0;
            kmer_cnt[partition] = 0;
        }
        memcpy(&(skm_buffer[partition][skm_buf_pos[partition]]), &(read[skm_offs[i]/BYTE_BASES]), skm_size_bytes);
        // process the first byte
        skm_buffer[partition][skm_buf_pos[partition]] &= 0b00111111;
        skm_buffer[partition][skm_buf_pos[partition]] |= (BYTE_BASES - skm_offs[i] % BYTE_BASES) << (2*BYTE_BASES);
        // process the last byte
        skm_buffer[partition][skm_buf_pos[partition] + skm_size_bytes - 1] &= 0b00111111;
        skm_buffer[partition][skm_buf_pos[partition] + skm_size_bytes - 1] |= ((skm_offs[i+1]-1+K_kmer) % BYTE_BASES) << (2*BYTE_BASES);
        // update counter
        skm_buf_pos[partition] += skm_size_bytes;
        skm_cnt[partition] ++;
        kmer_cnt[partition] += skm_offs[i+1] - skm_offs[i];
    }
}

void GenSuperkmerCPU (vector<ReadPtr> &reads,
    const T_kvalue K_kmer, const T_kvalue P_minimizer, bool HPC, 
    const int SKM_partitions, vector<SKMStoreNoncon*> skm_partition_stores)
{
    
    const int SKM_BUFFER_SIZE = 16384; // 64M in total with 4096 partitions
    T_skm_partsize skm_cnt[SKM_partitions];
    T_skm_partsize kmer_cnt[SKM_partitions];
    byte *skm_buffer[SKM_partitions]; // flush to store when buffer full
    int skm_buf_pos[SKM_partitions];
    int i;
    for (i=0; i<SKM_partitions; i++) {
        skm_buffer[i] = new byte[SKM_BUFFER_SIZE];//
        skm_buf_pos[i] = 0;
        skm_cnt[i] = 0;
        kmer_cnt [i] = 0;
    }
    for (ReadPtr &_read: reads) {
        // HPC encoding:
        T_read_len len = _read.len;
        byte *read = new byte[len];//
        if (HPC) len = _hpc_encoding(len, _read.read, read);
        else memcpy(read, _read.read, len);
        // Gen SKM offs:
        T_read_len *skm_offs = new T_read_len[len];//
        T_minimizer *minimizers = new T_minimizer[len];//
        T_read_len n_skms = gen_skm_offs (read, len, K_kmer, P_minimizer, skm_offs, minimizers);
        // Read compression:
        read_compression (read, len);
        // Save to skm buffer or dump to skm store:
        gen_skms (read, len, skm_offs, minimizers, n_skms, K_kmer, SKM_partitions,
            SKM_BUFFER_SIZE, skm_cnt, kmer_cnt, skm_buffer, skm_buf_pos,
            skm_partition_stores);

        delete read;//
        delete skm_offs;//
        delete minimizers;//
    }
    
    // dump skm buffer to store
    for (i=0; i<SKM_partitions; i++)
        SKMStoreNoncon::save_skms(skm_partition_stores[i], skm_cnt[i], kmer_cnt[i], skm_buffer[i], skm_buf_pos[i], SKM_BUFFER_SIZE);//
        // delete skm_buffer[i]; // No need to delete here. The right is passed to func save_skms.
    #ifndef TEST
    logger->log("---- BATCH\tCPU:\t#Reads "+to_string(reads.size())+" ----");
    #endif
}

/**
 * Phase 2: bucket (MSD radix + quick) sort for counting
*/
int _get_histogram (T_kmer *kmers, T_skm_partsize n_kmers, int right_shift, const T_kmer base_mask, const int NB, _out_ size_t *offs, bool _debug_output = false) {
    right_shift = right_shift > 0 ? right_shift : 0;
    // complexity: n+NB
    size_t i;
    for (i=0; i<n_kmers; i++)
        offs[((kmers[i] >> right_shift) & base_mask) + 1]++;
    // if (_debug_output) {
    //     for (i=0; i<NB; i++) cout<<offs[i]/100<<"\t";
    //     cout<<endl;
    // }
    int occupied_bins = 0;
    for (i=0; i<NB; i++) {
        occupied_bins += offs[i+1]>0;
        offs[i+1] += offs[i];
    }
    return occupied_bins; // 如果occupied_bins小于一定数值，或者当前区域kmer数量小于比如10000，用qsort。
}
void _inplace_reorder (_out_ T_kmer *kmers, T_skm_partsize n_kmers, int right_shift, const T_kmer base_mask, const int NB, size_t *offs) {
    right_shift = right_shift > 0 ? right_shift : 0;
    // complexity: n
    size_t processed[NB];
    memset(processed, 0, sizeof(size_t)*NB);
    int bin, target_bin;
    size_t i = 0;
    T_kmer tmp;
    for (bin = 0; bin < NB; bin ++) {
        i += processed[bin];
        while (i < offs[bin+1]) {
            target_bin = (kmers[i] >> right_shift) & base_mask;
            assert(target_bin < NB);
            if (target_bin > bin) {
                // swap i offs[loc]+processed[loc]
                tmp = kmers[offs[target_bin]+processed[target_bin]];
                kmers[offs[target_bin]+processed[target_bin]] = kmers[i];
                kmers[i] = tmp;
                processed[target_bin] ++;
            } else {
                processed[bin] ++; // no need
                i ++;
            }
        }
    }
}
void _reorder (_out_ T_kmer *kmers, T_skm_partsize n_kmers, int right_shift, const T_kmer base_mask, const int NB, size_t *offs) {
    right_shift = right_shift > 0 ? right_shift : 0;
    T_kmer *swap = new T_kmer [n_kmers];
    memcpy(swap, kmers, n_kmers * sizeof(T_kmer));

    size_t i = 0;
    T_kmer tmp;
    int target_bin;
    size_t pos[NB];
    memset(pos, 0, sizeof(pos[0]) * NB);
    for (i=0; i<n_kmers; i++) {
        target_bin = (swap[i] >> right_shift) & base_mask;
        kmers[offs[target_bin]+pos[target_bin]] = swap[i];
        pos[target_bin]++;
    }
    delete [] swap;
}
void sort_kmers (T_kmer *kmers, const T_skm_partsize n_kmers, const T_kvalue K_kmer, int i_base = 0) {
    if (i_base >= K_kmer || n_kmers <= 1) return;
    if (n_kmers < 4096) {
        std::sort(kmers, kmers+n_kmers);
        return;
    }
    
    // radix sort:
    const int BP_NBASE = 5;
    int NB = 1 << (BP_NBASE*2); // 4 ^ BP_NBASE = NB
    T_kmer base_mask = (1 << (BP_NBASE*2)) - 1;
    int bin;

    size_t offs[NB+1];
    memset(offs, 0, sizeof(size_t)*(NB+1));

    int occupied_bins = _get_histogram(kmers, n_kmers, (K_kmer-i_base-BP_NBASE)*2, base_mask, NB, offs, i_base==0);
    // if (n_kmers >= 67108864) // 0x4000000, 512M for ull kmer, 1G for 128bit kmer, basically only the first round will call _inplace_reorder
    if (false)
        _inplace_reorder (kmers, n_kmers, (K_kmer-i_base-BP_NBASE)*2, base_mask, NB, offs);
    else // will malloc n_kmers*sizeof(T_kmer) tmp space for swapping
        _reorder (kmers, n_kmers, (K_kmer-i_base-BP_NBASE)*2, base_mask, NB, offs);

    for (bin = 0; bin < NB; bin ++) {
        sort_kmers (kmers+offs[bin], offs[bin+1]-offs[bin], K_kmer, i_base+BP_NBASE);
    }
    return;
}
inline T_kmer _gen_rc_kmer (T_kmer kmer, T_kvalue k) {
    T_kvalue i;
    T_kmer rc_kmer = 0;
    kmer = ~kmer;
    for (i = 0; i < k; i++) {
        rc_kmer = (rc_kmer<<2) | (kmer & 0b11);
        kmer >>= 2;
    }
    return rc_kmer;
}
void _extract_kmers_cpu (SKMStoreNoncon &skms_store, T_kvalue k, _out_ T_kmer* kmers) {
    // load skms
    byte* skms;
    skms = new byte[skms_store.tot_size_bytes];//
    size_t i;
    if (skms_store.to_file) {
        FILE* fp;
        fp = fopen(skms_store.filename.c_str(), "rb");
        assert(fp);
        assert(fread(skms, 1, skms_store.tot_size_bytes, fp)==skms_store.tot_size_bytes);
        fclose(fp);
    }
    else {
        size_t skm_store_pos = 0;
        for (i=0; i<skms_store.skm_chunk_bytes.size(); i++) {
            memcpy(skms+skm_store_pos, skms_store.skm_chunks[i], skms_store.skm_chunk_bytes[i]);
            skm_store_pos += skms_store.skm_chunk_bytes[i];
        }
        assert(skms_store.tot_size_bytes == skm_store_pos);
    }
    // extract kmers
    size_t kmer_store_pos = 0;
    T_kmer kmer, rc_kmer;
    T_kmer kmer_mask = (T_kmer)(TKMAX>>(sizeof(T_kmer)*8-k*2));
    bool new_kmer_required = true;
    byte j; // = 0,1,2
    byte indicator, tmp;
    byte selector[4] = {0, 0b00000011, 0b00001111, 0b00111111};
    
    for (i=0, j=0; i<skms_store.tot_size_bytes;) { // i: index of byte, j: which base (0,1,2) in the current byte
        if (new_kmer_required) {
            // generate the first (k-1)-mer in the skm
            kmer = 0;
            indicator = skms[i] >> 6; // indicator
            T_kvalue len = indicator;
            kmer |= skms[i] & selector[indicator]; // e.g., skms[i] & 0b1111
            i++;
            while (len <= k-3) {
                kmer <<= 6; // for 3 bases
                kmer |= skms[i] & 0b111111;
                i++;
                len+=3;
            }
            j = k-len-1; // -1 is okay
            kmer <<= (k-len)*2;
            kmer |= (skms[i] >> ((3-(k-len))*2)) & selector[k-len];
            rc_kmer = _gen_rc_kmer(kmer, k);
            kmers[kmer_store_pos++] = min(kmer, rc_kmer);
            // cerr<<" "<<kmer<<endl;
            new_kmer_required = false;
        } else if (j < skms[i]>>6) {
            kmer = (kmer << 2) & kmer_mask;
            tmp = (skms[i] >> (2-j)*2);
            kmer |= tmp & 0b11;
            rc_kmer >>= 2;
            rc_kmer |= (T_kmer((~tmp) & 0b11)) << (2*k-2);
            kmers[kmer_store_pos++] = min(kmer, rc_kmer);
            // cerr<<" "<<kmer<<endl;
        }
        
        j++;
        if (j >= skms[i]>>6) { // overflow
            if (j!=3) new_kmer_required = true;
            j=0, i++;
        }
    }
    delete [] skms;//
    // cerr<<skms_store.id<<" "<<kmer_store_pos<<" "<<skms_store.skm_cnt<<" "<<skms_store.tot_size_bytes<<"|"<<skms_store.kmer_cnt<<endl;
    assert(kmer_store_pos == skms_store.kmer_cnt);
    return;
}
size_t KmerCountingCPU(T_kvalue k,
    SKMStoreNoncon *skms_store,
    T_kmer_cnt kmer_min_freq, T_kmer_cnt kmer_max_freq,
    _out_ vector<T_kmc> &kmc_result_curpart, int tid) {
    
    if (skms_store->kmer_cnt == 0) return 0;
    
    WallClockTimer wct;
    
    T_kmer *kmers = new T_kmer[skms_store->kmer_cnt];//
    // WallClockTimer wct_e;
    _extract_kmers_cpu(*skms_store, k, kmers); // 12%
    // double t1= wct_e.stop();
    // WallClockTimer wct_s;
    sort_kmers(kmers, skms_store->kmer_cnt, k); // 88%
    // sort(kmers, kmers+skms_store->kmer_cnt);
    // double t2= wct_s.stop();
    
    size_t i, cur_cnt = 1;
    T_kmer cur_kmer = kmers[0];
    size_t distinct_kmers = 1;
    size_t validation_cnt = 0;

    // string outfile = "/mnt/f/study/bio_dataset/tmp/" + to_string(skms_store->id) + ".txt";
    // FILE *fp = fopen(outfile.c_str(), "w");

    for (i=1; i<skms_store->kmer_cnt; i++) {
        if (kmers[i] != kmers[i-1]) {
            distinct_kmers++;
            // [SAVE] cur_kmer: cur_cnt
            // cerr<<cur_kmer<<": "<<cur_cnt<<endl;
            // if (cur_cnt>1) fprintf(fp, "%llu %llu\n", cur_kmer, cur_cnt);
            validation_cnt += cur_cnt;
            cur_cnt = 0;
            cur_kmer = kmers[i];
        }
        cur_cnt++;
    }
    // [SAVE] cur_kmer: cur_cnt
    // cerr<<cur_kmer<<": "<<cur_cnt<<endl;
    // fprintf(fp, "%llu %llu\n", cur_kmer, cur_cnt);
    // fclose(fp);
    validation_cnt += cur_cnt;
    // cerr<<validation_cnt<<" == "<<skms_store->kmer_cnt<<endl;
    assert(validation_cnt == skms_store->kmer_cnt);
    delete [] kmers;//
    skms_store->clear_skm_data();
    delete skms_store;//

    logger->log("CPU  \t(T"+to_string(tid)+"):\tPart "+to_string(skms_store->id)+" "+to_string(skms_store->tot_size_bytes)+"|"+to_string(skms_store->kmer_cnt)+" "+to_string(distinct_kmers)+" \t"+to_string(wct.stop()), Logger::LV_DEBUG);

    return distinct_kmers;
}


#ifdef TEST
int main() {
    char a[100]="AAAAAAAAACCCCCCCAAAAAAATGA";
    T_read_len skm_offs[100];
    T_minimizer minimizers[100];
    int skm_cnt = gen_skm_offs((byte*)(a), strlen(a), 10, 5, skm_offs, minimizers);
    for (int i=0; i<=skm_cnt; i++) {
        printf("%d,", skm_offs[i]);
    }
    printf("\n");
    for (int i=0; i<skm_cnt; i++) {
        printf("%u,", minimizers[i]);
    }
    printf("\n");
    read_compression((byte*)a, strlen(a));


    // byte b[100];
    // memset(b,0,sizeof(b));
    // _hpc_encoding(strlen(a), a, b);
    // printf("%s\n",b);
    return 0;
}
#endif