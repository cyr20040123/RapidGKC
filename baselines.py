import os
import time

FREE="freecache 115"
#SPEED="dd bs=1024k count=4096 if=/dev/zero of=~/tmp/test conv=fdatasync"

WRAPPER_F="{{ {TS} /usr/bin/time -v {PROG}; }} 2>&1 | cat >> ~/baselines/result/{LOGFILE}.txt"
PURERUN_F="/usr/bin/time -v {PROG}"
TASKSET_F="taskset {}"
LOGFILE_F="{METHOD}_{DATASET}_k{K}"

method_names = ["gerbil", "kmc", "kmc_disk", "chtkc", "chtkco"]
threads = [16,32,48,64]
kvalues = [16,28,55]
numa_set = ["0xFFFFFFFFFFFFFFFF", "0xFFFF0000FFFF", "0xFFFFFFFFFFFFFFFF"]
datasets = ["FV", "HG002", "NC"]

arg_for_data = {
    "FV":{"DATALIST":"Fvesca.lst", "DATAFOLDER":"~/biodataset/fvesca/", "AQ":"q", "MEM":"16"},
    "HG002":{"DATALIST":"HG002.lst", "DATAFOLDER":"~/biodataset/hg002/", "AQ":"a", "MEM":"50"},
    "NC":{"DATALIST":"HG002.lst", "DATAFOLDER":"~/biodataset/ncrassa/", "AQ":"a", "MEM":"100"}
}

PROGRAMS_F = {
    "kmc_disk": "kmc -k{K} -t{T} -ci1 -w -f{AQ} @{DATALIST} NA.res ~/tmp/kmc/",
    "kmc": "kmc -k{K} -t{T} -ci1 -w -r -f{AQ} @{DATALIST} NA.res ~/tmp/kmc/",
    "gerbil": "gerbil -i -t {T} -g -k {K} -l 8000 -o gerbil {DATAFOLDER} ~/tmp/gerbil/ ~/tmp/gerbil/output.txt",
    "chtkc": "chtkc count -k {K} -m {MEM}G -t {T} --rt=2 --f{AQ} -o ~/tmp/chtkc/result --filter-min=8000 {DATAFOLDER}*.fast{AQ}",
    "chtkco": "chtkco count -k {K} -m {MEM}G -t {T} --rt=2 --f{AQ} -o ~/tmp/chtkc/result --filter-min=8000 {DATAFOLDER}*.fast{AQ}"
}

def iospeed():
    cmd = "{ dd bs=1024k count=64 if=/dev/zero of=testspeed conv=fdatasync; } 2>&1 | cat > speed.tmp"
    os.system(cmd)
    fo = open('speed.tmp', 'r')
    ss = fo.read().split(' ')
    speed = int(ss[-2])
    fo.close()
    return speed

def overheating_check(thr=500):
    while(True):
        spd = iospeed()
        if (spd<thr):
            print("Speed =",spd,"<",thr,"MB/s, wait for 10 sec ...")
            time.sleep(10)
        else:
            break

def gen_command(method, dataset, numa, thread, kvalue, test_run=False):
    cmd = ""
    if (not test_run):
        cmd = WRAPPER_F.format(
            TS=TASKSET_F.format(numa_set[numa]),
            PROG=PROGRAMS_F[method].format(
                K=kvalue,
                T=thread,
                AQ=arg_for_data[dataset]['AQ'],
                DATALIST=arg_for_data[dataset]['DATALIST'],
                DATAFOLDER=arg_for_data[dataset]['DATAFOLDER'],
                MEM=arg_for_data[dataset]['MEM']
            ),
            LOGFILE=LOGFILE_F.format(METHOD=method, DATASET=dataset, K=kvalue)
        )
    else:
        cmd = PURERUN_F.format(
            PROGRAMS_F[method].format(
                K=kvalue,
                T=thread,
                AQ=arg_for_data[dataset]['AQ'],
                DATALIST=arg_for_data[dataset]['DATALIST'],
                DATAFOLDER=arg_for_data[dataset]['DATAFOLDER'],
                MEM=arg_for_data[dataset]['MEM']
            )
        )
    print(time.strftime('[%m-%d %H:%M:%S]',time.localtime(time.time())), method, dataset, numa, thread, kvalue, test_run)
    return cmd

def run(cmd):
    res = 0
    start_time = time.time()
    try:
        #res = os.system(cmd)
        print(cmd)
        pass
    except:
        print("ERROR WHEN RUNNING COMMAND:\n", cmd)
        exit(1)
    end_time = time.time()
    if res!=0 or end_time-start_time<4:
        print("ERROR WHEN RUNNING COMMAND:\n", cmd)
        exit(1)

def work(continue_dataset="", continue_method="", continue_k=0):
    do = False
    if (continue_k==0):
        do = True
    for dataset in datasets:
        print("\n\n======== Dataset",dataset,"========")
        for method in method_names:
            print("\n----",method,"----")
            run(FREE)
            for kvalue in kvalues:
                print("\nk =", kvalue)
                if (continue_dataset==dataset and continue_method==method and continue_k==kvalue):
                    do = True
                if (do):
                    #run(gen_command(method, dataset, 1, 16, 16, False))
                    run(gen_command(method, dataset, 2, 16, 16, False))
                    #run(gen_command(method, dataset, 1, 32, 16, False))
                    run(gen_command(method, dataset, 2, 32, 16, False))
                    overheating_check()
                    run(gen_command(method, dataset, 2, 48, 16, False))
                    overheating_check()
                    run(gen_command(method, dataset, 2, 64, 16, False))
                    overheating_check()

if __name__=='__main__':
    work()

'''
/usr/bin/time -v /data/ychengbu/baselines/chtkc/chtkc/build/chtkco count -k 28 -m 16G -t 48 --rt=6 --fq -o /ssddata/ychengbu/tmp/chtkc/result --filter-min=9999 /data/ychengbu/biodataset/Fvesca/*.fastq

CHTKC="""
{ taskset 0xFFFF0000FFFF /usr/bin/time -v /data/ychengbu/baselines/chtkc/chtkc/build/chtkco count -k 28 -m 16G -t 16 --rt=2 --fq -o /ssddata/ychengbu/tmp/chtkc/result --filter-min=9999 /data/ychengbu/biodataset/Fvesca/*.fastq; } 2>&1 | cat >> fvesca_chtkc_k28.txt
"""
KMC="""
{ taskset 0xFFFFFFFFFFFFFFFF /usr/bin/time -v kmc -k16 -t32 -ci1 -w -r -fa @HG002.txt NA.res /ssddata/ychengbu/tmp/kmctmp/; } 2>&1 | cat >> hg002_kmc_k16.txt
"""
KMC_DISK="""
{ taskset 0xFFFFFFFFFFFFFFFF /usr/bin/time -v kmc -k16 -t32 -ci1 -w -fa @HG002.txt NA.res /ssddata/ychengbu/tmp/kmctmp/; } 2>&1 | cat >> hg002_kmc_k16.txt
"""
GERBIL="""
{ taskset 0xFFFF0000FFFF /usr/bin/time -v /data/ychengbu/baselines/gerbil/build/gerbil -i -t 16 -g -k 16 -l 8000 -o gerbil /data/ychengbu/biodataset/Fvesca/ /ssddata/ychengbu/tmp/gerbil/ /data/ychengbu/tmp/output.txt; } 2>&1 | cat >> gerbil_fvesca_k16.txt
"""
'''
