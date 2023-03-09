#pragma once
#define _USE_RADULS_H
#include <cstdint>

namespace raduls
{
    using uchar = unsigned char;
	using int32 = int32_t;
	using uint32 = uint32_t;
	using int64 = int64_t;
	using uint64 = uint64_t;
    constexpr uint32 ALIGNMENT = 0x100;
	void CleanTmpArray (uint8_t* tmp, uint64_t n_recs, uint32_t rec_size, uint32_t n_threads);
    void RadixSortMSD (uint8_t* input, uint8_t* tmp, uint64_t n_recs, uint32_t rec_size, uint32_t key_size, uint32_t n_threads);
}