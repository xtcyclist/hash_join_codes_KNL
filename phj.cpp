#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef _NO_VECTOR
#ifndef _NO_VECTOR_HASHING
#define _NO_VECTOR_HASHING
#endif
#ifndef _NO_VECTOR_PARTITIONING
#define _NO_VECTOR_PARTITIONING
#endif
#endif

#ifdef _NO_VECTOR_HASHING
#ifdef _NO_VECTOR_PARTITIONING
#ifndef _NO_VECTOR
#define _NO_VECTOR
#endif
#endif
#endif

#ifndef _NO_VECTOR
#include <immintrin.h>
#endif
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <sched.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "rand.h"
#include "sched.h"
#include "hj.h"
#include "malloc.h"
#define R 0
#include <hbwmalloc.h>
#ifndef NEXT_POW_2

int BUFFER_SIZE=64;
int PREFETCH_DISTANCE=0;
int precision = 100;
/** 
 *  compute the next number, greater than or equal to 32-bit unsigned v.
 *  taken from "bit twiddling hacks":
 *  http://graphics.stanford.edu/~seander/bithacks.html
 */
#define NEXT_POW_2(V)                           \
	do {                                        \
		V--;                                    \
		V |= V >> 1;                            \
		V |= V >> 2;                            \
		V |= V >> 4;                            \
		V |= V >> 8;                            \
		V |= V >> 16;                           \
		V++;                                    \
	} while(0)
#endif

typedef struct rand_state_64 {
	uint64_t num[313];
	size_t index;
} rand64_t;

rand64_t *rand64_init(uint64_t seed)
{
	rand64_t *state = (rand64_t*)malloc(sizeof(rand64_t));
	uint64_t *n = state->num;
	size_t i;
	n[0] = seed;
	for (i = 0 ; i != 311 ; ++i)
		n[i + 1] = 6364136223846793005ull *
			(n[i] ^ (n[i] >> 62)) + i + 1;
	state->index = 312;
	return state;
}

__m512i simd_hash(__m512i k, __m512i factors, __m512i Nbins)
{
	k = _mm512_mullo_epi32(k, factors);
	__m512i permute_2 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
	__m512i blend_0 = _mm512_set1_epi32(0);
	__mmask16 blend_interleave = _mm512_int2mask(21845);
	__m512i Nbins2 = _mm512_permutevar_epi32 (permute_2,Nbins);
	Nbins=_mm512_mask_blend_epi32(blend_interleave,blend_0,Nbins);
	Nbins2=_mm512_mask_blend_epi32(blend_interleave,blend_0,Nbins2);
	__m512i k2=_mm512_permutevar_epi32 (permute_2,k);
	k=_mm512_mask_blend_epi32(blend_interleave,blend_0,k);
	k2=_mm512_mask_blend_epi32(blend_interleave,blend_0,k2);
	k=_mm512_mul_epu32 (k,Nbins);
	k2=_mm512_mul_epu32 (k2,Nbins2);
	k=_mm512_permutevar_epi32 (permute_2,k);
	k=_mm512_mask_blend_epi32(blend_interleave,k2,k);
	return k;
}
__m512i simd_hash_ratio(__m512i k, __m512i factors_1, __m512i factors_2, size_t partitions, double ratio)
{
	//threads are equally divided to the DDR and the HBM
	//ratio is 0, 0.1, 0.2, ..., 0.9, 1
	int cut = (int)((1-ratio)*precision);
	//first level hasing
	__m512i mask_ratio_par = _mm512_set1_epi32(precision);
	__m512i hash1 = simd_hash (k, factors_1, mask_ratio_par);
	//second level hashing
	__m512i mask_cut = _mm512_set1_epi32(cut);
	__m512i mask_partitions = _mm512_set1_epi32(partitions/2);
	__mmask16 is_hbm = ~_mm512_cmp_epi32_mask(hash1, mask_cut, _MM_CMPINT_LT);
	__m512i hash2 = simd_hash (k, factors_2, mask_partitions);
	hash2 = _mm512_mask_add_epi32 (hash2, is_hbm, hash2, mask_partitions);
	return hash2;
}
__m512i _mm512_fmadd_epi32(__m512i a, __m512i b, __m512i c)
{
	__m512i temp=_mm512_mullo_epi32(a,b);
	temp=_mm512_add_epi32 (temp,c);
	return temp;
}

uint64_t rand64_next(rand64_t *state)
{
	uint64_t x, *n = state->num;
	if (state->index == 312) {
		size_t i = 0;
		do {
			x = n[i] & 0xffffffff80000000ull;
			x |= n[i + 1] & 0x7fffffffull;
			n[i] = n[i + 156] ^ (x >> 1);
			n[i] ^= 0xb5026f5aa96619e9ull & -(x & 1);
		} while (++i != 156);
		n[312] = n[0];
		do {
			x = n[i] & 0xffffffff80000000ull;
			x |= n[i + 1] & 0x7fffffffull;
			n[i] = n[i - 156] ^ (x >> 1);
			n[i] ^= 0xb5026f5aa96619e9ull & -(x & 1);
		} while (++i != 312);
		state->index = 0;
	}
	x = n[state->index++];
	x ^= (x >> 29) & 0x5555555555555555ull;
	x ^= (x << 17) & 0x71d67fffeda60000ull;
	x ^= (x << 37) & 0xfff7eee000000000ull;
	x ^= (x >> 43);
	return x;
}

typedef struct rand_state_32 {
	uint32_t num[625];
	size_t index;
} rand32_t;

rand32_t *rand32_init(uint32_t seed)
{
	rand32_t *state = (rand32_t*)malloc(sizeof(rand32_t));
	uint32_t *n = state->num;
	size_t i;
	n[0] = seed;
	for (i = 0 ; i != 623 ; ++i)
		n[i + 1] = 0x6c078965 * (n[i] ^ (n[i] >> 30));
	state->index = 624;
	return state;
}

uint32_t rand32_next(rand32_t *state)
{
	uint32_t y, *n = state->num;
	if (state->index == 624) {
		size_t i = 0;
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i + 397] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 227);
		n[624] = n[0];
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i - 227] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 624);
		state->index = 0;
	}
	y = n[state->index++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680;
	y ^= (y << 15) & 0xefc60000;
	y ^= (y >> 18);
	return y;
}


uint64_t thread_time(void)
{
	struct timespec t;
	//assert(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0);
	return t.tv_sec * 1000 * 1000 * 1000 + t.tv_nsec;
}

uint64_t real_time(void)
{
	struct timespec t;
	//assert(clock_gettime(CLOCK_REALTIME, &t) == 0);
	return t.tv_sec * 1000 * 1000 * 1000 + t.tv_nsec;
}
double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
int hardware_threads(void)
{
	char name[64];
	struct stat st;
	int threads = -1;
	do {
		sprintf(name, "/sys/devices/system/cpu/cpu%d", ++threads);
	} while (stat(name, &st) == 0);
	return threads;
}

void bind_thread(int thread, int threads)
{
	size_t size = CPU_ALLOC_SIZE(threads);
	cpu_set_t *cpu_set = CPU_ALLOC(threads);
	//assert(cpu_set != NULL);
	CPU_ZERO_S(size, cpu_set);
	CPU_SET_S(thread, size, cpu_set);
	//assert(pthread_setaffinity_np(pthread_self(), size, cpu_set) == 0);
	CPU_FREE(cpu_set);
}
void *hma_malloc(size_t size, bool type)
{
	if (type) //ddr
	{
		return malloc(size);
	}
	else
	{
		return hbw_malloc(size);
	}
}
void hma_free(void *mem, bool type)
{
	if (type)
	{
		free(mem);
	}
	else
	{
		hbw_free(mem);
	}
}
void *mamalloc(size_t size)
{
	void *ptr = NULL;
	ptr=memalign(64, size);// ? NULL : ptr;
	return ptr;
}

void *align(const void *p)
{
	size_t i = 63 & (size_t) p;
	return (void*) (i ? p + 64 - i : p);
}

int power_of_2(uint64_t x)
{
	return x > 0 && (x & (x - 1)) == 0;
}

int odd_prime(uint64_t x)
{
	uint64_t d;
	for (d = 3 ; d * d <= x ; d += 2)
		if (x % d == 0) return 0;
	return 1;
}

#ifndef _NO_VECTOR_ASHING

void set(uint64_t *dst, size_t size, uint32_t value)
{
	uint64_t *dst_end = &dst[size];
	uint64_t *dst_aligned = (uint64_t *)align(dst);
	__m512i x = _mm512_set1_epi64(value);
	while (dst != dst_end && dst != dst_aligned)
		*dst++ = value;
	dst_aligned = &dst[(dst_end - dst) & ~7];
	while (dst != dst_aligned) {
		_mm512_store_epi64(dst, x);
		dst += 8;
	}
	while (dst != dst_end)
		*dst++ = value;
}

void build(const uint32_t *keys, const uint32_t *vals, size_t size,
		uint64_t *table, size_t buckets,
		const uint32_t factor[2], uint32_t empty)
{
	set(table, buckets, empty);
	// set constants
	__m512i mask_1 = _mm512_set1_epi32(1);
	__m512i mask_empty = _mm512_set1_epi32(empty);
	__m512i mask_factor_1 = _mm512_set1_epi32(factor[0]);
	__m512i mask_factor_2 = _mm512_set1_epi32(factor[1]);
	__m512i mask_buckets = _mm512_set1_epi32(buckets);
	__m512i mask_buckets_minus_1 = _mm512_set1_epi32(buckets - 1);
	__m512i mask_pack = _mm512_set_epi32(15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0);
	__mmask16 blend_0000 = _mm512_int2mask(0x0000);
	__mmask16 blend_AAAA = _mm512_int2mask(0xAAAA);
	__mmask16 blend_5555 = _mm512_int2mask(0x5555);
	// main loop
	size_t i = 0;
	size_t size_minus_16 = size - 16;
	__mmask16 k = _mm512_kxnor(k, k);
	__m512i key, val, off;

	if (size >= 16) do {
		// replace invalid keys & payloads
		//key = (key, k, &keys[i]);
		//key = _mm512_mask_loadunpackhi_epi32(key, k, &keys[i + 16]);
		key = _mm512_mask_expandloadu_epi32  (key, k, &keys[i]);
		val = _mm512_mask_expandloadu_epi32  (val, k, &vals[i]); 
		//val = _mm512_mask_loadunpacklo_epi32(val, k, &vals[i]);
		//val = _mm512_mask_loadunpackhi_epi32(val, k, &vals[i + 16]);
		// update 
		off = _mm512_mask_xor_epi32(off, k, off, off);
		i += _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
		// hash keys
		__m512i factors = _mm512_mask_blend_epi32(k, mask_factor_2, mask_factor_1);
		__m512i buckets = _mm512_mask_blend_epi32(k, mask_buckets_minus_1, mask_buckets);
		__m512i hash = simd_hash(key, factors, buckets);
		// combine with old offset and fix overflows
		off = _mm512_add_epi32(off, hash);
		k = _mm512_cmpge_epu32_mask(off, mask_buckets);
		off = _mm512_mask_sub_epi32(off, k, off, mask_buckets);
		// load keys from table and detect conflicts
		__m512i tab = _mm512_i32gather_epi32(off, table, 8);
		k = _mm512_cmpeq_epi32_mask(tab, mask_empty);
		_mm512_mask_i32scatter_epi32(table, k, off, mask_pack, 8);
		tab = _mm512_mask_i32gather_epi32(tab, k, off, table, 8);
		k = _mm512_mask_cmpeq_epi32_mask(k, tab, mask_pack);
		// mix keys and payloads in pairs
		__m512i key_tmp = _mm512_permutevar_epi32(mask_pack, key);
		__m512i val_tmp = _mm512_permutevar_epi32(mask_pack, val);
		__m512i lo = _mm512_mask_blend_epi32(blend_AAAA, key_tmp, _mm512_swizzle_epi32(val_tmp, _MM_SWIZ_REG_CDAB));
		__m512i hi = _mm512_mask_blend_epi32(blend_5555, val_tmp, _mm512_swizzle_epi32(key_tmp, _MM_SWIZ_REG_CDAB));
		// store valid pairs
		_mm512_mask_i32loscatter_epi64(table, k, off, lo, 8);
		__mmask16 rev_k = _mm512_kunpackb (k,k>>8);
		__m512i rev_off = _mm512_permute4f128_epi32(off, _MM_PERM_BADC);
		_mm512_mask_i32loscatter_epi64(table, rev_k, rev_off, hi, 8);
		off = _mm512_add_epi32(off, mask_1);
	} while (i <= size_minus_16);
	// save last items
	uint32_t keys_last[32];
	uint32_t vals_last[32];
	k = _mm512_knot(k);
	_mm512_mask_compressstoreu_epi32(&keys_last[0],  k, key);
	_mm512_mask_compressstoreu_epi32(&vals_last[0],  k, val);
	//_mm512_mask_packstorelo_epi32(&keys_last[0],  k, key);
	//_mm512_mask_packstorehi_epi32(&keys_last[16], k, key);
	//_mm512_mask_packstorelo_epi32(&vals_last[0],  k, val);
	//_mm512_mask_packstorehi_epi32(&vals_last[16], k, val);
	size_t j = _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
	for (; i != size ; ++i, ++j) {
		keys_last[j] = keys[i];
		vals_last[j] = vals[i];
	}
	// process last items in scalar code
	for (i = 0 ; i != j ; ++i) {
		uint32_t k = keys_last[i];
		uint64_t p = vals_last[i];
		p = (p << 32) | k;
		uint64_t h1 = (uint32_t) (k * factor[0]);
		uint64_t h2 = (uint32_t) (k * factor[1]);
		h1 = (h1 * buckets) >> 32;
		h2 = ((h2 * (buckets - 1)) >> 32) + 1;
		while (empty != (uint32_t) table[h1]) {
			h1 += h2;
			if (h1 >= buckets)
				h1 -= buckets;
		}
		table[h1] = p;
	}
}

size_t probe(const uint32_t *keys, const uint32_t *vals, size_t size,
		const uint64_t *table, size_t buckets,
		const uint32_t factor[2], uint32_t empty,
		uint32_t *keys_buf, uint32_t *vals_buf, uint32_t *tabs_buf,
		uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out,
		size_t offset, size_t buffer_size, size_t block_size,
		size_t block_limit, volatile size_t *counter, int flush)
{
	// generate masks
	__m512i mask_1 = _mm512_set1_epi32(1);
	__m512i mask_empty = _mm512_set1_epi32(empty);
	__m512i mask_factor_1 = _mm512_set1_epi32(factor[0]);
	__m512i mask_factor_2 = _mm512_set1_epi32(factor[1]);
	__m512i mask_buckets = _mm512_set1_epi32(buckets);
	__m512i mask_buckets_minus_1 = _mm512_set1_epi32(buckets - 1);
	__m512i mask_unpack = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
	__mmask16 blend_0000 = _mm512_int2mask(0x0000);
	__mmask16 blend_AAAA = _mm512_int2mask(0xAAAA);
	__mmask16 blend_5555 = _mm512_int2mask(0x5555);
	// main loop
	size_t b, i = 0;
	size_t j = offset &  (buffer_size - 1);
	size_t o = offset & ~(buffer_size - 1);
	const size_t size_vec = size - 16;
	__mmask16 k = _mm512_kxnor(k, k);
	__m512i key, val, off;
	if (size >= 16) do {
		// replace invalid keys & payloads
		key = _mm512_mask_expandloadu_epi32 (key, k, &keys[i]);
		val = _mm512_mask_expandloadu_epi32 (val, k, &vals[i]);
		//key = _mm512_mask_loadunpacklo_epi32(key, k, &keys[i]);
		//key = _mm512_mask_loadunpackhi_epi32(key, k, &keys[i + 16]);
		//val = _mm512_mask_loadunpacklo_epi32(val, k, &vals[i]);
		//val = _mm512_mask_loadunpackhi_epi32(val, k, &vals[i + 16]);
		// update offsets
		off = _mm512_mask_xor_epi32(off, k, off, off);
		i += _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
		// hash keys using either 1st or 2nd function
		__m512i factors = _mm512_mask_blend_epi32(k, mask_factor_2, mask_factor_1);
		__m512i buckets = _mm512_mask_blend_epi32(k, mask_buckets_minus_1, mask_buckets);
		//__m512i hash = _mm512_mullo_epi32(key, factors);
		//hash = _mm512_mulhi_epu16(hash, buckets);
		__m512i hash = simd_hash(key, factors, buckets);
		// combine with old offset and fix overflows
		off = _mm512_add_epi32(off, hash);
		k = _mm512_cmpge_epu32_mask(off, mask_buckets);
		off = _mm512_mask_sub_epi32(off, k, off, mask_buckets);
		// load keys from table and update offsets
		__m512i lo = _mm512_i32logather_epi64(off, table, 8);
		__m512i rev = _mm512_permute4f128_epi32(off, _MM_PERM_BADC);
		__m512i hi = _mm512_i32logather_epi64(rev, table, 8);
		off = _mm512_add_epi32(off, mask_1);
		// split keys and payloads
		__m512i tab_key = _mm512_mask_blend_epi32(blend_AAAA, lo, _mm512_swizzle_epi32(hi, _MM_SWIZ_REG_CDAB));
		__m512i tab_val = _mm512_mask_blend_epi32(blend_5555, hi, _mm512_swizzle_epi32(lo, _MM_SWIZ_REG_CDAB));
		tab_key = _mm512_permutevar_epi32(mask_unpack, tab_key);
		tab_val = _mm512_permutevar_epi32(mask_unpack, tab_val);
		// compare
		__mmask16 m = _mm512_cmpeq_epi32_mask(tab_key, key);
		k = _mm512_cmpeq_epi32_mask(tab_key, mask_empty);
#ifdef _UNIQUE
		k = _mm512_kor(k, m);
#endif
		// partitions_aligned store matches
		_mm512_mask_compressstoreu_epi32(&keys_buf[j +  0], m, key);
		_mm512_mask_compressstoreu_epi32(&vals_buf[j +  0], m, val);
		_mm512_mask_compressstoreu_epi32(&tabs_buf[j +  0], m, tab_val);
		//_mm512_mask_packstorelo_epi32(&keys_buf[j +  0], m, key);
		//_mm512_mask_packstorehi_epi32(&keys_buf[j + 16], m, key);
		//_mm512_mask_packstorelo_epi32(&vals_buf[j +  0], m, val);
		//_mm512_mask_packstorehi_epi32(&vals_buf[j + 16], m, val);
		//_mm512_mask_packstorelo_epi32(&tabs_buf[j +  0], m, tab_val);
		//_mm512_mask_packstorehi_epi32(&tabs_buf[j + 16], m, tab_val);
		j += _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, m));
		if (j >= buffer_size) {
			for (b = 0 ; b != buffer_size ; b += 16, o += 16) {
				__m512 x = _mm512_load_ps(&keys_buf[b]);
				__m512 y = _mm512_load_ps(&vals_buf[b]);
				__m512 z = _mm512_load_ps(&tabs_buf[b]);
				_mm512_stream_ps (&keys_out[o], x);
				_mm512_stream_ps (&vals_out[o], y);
				_mm512_stream_ps (&tabs_out[o], z);
			}
			j -= buffer_size;
			if (j) {
				__m512i x = _mm512_load_epi32(&keys_buf[b]);
				__m512i y = _mm512_load_epi32(&vals_buf[b]);
				__m512i z = _mm512_load_epi32(&tabs_buf[b]);
				_mm512_store_epi32 (keys_buf, x);
				_mm512_store_epi32 (vals_buf, y);
				_mm512_store_epi32 (tabs_buf, z);
			}
			if ((o & (block_size - 1)) == 0) {
				o = __sync_fetch_and_add(counter, 1);
				assert(o <= block_limit);
				o *= block_size;
			}
		}
	} while (i <= size_vec);
	off = _mm512_sub_epi32(off, mask_1);
	// save last items
	uint32_t keys_last[32];
	uint32_t vals_last[32];
	uint32_t offs_last[32];
	k = _mm512_knot(k);
	_mm512_mask_compressstoreu_epi32 (&keys_last[0],  k, key);
	_mm512_mask_compressstoreu_epi32 (&vals_last[0],  k, val);
	_mm512_mask_compressstoreu_epi32 (&offs_last[0],  k, off);
	//_mm512_mask_packstorelo_epi32(&keys_last[0],  k, key);
	//_mm512_mask_packstorehi_epi32(&keys_last[16], k, key);
	//_mm512_mask_packstorelo_epi32(&vals_last[0],  k, val);
	//_mm512_mask_packstorehi_epi32(&vals_last[16], k, val);
	//_mm512_mask_packstorelo_epi32(&offs_last[0],  k, off);
	//_mm512_mask_packstorehi_epi32(&offs_last[16], k, off);
	size_t l = _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
	for (; i != size ; ++i, ++l) {
		keys_last[l] = keys[i];
		vals_last[l] = vals[i];
		offs_last[l] = buckets;
	}
	// process last items in scalar code
	uint32_t factor_1 = factor[0];
	uint32_t factor_2 = factor[1];
	for (i = 0 ; i != l ; ++i) {
		uint32_t key = keys_last[i];
		uint32_t val = vals_last[i];
		uint64_t h1 = offs_last[i];
		uint64_t h2 = (uint32_t) (key * factor_2);
		h2 = ((h2 * (buckets - 1)) >> 32) + 1;
		if (h1 == buckets) {
			h1 = (uint32_t) (key * factor_1);
			h1 = (h1 * buckets) >> 32;
		} else {
			h1 += h2;
			if (h1 >= buckets)
				h1 -= buckets;
		}
		uint64_t tab = table[h1];
		while (empty != (uint32_t) tab) {
			if (key == (uint32_t) tab) {
				keys_buf[j] = key;
				vals_buf[j] = val;
				tabs_buf[j] = tab >> 32;
				if (++j == buffer_size) {
					for (j = b = 0 ; b != buffer_size ; b += 16, o += 16) {
						__m512 x = _mm512_load_ps(&keys_buf[b]);
						__m512 y = _mm512_load_ps(&vals_buf[b]);
						__m512 z = _mm512_load_ps(&tabs_buf[b]);
						_mm512_stream_ps(&keys_out[o], x);
						_mm512_stream_ps(&vals_out[o], y);
						_mm512_stream_ps(&tabs_out[o], z);
					}
					if ((o & (block_size - 1)) == 0) {
						o = __sync_fetch_and_add(counter, 1);
						assert(o <= block_limit);
						o *= block_size;
					}
				}
			}
			h1 += h2;
			if (h1 >= buckets)
				h1 -= buckets;
			tab = table[h1];
		}
	}
	if (!flush) o += j;
	else for (b = 0 ; b != j ; ++b, ++o) {
		keys_out[o] = keys_buf[b];
		vals_out[o] = vals_buf[b];
		tabs_out[o] = tabs_buf[b];
	}
	return o;
}

#else


#endif
void build_s(const uint32_t *keys, const uint32_t *vals, size_t size,
		uint64_t *table, size_t buckets,
		const uint32_t factor[2], uint32_t empty)
{
	size_t i;
	uint32_t factor_1 = factor[0];
	uint32_t factor_2 = factor[1];
	for (i = 0 ; i != buckets ; ++i)
		table[i] = empty;
	for (i = 0 ; i != size ; ++i) {
		uint32_t k = keys[i];
		uint64_t p = vals[i];
		p = (p << 32) | k;
		uint64_t h1 = (uint32_t) (k * factor_1);
		h1 = (h1 * buckets) >> 32;
		if (empty != (uint32_t) table[h1]) {
			uint64_t h2 = (uint32_t) (k * factor_2);
			h2 = ((h2 * (buckets - 1)) >> 32) + 1;
			do {
				h1 += h2;
				if (h1 >= buckets)
					h1 -= buckets;
			} while (empty != (uint32_t) table[h1]);
		}
		table[h1] = p;
	}
}

size_t probe_s(const uint32_t *keys, const uint32_t *vals, size_t size,
		const uint64_t *table, size_t buckets,
		const uint32_t factor[2], uint32_t empty,
		uint32_t *keys_buf, uint32_t *vals_buf, uint32_t *tabs_buf,
		uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out,
		size_t offset, size_t buffer_size, size_t block_size,
		size_t block_limit, volatile size_t *counter, int flush)
{
	size_t i, o = offset;
	uint32_t factor_1 = factor[0];
	uint32_t factor_2 = factor[1];
	for (i = 0 ; i != size ; ++i) {
		uint32_t k = keys[i];
		uint32_t v = vals[i];
		uint64_t h1 = (uint32_t) k * factor_1;
		h1 = (h1 * buckets) >> 32;
		uint64_t t = table[h1];
		if (empty != (uint32_t) t) {
			uint64_t h2 = (uint32_t) k * factor_2;
			h2 = ((h2 * (buckets - 1)) >> 32) + 1;
			do {
				if (k == (uint32_t) t) {
					tabs_out[o] = t >> 32;
					vals_out[o] = v;
					keys_out[o] = k;
					if ((++o & (block_size - 1)) == 0) {
						o = __sync_fetch_and_add(counter, 1);
						assert(o <= block_limit);
						o *= block_size;
					}
#ifdef _UNIQUE
					break;
#endif
				}
				h1 += h2;
				if (h1 >= buckets)
					h1 -= buckets;
				t = table[h1];
			} while (empty != (uint32_t) t);
		}
	}
	return o;
}
void flush_shared(const uint32_t *counts, const uint32_t *offsets, const uint64_t *buffers,
		uint32_t *keys_out, uint32_t *vals_out, uint32_t *keys_out_hbm, uint32_t *vals_out_hbm, double ratio, size_t partitions)
{
	size_t p;
	uint32_t *current_keys_out, *current_vals_out;
	for (p = 0 ; p != partitions ; ++p) {
		current_keys_out=p<partitions/2?keys_out:keys_out_hbm;
		current_vals_out=p<partitions/2?vals_out:vals_out_hbm;
		const uint64_t *buf = &buffers[p * BUFFER_SIZE];
		size_t o = offsets[p];
		size_t c = counts[p];
		size_t e = o & (BUFFER_SIZE-1);
		size_t b = e > c ? e - c : 0;
		o -= e - b;
		while (b != e) {
			uint64_t key_val = buf[b++];
			current_keys_out[o] = key_val;
			current_vals_out[o] = key_val >> 32;
			o++;
		}
	}
}
void flush(const uint32_t *counts, const uint32_t *offsets, const uint64_t *buffers,
		uint32_t *keys_out, uint32_t *vals_out, size_t partitions)
{
	size_t p;
	for (p = 0 ; p != partitions ; ++p) {
		const uint64_t *buf = &buffers[p * BUFFER_SIZE];
		size_t o = offsets[p];
		size_t c = counts[p];
		size_t e = o & (BUFFER_SIZE-1);
		size_t b = e > c ? e - c : 0;
		o -= e - b;
		while (b != e) {
			uint64_t key_val = buf[b++];
			keys_out[o] = key_val;
			vals_out[o] = key_val >> 32;
			o++;
		}
	}
}


#ifndef _NO_VECTOR_PARTITIONING

void histogram(const uint32_t *keys, size_t size, uint32_t *counts,
		uint32_t factor, size_t partitions)
{
	// partition vector space
	uint32_t parts_space[31];
	uint32_t *parts = (uint32_t *)align(parts_space);
	// create masks
	__mmask16 blend_0 = _mm512_int2mask(0);
	__m512i mask_0 = _mm512_set1_epi32(0);
	__m512i mask_1 = _mm512_set1_epi32(1);
	__m512i mask_16 = _mm512_set1_epi32(16);
	__m512i mask_255 = _mm512_set1_epi32(255);
	__m512i mask_factor = _mm512_set1_epi32(factor);
	__m512i mask_partitions = _mm512_set1_epi32(partitions);
	__m512i mask_lanes = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	// reset counts
	size_t p, partitions_x16 = partitions << 4;
	uint32_t all_counts_space[partitions_x16 + 127];
	uint32_t *all_counts = (uint32_t *)align(all_counts_space);
	for (p = 0 ; p < partitions_x16 ; p += 16)
		_mm512_store_epi32(&all_counts[p], mask_0);
	for (p = 0 ; p != partitions ; ++p)
		counts[p] = 0;
	// before alignment
	const uint32_t *keys_end = &keys[size];
	const uint32_t *keys_aligned = (uint32_t *)align(keys);
	while (keys != keys_end && keys != keys_aligned) {
		uint32_t key = *keys++;
		p = (uint32_t) (key * factor);
		p = (p * partitions) >> 32;
		counts[p]++;
	}

	// aligned
	keys_aligned = &keys[(keys_end - keys) & -16];
	while (keys != keys_aligned) {
		//printf ("align\n");
		__m512i key = _mm512_load_epi32(keys);
		keys += 16;
		//__m512i part = _mm512_mullo_epi32(key, mask_factor);
		//part = _mm512_mulhi_epi32(part, mask_partitions);
		__m512i part = simd_hash(key, mask_factor, mask_partitions);
		__m512i part_lanes = _mm512_fmadd_epi32(part, mask_16, mask_lanes);
		__m512i count = _mm512_i32gather_epi32(part_lanes, all_counts, 4);
		__mmask16 k = _mm512_cmpeq_epi32_mask(count, mask_255);
		count = _mm512_add_epi32(count, mask_1);
		count = _mm512_and_epi32(count, mask_255);
		_mm512_i32scatter_epi32(all_counts, part_lanes, count, 4);
		if (!_mm512_kortestz(k, k)) {
			_mm512_store_epi32(parts, part);
			size_t mask = _mm512_kconcatlo_64(blend_0, k);
			size_t b = _mm_tzcnt_64(mask);
			do {
				p = parts[b];
				counts[p] += 256;
				//b = _mm_tzcnti_64(b, mask);
				mask=mask&(~(1<<b));
				b = _mm_tzcnt_64(mask);
			} while (b != 64);
		}
	}
	// after alignment
	while (keys != keys_end) {
		uint32_t key = *keys++;
		p = (uint32_t) (key * factor);
		p = (p * partitions) >> 32;
		counts[p]++;
	}
	// merge counts
	for (p = 0 ; p != partitions ; ++p) {
		__m512i sum = _mm512_load_epi32(&all_counts[p << 4]);
		counts[p] += _mm512_reduce_add_epi32(sum);
	}
#ifdef BG
	size_t i;
	for (i = p = 0 ; p != partitions ; ++p)
		i += counts[p];
	//assert(i == size);
#endif
}
void histogram_shared(const uint32_t *keys, size_t size, uint32_t *counts,
		uint32_t factor, uint32_t factor2, size_t partitions, double ratio)
{
	// partition vector space
	uint32_t parts_space[31];
	uint32_t *parts = (uint32_t *)align(parts_space);
	// create masks
	__mmask16 blend_0 = _mm512_int2mask(0);
	__m512i mask_0 = _mm512_set1_epi32(0);
	__m512i mask_1 = _mm512_set1_epi32(1);
	__m512i mask_16 = _mm512_set1_epi32(16);
	__m512i mask_255 = _mm512_set1_epi32(255);
	__m512i mask_factor = _mm512_set1_epi32(factor);
	__m512i mask_factor_2 = _mm512_set1_epi32(factor2);
	__m512i mask_partitions = _mm512_set1_epi32(partitions);
	__m512i mask_lanes = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	// reset counts
	size_t p, partitions_x16 = partitions << 4;
	uint32_t all_counts_space[partitions_x16 + 127];
	uint32_t *all_counts = (uint32_t *)align(all_counts_space);
	for (p = 0 ; p < partitions_x16 ; p += 16)
		_mm512_store_epi32(&all_counts[p], mask_0);
	for (p = 0 ; p != partitions ; ++p)
		counts[p] = 0;
	// before alignment
	const uint32_t *keys_end = &keys[size];
	const uint32_t *keys_aligned = (uint32_t *)align(keys);
	while (keys != keys_end && keys != keys_aligned) {
		uint32_t key = *keys++;
		p = (uint32_t) (key * factor);
		int cut = (int)((1-ratio)*precision);
		p = (p * precision) >> 32;
		size_t p2 = (uint32_t) (key * factor2);
		p2 = (p2 * partitions/2) >> 32;
		if (p>=cut)
		{
			p=p2+partitions/2;
		}
		else
		{
			p=p2;
		}
		counts[p]++;
	}

	// aligned
	keys_aligned = &keys[(keys_end - keys) & -16];
	while (keys != keys_aligned) {
		//printf ("align\n");
		__m512i key = _mm512_load_epi32(keys);
		keys += 16;
		//__m512i part = _mm512_mullo_epi32(key, mask_factor);
		//part = _mm512_mulhi_epi32(part, mask_partitions);
		//__m512i part = simd_hash(key, mask_factor, mask_partitions);
		__m512i part = simd_hash_ratio(key, mask_factor, mask_factor_2, partitions, ratio);
		__m512i part_lanes = _mm512_fmadd_epi32(part, mask_16, mask_lanes);
		__m512i count = _mm512_i32gather_epi32(part_lanes, all_counts, 4);
		__mmask16 k = _mm512_cmpeq_epi32_mask(count, mask_255);
		count = _mm512_add_epi32(count, mask_1);
		count = _mm512_and_epi32(count, mask_255);
		_mm512_i32scatter_epi32(all_counts, part_lanes, count, 4);
		if (!_mm512_kortestz(k, k)) {
			_mm512_store_epi32(parts, part);
			size_t mask = _mm512_kconcatlo_64(blend_0, k);
			size_t b = _mm_tzcnt_64(mask);
			do {
				p = parts[b];
				counts[p] += 256;
				//b = _mm_tzcnti_64(b, mask);
				mask=mask&(~(1<<b));
				b = _mm_tzcnt_64(mask);
			} while (b != 64);
		}
	}
	// after alignment
	while (keys != keys_end) {
		uint32_t key = *keys++;
		p = (uint32_t) (key * factor);
		int cut = (int)((1-ratio)*precision);
		p = (p * precision) >> 32;
		size_t p2 = (uint32_t) (key * factor2);
		p2 = (p2 * partitions/2) >> 32;
		if (p>=cut)
		{
			p=p2+partitions/2;
		}
		else
		{
			p=p2;
		}
		counts[p]++;
	}
	// merge counts
	for (p = 0 ; p != partitions ; ++p) {
		__m512i sum = _mm512_load_epi32(&all_counts[p << 4]);
		counts[p] += _mm512_reduce_add_epi32(sum);
	}
#ifdef BG
	size_t i;
	for (i = p = 0 ; p != partitions ; ++p)
		i += counts[p];
	//assert(i == size);
#endif
}
void partition_shared(const uint32_t *keys, const uint32_t *vals, size_t size,
		uint32_t *offsets, uint64_t *buffers, uint32_t *keys_out, 
		uint32_t *vals_out, uint32_t *keys_out_hbm, uint32_t *vals_out_hbm, double ratio, uint32_t factor, uint32_t factor2, size_t partitions)
{
	size_t i, p;
	size_t partitions_ddr = partitions / 2 ;//* (1-ratio);
	uint32_t *current_keys_out, *current_vals_out;
	assert(partitions > 1);
	assert(keys_out == align(keys_out));
	assert(vals_out == align(vals_out));
	// conflict space
	uint32_t conflicts_space[partitions + 15];
	uint32_t *conflicts = (uint32_t *)align(conflicts_space);
	// partition vector space
	uint32_t parts_space[31];
	uint32_t *parts = (uint32_t *)align(parts_space);
	// generate masks
	__m512i mask_factor = _mm512_set1_epi32(factor);
	__m512i mask_factor_2 = _mm512_set1_epi32(factor2);
	__m512i mask_partitions = _mm512_set1_epi32(partitions);
	__m512i mask_pack = _mm512_set_epi32(15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0);
	__m512i mask_unpack = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
	__m512i mask_16 = _mm512_set1_epi32(BUFFER_SIZE);
	__m512i mask_15 = _mm512_set1_epi32(BUFFER_SIZE-1);
	__m512i mask_1 = _mm512_set1_epi32(1);
	__mmask16 blend_AAAA = _mm512_int2mask(0xAAAA);
	__mmask16 blend_5555 = _mm512_int2mask(0x5555);
	__mmask16 blend_0000 = _mm512_int2mask(0x0000);
	// unaligned loop
	size_t size_minus_16 = size - 16;
	__mmask16 k = _mm512_kxnor(k, k);
	__m512i key, val;
	i = 0;
	if (size >= 16) do {
		// load new keys and payloads
		key = _mm512_mask_expandloadu_epi32 (key, k, &keys[i]);
		val = _mm512_mask_expandloadu_epi32 (val, k, &vals[i]);
		//_mm_prefetch((char*)&keys[i+16*PREFETCH_DISTANCE], _MM_HINT_T0);
		//_mm_prefetch((char*)&vals[i+16*PREFETCH_DISTANCE], _MM_HINT_T0);
		// hash keys
		//__m512i part = simd_hash(key, mask_factor, mask_partitions);//= _mm512_mullo_epi32(key, mask_factor);
		__m512i part = simd_hash_ratio(key, mask_factor, mask_factor_2, partitions, ratio);//  mask_factor_2
		// detect conflicts
		_mm512_i32scatter_epi32(conflicts, part, mask_pack, 4);
		__m512i back = _mm512_i32gather_epi32(part, conflicts, 4);
		size_t c = _mm512_kconcatlo_64(blend_0000, k);
		k = _mm512_cmpeq_epi32_mask(back, mask_pack);//1, if equal
		// split the data into low and high part
		__m512i key_tmp = _mm512_permutevar_epi32(mask_pack, key);//shuffle across lanes according to the index
		__m512i val_tmp = _mm512_permutevar_epi32(mask_pack, val);
		__m512i lo = _mm512_mask_blend_epi32(blend_AAAA, key_tmp, _mm512_swizzle_epi32(val_tmp, _MM_SWIZ_REG_CDAB));
		__m512i hi = _mm512_mask_blend_epi32(blend_5555, val_tmp, _mm512_swizzle_epi32(key_tmp, _MM_SWIZ_REG_CDAB));
		// update offsets and detect conflits
		__m512i offset = _mm512_mask_i32gather_epi32(offset, k, part, offsets, 4);
		__m512i offset_plus_1 = _mm512_add_epi32(offset, mask_1);
		_mm512_mask_i32scatter_epi32(offsets, k, part, offset_plus_1, 4);
		// compute block offsets
		offset = _mm512_and_epi32(offset, mask_15);
		__mmask16 eq = _mm512_mask_cmpeq_epi32_mask(k, offset, mask_15);
		offset = _mm512_fmadd_epi32(part, mask_16, offset);//mask_16 is the buffer size
		// write interleaved keys and payloads to buffers
		_mm512_mask_i32loscatter_epi64(buffers, k, offset, lo, 8);
		offset = _mm512_permute4f128_epi32(offset, _MM_PERM_BCDC);
		__mmask16 r_k = _mm512_kunpackb (k,k>>8);
		_mm512_mask_i32loscatter_epi64(buffers, r_k, offset, hi, 8);
		c = _mm_countbits_64(c);
		// flush full blocks (taken ~ 65%)
		if (!_mm512_kortestz(eq, eq)) {
			_mm512_store_epi32(parts, part);//needs to be in cache, not stream here
			size_t mask = _mm512_kconcatlo_64(blend_0000, eq);
			size_t b = _mm_tzcnt_64(mask);
			do {
				size_t l = parts[b];
				size_t o = offsets[l];
				int i,loop=BUFFER_SIZE/16;
				//l <<= 4;
				current_keys_out=l<partitions_ddr?keys_out:keys_out_hbm;
				current_vals_out=l<partitions_ddr?vals_out:vals_out_hbm;
				l*=BUFFER_SIZE;//l <<= 5;
				for (i=0;i<loop;++i)
				{
					lo = _mm512_load_epi64(&buffers[l + 0 + 16 * i]);
					hi = _mm512_load_epi64(&buffers[l + 8 + 16 * i]);
					key_tmp = _mm512_mask_blend_epi32(blend_AAAA, lo, _mm512_swizzle_epi32(hi, _MM_SWIZ_REG_CDAB));
					val_tmp = _mm512_mask_blend_epi32(blend_5555, hi, _mm512_swizzle_epi32(lo, _MM_SWIZ_REG_CDAB));
					key_tmp = _mm512_permutevar_epi32(mask_unpack, key_tmp);
					val_tmp = _mm512_permutevar_epi32(mask_unpack, val_tmp);
					_mm512_stream_ps(&current_keys_out[o - BUFFER_SIZE + 16 * i], _mm512_castsi512_ps(key_tmp));
					_mm512_stream_ps(&current_vals_out[o - BUFFER_SIZE + 16 * i], _mm512_castsi512_ps(val_tmp));
				}
				mask=mask&(~(1<<b));
				b = _mm_tzcnt_64(mask);
			} while (b != 64);
		}
		i += c;
	} while (i <= size_minus_16);
	// store last items in stack
	uint32_t tmp_key[32];
	uint32_t tmp_rid[32];
	k = _mm512_knot(k);
	_mm512_mask_compressstoreu_epi32(&tmp_key[0],  k, key);
	_mm512_mask_compressstoreu_epi32(&tmp_rid[0],  k, val);
	//_mm512_mask_packstorelo_epi32(&tmp_key[0],  k, key);
	//_mm512_mask_packstorehi_epi32(&tmp_key[16], k, key);
	//_mm512_mask_packstorelo_epi32(&tmp_rid[0],  k, val);
	//_mm512_mask_packstorehi_epi32(&tmp_rid[16], k, val);
	size_t c = _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
	assert(size - i + c <= 32);
	for (; i != size ; ++i, ++c) {
		tmp_key[c] = keys[i];
		tmp_rid[c] = vals[i];
	}
	// partition last items using scalar code
	for (i = 0 ; i != c ; ++i) {
		uint64_t kv = tmp_rid[i];
		kv = (kv << 32) | tmp_key[i];
		p = (uint32_t) (factor * (uint32_t) kv);
		//p = (p * partitions) >> 32;
		int cut = (int)((1-ratio)*precision);
		p = (p * precision) >> 32;
		size_t p2 = (uint32_t) (factor2 * (uint32_t) kv);
		p2 = (p2 * partitions/2) >> 32;
		if (p>=cut)
		{
			p=p2+partitions/2;
		}
		else
		{
			p=p2;
		}
		current_keys_out=p<partitions_ddr?keys_out:keys_out_hbm;
		current_vals_out=p<partitions_ddr?vals_out:vals_out_hbm;
		uint64_t *buf = &buffers[p * BUFFER_SIZE];
		size_t o = offsets[p]++;
		size_t b = o & (BUFFER_SIZE-1);
		buf[b] = kv;
		if (b == BUFFER_SIZE-1) {
			int iii=0,loop=BUFFER_SIZE/16;
			for (iii=0;iii<loop;++iii)
			{
				__m512i lo_out = _mm512_load_epi64(&buf[0+16*iii]);
				__m512i hi_out = _mm512_load_epi64(&buf[8+16*iii]);
				key = _mm512_mask_blend_epi32(blend_AAAA, lo_out, _mm512_swizzle_epi32(hi_out, _MM_SWIZ_REG_CDAB));
				val = _mm512_mask_blend_epi32(blend_5555, hi_out, _mm512_swizzle_epi32(lo_out, _MM_SWIZ_REG_CDAB));
				key = _mm512_permutevar_epi32(mask_unpack, key);
				val = _mm512_permutevar_epi32(mask_unpack, val);
				_mm512_stream_ps(&current_keys_out[o - BUFFER_SIZE + 1 +16 * iii], _mm512_castsi512_ps(key));
				_mm512_stream_ps(&current_vals_out[o - BUFFER_SIZE + 1 +16 * iii], _mm512_castsi512_ps(val));
			}
		}
	}
}
void partition(const uint32_t *keys, const uint32_t *vals, size_t size,
		const uint32_t *counts, uint32_t *keys_out, uint32_t *vals_out,
		uint32_t factor, size_t partitions)
{
	size_t i, p;
#ifdef BG
	uint64_t key_checksum = 0;
	uint64_t val_checksum = 0;
	for (i = 0 ; i != size ; ++i) {
		key_checksum += keys[i];
		val_checksum += vals[i];
	}
#endif
	// conflict space
	uint32_t conflicts_space[partitions + 15];
	uint32_t *conflicts = (uint32_t *)align(conflicts_space);
	// partition vector space
	uint32_t parts_space[31];
	uint32_t *parts = (uint32_t *)align(parts_space);
	// buffer space
	uint64_t buffers_space[(partitions * BUFFER_SIZE) + 7];
	uint64_t *buffers = (uint64_t *)align(buffers_space);
	// offset space
	uint32_t offsets_space[partitions + 15];
	uint32_t *offsets = (uint32_t *)align(offsets_space);
	// compute offset to align output
	uint32_t *keys_out_aligned = (uint32_t *)align(keys_out);
	uint32_t *vals_out_aligned = (uint32_t *)align(vals_out);
	size_t to_align_keys = keys_out_aligned - keys_out;
	size_t to_align_vals = vals_out_aligned - vals_out;
	assert(to_align_keys == to_align_vals);
	size_t align_output = 16 - to_align_keys;
	keys_out -= align_output;
	vals_out -= align_output;
	// initialize offsets
	for (i = p = 0 ; p != partitions ; ++p) {
		offsets[p] = i + align_output;
		i += counts[p];
	}
	//assert(i == size);
	// generate masks
	__m512i mask_factor = _mm512_set1_epi32(factor);
	__m512i mask_partitions = _mm512_set1_epi32(partitions);
	__m512i mask_pack = _mm512_set_epi32(15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 0);
	__m512i mask_unpack = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
	__m512i mask_16 = _mm512_set1_epi32(BUFFER_SIZE);
	__m512i mask_15 = _mm512_set1_epi32(BUFFER_SIZE-1);
	__m512i mask_1 = _mm512_set1_epi32(1);
	__mmask16 blend_AAAA = _mm512_int2mask(0xAAAA);
	__mmask16 blend_5555 = _mm512_int2mask(0x5555);
	__mmask16 blend_0000 = _mm512_int2mask(0x0000);
	// unaligned loop
	size_t size_minus_16 = size - 16;
	__mmask16 k = _mm512_kxnor(k, k);
	__m512i key, val;
	i = 0;
	if (size >= 16) do {
		// load new keys and payloads
		key = _mm512_mask_expandloadu_epi32 (key, k, &keys[i]);
		val = _mm512_mask_expandloadu_epi32 (val, k, &vals[i]);
		//_mm_prefetch((char*)&keys[i+16*PREFETCH_DISTANCE], _MM_HINT_T0);
		//_mm_prefetch((char*)&vals[i+16*PREFETCH_DISTANCE], _MM_HINT_T0);
		//key = _mm512_mask_loadunpacklo_epi32(key, k, &keys[i]);
		//key = _mm512_mask_loadunpackhi_epi32(key, k, &keys[i + 16]);
		//val = _mm512_mask_loadunpacklo_epi32(val, k, &vals[i]);
		//val = _mm512_mask_loadunpackhi_epi32(val, k, &vals[i + 16]);
		// hash keys
		__m512i part = simd_hash(key, mask_factor, mask_partitions);//= _mm512_mullo_epi32(key, mask_factor);
		//part = _mm512_mulhi_epi32(part, mask_partitions);
		// detect conflicts
		_mm512_i32scatter_epi32(conflicts, part, mask_pack, 4);
		__m512i back = _mm512_i32gather_epi32(part, conflicts, 4);
		size_t c = _mm512_kconcatlo_64(blend_0000, k);
		k = _mm512_cmpeq_epi32_mask(back, mask_pack);
		// split the data into low and high part
		__m512i key_tmp = _mm512_permutevar_epi32(mask_pack, key);
		__m512i val_tmp = _mm512_permutevar_epi32(mask_pack, val);
		__m512i lo = _mm512_mask_blend_epi32(blend_AAAA, key_tmp, _mm512_swizzle_epi32(val_tmp, _MM_SWIZ_REG_CDAB));
		__m512i hi = _mm512_mask_blend_epi32(blend_5555, val_tmp, _mm512_swizzle_epi32(key_tmp, _MM_SWIZ_REG_CDAB));
		// update offsets and detect conflits
		__m512i offset = _mm512_mask_i32gather_epi32(offset, k, part, offsets, 4);
		__m512i offset_plus_1 = _mm512_add_epi32(offset, mask_1);
		_mm512_mask_i32scatter_epi32(offsets, k, part, offset_plus_1, 4);
		// compute block offsets
		offset = _mm512_and_epi32(offset, mask_15);
		__mmask16 eq = _mm512_mask_cmpeq_epi32_mask(k, offset, mask_15);
		offset = _mm512_fmadd_epi32(part, mask_16, offset);
		// write interleaved keys and payloads to buffers
		_mm512_mask_i32loscatter_epi64(buffers, k, offset, lo, 8);
		offset = _mm512_permute4f128_epi32(offset, _MM_PERM_BCDC);
		__mmask16 r_k = _mm512_kunpackb (k,k>>8);
		//__mmask16 r_k = _mm512_kmerge2l1h(k, k);
		_mm512_mask_i32loscatter_epi64(buffers, r_k, offset, hi, 8);
		c = _mm_countbits_64(c);
		// flush full blocks (taken ~ 65%)
		if (!_mm512_kortestz(eq, eq)) {
			_mm512_store_epi32(parts, part);
			size_t mask = _mm512_kconcatlo_64(blend_0000, eq);
			size_t b = _mm_tzcnt_64(mask);
			do {
				size_t l = parts[b];
				size_t o = offsets[l];
				if (o != BUFFER_SIZE) {
					l *= BUFFER_SIZE;
					int i,loop=BUFFER_SIZE/16;
					for(i=0;i<loop;++i)
					{
						lo = _mm512_load_epi64(&buffers[l + 0 + 16 * i]);
						hi = _mm512_load_epi64(&buffers[l + 8 + 16 * i]);
						key_tmp = _mm512_mask_blend_epi32(blend_AAAA, lo, _mm512_swizzle_epi32(hi, _MM_SWIZ_REG_CDAB));
						val_tmp = _mm512_mask_blend_epi32(blend_5555, hi, _mm512_swizzle_epi32(lo, _MM_SWIZ_REG_CDAB));
						key_tmp = _mm512_permutevar_epi32(mask_unpack, key_tmp);
						val_tmp = _mm512_permutevar_epi32(mask_unpack, val_tmp);
						_mm512_stream_ps(&keys_out[o - BUFFER_SIZE + 16 * i], _mm512_castsi512_ps(key_tmp));
						_mm512_stream_ps(&vals_out[o - BUFFER_SIZE + 16 * i], _mm512_castsi512_ps(val_tmp));
					}
				} else {
					uint64_t *buf = &buffers[l * BUFFER_SIZE];
					o = align_output;
					while (l) o += counts[--l];
					while (o != BUFFER_SIZE) {
						uint64_t key_val = buf[o];
						keys_out[o] = key_val;
						vals_out[o] = key_val >> 32;
						o++;
					}
				}
				//b = _mm_tzcnti_64(b, mask);
				mask=mask&(~(1<<b));
				b = _mm_tzcnt_64(mask);
			} while (b != 64);
		}
		i += c;
	} while (i <= size_minus_16);
	// store last items in stack
	uint32_t tmp_key[32];
	uint32_t tmp_val[32];
	k = _mm512_knot(k);
	_mm512_mask_compressstoreu_epi32(&tmp_key[0],  k, key);
	_mm512_mask_compressstoreu_epi32(&tmp_val[0],  k, val);
	//_mm512_mask_packstorelo_epi32(&tmp_key[0],  k, key);
	//_mm512_mask_packstorehi_epi32(&tmp_key[16], k, key);
	//_mm512_mask_packstorelo_epi32(&tmp_val[0],  k, val);
	//_mm512_mask_packstorehi_epi32(&tmp_val[16], k, val);
	size_t c = _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
	for (; i != size ; ++i, ++c) {
		tmp_key[c] = keys[i];
		tmp_val[c] = vals[i];
	}
	// partition last items using scalar code
	for (i = 0 ; i != c ; ++i) {
		uint64_t kv = tmp_val[i];
		kv = (kv << 32) | tmp_key[i];
		p = (uint32_t) (factor * (uint32_t) kv);
		p = (p * partitions) >> 32;
		uint64_t *buf = &buffers[p * BUFFER_SIZE];
		size_t o = offsets[p]++;
		size_t b = o & (BUFFER_SIZE-1);
		buf[b] = kv;
		if (b != BUFFER_SIZE-1) ;
		else if (o != BUFFER_SIZE-1) {
			int iii,loop=BUFFER_SIZE/16;
			for (iii=0;iii<loop;++iii)
			{
				__m512i lo = _mm512_load_epi64(&buf[0+16*iii]);
				__m512i hi = _mm512_load_epi64(&buf[8+16*iii]);
				key = _mm512_mask_blend_epi32(blend_AAAA, lo, _mm512_swizzle_epi32(hi, _MM_SWIZ_REG_CDAB));
				val = _mm512_mask_blend_epi32(blend_5555, hi, _mm512_swizzle_epi32(lo, _MM_SWIZ_REG_CDAB));
				key = _mm512_permutevar_epi32(mask_unpack, key);
				val = _mm512_permutevar_epi32(mask_unpack, val);
				_mm512_store_ps(&keys_out[o - BUFFER_SIZE+1+16*iii], _mm512_castsi512_ps(key));
				_mm512_store_ps(&vals_out[o - BUFFER_SIZE+1+16*iii], _mm512_castsi512_ps(val));
			}
		} else {
			o = align_output;
			while (p) o += counts[--p];
			while (o != BUFFER_SIZE) {
				uint64_t key_val = buf[o];
				keys_out[o] = key_val;
				vals_out[o] = key_val >> 32;
				o++;
			}
		}
	}
	flush(counts, offsets, buffers, keys_out, vals_out, partitions);
#ifdef BG
	keys_out += align_output;
	vals_out += align_output;
	for (i = 0 ; i != size ; ++i) {
		key_checksum -= keys_out[i];
		val_checksum -= vals_out[i];
	}
	//assert(key_checksum == 0);
	//assert(val_checksum == 0);
	size_t pp = 0;
	for (i = 0 ; i != size ; ++i) {
		p = (uint32_t) (keys_out[i] * factor);
		p = (p * partitions) >> 32;
		//assert(pp <= p);
		pp = p;
	}
#endif
}



void copy(uint32_t *dst, const uint32_t *src, size_t size)
{
	uint32_t *dst_end = &dst[size];
	uint32_t *dst_aligned = (uint32_t *)align(dst);
	__mmask16 k = _mm512_kxnor(k, k);
	while (dst != dst_end && dst != dst_aligned)
		*dst++ = *src++;
	dst_aligned = &dst[(dst_end - dst) & -16];
	if (src == align(src))
		while (dst != dst_aligned) {
			__m512 x = _mm512_load_ps(src);
			_mm512_store_ps(dst, x);
			src += 16;
			dst += 16;
		}
	else
		while (dst != dst_aligned) {
			__m512 x;
			x = _mm512_mask_loadu_ps(x, k, src);
			src += 16;
			//x = _mm512_loadunpackhi_ps(x, src);
			_mm512_store_ps(dst, x);
			dst += 16;
		}
	while (dst != dst_end)
		*dst++ = *src++;
}

size_t interleave(uint32_t **counts, uint32_t *offsets, uint32_t *aggr_counts,
		size_t partitions, size_t thread, size_t threads)
{
	size_t i = 0, p = 0;
	assert(offsets == align(offsets));
	assert(aggr_counts == align(aggr_counts));
	while (p != partitions) {
		size_t s = partitions - p;
		if (s > 16) s = 16;
		__mmask16 k = _mm512_int2mask((1 << s) - 1);
		__m512i sum = _mm512_xor_epi32(sum, sum);
		size_t t = 0;
		if (thread) do {
			__m512i cur = _mm512_mask_load_epi32(cur, k, &counts[t][p]);
			sum = _mm512_add_epi32(sum, cur);
		} while (++t != thread);
		_mm512_mask_store_epi32(&offsets[p], k, sum);
		do {
			__m512i cur = _mm512_mask_load_epi32(cur, k, &counts[t][p]);
			sum = _mm512_add_epi32(sum, cur);
		} while (++t != threads);
		_mm512_mask_store_epi32(&aggr_counts[p], k, sum);
		do {
			offsets[p] += i;
			i += aggr_counts[p++];
		} while (--s);
	}
	return i;
}

#else

void histogram(const uint32_t *keys, size_t size, uint32_t *counts,
		uint32_t factor, size_t partitions)
{
	size_t i, p;
	for (p = 0 ; p != partitions ; ++p)
		counts[p] = 0;
	for (i = 0 ; i != size ; ++i) {
		p = (uint32_t) (keys[i] * factor);
		p = (p * partitions) >> 32;
		counts[p]++;
	}
}

#ifndef _ALIGNED

void partition(const uint32_t *keys, const uint32_t *vals, size_t size,
		const uint32_t *counts, uint32_t *keys_out, uint32_t *vals_out,
		uint32_t factor, size_t partitions)
{
	size_t i, p;
	uint32_t offsets[partitions];
	uint64_t buffers_space[(partitions << 4) + 7];
	uint64_t *buffers = (uint64_t *)align(buffers_space);
	for (i = p = 0 ; p != partitions ; ++p) {
		offsets[p] = i;
		i += counts[p];
	}
	//assert(i == size);
	for (i = 0 ; i != size ; ++i) {
		uint32_t key = keys[i];
		uint64_t key_val = vals[i];
		key_val = (key_val << 32) | key;
		p = (uint32_t) (key * factor);
		p = (p * partitions) >> 32;
		size_t o = offsets[p]++;
		size_t b = o & 15;
		uint64_t *buf = &buffers[p << 4];
		buf[b] = key_val;
		if (b == 15) {
			uint32_t *k_out = &keys_out[o - 15];
			uint32_t *v_out = &vals_out[o - 15];
			for (b = 0 ; b != 16 ; ++b) {
				key_val = buf[b];
				k_out[b] = key_val;
				v_out[b] = key_val >> 32;
			}
		}
	}
	flush(counts, offsets, buffers, keys_out, vals_out, partitions);
}

#else

void partition(const uint32_t *keys, const uint32_t *vals, size_t size,
		const uint32_t *counts, uint32_t *keys_out, uint32_t *vals_out,
		uint32_t factor, size_t partitions)
{
	size_t align_output = 0;
	uint32_t *keys_out_aligned = (uint32_t *)align(keys_out);
	uint32_t *vals_out_aligned = (uint32_t *)align(vals_out);

	//assert(keys_out_aligned - keys_out ==
	vals_out_aligned - vals_out);
	if (keys_out != keys_out_aligned) {
		//assert(keys_out_aligned - keys_out < 16);
		align_output = 16 - (keys_out_aligned - keys_out);
	}
	keys_out -= align_output;
	vals_out -= align_output;
	size_t i, p;
	uint32_t offsets[partitions];
	uint64_t buffers_space[(partitions << 4) + 7];
	uint64_t *buffers = (uint64_t *)align(buffers_space);
	for (i = p = 0 ; p != partitions ; ++p) {
		offsets[p] = i + align_output;
		i += counts[p];
	}
	//assert(i == size);
	for (i = 0 ; i != size ; ++i) {
		uint32_t key = keys[i];
		uint64_t key_val = vals[i];
		key_val = (key_val << 32) | key;
		p = (uint32_t) (key * factor);
		p = (p * partitions) >> 32;
		size_t o = offsets[p]++;
		size_t b = o & 15;
		uint64_t *buf = &buffers[p << 4];
		buf[b] = key_val;
		if (b != 15) ;
		else if (o != 15) {
			uint32_t *k_out = &keys_out[o - 15];
			uint32_t *v_out = &vals_out[o - 15];
			for (b = 0 ; b != 16 ; ++b) {
				key_val = buf[b];
				k_out[b] = key_val;
				v_out[b] = key_val >> 32;
			}
		} else {
			o = align_output;
			while (p) o += counts[--p];
			for (; o != 16 ; ++o) {
				key_val = buf[o];
				keys_out[o] = key_val;
				vals_out[o] = key_val >> 32;
			}
		}
	}
	flush(counts, offsets, buffers, keys_out, vals_out, partitions);
}

#endif

void partition_shared(const uint32_t *keys, const uint32_t *vals, size_t size,
		uint32_t *offsets, uint64_t *buffers, uint32_t *keys_out,
		uint32_t *vals_out, uint32_t factor, size_t partitions)
{
	size_t i, p;
	for (i = 0 ; i != size ; ++i) {
		uint32_t key = keys[i];
		uint64_t key_val = vals[i];
		key_val = (key_val << 32) | key;
		p = (uint32_t) (key * factor);
		p = (p * partitions) >> 32;
		size_t o = offsets[p]++;
		size_t b = o & 15;
		uint64_t *buf = &buffers[p << 4];
		buf[b] = key_val;
		if (b == 15) {
			uint32_t *k_out = &keys_out[o - 15];
			uint32_t *v_out = &vals_out[o - 15];
			for (b = 0 ; b != 16 ; ++b) {
				key_val = buf[b];
				k_out[b] = key_val;
				v_out[b] = key_val >> 32;
			}
		}
	}
}

void copy(uint32_t *dst, const uint32_t *src, size_t size)
{
	size_t i;
	for (i = 0 ; i != size ; ++i)
		dst[i] = src[i];
}

size_t interleave(uint32_t **counts, uint32_t *offsets, uint32_t *aggr_counts,
		size_t partitions, size_t thread, size_t threads)
{
	size_t i, p, t;
	for (i = p = 0 ; p != partitions ; ++p) {
		size_t s = i;
		for (t = 0 ; t != thread ; ++t)
			i += counts[t][p];
		offsets[p] = i;
		for (; t != threads ; ++t)
			i += counts[t][p];
		aggr_counts[p] = i - s;
	}
	return i;
}

#endif

typedef struct {
	size_t beg;
	size_t end;
} pair_size_t;

int pair_cmp(const void *x, const void *y)
{
	size_t a = ((pair_size_t*) x)->beg;
	size_t b = ((pair_size_t*) y)->beg;
	return a < b ? -1 : a > b ? 1 : 0;
}

size_t close_gaps(uint32_t *keys, uint32_t *vals, uint32_t *tabs,
		const size_t *offsets, size_t count,
		size_t block_size, volatile size_t *counter)
{
	//assert(power_of_2(block_size));
	pair_size_t *holes = (pair_size_t *)malloc(count * sizeof(pair_size_t));
	size_t i, c = 0, l = 0, h = count - 1, s = 0;
	for (i = 0 ; i != count ; ++i) {
		holes[i].beg = offsets[i];
		holes[i].end = (offsets[i] & ~(block_size - 1)) + block_size;
		s += holes[i].end - holes[i].beg;
	}
	qsort(holes, count, sizeof(pair_size_t), pair_cmp);
	size_t blocks = holes[h].beg / block_size + 1;
	size_t src = holes[h].end;
	i = __sync_fetch_and_add(counter, 1);
	while (l <= h) {
		size_t fill = src - holes[h].end;
		if (fill == 0) {
			src = holes[h].beg;
			if (!h--) break;
			continue;
		}
		size_t hole = holes[l].end - holes[l].beg;
		if (hole == 0) {
			l++;
			continue;
		}
		size_t cnt = fill < hole ? fill : hole;
		size_t dst = holes[l].beg;
		holes[l].beg += cnt;
		src -= cnt;
		if (c++ == i) {
			copy(&keys[dst], &keys[src], cnt);
			copy(&vals[dst], &vals[src], cnt);
			copy(&tabs[dst], &tabs[src], cnt);
			i = __sync_fetch_and_add(counter, 1);
		}
	}
	free(holes);
	//assert(src == blocks * block_size - s);
	return src;
}

size_t thread_beg(size_t size, size_t alignment, size_t thread, size_t threads)
{
	//assert(power_of_2(alignment));
	size_t part = (size / threads) & ~(alignment - 1);
	return part * thread;
}

size_t thread_end(size_t size, size_t alignment, size_t thread, size_t threads)
{
	//assert(power_of_2(alignment));
	size_t part = (size / threads) & ~(alignment - 1);
	if (thread + 1 == threads) return size;
	return part * (thread + 1);
}

void swap(uint32_t **x, uint32_t **y)
{
	uint32_t *t = *x; *x = *y; *y = t;
}

size_t max(size_t x, size_t y)
{
	return x > y ? x : y;
}

size_t min(size_t x, size_t y)
{
	return x < y ? x : y;
}

void shuffle(uint32_t *data, size_t size, rand32_t *gen)
{
	size_t i, j;
	for (i = 0 ; i != size ; ++i) {
		j = rand32_next(gen);
		j *= size - i;
		j >>= 32;
		j += i;
		uint32_t t = data[i];
		data[i] = data[j];
		data[j] = t;
	}
}

void unique(uint32_t *keys, size_t size, volatile uint32_t *table, size_t buckets,
		uint32_t factor, uint32_t empty, rand32_t *gen)
{
	//assert(odd_prime(buckets));
	size_t i = 0;
	while (i != size) {
		uint32_t key;
		do {
			key = rand32_next(gen);
		} while (key == empty);
		size_t h = (uint32_t) (key * factor);
		h = (h * buckets) >> 32;
		uint32_t tab = table[h];
		while (tab != key) {
			if (tab == empty) {
				tab = __sync_val_compare_and_swap(&table[h], empty, key);
				if (tab == empty) {
					keys[i++] = key;
					break;
				}
				if (tab == key) break;
			}
			if (++h == buckets) h = 0;
			tab = table[h];
		}
	}
}

int uint32_cmp(const void *x, const void *y)
{
	uint32_t a = *((uint32_t*) x);
	uint32_t b = *((uint32_t*) y);
	return a < b ? -1 : a > b ? 1 : 0;
}
void *generate_data_for_join(void *arg)
{
	info_t_hj *d = (info_t_hj*) arg;
	////assert(pthread_equal(pthread_self(), d->id));
	pthread_barrier_t *barrier = d->barrier;
	//bind_thread(d->thread, d->threads);
	uint32_t *inner_keys_in  = d->inner_keys[0];
	uint32_t *inner_keys_out = d->inner_keys[1];
	uint32_t *inner_vals_in  = d->inner_vals[0];
	uint32_t *inner_vals_out = d->inner_vals[1];
	uint32_t *outer_keys_in  = d->outer_keys[0];
	uint32_t *outer_keys_out = d->outer_keys[1];
	uint32_t *outer_vals_in  = d->outer_vals[0];
	uint32_t *outer_vals_out = d->outer_vals[1];
	size_t i, p, o, j, u, t, f, h;
	size_t thread  = d->thread;
	size_t threads = d->threads;
	rand32_t *gen = rand32_init(d->seed);
	for (p = 0 ; p != 8 ; ++p)
		d->times[p] = 0;
	// generate unique items

	size_t inner_beg = thread_beg(d->inner_tuples, 16, thread, threads);
	size_t inner_end = thread_end(d->inner_tuples, 16, thread, threads);
	//assert(u == inner_distinct_end);
	size_t outer_beg = thread_beg(d->outer_tuples, 16, thread, threads);
	size_t outer_end = thread_end(d->outer_tuples, 16, thread, threads);

	// generate payloads and outputs
	pthread_barrier_wait(barrier++);
	uint32_t inner_factor = d->inner_factor;
	for (i = inner_beg ; i != inner_end ; ++i) {
		//inner_vals_in[i] = inner_keys_in[i] * inner_factor;
		inner_keys_out[i] = 0xCAFEBABE;
		inner_vals_out[i] = 0xDEADBEEF;
	}
	uint32_t outer_factor = d->outer_factor;
	for (o = outer_beg ; o != outer_end ; ++o) {
		//outer_vals_in[o] = outer_keys_in[o] * outer_factor;
		outer_keys_out[o] = 0xBABECAFE;
		outer_vals_out[o] = 0xBEEFDEAD;
	}
	size_t join_beg = thread_beg(d->block_limit * d->block_size, 1, thread, threads);
	size_t join_end = thread_end(d->block_limit * d->block_size, 1, thread, threads);
	uint32_t *join_keys = d->join_keys;
	uint32_t *join_vals = d->join_outer_vals;
	uint32_t *join_tabs = d->join_inner_vals;
	for (j = join_beg ; j != join_end ; ++j) {
		//join_keys[j] = 0xCAFEBABE;
		join_vals[j] = 0xDEADBEEF;
		join_tabs[j] = 0x12345678;
	}
}
void *run_hj(void *arg)
{
	info_t_hj *d = (info_t_hj*) arg;
	//assert(pthread_equal(pthread_self(), d->id));
	pthread_barrier_t *barrier = d->barrier;
	//bind_thread(d->thread, d->threads);
	uint32_t *inner_keys_in  = d->inner_keys[0];
	uint32_t *inner_keys_out = d->inner_keys[1];
	uint32_t *inner_vals_in  = d->inner_vals[0];
	uint32_t *inner_vals_out = d->inner_vals[1];
	uint32_t *outer_keys_in  = d->outer_keys[0];
	uint32_t *outer_keys_out = d->outer_keys[1];
	uint32_t *outer_vals_in  = d->outer_vals[0];
	uint32_t *outer_vals_out = d->outer_vals[1];

	uint32_t *inner_keys_in_hbm  = d->inner_keys[2];
	uint32_t *inner_keys_out_hbm = d->inner_keys[3];
	uint32_t *inner_vals_in_hbm  = d->inner_vals[2];
	uint32_t *inner_vals_out_hbm = d->inner_vals[3];
	uint32_t *outer_keys_in_hbm  = d->outer_keys[2];
	uint32_t *outer_keys_out_hbm = d->outer_keys[3];
	uint32_t *outer_vals_in_hbm  = d->outer_vals[2];
	uint32_t *outer_vals_out_hbm = d->outer_vals[3];

	size_t i, p, o, j, u, t, f, h;
	size_t thread  = d->thread;
	size_t threads = d->threads;
	rand32_t *gen = rand32_init(d->seed);
	//for (p = 0 ; p != 8 ; ++p)
	//	d->times[p] = 0;
	//uint64_t outer_checksum = 0;
	//size_t join_distinct = d->join_distinct;
	//size_t inner_distinct = d->inner_distinct;
	//size_t outer_distinct = d->outer_distinct;
	//size_t distinct = inner_distinct + outer_distinct - join_distinct;
	//size_t distinct_beg = thread_beg(distinct, 1, thread, threads);
	//size_t distinct_end = thread_end(distinct, 1, thread, threads);
	size_t inner_beg = thread_beg(d->inner_tuples, 16, thread, threads);
	size_t inner_end = thread_end(d->inner_tuples, 16, thread, threads);
	//size_t inner_distinct_beg = thread_beg(inner_distinct, 16, thread, threads);
	//size_t inner_distinct_end = thread_end(inner_distinct, 16, thread, threads);
	//uint64_t inner_checksum = 0;
	//uint32_t *outer_unique = &d->unique[inner_distinct - join_distinct];
	size_t outer_beg = thread_beg(d->outer_tuples, 16, thread, threads);
	size_t outer_end = thread_end(d->outer_tuples, 16, thread, threads);
	//size_t outer_distinct_beg = thread_beg(outer_distinct, 16, thread, threads);
	//size_t outer_distinct_end = thread_end(outer_distinct, 16, thread, threads);
	size_t join_beg = thread_beg(d->block_limit * d->block_size, 1, thread, threads);
	size_t join_end = thread_end(d->block_limit * d->block_size, 1, thread, threads);
	uint32_t *join_keys = d->join_keys;
	uint32_t *join_vals = d->join_outer_vals;
	uint32_t *join_tabs = d->join_inner_vals;
	uint32_t inner_factor = d->inner_factor;
	uint32_t outer_factor = d->outer_factor;
	//d->inner_checksum[thread] = inner_checksum;
	//d->outer_checksum[thread] = outer_checksum;
	uint64_t inner_sum = d->inner_sum;
	uint64_t outer_sum = d->outer_sum;
	// start timinig
	pthread_barrier_wait(barrier++);
	//uint64_t tt = thread_time();
	//uint64_t rt = real_time();
	double second=mysecond();
	//double nosynctime=mysecond();
	//double nosyncsum=0;
	// shared partitioning
#ifdef TIMELOG
d->timelog[d->timestep++]=mysecond();
#endif
	if (threads > 1) {
		size_t partitions = threads;
		uint64_t buffers_space[threads * BUFFER_SIZE * 2 + 7];
		uint64_t *buffers = (uint64_t *)align(buffers_space);
		uint64_t *inner_buffers = &buffers[partitions *  0];
		uint64_t *outer_buffers = &buffers[partitions * BUFFER_SIZE];
		size_t partitions_aligned = (partitions + 15) & -16;
		uint32_t counts_space[partitions_aligned * 6 + 15];
		uint32_t *counts = (uint32_t *)align(counts_space);
		uint32_t *inner_offsets = &counts[partitions_aligned * 0];
		uint32_t *outer_offsets = &counts[partitions_aligned * 1];
		uint32_t *inner_counts  = &counts[partitions_aligned * 2];
		uint32_t *outer_counts  = &counts[partitions_aligned * 3];
		d->inner_counts[thread] = inner_counts;
		d->outer_counts[thread] = outer_counts;
		uint32_t factor = d->thread_factor;
		uint32_t factor2 = d->thread_factor2;
		histogram_shared(&inner_keys_in[inner_beg], inner_end - inner_beg, inner_counts, factor, factor2, partitions, d->ratio);
		histogram_shared(&outer_keys_in[outer_beg], outer_end - outer_beg, outer_counts, factor, factor2, partitions, d->ratio);
		//nosynctime=mysecond()-nosynctime;nosyncsum+=nosynctime;//NO SYNC TIME
		pthread_barrier_wait(barrier++);
		//nosynctime=mysecond();
		inner_counts = &counts[partitions_aligned * 4];
		outer_counts = &counts[partitions_aligned * 5];
		i = interleave(d->inner_counts, inner_offsets, inner_counts, partitions, thread, threads);
		o = interleave(d->outer_counts, outer_offsets, outer_counts, partitions, thread, threads);
		assert(i == d->inner_tuples);
		assert(o == d->outer_tuples);
		partition_shared(&inner_keys_in[inner_beg], &inner_vals_in[inner_beg], inner_end - inner_beg,
				inner_offsets, inner_buffers, inner_keys_out, inner_vals_out, inner_keys_out_hbm, inner_vals_out_hbm, d->ratio, factor, factor2, partitions);
		partition_shared(&outer_keys_in[outer_beg], &outer_vals_in[outer_beg], outer_end - outer_beg,
				outer_offsets, outer_buffers, outer_keys_out, outer_vals_out, outer_keys_out_hbm, outer_vals_out_hbm, d->ratio, factor, factor2, partitions);
		//nosynctime=mysecond()-nosynctime;nosyncsum+=nosynctime;//NO SYNC TIME
		pthread_barrier_wait(barrier++);
		//nosynctime=mysecond();
		flush_shared(d->inner_counts[thread], inner_offsets, inner_buffers, inner_keys_out, inner_vals_out, inner_keys_out_hbm, inner_vals_out_hbm, d->ratio, partitions);
		flush_shared(d->outer_counts[thread], outer_offsets, outer_buffers, outer_keys_out, outer_vals_out, outer_keys_out_hbm, outer_vals_out_hbm, d->ratio, partitions);
		swap(&inner_keys_in, &inner_keys_out);
		swap(&inner_vals_in, &inner_vals_out);
		swap(&outer_keys_in, &outer_keys_out);
		swap(&outer_vals_in, &outer_vals_out);
		swap(&inner_keys_in_hbm, &inner_keys_out_hbm);
		swap(&inner_vals_in_hbm, &inner_vals_out_hbm);
		swap(&outer_keys_in_hbm, &outer_keys_out_hbm);
		swap(&outer_vals_in_hbm, &outer_vals_out_hbm);
		inner_beg = 0;
		outer_beg = 0;
		for (t = 0 ; t != thread ; ++t) {
			inner_beg += inner_counts[t];
			outer_beg += outer_counts[t];
		}
		inner_end = inner_beg + inner_counts[t];
		outer_end = outer_beg + outer_counts[t];
		pthread_barrier_wait(barrier++);

	}
#ifdef TIMELOG
d->timelog[d->timestep++]=mysecond();
#endif
	bool is_ddr = (thread<threads/2);
	if (!is_ddr)
	{
		inner_keys_in=inner_keys_in_hbm;
		inner_vals_in=inner_vals_in_hbm;
		outer_keys_in=outer_keys_in_hbm;
		outer_vals_in=outer_vals_in_hbm;
		inner_keys_out=inner_keys_out_hbm;
		inner_vals_out=inner_vals_out_hbm;
		outer_keys_out=outer_keys_out_hbm;
		outer_vals_out=outer_vals_out_hbm;
	}
	//else
	//	printf ("%d\t%d\t%d\n", thread, inner_end - inner_beg, outer_end- outer_beg);
	d->inner_thread_tuples = inner_end - inner_beg;
	d->outer_thread_tuples = outer_end - outer_beg;
	// local partitioning
	size_t partitions = (inner_end - inner_beg) / d->hash_table_limit;
	size_t passes = 0;
	if      (partitions > 1000000) passes = 4;
	else if (partitions > 20000)   passes = 3;
	else if (partitions > 400)     passes = 2;
	else if (partitions > 10)      passes = 1;
	//passes=4;
	//printf ("%d\n", partitions);
	assert(passes > 0 || threads > 1);
	for (p = 0 ; p != passes ; ++p)
		d->fanout[p] = pow(partitions, 1.0 / passes);
	d->fanout[p] = 1;
	if (passes) {
		size_t fanout_product = 1;
		for (p = 0 ; p != passes - 1 ; ++p)
			fanout_product *= d->fanout[p];
		d->fanout[p] = partitions / fanout_product;
	}
	uint32_t *inner_counts = (uint32_t *)hma_malloc(sizeof(uint32_t),is_ddr);
	uint32_t *outer_counts = (uint32_t *)hma_malloc(sizeof(uint32_t),is_ddr);
	inner_counts[0] = inner_end - inner_beg;
	outer_counts[0] = outer_end - outer_beg;
	partitions = 1;
	size_t size;
	uint32_t *counts;
	uint32_t *keys_in, *keys_out;
	uint32_t *vals_in, *vals_out;
	uint32_t factor_1st = 0;
	for (f = 0 ; d->fanout[f] != 1 ; ++f) {
		size_t fanout = d->fanout[f];
		uint32_t *inner_counts_next = (uint32_t *)hma_malloc(partitions * fanout * sizeof(uint32_t),is_ddr);
		uint32_t *outer_counts_next = (uint32_t *)hma_malloc(partitions * fanout * sizeof(uint32_t),is_ddr);
		uint32_t factor = rand32_next(gen) | 1;
		if (f == 0) factor_1st = factor;
		i = inner_beg;
		o = outer_beg;
		for (p = 0 ; p != partitions ; ++p) {
			size = inner_counts[p];
			keys_in = &inner_keys_in[i];
			vals_in = &inner_vals_in[i];
			keys_out = &inner_keys_out[i];
			vals_out = &inner_vals_out[i];
			counts = &inner_counts_next[p * fanout];
			histogram(keys_in, size, counts, factor, fanout);
			partition(keys_in, vals_in, size, counts,
					keys_out, vals_out, factor, fanout);
			i += size;
			size = outer_counts[p];
			keys_in = &outer_keys_in[o];
			vals_in = &outer_vals_in[o];
			keys_out = &outer_keys_out[o];
			vals_out = &outer_vals_out[o];
			counts = &outer_counts_next[p * fanout];
			histogram(keys_in, size, counts, factor, fanout);
			partition(keys_in, vals_in, size, counts,
					keys_out, vals_out, factor, fanout);
			o += size;
		}
		assert(i == inner_end);
		assert(o == outer_end);
		hma_free(inner_counts,is_ddr);
		hma_free(outer_counts,is_ddr);
		partitions *= fanout;
		inner_counts = inner_counts_next;
		outer_counts = outer_counts_next;
		swap(&inner_keys_in, &inner_keys_out);
		swap(&inner_vals_in, &inner_vals_out);
		swap(&outer_keys_in, &outer_keys_out);
		swap(&outer_vals_in, &outer_vals_out);
		//ptt = ntt;
		//ntt = thread_time();
		//d->times[f + 1] = ntt - ptt;
	}
#ifdef TIMELOG
d->timelog[d->timestep++]=mysecond();
#endif
	// cache resident joins

/*	double inverse_load = 1.0 / d->hash_table_load;
	size_t max_buckets = 0;
	uint64_t *table = NULL;
	uint32_t factors[2];
	do {
		factors[0] = rand32_next(gen) | 1;
		factors[1] = rand32_next(gen) | 1;
	} while (((factors[0] - factors[1]) & 3) == 0);
	uint32_t *keys_buf = (uint32_t *)hma_malloc((d->buffer_size + 16) * sizeof(uint32_t),is_ddr);
	uint32_t *vals_buf = (uint32_t *)hma_malloc((d->buffer_size + 16) * sizeof(uint32_t),is_ddr);
	uint32_t *tabs_buf = (uint32_t *)hma_malloc((d->buffer_size + 16) * sizeof(uint32_t),is_ddr);
	size_t offset = __sync_fetch_and_add(d->block_counter, 1);
	assert(offset <= d->block_limit);
	offset *= d->block_size;
	i = inner_beg;
	o = outer_beg;
	for (p = 0 ; p != partitions ; ++p) {
		uint32_t empty = 0;
		if (p == 0)
			if (threads == 1)
				do {
					h = (uint32_t) (++empty * factor_1st);
					h = (h * d->fanout[0]) >> 32;
				} while (h == 0);
			else if (thread == 0)
				do {
					h = (uint32_t) (++empty * d->thread_factor);
					h = (h * threads) >> 32;
				} while (h == 0);
		size = inner_counts[p];
		size_t buckets = size * inverse_load;
		if (buckets > max_buckets) {
			for (buckets |= 1 ; !odd_prime(buckets) ; buckets += 2);
			max_buckets = buckets;
			if(is_ddr)
				table = (uint64_t*)realloc(table, buckets * sizeof(uint64_t));
			else
				table = (uint64_t*)hbw_realloc(table, buckets * sizeof(uint64_t));
		} else if (buckets * 1.2 > max_buckets)
			for (buckets |= 1 ; !odd_prime(buckets) ; buckets += 2);
		else buckets = max_buckets;
		keys_in = &inner_keys_in[i];
		vals_in = &inner_vals_in[i];
		build(keys_in, vals_in, size, table, buckets, factors, empty);
		i += size;
		size = outer_counts[p];
		keys_in = &outer_keys_in[o];
		vals_in = &outer_vals_in[o];
		offset = probe(keys_in, vals_in, size, table, buckets, factors, empty,
				keys_buf, vals_buf, tabs_buf,
				join_keys, join_vals, join_tabs,
				offset, d->buffer_size, d->block_size, d->block_limit,
				d->block_counter, p + 1 == partitions);

		o += size;
	} */
#ifdef TIMELOG
d->timelog[d->timestep++]=mysecond();
#endif

	//d->final_offsets[thread] = offset;
	//hma_free(table,is_ddr);
	//hma_free(keys_buf,is_ddr);
	//hma_free(vals_buf,is_ddr);
	//hma_free(tabs_buf,is_ddr);
	//hma_free(inner_counts,is_ddr);
	//hma_free(outer_counts,is_ddr);

	
	//pthread_barrier_wait(barrier++);

	
	//d->join_tuples = close_gaps(join_keys, join_vals, join_tabs, d->final_offsets,threads, d->block_size, d->close_gaps_counter);

	//assert(d->join_tuples <= d->block_size * d->block_limit);

	// finish timinig
	second=mysecond()-second;
	d->time=second;
	pthread_exit(NULL);
}
int get_cpu_id (int i)
{
#ifdef COMPACT
	//printf ("%d\n", (int)(i/4)+(i%4)*64);
	return (int)(i/4)+(i%4)*64;
#else
	return i;
#endif
}
int main(int argc, char **argv)
{
	// arguments
	int t, threads = argc > 1 ? atoi(argv[1]) : hardware_threads();
	size_t outer_tuples = argc > 2 ? atoll(argv[2]) : 200 * 1000 * 1000;
	size_t inner_tuples = argc > 3 ? atoll(argv[3]) : 200 * 1000 * 1000;
	double ratio= argc > 4 ? std::stod(argv[4]):1;
	double selc= 1;//argc > 6 ? std::stod(argv[6]):1;
	BUFFER_SIZE= 64;
	//PREFETCH_DISTANCE= argc > 5 ? atoll(argv[5]):0;
	size_t outer_distinct = min(inner_tuples, outer_tuples);
	size_t inner_distinct = min(inner_tuples, outer_tuples);
	size_t join_distinct  = min(inner_distinct, outer_distinct);
	double outer_repeats = outer_tuples * 1.0 / outer_distinct;
	double inner_repeats = inner_tuples * 1.0 / inner_distinct;
	size_t join_tuples = outer_repeats * inner_repeats * join_distinct;
	// other parameters
	double hash_table_load = 0.4;
	size_t hash_table_limit = 6400;
	size_t buffer_size = 256;
	size_t block_size = buffer_size * 256;
	// estimate output size
	size_t block_limit = join_tuples * 1.05 / block_size + threads * 2;
	size_t max_join_tuples = block_size * block_limit;
	size_t dummy_tuples = max(max(inner_tuples, outer_tuples), max_join_tuples);
	// print info
	//assert(inner_distinct <= inner_tuples);
	//assert(outer_distinct <= outer_tuples);
	//assert(join_distinct <= min(inner_distinct, outer_distinct));
	//assert(threads > 0 && threads <= hardware_threads());
	//assert(power_of_2(block_size) && power_of_2(buffer_size));
	//assert(buffer_size <= block_size);
#ifdef BG
	fprintf(stderr, "Debugging mode!\n");
#endif
#ifdef _UNIQUE
	fprintf(stderr, "Enforcing unique keys\n");
	//assert(inner_tuples == inner_distinct);
#endif
#ifdef _NO_VECTOR_PARTITIONING
	fprintf(stderr, "Vectorized partitioning disabled!\n");
#endif
#ifdef _NO_VECTOR_HASHING
	fprintf(stderr, "Vectorized hashing disabled!\n");
#endif
	//fprintf(stderr, "Threads: %d\n", threads);
	//fprintf(stderr, "Inner tuples: %9ld\n", inner_tuples);
	//fprintf(stderr, "Outer tuples: %9ld\n", outer_tuples);
	//fprintf(stderr, "Inner distinct values: %9ld\n", inner_distinct);
	//fprintf(stderr, "Outer distinct values: %9ld\n", outer_distinct);
	//fprintf(stderr, "Join  distinct values: %9ld\n", join_distinct);
	//fprintf(stderr, "Join tuples (expected): %9ld\n", join_tuples);
	// compute space
	double space = inner_tuples * 16 +
		outer_tuples * 16 +
		max_join_tuples * 12;
	space /= 1024 * 1024 * 1024;
	//fprintf(stderr, "Space: %.2f GB\n", space);
	//assert(space <= 14.5);
	// parameters
	srand(time(NULL));
	uint32_t factors[5];
	for (t = 0 ; t != 5 ; ++t)
		factors[t] = (rand() << 1) | 1;
	// inner side & buffers on DDR
	uint32_t *inner_keys_1 = (uint32_t *)mamalloc(inner_tuples * sizeof(uint32_t));
	uint32_t *inner_keys_2 = (uint32_t *)mamalloc(inner_tuples * sizeof(uint32_t));
	uint32_t *inner_vals_1 = (uint32_t *)mamalloc(inner_tuples * sizeof(uint32_t));
	uint32_t *inner_vals_2 = (uint32_t *)mamalloc(inner_tuples * sizeof(uint32_t));
	// outer side & buffers on DDR
	uint32_t *outer_keys_1 = (uint32_t *)mamalloc(outer_tuples * sizeof(uint32_t));
	uint32_t *outer_keys_2 = (uint32_t *)mamalloc(outer_tuples * sizeof(uint32_t));
	uint32_t *outer_vals_1 = (uint32_t *)mamalloc(outer_tuples * sizeof(uint32_t));
	uint32_t *outer_vals_2 = (uint32_t *)mamalloc(outer_tuples * sizeof(uint32_t));
	// inner side & buffers on HBM
	uint32_t *inner_keys_1_hbm = (uint32_t *)hbw_malloc(inner_tuples * sizeof(uint32_t));
	uint32_t *inner_keys_2_hbm = (uint32_t *)hbw_malloc(inner_tuples * sizeof(uint32_t));
	uint32_t *inner_vals_1_hbm = (uint32_t *)hbw_malloc(inner_tuples * sizeof(uint32_t));
	uint32_t *inner_vals_2_hbm = (uint32_t *)hbw_malloc(inner_tuples * sizeof(uint32_t));
	// outer side & buffers on HBM
	uint32_t *outer_keys_1_hbm = (uint32_t *)hbw_malloc(outer_tuples * sizeof(uint32_t));
	uint32_t *outer_keys_2_hbm = (uint32_t *)hbw_malloc(outer_tuples * sizeof(uint32_t));
	uint32_t *outer_vals_1_hbm = (uint32_t *)hbw_malloc(outer_tuples * sizeof(uint32_t));
	uint32_t *outer_vals_2_hbm = (uint32_t *)hbw_malloc(outer_tuples * sizeof(uint32_t));
	// join result
	uint32_t *join_keys       = (uint32_t *)mamalloc(max_join_tuples * sizeof(uint32_t));
	uint32_t *join_inner_vals = (uint32_t *)mamalloc(max_join_tuples * sizeof(uint32_t));
	uint32_t *join_outer_vals = (uint32_t *)mamalloc(max_join_tuples * sizeof(uint32_t));
	// unique table and unique items
	size_t distinct = outer_distinct + inner_distinct - join_distinct;
	size_t unique_table_buckets = distinct * 2 + 1;
	while (!odd_prime(unique_table_buckets)) unique_table_buckets += 2;
	uint32_t *unique = (uint32_t *)malloc(distinct * sizeof(uint32_t));
	uint32_t *unique_table = (uint32_t *)calloc(unique_table_buckets, sizeof(uint32_t));
	// run threads
	int b, barriers = 64;
	pthread_barrier_t barrier[barriers];
	for (b = 0 ; b != barriers ; ++b)
		pthread_barrier_init(&barrier[b], NULL, threads);
	info_t_hj info[threads];
	size_t final_offsets[threads];
	uint64_t inner_checksum[threads];
	uint64_t outer_checksum[threads];
	uint32_t *inner_counts[threads];
	uint32_t *outer_counts[threads];
	volatile size_t block_counter = 0;
	volatile size_t close_gaps_counter = 0;
	int public_seed=rand();
	pthread_attr_t attr;
	cpu_set_t cpuset;
	pthread_attr_init(&attr);

	//data generation
	FILE *f_inner_keys;//=fopen("../data/inner_keys.data", "rb");
	FILE *f_inner_vals;//=fopen("../data/inner_vals.data", "rb");
	FILE *f_outer_keys;//=fopen("../data/outer_keys.data", "rb");
	FILE *f_outer_vals;//=fopen("../data/outer_vals.data", "rb");
	
	std::string name1,name2,name3,name4;
	name1.append("/home/s/shuhao-z/join_alg/data/ik");
	name2.append("/home/s/shuhao-z/join_alg/data/iv");
	name3.append("/home/s/shuhao-z/join_alg/data/ok");
	name4.append("/home/s/shuhao-z/join_alg/data/ov");
	name1.append("_");
	name2.append("_");
	name3.append("_");
	name4.append("_");
	name1.append(std::to_string(inner_tuples));	
	name2.append(std::to_string(inner_tuples));	
	name3.append(std::to_string(outer_tuples));	
	name4.append(std::to_string(outer_tuples));	
	name1.append("_");
	name2.append("_");
	name3.append("_");
	name4.append("_");
	name1.append(std::to_string(selc));	
	name2.append(std::to_string(selc));	
	name3.append(std::to_string(selc));	
	name4.append(std::to_string(selc));	
	name1.append(".txt");
	name2.append(".txt");
	name3.append(".txt");
	name4.append(".txt");
	
	f_inner_keys=fopen(name1.c_str(), "rb");
	f_inner_vals=fopen(name2.c_str(), "rb");
	f_outer_keys=fopen(name3.c_str(), "rb");
	f_outer_vals=fopen(name4.c_str(), "rb");		

	fread(inner_keys_1, inner_tuples, sizeof(int), f_inner_keys);
	fread(inner_vals_1, inner_tuples, sizeof(int), f_inner_vals);
	fread(outer_keys_1, outer_tuples, sizeof(int), f_outer_keys);
	fread(outer_vals_1, outer_tuples, sizeof(int), f_outer_vals);
	
	//data generation
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(0,&set);

	for (t = 0 ; t != threads ; ++t) {
		info[t].thread = t;
		int cpu_idx=get_cpu_id(t);
		CPU_ZERO(&cpuset);
		CPU_SET(cpu_idx,&cpuset);
		pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
		//CPU_SET(get_cpu_id(t), &cpuset);
		//pthread_setaffinity_np(info[t].id, sizeof(cpu_set_t), &cpuset);
		info[t].threads = threads;
		info[t].seed = public_seed;
		info[t].join_tuples = join_tuples;
		info[t].block_limit = block_limit;
		info[t].outer_tuples = outer_tuples;
		info[t].inner_tuples = inner_tuples;
		info[t].join_distinct = join_distinct;
		info[t].inner_distinct = inner_distinct;
		info[t].outer_distinct = outer_distinct;
		info[t].inner_keys[0] = inner_keys_1;
		info[t].inner_keys[1] = inner_keys_2;
		info[t].inner_vals[0] = inner_vals_1;
		info[t].inner_vals[1] = inner_vals_2;
		info[t].outer_keys[0] = outer_keys_1;
		info[t].outer_keys[1] = outer_keys_2;
		info[t].outer_vals[0] = outer_vals_1;
		info[t].outer_vals[1] = outer_vals_2;

		info[t].inner_keys[2] = inner_keys_1_hbm;
		info[t].inner_keys[3] = inner_keys_2_hbm;
		info[t].inner_vals[2] = inner_vals_1_hbm;
		info[t].inner_vals[3] = inner_vals_2_hbm;
		info[t].outer_keys[2] = outer_keys_1_hbm;
		info[t].outer_keys[3] = outer_keys_2_hbm;
		info[t].outer_vals[2] = outer_vals_1_hbm;
		info[t].outer_vals[3] = outer_vals_2_hbm;

		info[t].ratio = ratio;

		info[t].inner_counts = inner_counts;
		info[t].outer_counts = outer_counts;
		info[t].final_offsets = final_offsets;
		info[t].join_keys = join_keys;
		info[t].join_inner_vals = join_inner_vals;
		info[t].join_outer_vals = join_outer_vals;
		info[t].inner_checksum = inner_checksum;
		info[t].outer_checksum = outer_checksum;
		info[t].unique = unique;
		info[t].unique_table = unique_table;
		info[t].unique_table_buckets = unique_table_buckets;
		info[t].unique_factor = factors[0];
		info[t].thread_factor = factors[1];
		info[t].thread_factor2 = factors[4] ;
		info[t].inner_factor  = factors[2];
		info[t].outer_factor  = factors[3];
		info[t].hash_table_load = hash_table_load;
		info[t].hash_table_limit = hash_table_limit;
		info[t].buffer_size = buffer_size;
		info[t].block_size = block_size;
		info[t].block_counter = &block_counter;
		info[t].close_gaps_counter = &close_gaps_counter;
		info[t].barrier = barrier;
		info[t].timestep=0;
		pthread_create(&info[t].id, &attr, run_hj, (void*) &info[t]);
	}
	for (t = 0 ; t != threads ; ++t)
		pthread_join(info[t].id, NULL);
	for (b = 0 ; b != barriers ; ++b)
		pthread_barrier_destroy(&barrier[b]);
	//assert(block_counter <= block_limit);
	join_tuples = info[0].join_tuples;
	//fprintf(stderr, "Join tuples (measured): %9ld\n", join_tuples);
	//fprintf(stderr, "Join selectivity: %.3f\n", join_tuples * 1.0 / (inner_tuples + outer_tuples));

	double ttime=0.0;
	for (t=0;t<threads;++t)
	{
		if (info[t].time>ttime)
			ttime=info[t].time;
	}
	//ttime;
	printf ("%lf\t%lf\t%lf\t", ttime, info[0].time, info[128].time);
	printf ("\n");
	#ifdef TIMELOG
	for (t=1;t<4;++t)
	{
		printf ("%lf\t", info[0].timelog[t]-info[0].timelog[t-1]);
	}
        for (t=1;t<4;++t)
        {
                printf ("%lf\t", info[128].timelog[t]-info[128].timelog[t-1]);
        }
	printf ("\n");
	#endif
	free(join_keys);
	free(join_inner_vals);
	free(join_outer_vals);
	free(inner_keys_1);
	free(inner_keys_2);
	free(inner_vals_1);
	free(inner_vals_2);
	free(outer_keys_1);
	free(outer_keys_2);
	free(outer_vals_1);
	free(outer_vals_2);
	//free hbm
	hbw_free(inner_keys_1_hbm);
	hbw_free(inner_keys_2_hbm);
	hbw_free(inner_vals_1_hbm);
	hbw_free(inner_vals_2_hbm);
	hbw_free(outer_keys_1_hbm);
	hbw_free(outer_keys_2_hbm);
	hbw_free(outer_vals_1_hbm);
	hbw_free(outer_vals_2_hbm);
	return EXIT_SUCCESS;
} 
