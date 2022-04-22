#ifndef __GPU_VECTOR_CUH
#define __GPU_VECTOR_CUH

#include "gpuVector.h"

#ifdef __USE_GVECTOR_LAZY_EVALUATION

#include"cuda_runtime.h"
#include"iostream"
//#include"lib.cuh"
#include"type_traits"
#include <algorithm>

#define cuda_error_check do{ \
	auto err = cudaGetLastError(); \
	if (err != 0) { \
		printf("\x1b[31mCUDA error occured at line %d in file %s, error type %s \x1b[0m\n", __LINE__,__FILE__, cudaGetErrorName(err));\
	} \
}while(0)

/*
	===========     define computation kernel      =========== 
*/
namespace gv {
	template<typename T>
	__global__ void init_array_kernel(T* array, T value, int array_size) {
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
		if (tid < array_size) {
			array[tid] = value;
		}
	}
	// define some kernel function
	template <typename T, unsigned int blockSize>
	__device__ void warpReduce(volatile T *sdata, unsigned int tid) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}

	template <typename T, unsigned int blockSize>
	__device__ void warpMax(volatile T *sdata, unsigned int tid) {
		if (blockSize >= 64) { T s = sdata[tid + 32]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 32) { T s = sdata[tid + 16]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 16) { T s = sdata[tid + 8]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 8) { T s = sdata[tid + 4]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 4) { T s = sdata[tid + 2]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 2) { T s = sdata[tid + 1]; if (sdata[tid] < s) sdata[tid] = s; };
	}

	template <typename T, unsigned int blockSize>
	__device__ void warpMin(volatile T *sdata, unsigned int tid) {
		if (blockSize >= 64) { T s = sdata[tid + 32]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 32) { T s = sdata[tid + 16]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 16) { T s = sdata[tid + 8]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 8) { T s = sdata[tid + 4]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 4) { T s = sdata[tid + 2]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 2) { T s = sdata[tid + 1]; if (sdata[tid] > s) sdata[tid] = s; };
	}
	template<typename T, typename Tout, typename Lam>
	__global__ void map(T* g_data, Tout* dst, int n, Lam func) {
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
		if (tid < n) {
			dst[tid] = func(g_data[tid]);
		}
	}

	template<typename T, typename Lam>
	__global__ void map(T* dst, int n, Lam func) {
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
		if (tid < n) {
			dst[tid] = func(tid);
		}
	}

	template<typename Lam>
	__global__ void map(int n, Lam func) {
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
		if (tid < n) {
			func(tid);
		}
	}
	template<typename Scalar, typename graph_t>
	__global__ void compute_graph_kernel(Scalar* dst, int array_size, graph_t graph) {
		int tid = blockDim.x*blockIdx.x + threadIdx.x;
		if (tid >= array_size) return;
		dst[tid] = graph.eval(tid);
	}

	template<typename Tin, int blockSize = 512, typename Tout = Tin>
	__global__ void block_sum_kernel(const Tin* pdata, Tout* odata, size_t n) {
		__shared__ Tout sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockDim.x*blockIdx.x;
		Tout s = 0;
		// load data to block
		if (element_id < n) {
			s = pdata[element_id];
		}
		sdata[tid] = s;
		__syncthreads();

		// block reduce sum
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) { warpReduce<Tout, blockSize>(sdata, tid); }
		if (tid == 0) odata[blockIdx.x] = sdata[0];
	}
	template<typename T, int blockSize = 512>
	__global__ void block_max_kernel(const T* indata, T* odata, size_t n) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = -1e30;
		if (element_id < n) {
			s = indata[element_id];
		}
		sdata[tid] = s;
		__syncthreads();

		// block max 
		if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpMax<T, blockSize>(sdata, tid);
		if (tid == 0) odata[blockIdx.x] = sdata[0];
	}

	template<typename T, int blockSize = 512>
	__global__ void block_maxabs_kernel(const T* indata, T* odata, size_t n) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = -1e30;
		if (element_id < n) {
			s = abs(indata[element_id]);
		}
		sdata[tid] = s;
		__syncthreads();

		// block max 
		if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpMax<T, blockSize>(sdata, tid);
		if (tid == 0) odata[blockIdx.x] = sdata[0];
	}

	template<typename T, int blockSize = 512>
	__global__ void block_min_kernel(const T* indata, T* odata, size_t n) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = 1e30;
		if (element_id < n) {
			s = indata[element_id];
		}
		sdata[tid] = s;
		__syncthreads();

		// block max 
		if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpMin<T, blockSize>(sdata, tid);
		if (tid == 0) odata[blockIdx.x] = sdata[0];
	}

	template<typename T, int blockSize = 512>
	__global__ void block_dot_kernel(const T* v1p, const T* v2p, T* odata, size_t n) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = 0.f;
		// load data to block
		if (element_id < n) {
			s = v1p[element_id] * v2p[element_id];
		}
		sdata[tid] = s;
		__syncthreads();

		// block reduce sum
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
		if (tid == 0) odata[blockIdx.x] = sdata[0];
	}


	template<typename T, typename graph_t1, typename graph_t2, int blockSize = 512>
	__global__ void dot_graph_kernel(T* dump, int array_size, graph_t1 g1, graph_t2 g2) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = 0.f;
		// load data to block
		if (element_id < array_size) {
			s = g1.eval(element_id) * g2.eval(element_id);
		}
		sdata[tid] = s;
		__syncthreads();

		// block reduce sum
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
		if (tid == 0) dump[blockIdx.x] = sdata[0];
	}

	template<typename T, typename graph_t, int blockSize = 512>
	__global__ void sum_graph_kernel(T* dump, int array_size, graph_t graph) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = 0.f;
		// load data to block
		if (element_id < array_size) {
			s = graph.eval(element_id);
		}
		sdata[tid] = s;
		__syncthreads();

		// block reduce sum
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
		if (tid == 0) dump[blockIdx.x] = sdata[0];
	}

	template<typename T, typename graph_t, int blockSize = 512>
	__global__ void sqrnorm_graph_kernel(T* dump, int array_size, graph_t graph) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = 0.f;
		// load data to block
		if (element_id < array_size) {
			T val = graph.eval(element_id);
			s = val * val;
		}
		sdata[tid] = s;
		__syncthreads();

		// block reduce sum
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpReduce<T, blockSize>(sdata, tid);
		if (tid == 0) dump[blockIdx.x] = sdata[0];
	}

	template<typename T, typename graph_t, int blockSize = 512>
	__global__ void max_graph_kernel(T* odata, size_t n, graph_t graph) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = -1e30;
		if (element_id < n) {
			s = graph.eval(element_id);
		}
		sdata[tid] = s;
		__syncthreads();

		// block max 
		if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] < v) sdata[tid] = v; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpMax<T, blockSize>(sdata, tid);
		if (tid == 0) odata[blockIdx.x] = sdata[0];
	}

	template<typename T, typename graph_t, int blockSize = 512>
	__global__ void min_graph_kernel(T* odata, size_t n, graph_t graph) {
		__shared__ T sdata[blockSize];
		if (blockDim.x != blockSize) {
			printf("error block size does not match at line %d ! \n", __LINE__);
		}
		int tid = threadIdx.x;
		size_t element_id = threadIdx.x + blockIdx.x*blockDim.x;
		T s = 1e30;
		if (element_id < n) {
			s = graph.eval(element_id);
		}
		sdata[tid] = s;
		__syncthreads();

		// block max 
		if (blockSize >= 512) { if (tid < 256) { T v = sdata[tid + 256]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { T v = sdata[tid + 128]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { T v = sdata[tid + 64]; if (sdata[tid] > v) sdata[tid] = v; } __syncthreads(); }

		// use warpReduce to sum last 64 component 
		if (tid < 32) warpMin<T, blockSize>(sdata, tid);
		if (tid == 0) odata[blockIdx.x] = sdata[0];
	}

	__host__ void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size) {
		*block_size = prefer_block_size;
		*block_num = (num_tasks + prefer_block_size - 1) / prefer_block_size;
	}

	// dump_array_sum makes original array dirty, make sure dump is large enough
	template<typename T, int blockSize = 512>
	T dump_array_sum(T* dump, size_t n) {
		T sum;
		if (n <= 1) {
			cudaMemcpy(&sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
			return sum;
		}
		size_t grid_dim, block_dim;
		T* block_dump2 = dump;
		T* block_dump1 = dump + ((n + 63) / 2 / 32) * 32;
		do {
#if 0
			// input : dump1  output : dump2 
			std::swap(block_dump1, block_dump2);
			make_kernel_param(&grid_dim, &block_dim, n, blockSize);
			block_sum_kernel<T, blockSize> << <grid_dim, block_dim >> > (block_dump1, block_dump2, n);
			// error may occurred because of the inideal parallel block, the block result will overwrite latter data
#else
			make_kernel_param(&grid_dim, &block_dim, n, blockSize);
			block_sum_kernel<T, blockSize> << <grid_dim, block_dim >> > (block_dump2, block_dump2, n);
			// if the early block is excuted first, latter data will not be overwritten
#endif
		} while ((n = (n + blockSize - 1) / blockSize) > 1);
		cudaMemcpy(&sum, block_dump2, sizeof(T), cudaMemcpyDeviceToHost);
		return sum;
	}

	template<typename T>
	T dot(const T* indata1, const T* indata2, T* dump_buf, size_t n, T* dot_dst = nullptr) {
		constexpr int blockSize = 512;
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, n, blockSize);
		block_dot_kernel << <grid_dim, block_dim >> > (indata1, indata2, dump_buf, n);

		if (n <= blockSize) {
			T sum;
			cudaMemcpy(&sum, dump_buf, sizeof(T), cudaMemcpyDeviceToHost);
			if (dot_dst != nullptr)
				cudaMemcpy(dot_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
			//cuda_error_check;
			return sum;
		}
		else {
			T sum = dump_array_sum(dump_buf, (n + blockSize - 1) / blockSize);
			if (dot_dst != nullptr)
				cudaMemcpy(dot_dst, &sum, sizeof(T), cudaMemcpyHostToDevice);
			//cuda_error_check;
			return sum;
		}
	}

	template<typename T>
	T parallel_max(const T* indata, T* dump, size_t array_size, T* max_dst = nullptr) {
		constexpr int blockSize = 512;
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
		block_max_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

		if (array_size <= blockSize) {
			T max_num;
			cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
			if (max_dst != nullptr)
				cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
			return max_num;
		}
		else {
			T max_num = dump_max(dump, (array_size + blockSize - 1) / blockSize);
			if (max_dst != nullptr)
				cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
			return max_num;
		}
	}

	template<typename T>
	T parallel_maxabs(const T* indata, T* dump, size_t array_size, T* max_dst = nullptr) {
		constexpr int blockSize = 512;
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
		block_maxabs_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

		if (array_size <= blockSize) {
			T max_num;
			cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
			if (max_dst != nullptr)
				cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
			return max_num;
		}
		else {
			T max_num = dump_max(dump, (array_size + blockSize - 1) / blockSize);
			if (max_dst != nullptr)
				cudaMemcpy(max_dst, &max_num, sizeof(T), cudaMemcpyHostToDevice);
			return max_num;
		}
	}

	template<typename T>
	T parallel_min(const T* indata, T* dump, size_t array_size, T* min_dst = nullptr) {
		constexpr int blockSize = 512;
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
		block_min_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

		if (array_size <= blockSize) {
			T min_num;
			cudaMemcpy(&min_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
			if (min_dst != nullptr)
				cudaMemcpy(min_dst, &min_num, sizeof(T), cudaMemcpyHostToDevice);
			return min_num;
		}
		else {
			T min_num = dump_min(dump, (array_size + blockSize - 1) / blockSize);
			if (min_dst != nullptr)
				cudaMemcpy(min_dst, &min_num, sizeof(T), cudaMemcpyHostToDevice);
			return min_num;
		}
	}

	template<typename T>
	T parallel_sum(const T* indata, T* dump, size_t array_size, T* sum_dst = nullptr) {
		constexpr int blockSize = 512;
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, array_size, blockSize);
		block_sum_kernel << <grid_dim, block_dim >> > (indata, dump, array_size);

		if (array_size <= blockSize) {
			T array_sum;
			cudaMemcpy(&array_sum, dump, sizeof(T), cudaMemcpyDeviceToHost);
			if (sum_dst != nullptr) {
				cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
			}
			return array_sum;
		}
		else {
			T array_sum = dump_array_sum(dump, (array_size + blockSize - 1) / blockSize);
			if (sum_dst != nullptr) {
				cudaMemcpy(sum_dst, &array_sum, sizeof(T), cudaMemcpyHostToDevice);
			}
			return array_sum;
		}
	}

	template<typename T, int blockSize = 512>
	T dump_max(T* dump, size_t n) {
		T max_num = -1e30;
		if (n <= 1) {
			cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
			return max_num;
		}
		size_t grid_dim, block_dim;
		do {
			make_kernel_param(&grid_dim, &block_dim, n, blockSize);
			block_max_kernel<T, blockSize> << <grid_dim, block_dim >> > (dump, dump, n);
			//cudaDeviceSynchronize();
			//cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
			//std::cout << "current max num " << max_num << std::endl;
		} while ((n = (n + blockSize - 1) / blockSize) > 1);
		cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return max_num;
	}

	template<typename T, int blockSize = 512>
	T dump_min(T* dump, size_t n) {
		T min_num = 1e30;
		if (n <= 1) {
			cudaMemcpy(&min_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
			return min_num;
		}
		size_t grid_dim, block_dim;
		do {
			make_kernel_param(&grid_dim, &block_dim, n, blockSize);
			block_min_kernel<T, blockSize> << <grid_dim, block_dim >> > (dump, dump, n);
			//cudaDeviceSynchronize();
			//cudaMemcpy(&max_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
			//std::cout << "current max num " << max_num << std::endl;
		} while ((n = (n + blockSize - 1) / blockSize) > 1);
		cudaMemcpy(&min_num, dump, sizeof(T), cudaMemcpyDeviceToHost);
		return min_num;
	}

	template<typename T>
	void init_array(T* dev_array, T value, int array_size) {
		size_t grid_dim;
		size_t block_dim;
		make_kernel_param(&grid_dim, &block_dim, array_size, 512);
		init_array_kernel << <grid_dim, block_dim >> > (dev_array, value, array_size);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

};

/*
	===========     define vector and expression class      =========== 
*/
namespace gv {

	template<typename Scalar> class gVectorMap;
	template<typename Scalar> class gVector;
	template<typename Scalar> class gElementProxy;

	template<typename... > struct is_expression;
	template<typename, typename> struct min_exp_t;
	template<typename, typename> struct max_exp_t;
	template<typename> struct sqrt_exp_t;
	template<typename, typename > struct map_exp_t;
	template<typename T = float, typename std::enable_if<std::is_scalar<T>::value, int >::type = 0> struct scalar_t;
	template<typename Scalar, typename T = gVector<Scalar>, typename std::enable_if<std::is_same<T, gVector<Scalar>>::value, int>::type = 0> struct var_t;
	template<typename, typename >struct dot_exp_t;

	template<typename Scalar>
	class gVector {
	public:
		typedef Scalar Scalar;
		static gVector buf_vector;
	private:
		Scalar* _data = nullptr;
		size_t _size = 0;

		void build(size_t dim);

		template<typename>friend  class gVectorMap;

		template<typename Lambda>friend  void apply_vector(gVector& v1, const gVector& v2, Lambda func);

	protected:
		gVector(Scalar* data_ptr, size_t size) :_data(data_ptr), _size(size) {}

	protected:
		auto& _Get_data(void) { return _data; }
		auto& _Get_size(void) { return _size; }
	public:
		Scalar*& data() { return _data; }

		const Scalar* data() const { return _data; }

		size_t size() const { return _size; }

		void clear(void);

		bool empty(void) const { return _size == 0; }

		void swap(gVector& v2);

		void set(Scalar val) { init_array(data(), val, size()); }

		void set(int* filter, Scalar val);

		void set(const Scalar* host_ptr);

		gVector(void) :_size(0), _data(nullptr) {}

		//gVector(Scalar* host_ptr, size_t size);

		gVector(const gVector& v);

		//gVector(gVector&& v) noexcept {
		//	_data = v.data(); _size = v._size;
		//	v._data = nullptr; v._size = 0;
		//	std::cout << "Move constructor called" << std::endl;
		//}

		explicit gVector(size_t dim, Scalar default_value = 0) {
			_size = dim;
			cudaMalloc(&_data, _size * sizeof(Scalar));
			init_array(_data, default_value, _size);
			cuda_error_check;
		}

		virtual ~gVector(void) {
			//cuda_error_check;
			if (_data != nullptr) { cudaFree(_data); }
			cuda_error_check;
		}


	public:

		void resize(size_t dim) { build(dim); }

		// for init, this version will not free buf
		void resize(size_t dim, int) {
			cudaMalloc(&_data, dim * sizeof(Scalar));
			_size = dim;
		}


	public:
		const gVector& operator=(const gVector& v2);

		void download(Scalar* host_ptr) const;

		const gVector& operator+=(const gVector& v2);

		const gVector& operator-=(const gVector& v2);

		const gVector& operator*=(const gVector& v2);

		const gVector& operator/=(const gVector& v2);

		const gVector& operator/=(Scalar s);

		const gVector& operator*=(Scalar s);

		template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
		const gVector& operator+=(const expr_t& expr) {
			size_t expr_dim = expr.size();
			if (expr_dim != _size) {
				throw std::string("unmatched vector size !");
			}
			((*this) + expr).launch(_data, expr_dim);
			return *this;
		}

		template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
		const gVector& operator-=(const expr_t& expr) {
			size_t expr_dim = expr.size();
			if (expr_dim != _size) {
				throw std::string("unmatched vector size !");
			}
			((*this) - expr).launch(_data, expr_dim);
			return *this;
		}

		template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
		const gVector& operator/=(const expr_t& expr) {
			size_t expr_dim = expr.size();
			if (expr_dim != _size) {
				throw std::string("unmatched vector size !");
			}
			((*this) / expr).launch(_data, expr_dim);
			return *this;
		}

		template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
		const gVector& operator*=(const expr_t& expr) {
			size_t expr_dim = expr.size();
			if (expr_dim != _size) {
				throw std::string("unmatched vector size !");
			}
			((*this) * expr).launch(_data, expr_dim);
			return *this;
		}


		template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0 >
		const gVector& operator=(const expr_t& expr) {
			size_t expr_dim = expr.size();
			if (expr_dim != size()) {
				clear();
				build(expr_dim);
			}
			expr.launch(_data, expr_dim);
			return *this;
		}

		template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0>
		gVector(const expr_t& expr) {
			size_t expr_dim = expr.size();
			resize(expr_dim, 0);
			expr.launch(_data, expr_dim);
		}

		template<typename expr_t>
		const gVector& operator*=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
			auto new_expr = (*this)*expr;
			size_t expr_dim = expr.size();
			new_expr.launch(_data, expr_dim);
		}

		template<typename expr_t>
		const gVector& operator/=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
			auto new_expr = (*this) / expr;
			size_t expr_dim = expr.size();
			new_expr.launch(_data, expr_dim);
		}

		template<typename expr_t>
		const gVector& operator+=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
			auto new_expr = (*this) + expr;
			size_t expr_dim = expr.size();
			new_expr.launch(_data, expr_dim);
		}

		template<typename expr_t>
		const gVector& operator-=(const typename std::enable_if<expr_t::is_exp, expr_t>::type& expr) {
			auto new_expr = (*this) - expr;
			size_t expr_dim = expr.size();
			new_expr.launch(_data, expr_dim);
		}

		//Scalar operator[](int eid) const;

		gElementProxy<Scalar> operator[](int eid);

		void invInPlace(void);

		void maximize(Scalar s);

		void maximize(const gVector& v2);

		void minimize(Scalar s);

		void minimize(const gVector& v2);

		template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0, typename vec_t = gVector>
		min_exp_t<var_t<Scalar>, opExp_t> min(const opExp_t& op2) const {
			return min_exp_t<var_t<Scalar>, opExp_t>(var_t<Scalar>(*this), op2);
		}

		template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0, typename vec_t = gVector>
		max_exp_t<var_t<Scalar>, opExp_t> max(const opExp_t& op2) const {
			return max_exp_t<var_t<Scalar>, opExp_t>(var_t<Scalar>(*this), op2);
		}

		template<typename vec_t = gVector, typename Scalar_type = Scalar,
			typename std::enable_if<std::is_scalar<Scalar_type>::value, int>::type = 0>
			min_exp_t<var_t<Scalar>, scalar_t<Scalar_type>> min(Scalar_type op2) const {
			return min_exp_t<var_t<Scalar>, scalar_t<Scalar_type>>(var_t<Scalar>(*this), op2);
		}

		template<typename vec_t = gVector, typename Scalar_type = Scalar,
			typename std::enable_if<std::is_scalar<Scalar_type>::value, int>::type = 0>
			max_exp_t<var_t<Scalar>, scalar_t<Scalar_type>> max(Scalar_type op2)const {
			return max_exp_t<var_t<Scalar>, scalar_t<Scalar_type>>(var_t<Scalar>(*this), op2);
		}

		template<typename Lambda, typename vec_t = gVector>
		map_exp_t<var_t<Scalar>, Lambda> fmap(Lambda func) {
			return map_exp_t<var_t<Scalar>, Lambda>(var_t<Scalar>(*this), func);
		}

		Scalar max(void) const;

		Scalar min(void) const;

		Scalar min_positive(void) const;

		Scalar norm(void) const {
			return sqrt(dot(*this));
		}

		Scalar infnorm(void) const { return parallel_maxabs(_data, buf_vector.data(), size()); }

		Scalar sqrnorm(void) const { return dot(*this); }

		void Sqrt(void);

		template<typename Lambda, typename vec_t = gVector>
		void mapInplace(Lambda func) {
			auto expr = fmap<Lambda, vec_t>(func);
			expr.launch(_data, _size);
		}

		void clamp(Scalar lower, Scalar upper);

		void clamp(Scalar* lower, Scalar* upper);

		void clamp(gVector& vl, gVector& vu) { clamp(vl.data(), vu.data()); }

		gVector slice(int start, int end) const;

		gVector concated_one(const gVector& v2) const;

		gVector concated_one(Scalar val) const;

		void concate_one(const gVector& v2);

		void concate_one(Scalar val);

		template<typename Arg0, typename... Args>
		gVector concated(Arg0 arg0, Args... args) {
			return concated_one(arg0).concated(args...);
		}

		template<typename Arg0, typename... Args>
		void concate(const Arg0& arg0, Args... args) {
			concate_one(arg0);
			concate(args...);
		}

		void concate(void) { return; }

		gVector concated(void) const { return *this; }

		std::vector<Scalar> slice2host(int start, int end) const;

		static void Init(size_t max_vec_size) {
			if (std::is_same<Scalar, double>::value) {
				cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
			} else if (std::is_same<Scalar, float>::value) {
				cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
			}
			if (max_vec_size > buf_vector.size()) {
				buf_vector.resize(max_vec_size);
			}
		}


		static Scalar* get_dump_buf(void) { return buf_vector.data(); }

		Scalar sum(void) const;

		Scalar dot(const gVector& v2) const;

		template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0, typename T = gVector>
		Scalar dot(const opExp_t& ex) {
			return var_t<Scalar>(*this).dot(ex);
		}

		Scalar* begin(void) { return _data; }

		Scalar* end(void) { return _data + _size; }

		static gVectorMap<Scalar> Map(Scalar* ptr, size_t size) { return gVectorMap<Scalar>(ptr, size); }
	};


	template<typename Scalar> gVector<Scalar> gVector<Scalar>::buf_vector;

	template<typename Scalar = float>
	class gVectorMap
		:public gVector<Scalar>
	{
	public:
		gVectorMap(Scalar* data_ptr, size_t size) :gVector<Scalar>(data_ptr, size) {}

		const gVectorMap<Scalar>& operator=(const gVector<Scalar>& v2) const {
			cudaMemcpy(_data, v2.data(), v2.size(), cudaMemcpyDeviceToDevice);
			cuda_error_check;
			return *this;
		}

		template<typename expr_t, typename std::enable_if<is_expression<expr_t>::value, int>::type = 0>
		const gVectorMap& operator=(const expr_t& expr) {
			size_t expr_dim = expr.size();
			if (expr_dim != _size) {
				throw std::string("unmatched vector size !");
			}
			expr.launch(_data, expr_dim);
			return *this;
		}

		gVectorMap(const gVectorMap& vm2) = delete;

		~gVectorMap(void) override {
			_Get_data() = nullptr;
			_Get_size() = 0;
		}

	};

	template<typename Scalar = float>
	class gElementProxy {
		gVector<Scalar>::Scalar* address;
	public:
		explicit gElementProxy(Scalar* ptr) : address(ptr) {}

		const gElementProxy& gElementProxy::operator=(Scalar val) {
			cudaMemcpy(address, &val, sizeof(Scalar), cudaMemcpyHostToDevice);
			return (*this);
		}

		operator Scalar(void) const {
			Scalar val;
			cudaMemcpy(&val, address, sizeof(Scalar), cudaMemcpyDeviceToHost);
			return val;
		}
	};

	gVector<double> buf_vector;

	template<typename Scalar>
	gElementProxy<Scalar> gVector<Scalar>::operator[](int eid) {
		return gElementProxy(_data + eid);
	}

	template<typename Scalar>
	void gVector<Scalar>::build(size_t dim) {
		if (_size != dim) {
			clear();
			cudaMalloc(&_data, dim * sizeof(Scalar)); _size = dim;
			cuda_error_check;
		}
		else {
			return;
		}
	}

	template<typename Scalar>
	void gVector<Scalar>::clear(void) {
		if (_data == nullptr && _size == 0) { return; }
		cudaFree(_data);
		_size = 0;
		_data = nullptr;
	}

	template<typename Lambda, typename Scalar>
	void apply_vector(gVector<Scalar>& v1, const gVector<Scalar>& v2, Lambda func) {
		if (v2.size() != v1.size()) printf("warning : using two vectors with unmatched size !");
		Scalar* v1data = v1.data();
		const Scalar* v2data = v2.data();
		auto merge = [=] __device__(int eid) {
			v1data[eid] = func(v1data[eid], v2data[eid]);
		};
		size_t gridSize, blockSize;
		make_kernel_param(&gridSize, &blockSize, v1.size(), 512);
		traverse_noret << <gridSize, blockSize >> > (v1.size(), merge);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	gVector<Scalar>::gVector(const gVector<Scalar>& v) {
		cudaMalloc(&_data, v.size() * sizeof(Scalar));
		_size = v.size();
		cudaMemcpy(_data, v.data(), _size * sizeof(Scalar), cudaMemcpyDeviceToDevice);
		cuda_error_check;
	}

	template<typename Scalar>
	const gVector<Scalar>& gVector<Scalar>::operator=(const gVector<Scalar>& v2) {
		if (size() != v2.size()) {
			clear();
			build(v2.size());
		}
		cudaMemcpy(data(), v2.data(), sizeof(Scalar)*v2.size(), cudaMemcpyDeviceToDevice);
		cuda_error_check;
		return (*this);
	}

	template<typename Scalar>
	void gVector<Scalar>::download(Scalar* host_ptr) const {
		cudaMemcpy(host_ptr, data(), sizeof(Scalar)*size(), cudaMemcpyDeviceToHost);
		cuda_error_check;
	}

	template<typename Scalar>
	void gVector<Scalar>::set(const Scalar* host_ptr) {
		cudaMemcpy(data(), host_ptr, sizeof(Scalar)*size(), cudaMemcpyHostToDevice);
		cuda_error_check;
	}

	template<typename Scalar>
	const gVector<Scalar>& gVector<Scalar>::operator+=(const gVector<Scalar>& v2) {
		auto add = [=] __device__(Scalar v1, Scalar v2) {
			return v1 + v2;
		};
		apply_vector(*this, v2, add);
		return *this;
	}

	template<typename Scalar>
	const gVector<Scalar>& gVector<Scalar>::operator-=(const gVector<Scalar>& v2) {
		auto minus = [=] __device__(Scalar v1, Scalar v2) {
			return v1 - v2;
		};
		apply_vector(*this, v2, minus);
		return *this;
	}

	template<typename Scalar>
	const gVector<Scalar>& gVector<Scalar>::operator*=(const gVector<Scalar>& v2) {
		auto multiply = [=] __device__(Scalar v1, Scalar v2) {
			return v1 * v2;
		};
		apply_vector(*this, v2, multiply);
		return *this;
	}

	template<typename Scalar>
	const gVector<Scalar>& gVector<Scalar>::operator/=(const gVector<Scalar>& v2) {
		auto divide = [=]__device__(Scalar v1, Scalar v2) {
			return v1 / v2;
		};
		apply_vector(*this, v2, divide);
		return *this;
	}

	template<typename Scalar>
	const gVector<Scalar>& gVector<Scalar>::operator/=(Scalar s) {
		Scalar* ptr = _data;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map << <grid_size, block_size >> > (_data, size(), [=] __device__(int tid) { return ptr[tid] / s; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	template<typename Scalar>
	void gVector<Scalar>::invInPlace(void) {
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map << <grid_size, block_size >> > (ptr, size(), [=] __device__(int tid) { return 1 / ptr[tid]; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return;
	}

	template<typename Scalar>
	void gVector<Scalar>::swap(gVector<Scalar>& v2) {
		if (_size != v2.size()) {
			throw std::string("size does not match !");
		}
		auto ptr = _data;
		_data = v2._data;
		v2._data = ptr;
	}

	template<typename Scalar>
	const gVector<Scalar>& gVector<Scalar>::operator*=(Scalar s) {
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(data(), size(), [=] __device__(int tid) { return ptr[tid] * s; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	__device__ bool read_bit(int* flag, int offset) {
		int bit32 = flag[offset / 32];
		return bit32 & (offset % 32);
	}

	template<typename Scalar>
	void gVector<Scalar>::set(int* filter, Scalar val)
	{
		int len = size();
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, len, 512);
		map << <grid_size, block_size >> > (size(), [=] __device__(int tid) {
			if (read_bit(filter, tid)) { ptr[tid] = val; }
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	void gVector<Scalar>::maximize(Scalar s)
	{
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), [=]  __device__(int tid) {
			Scalar v = ptr[tid]; return v > s ? v : s;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	void gVector<Scalar>::maximize(const gVector<Scalar>& v2)
	{
		Scalar* v1data = data();
		const Scalar* v2data = v2.data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(v1data, size(), [=] __device__(int tid) {
			Scalar val1 = v1data[tid];
			Scalar val2 = v2data[tid];
			return val1 > val2 ? val1 : val2;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	void gVector<Scalar>::minimize(Scalar s)
	{
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), [=] __device__(int tid) {
			Scalar v = ptr[tid];
			return v < s ? v : s;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	void gVector<Scalar>::minimize(const gVector<Scalar>& v2)
	{
		Scalar* v1data = data();
		const Scalar* v2data = v2.data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(v1data, size(), [=] __device__(int tid) {
			Scalar val1 = v1data[tid];
			Scalar val2 = v2data[tid];
			return val1 < val2 ? val1 : val2;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	Scalar gVector<Scalar>::sum(void) const
	{
		//gVector tmp((size() + 511) / 512);
		Scalar res = parallel_sum(data(), buf_vector.data(), size());
		return res;
	}

	template<typename Scalar>
	void gVector<Scalar>::clamp(Scalar lower, Scalar upper)
	{
		Scalar* ptr = data();
		auto clamp_kernel = [=] __device__(int eid) {
			Scalar val = ptr[eid];
			if (lower > val) return lower;
			if (upper < val) return upper;
			return  val;
		};
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), clamp_kernel);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	void gVector<Scalar>::clamp(Scalar* lower, Scalar* upper)
	{
		Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, _size, 512);
		map<<<grid_size,block_size>>>(ptr, size(), [=] __device__(int eid) {
			Scalar val = ptr[eid];
			Scalar low = lower[eid], up = upper[eid];
			if (low > val) return low;
			if (up < val) return up;
			return val;
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	gVector<Scalar> gVector<Scalar>::slice(int start, int end) const
	{
		gVector res(end - start);
		const Scalar* ptr = data();
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, res.size(), 512);
		map << <grid_size, block_size >> > (res.data(), res.size(), [=]  __device__(int eid) {
			return ptr[eid + start];
		});
		cudaDeviceSynchronize();
		cuda_error_check;
		return res;
	}


	template<typename Scalar>
	std::vector<Scalar> gVector<Scalar>::slice2host(int start, int end) const
	{
		if (end < start || start < 0 || end >= size()) {
			throw std::string("invalid indices !");
		}
		std::vector<Scalar> res(end - start);
		cudaMemcpy(res.data(), data(), sizeof(Scalar)*size(), cudaMemcpyDeviceToHost);
		cuda_error_check;
		return res;
	}

	template<typename Scalar>
	Scalar gVector<Scalar>::dot(const gVector<Scalar>& v2) const {
		return gv::dot(data(), v2.data(), buf_vector.data(), size());
	}

	template<typename Scalar>
	Scalar gVector<Scalar>::max(void) const {
		return parallel_max(data(), buf_vector.data(), size());
	}

	template<typename Scalar>
	Scalar gVector<Scalar>::min(void) const {
		return parallel_min(data(), buf_vector.data(), size());
	}

	template<typename Scalar>
	Scalar gVector<Scalar>::min_positive(void) const {
		gVector tmp(size());
		Scalar* src = _data;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, size(), 512);
		map << <grid_size, block_size >> > (tmp.data(), size(), [=] __device__(int eid) {
			Scalar val = src[eid];
			if (val < 0) {
				val = 1e30;
			}
			return val;
		});
		cudaDeviceSynchronize();
		cuda_error_check;

		return tmp.min();
	}

	template<typename Scalar>
	void gVector<Scalar>::Sqrt(void) {
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, size(), 512);
		Scalar* src = _data;
		map << <grid_size, block_size >> > (_data, size(), [=] __device__(int tid) {
			return sqrt(src[tid]);
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename Scalar>
	gVector<Scalar> gVector<Scalar>::concated_one(const gVector<Scalar>& v2) const {
		gVector result(size() + v2.size());
		cudaMemcpy(result.data(), data(), sizeof(Scalar)*size(), cudaMemcpyDeviceToDevice);
		cudaMemcpy(result.data() + size(), v2.data(), sizeof(Scalar)*v2.size(), cudaMemcpyDeviceToDevice);
		cuda_error_check;
		return result;
	}

	template<typename Scalar>
	gVector<Scalar> gVector<Scalar>::concated_one(Scalar val) const {
		gVector result(size() + 1);
		cudaMemcpy(result.data(), data(), size() * sizeof(Scalar), cudaMemcpyDeviceToDevice);
		result[size()] = val;
		cuda_error_check;
		return result;
	}

	template<typename Scalar>
	void gVector<Scalar>::concate_one(const gVector<Scalar>& v2) {
		gVector old_vec = *this;
		size_t new_size = v2.size() + size();
		clear();
		build(new_size);
		cudaMemcpy(data(), old_vec.data(), sizeof(Scalar)*old_vec.size(), cudaMemcpyDeviceToDevice);
		cudaMemcpy(data() + old_vec.size(), v2.data(), sizeof(Scalar)*v2.size(), cudaMemcpyDeviceToDevice);
		cuda_error_check;
	}

	template<typename Scalar>
	void gVector<Scalar>::concate_one(Scalar val) {
		gVector old_vec = *this;
		size_t new_size = size() + 1;
		clear();
		build(new_size);
		cudaMemcpy(data(), old_vec.data(), old_vec.size() * sizeof(Scalar), cudaMemcpyDeviceToDevice);
		cudaMemcpy(data() + old_vec.size(), &val, sizeof(Scalar), cudaMemcpyHostToDevice);
		cuda_error_check;
	}

/*
************************************************************************************
*********************           expression definition          *********************
************************************************************************************
*/

	template<typename T>
	struct scalar_type {
		typedef typename T::value_type type;
	};

	template<typename Scalar, typename subExp_t>
	struct exp_t
		//:public exp_base_t
	{
		static constexpr bool is_exp = true;
		typedef Scalar value_type;
		void launch(Scalar* dst, int n) const {
			const subExp_t* p_graph = static_cast<const subExp_t*>(this);
			subExp_t graph = *p_graph;
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, n, 512);
			//std::cout << "launcing with size " << n << std::endl;
			//std::cout << "result at " << dst << std::endl;
			cuda_error_check;
			//std::cout << typeid(graph).name() << std::endl;
			compute_graph_kernel << <grid_size, block_size >> > (dst, n, graph);
			cudaDeviceSynchronize();
			cuda_error_check;
		}

		template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0>
		min_exp_t<subExp_t, opExp_t> min(const opExp_t& op2) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return min_exp_t<subExp_t, opExp_t>(*p_ex, op2);
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		min_exp_t<subExp_t, scalar_t<T>> min(T s) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return min_exp_t<subExp_t, scalar_t<T>>(*p_ex, scalar_t<T>(s));
		}

#if 1
		template<typename Scalar, typename T, typename std::enable_if<std::is_same<T, gVector<Scalar>>::value, int>::type = 0>
		min_exp_t<subExp_t, var_t<T>> min(const T& s) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return min_exp_t<subExp_t, var_t<T>>(*p_ex, var_t<T>(s));
		}
#else
		template<typename T, typename std::enable_if<std::is_same<T, gVector>::value, int>::type = 0>
		min_exp_t<var_t, var_t> min(const T& s) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return min_exp_t<var_t, var_t>(*p_ex, var_t(s));
		}
#endif

		template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0>
		max_exp_t<subExp_t, opExp_t> max(const opExp_t& op2) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return max_exp_t<subExp_t, opExp_t>(*p_ex, op2);
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		max_exp_t<subExp_t, scalar_t<T>> max(T s) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return max_exp_t<subExp_t, scalar_t<T>>(*p_ex, scalar_t<T>(s));
		}

		template<typename Scalar, typename T, typename std::enable_if<std::is_same<T, gVector<Scalar>>::value, int>::type = 0>
		max_exp_t<subExp_t, var_t<T>> max(const T& s) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return max_exp_t<subExp_t, var_t<T>>(*p_ex, var_t<T>(s));
		}

		template<typename Lambda>
		map_exp_t<subExp_t, Lambda> map(Lambda func) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			return map_exp_t<subExp_t, Lambda>(*p_ex, func);
		}

		template<typename opExp_t, typename std::enable_if<is_expression<opExp_t>::value, int>::type = 0>
		Scalar dot(const opExp_t& op2) const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			subExp_t graph1 = *p_ex;
			opExp_t graph2 = op2;
			Scalar* pbuf = gVector<Scalar>::get_dump_buf();
			//printf("pbuf = %p\n", pbuf);
			int n = op2.size();
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, n, 512);
			cuda_error_check;
			dot_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph1, graph2);
			cudaDeviceSynchronize();
			cuda_error_check;
			n = (n + 511) / 512;
			return dump_array_sum(pbuf, n);
		}

		Scalar sum(void)const {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			subExp_t graph = *p_ex;
			Scalar* pbuf = gVector<Scalar>::get_dump_buf();
			int n = graph.size();
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, n, 512);
			cuda_error_check;
			sum_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
			cudaDeviceSynchronize();
			cuda_error_check;
			n = (n + 511) / 512;
			return dump_array_sum(pbuf, n);
		}

		Scalar sqrnorm(void) {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			subExp_t graph = *p_ex;
			Scalar* pbuf = gVector<Scalar>::get_dump_buf();
			int n = graph.size();
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, n, 512);
			cuda_error_check;
			sqrnorm_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
			cudaDeviceSynchronize();
			cuda_error_check;
			n = (n + 511) / 512;
			return dump_array_sum(pbuf, n);
		}

		Scalar max(void) {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			subExp_t graph = *p_ex;
			Scalar* pbuf = gVector<Scalar>::get_dump_buf();
			int n = graph.size();
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, n, 512);
			cuda_error_check;
			max_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
			cudaDeviceSynchronize();
			cuda_error_check;
			n = (n + 511) / 512;
			return dump_max(pbuf, n);
		}

		Scalar min(void) {
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			subExp_t graph = *p_ex;
			Scalar* pbuf = gVector<Scalar>::get_dump_buf();
			int n = graph.size();
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, n, 512);
			cuda_error_check;
			min_graph_kernel << <grid_size, block_size >> > (pbuf, n, graph);
			cudaDeviceSynchronize();
			cuda_error_check;
			n = (n + 511) / 512;
			return dump_min(pbuf, n);
		}

		Scalar norm(void) {
			return sqrt(sqrnorm());
		}

		void toMatlab(const char* name) {
#if defined(__GVECTOR_WITH_MATLAB)  
			const subExp_t* p_ex = static_cast<const subExp_t*>(this);
			subExp_t graph = *p_ex;
			gVector vec = graph;
			vec.toMatlab(name);
#endif
		}

	};

	template<typename exp> struct exp_scalar { typedef typename exp::value_type type; };
	template<typename exp> using exp_scalar_t = exp_scalar<exp>::type;
	template<typename _T1, typename _T2> struct scalar_result { typedef decltype(_T1{ 0 }*_T2{ 0 }) type; };
	template<typename _T1, typename _T2> using  scalar_result_t = scalar_result<_T1, _T2>::type;

template<typename  Scalar, typename T /*= gVector<Scalar>*/, typename std::enable_if<std::is_same<T, gVector<Scalar>>::value, int>::type /*= 0*/>
struct var_t
	:public exp_t<Scalar, var_t<Scalar, T> >
{
	const Scalar* ptr;
	int vec_dim;
	__host__ __device__ var_t(const Scalar* ptr_) :ptr(ptr_) {}
	__host__ __device__ var_t(const gVector<Scalar>& var) : ptr(var.data()), vec_dim(var.size()) {}
	__device__ Scalar eval(int eid)const {
		return ptr[eid];
	}
	__host__ __device__ int size(void)const {
		return vec_dim;
	}
};

template<typename T /*= Scalar*/, typename std::enable_if<std::is_scalar<T>::value, int >::type /*= 0*/>
struct scalar_t
	:public exp_t<T, scalar_t<T>>
{
	T scalar;
	__host__ __device__ scalar_t(T s) :scalar(s) {}
	__device__ T eval(int eid) const {
		return scalar;
	}
	__host__ __device__ int size(void) const {
		return 0;
	}
};

template<typename subExp_t, typename opExp_t>
struct unary_exp_t
	:public exp_t<typename exp_scalar_t<subExp_t>, subExp_t>
{
	typedef exp_t<typename exp_scalar_t<subExp_t>, subExp_t>::value_type Scalar;
	opExp_t exp;
	__host__ __device__ unary_exp_t(const opExp_t& opexp) :exp(opexp) {}
	__host__ __device__ int size(void) const {
		return exp.size();
	}
};

template<typename opExp_t>
struct negat_exp_t
	:public unary_exp_t<negat_exp_t<opExp_t>, opExp_t>
{
	//typedef value_type Scalar;
	typedef unary_exp_t<negat_exp_t<opExp_t>, opExp_t> baseType;
	typedef typename baseType::Scalar Scalar;
	__host__ __device__ negat_exp_t(const opExp_t& ex) : unary_exp_t<negat_exp_t<opExp_t>, opExp_t>(ex) {}
	__device__ Scalar eval(int eid) const {
		return -unary_exp_t<negat_exp_t<opExp_t>, opExp_t>::exp.eval(eid);
	}
};

template<typename opExp_t>
struct sqrt_exp_t
	:public unary_exp_t<sqrt_exp_t<opExp_t>, opExp_t>
{
	typedef unary_exp_t<sqrt_exp_t<opExp_t>, opExp_t> baseType;
	typedef typename baseType::Scalar Scalar;
	__host__ __device__ sqrt_exp_t(const opExp_t& ex) :unary_exp_t<sqrt_exp_t<opExp_t>, opExp_t>(ex) {}
	__device__ Scalar eval(int eid) const {
		return sqrt(unary_exp_t<sqrt_exp_t<opExp_t>, opExp_t>::exp.eval(eid));
	}
};

template<typename opExp_t, typename Lambda>
struct map_exp_t
	:public unary_exp_t<map_exp_t<opExp_t, Lambda>, opExp_t>
{
	typedef unary_exp_t<map_exp_t<opExp_t, Lambda>, opExp_t> baseType;
	typedef typename baseType::Scalar Scalar;
	Lambda _map;
	__host__ __device__ map_exp_t(const opExp_t& ex, Lambda map)
		: unary_exp_t<map_exp_t<opExp_t, Lambda>, opExp_t>(ex), _map(map)
	{ }
	__device__ Scalar eval(int eid) const {
		return _map(unary_exp_t<map_exp_t<opExp_t, Lambda>, opExp_t>::exp.eval(eid));
	}
};

template<typename subExp_t, typename opExp1_t, typename opExp2_t>
struct binary_exp_t
	:public exp_t<
	scalar_result_t<typename exp_scalar_t<opExp1_t>, typename exp_scalar_t<opExp2_t>>,
	subExp_t>
{
	typedef exp_t< 
		scalar_result_t<typename exp_scalar_t<opExp1_t>,
		typename exp_scalar_t<opExp2_t>>,
		subExp_t>::value_type Scalar;
	opExp1_t exp1;
	opExp2_t exp2;
	__host__ __device__ binary_exp_t(const opExp1_t& op1, const opExp2_t& op2) : exp1(op1), exp2(op2) {}
	__host__ __device__ int size(void) const {
		return (std::max)(exp1.size(), exp2.size());
	}
};

template<typename opExp1_t, typename opExp2_t>
struct add_exp_t
	:public binary_exp_t<add_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	//typedef value_type Scalar;
	typedef binary_exp_t<add_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ add_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<add_exp_t, opExp1_t, opExp2_t>(op1, op2) {}
	// add_exp_t(const add_exp_t<opExp1_t,opExp2_t>& ex): binary_exp_t<add_exp_t,opExp1_t,opExp2_t>(ex.exp1,ex.exp2){}
	__device__ baseType::Scalar eval(int eid) const {
		return baseType::exp1.eval(eid) + baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t, typename opExp2_t>
struct minus_exp_t
	:public binary_exp_t<minus_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	//typedef value_type Scalar;
	typedef binary_exp_t<minus_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ minus_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<minus_exp_t, opExp1_t, opExp2_t >(op1, op2) {}

	__device__ baseType::Scalar eval(int eid) const {
		return baseType::exp1.eval(eid) - baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t, typename opExp2_t>
struct div_exp_t
	:public binary_exp_t<div_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	//typedef value_type Scalar;
	typedef binary_exp_t<div_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ div_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<div_exp_t, opExp1_t, opExp2_t >(op1, op2) {}

	__device__ baseType::Scalar eval(int eid)const {
		return baseType::exp1.eval(eid) / baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t, typename opExp2_t>
struct multiply_exp_t
	:public binary_exp_t<multiply_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	//typedef value_type Scalar;
	typedef  binary_exp_t<multiply_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ multiply_exp_t(const opExp1_t& op1, const opExp2_t& op2) : binary_exp_t<multiply_exp_t/*<opExp1_t, opExp2_t>*/, opExp1_t, opExp2_t>(op1, op2) {}
	// multiply_exp_t(const multiply_exp_t& ex) :baseType(ex.exp1, ex.exp2) {}
	__device__ baseType::Scalar eval(int eid) const {
		return baseType::exp1.eval(eid)*baseType::exp2.eval(eid);
	}
};

template<typename opExp1_t, typename opExp2_t>
struct pow_exp_t
	:public binary_exp_t<pow_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	//typedef value_type Scalar;
	typedef binary_exp_t<pow_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	__host__ __device__ pow_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<pow_exp_t, opExp1_t, opExp2_t >(op1, op2) {}

	__device__ typename baseType::Scalar eval(int eid) const {
		return std::is_same<typename baseType::Scalar, float>::value ? powf(baseType::exp1.eval(eid), baseType::exp2.eval(eid)) : pow(baseType::exp1.eval(eid), baseType::exp2.eval(eid));
	}
};

template<typename opExp1_t, typename opExp2_t>
struct min_exp_t
	:public binary_exp_t<min_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	//typedef value_type Scalar;
	typedef binary_exp_t<min_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	typedef baseType::Scalar Scalar;
	__host__ __device__ min_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<min_exp_t, opExp1_t, opExp2_t >(op1, op2) {}
	__device__ Scalar eval(int eid) const {
		Scalar val1 = baseType::exp1.eval(eid);
		Scalar val2 = baseType::exp2.eval(eid);
		return val1 < val2 ? val1 : val2;
	}
};

template<typename opExp1_t, typename opExp2_t>
struct max_exp_t
	:public binary_exp_t<max_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t>
{
	//typedef value_type Scalar;
	typedef binary_exp_t<max_exp_t<opExp1_t, opExp2_t>, opExp1_t, opExp2_t> baseType;
	typedef typename baseType::Scalar Scalar;
	__host__ __device__ max_exp_t(const opExp1_t& op1, const opExp2_t& op2) :binary_exp_t<max_exp_t, opExp1_t, opExp2_t >(op1, op2) {}
	__device__ Scalar eval(int eid) const {
		Scalar val1 = baseType::exp1.eval(eid);
		Scalar val2 = baseType::exp2.eval(eid);
		return val1 > val2 ? val1 : val2;
	}
};


/*******************************************************************************************
*                         negate      expression                                           *
********************************************************************************************/
template<typename Scalar>
negat_exp_t<var_t<Scalar>> operator-(const gVector<Scalar>& gvec) {
	return negat_exp_t<var_t<Scalar>>(var_t<Scalar>(gvec));
}

#define DEFINE_BINEXP_SCALAR_VAR(op,opsign)  \
template<typename Scalar1, typename Scalar2,  \
std::enable_if_t<std::is_scalar_v<Scalar1>,int> = 0,  \
std::enable_if_t<std::is_scalar_v<Scalar2>,int> = 0  \
> \
op##_exp_t<scalar_t<Scalar1>, var_t<Scalar2>> operator opsign( \
	Scalar1 s,  \
	const gVector<Scalar2>& v2  \
	) {  \
	return op##_exp_t<scalar_t<Scalar1>, var_t<Scalar2>>(scalar_t<Scalar1>(s), var_t<Scalar2>(v2)); \
}

#define DEFINE_BINEXP_VAR_SCALAR(op,opsign)  \
template<typename Scalar1,typename Scalar2 , \
std::enable_if_t<std::is_scalar_v<Scalar1>,int> = 0, \
std::enable_if_t<std::is_scalar_v<Scalar2>,int> = 0 \
>  \
op##_exp_t<var_t<Scalar1>, scalar_t<Scalar2>> operator opsign(  \
	const gVector<Scalar1>& v1,  \
	Scalar2 v2   \
	) {  \
	return op##_exp_t<var_t<Scalar1>, scalar_t<Scalar2>>(var_t<Scalar1>(v1), scalar_t<Scalar2>(v2)); \
}

#define DEFINE_BINEXP_VAR_VAR(op,opsign)  \
template<typename Scalar1,typename Scalar2, \
typename std::enable_if<std::is_scalar_v<Scalar1>,int>::type = 0, \
typename std::enable_if<std::is_scalar_v<Scalar2>,int>::type = 0 \
>  \
op##_exp_t<var_t<Scalar1>, var_t<Scalar2>> operator opsign(  \
	const gVector<Scalar1>& v1,  \
	const gVector<Scalar2>& v2   \
	) {  \
	return op##_exp_t<var_t<Scalar1>, var_t<Scalar2>>(var_t<Scalar1>(v1), var_t<Scalar2>(v2)); \
}

#define DEFINE_BINEXP_EXP_VAR(op,opsign)  \
template<typename Scalar, typename opExp1_t,  \
	std::enable_if_t<std::is_scalar_v<Scalar>, int> = 0, \
	std::enable_if_t<is_expression<opExp1_t>::value, int> = 0 > \
	op##_exp_t<opExp1_t, var_t<Scalar>> operator opsign( \
		const opExp1_t& op1, const gVector<Scalar>& v2) { \
	static_assert(opExp1_t::is_exp, "Not a vector expression"); \
		return op##_exp_t<opExp1_t, var_t<Scalar>>(op1, var_t<Scalar>(v2)); \
}

#define DEFINE_BINEXP_VAR_EXP(op,opsign)  \
template<typename Scalar, typename opExp2_t, \
    std::enable_if_t<std::is_scalar_v<Scalar>,int> = 0, \
	std::enable_if_t<is_expression<opExp2_t>::value, int> = 0 > \
	op##_exp_t<var_t<Scalar>, opExp2_t> operator opsign( \
		const gVector<Scalar>& v1, \
		const opExp2_t& op2 \
		) {  \
	static_assert(opExp2_t::is_exp, "Not a vector expression"); \
	return op##_exp_t<var_t<Scalar>, opExp2_t>(var_t<Scalar>(v1), op2); \
}

#define DEFINE_BINEXP_SCALAR_EXP(op,opsign)  \
template<typename Scalar, typename opExp2_t,  \
    std::enable_if_t<std::is_scalar_v<Scalar>,int> = 0, \
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 >  \
	op##_exp_t<scalar_t<Scalar>, opExp2_t > operator opsign(  \
		Scalar s1,  \
		const opExp2_t& op2  \
		) { \
	static_assert(opExp2_t::is_exp, "Not a vector expression"); \
	return op##_exp_t<scalar_t<Scalar>, opExp2_t>(scalar_t<Scalar>(s1), op2); \
}

#define DEFINE_BINEXP_EXP_SCALAR(op,opsign)  \
template<typename Scalar, typename opExp1_t,  \
    std::enable_if_t<std::is_scalar_v<Scalar>,int> = 0, \
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0 > \
	op##_exp_t<opExp1_t, scalar_t<Scalar>> operator opsign( \
		const opExp1_t& op1, \
		Scalar s2 \
		) { \
	static_assert(opExp1_t::is_exp, "Not a vector expression"); \
	return op##_exp_t<opExp1_t, scalar_t<Scalar>>(op1, scalar_t<Scalar>(s2)); \
}

#define DEFINE_BINEXP_EXP_EXP(op, opsign) \
template<typename opExp1_t, typename opExp2_t, \
	typename std::enable_if<is_expression<opExp1_t>::value, int>::type = 0, \
	typename std::enable_if<is_expression<opExp2_t>::value, int>::type = 0 > \
	op##_exp_t<opExp1_t, opExp2_t> operator opsign(  \
		const opExp1_t& op1,  \
		const opExp2_t& op2) {  \
	static_assert(opExp1_t::is_exp, "Not a vector expression");  \
	static_assert(opExp2_t::is_exp, "Not a vector expression");  \
	return op##_exp_t<opExp1_t, opExp2_t>(op1, op2); \
}

#define DEFINE_BINEXP(op,opsign)    DEFINE_BINEXP_SCALAR_VAR(op,opsign)  \
									DEFINE_BINEXP_VAR_SCALAR(op,opsign)  \
									DEFINE_BINEXP_VAR_VAR(op, opsign)    \
									DEFINE_BINEXP_EXP_VAR(op, opsign)    \
									DEFINE_BINEXP_VAR_EXP(op, opsign)    \
									DEFINE_BINEXP_SCALAR_EXP(op, opsign) \
									DEFINE_BINEXP_EXP_SCALAR(op, opsign) \
									DEFINE_BINEXP_EXP_EXP(op, opsign) 


//=============================================================================================================================

template<typename _T,typename _T2>
struct is_expression_impl {
	static constexpr bool value = false;
};

template<typename... _T>
struct is_expression<add_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<minus_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<multiply_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<div_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<pow_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<negat_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<sqrt_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<map_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<max_exp_t<_T...>> {
	static constexpr bool value = true;
};

template<typename... _T>
struct is_expression<min_exp_t<_T...>> {
	static constexpr bool value = true;
};
//=============================================================================================================================


template<typename opExp_t>
negat_exp_t<opExp_t> operator-(
	const typename std::enable_if<opExp_t::is_exp, opExp_t>::type&
	exp) {
	return negat_exp_t<opExp_t>(exp);
}

// bindary expression definition
DEFINE_BINEXP(add, +);
DEFINE_BINEXP(minus, -);
DEFINE_BINEXP(multiply, *);
DEFINE_BINEXP(div, / );
DEFINE_BINEXP(pow, ^);

};

#endif

#endif


