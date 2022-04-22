# gpuVector
a header only CUDA based  library for algebraic vector computation.



## Usage

Download the file `gpuVector/gpuVector.cuh`,  and include it in you `.cu` files.



### initialization

The initialization function `gVector::Init(size_t len)` should be called before any usage. This function preallocates buffer for many reduce operation, say `sum`, `max`, `norm` , etc. 

The buffer length `len`，should be at least `max_len/512` , where `max_len`  is the maximal vector length you will create in you application . See the example:

```cpp
// test.cu
#include "gpuVector.cuh"
#include <vector>

int main(void) {
    gVector<float>::Init(1000);  // intialize library
    
    // initilize vector
    gv::gVector<float> v0(); // zero length vector will be created 
    
    gv::gVector<float> v1(10000); // create float vector of length 10000, default value is 0
    
    gv::gVector<float> v2(10000,2); // create float vector of length 10000, set default value to 2.f
    
    // or initialze values manually by
    v2.set(1.5); // set all elements of v2 to 1.5 
    
    // transfer data from host to vector
    std::vector<float> vhost(10000);
    v1.set(vhost.data());  // set v1 to be the same as vhost, the length of vhost must not be less than length of v1
    
    // resize the length of a vector
    v2.resize(1000);  // v2 will have new length of 1000
}

```

### 

### Algebraic computation and Lazy evaluation

All the vectors with same length can do algebraic computation like `+,-,*,/`, and these computation are all **element wise operation**. The composition of these operation is just an expression, no computation is down until the value of the expression is needed and the computation will be launched to a single kernel. For example:

```cpp
gVector<float> v1(1000),v2(1000),v3(1000);
gVector<float> v= 3.f * (v1 + v2) * v3 / v1 + v2^3 -1.0f;  
```

The expression in second line is lazy evaluated,  when construct `v`, the expression is computed in a single kernel and write value back to `v`. This feature greatly reduce the memory bandwidth since no intermediate results are stored.



### Other handy operation

Most the functions of `gVector`  can be divided into two categories, *map* and *reduce*. 

#### map

* `minimize/maximize`: clamp the vector with a given upper/lower bound, they are down in place.
* `min/max`: computes the small or larger value of two vectors and generate an expression of the result.
* `clamp`: clamp the vector with two value or other two vectors.
* `Sqrt`: element-wise square root .
* `pow`: element-wise power
* `dot`: computes the inner product of two vectors.
* `map`:  map a vector to another by given a lambda function



#### reduce 

* `min/max`: computes the minimal/maximal value in a vector
* `min_positive`: computes the minimal positive value in a vector
* `infnorm`: computes the infinity norm of a vector, i.e., the element with maximal absolute value.
* `norm`: computes the 2-norm of a vector
* `sqrnorm`: com
* `sum`: compute the summation of all elements in a vector.

Most of these operation can also be done on an expression, and they are still lazy evaluated:

```cpp
gVector<double> v1(1000),v2(1000)
double mi = ((v1.Sqrt() + v2) / v1).min();
gVector<double> v3 = v1.clamp(v1+v2^2,1+v1^2+v2);
```



## Compilation

The library should be compiled by `nvcc` with the following additional flags:

```shell
--std=c++17 --extended-lambda 
```

the examples are tested on CUDA10.
