#include"cuda_runtime.h"
#include"vector"
#include"gpuVector.cuh"

//#define __DEBUG_GVECTOR

using namespace gv;

void test_gVector(void) {
	gVector<float>::Init(100000);
	{
		gVector<float> v1(10000), v2(10000), v3(10000);
		v1.set(1);
		v2.set(1.5);
		v3.set(3);
		//gVector<float> v4 = v1 + v2;
		gVector<float> vx = v1 * v2 * v3 * v3;
		//constexpr bool val = decltype(v1 - v2)::is_exp;
		gVector<float> v4 = (v1 - v2) / v3 + v1 * v1 - 3;

		std::cout << "v4.max      " << v4.max() << std::endl;
		std::cout << "v4.min      " << v4.min() << std::endl;
		std::cout << "v4.sum      " << v4.sum() << std::endl;
		std::cout << "||v4||      " << v4.norm() << std::endl;
		std::cout << "||v4 / 2||  " << (v4 / 2.f).norm() << std::endl;


		// test dot
		std::cout << "v2'*(v1+v3) = " << v2.dot(v1 + v3) << std::endl;

		// test map
		gVector<float> v5 = gVector<float>::Map(v4.data(), 1000);
		std::cout << "v5.max         " << v5.max() << std::endl;
		std::cout << "v5.min         " << v5.min() << std::endl;
		std::cout << "||v5||         " << v5.norm() << std::endl;
		std::cout << "||v5 * 2||     " << (v5 * 2.f).norm() << std::endl;

		// test index access
		gVector<float> v6 = gVector<float>::Map(v5.data(), 10);
		std::cout << "old value : " << std::endl;
		for (int i = 0; i < 10; i++) {
			float val = v6[i];
			printf("v6[%d] = %f\n", i, val);
			v6[i] = val - 1;
		}

		std::cout << "new value : " << std::endl;
		for (int i = 0; i < 10; i++) {
			float val = v6[i];
			printf("v6[%d] = %f\n", i, val);
		}

		// test max/minimize 
		v5.maximize(1);
		printf("max(v5,1).max = %f\n", v5.max());
		printf("max(v5,1).min = %f\n", v5.min());

		v5.minimize(-1);
		printf("min(v5, -1).max = %f\n", v5.max());
		printf("min(v5, -1).min = %f\n", v5.min());

		printf("||max(v5,2)|| = %f\n", v5.max(2).norm());

		// test concate
		gVector<float> v7 = v1.concated(v2, v3);
		printf("v7.size = %d\n", v7.size());
		printf("v7.max  = %f \n", v7.max());
		printf("v7.min  = %f \n", v7.min());
		cuda_error_check;

		//gVector<float> v8;
		//v8.concate(gVector<float>(2, 1), gVector<float>(2, 2), gVector<float>(2, 3), 4, 4);
		//printf("v8 = \n");
		//for (int i = 0; i < v8.size(); i++) {
		//	printf("%f\n", float(v8[i]));
		//}
	}
}




