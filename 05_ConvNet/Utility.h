#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>
#include <iostream>


#define namedPrint(x) \
	std::cout << #x << " : " << x << std::endl;

using namespace std;
using namespace cv;

class ForEach : public ParallelLoopBody{
	private:
		float* p;
		function<float(float)> f;
	public:
		ForEach(uchar* ptr, function<float(float)> f);
		ForEach(uchar* ptr, float (*f)(float));
		virtual void operator()(const Range& r) const;
};

extern bool isnan(Mat&);
extern bool isnan(std::vector<Mat>&);
extern float sigmoid(float);
extern float sigmoidPrime(float);
extern float softplus(float);
extern float softplusPrime(float);
extern float ReLU(float);
extern float ReLUPrime(float);
extern float tanhPrime(float);
void softMax(Mat&,Mat&);

extern void correlate(Mat& I, Mat& O, Mat& W, bool flip=false);
extern Mat ave_pool(Mat& m, Size s);
extern int argmax(Mat&); 


extern float constrain(float x, float l, float h);

extern float ETA;
extern float DECAY;
extern bool TEST_STATE; 
#endif

