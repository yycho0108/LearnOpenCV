#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>


#define ETA 0.6
#define DECAY 0.01

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
extern float sigmoid(float);
extern float sigmoidPrime(float);
extern float softplus(float);
extern float softplusPrime(float);
extern float ReLU(float);
extern float ReLUPrime(float);
extern float tanhPrime(float);
void softMax(Mat&,Mat&);

extern void correlate(cv::InputArray I, cv::OutputArray O,cv::InputArray W, bool flip=false);
extern Mat ave_pool(Mat& m, Size s);
extern int argmax(Mat&); 

#endif

