#include "Utility.h"

ForEach::ForEach(uchar* ptr, function<float(float)> f)
	:p((float*)ptr),f(f){
	
}
ForEach::ForEach(uchar* ptr, float (*f)(float))
	:p((float*)ptr),f(f){

}

void ForEach::operator()(const Range& r) const{
	for(int i= r.start; i != r.end; ++i){
		p[i] = f(p[i]);
	}
}

bool isnan(Mat& m){
	for(auto i = m.begin<float>(); i != m.end<float>(); ++i){
		if(isnan(*i) || isinf(*i)){
			return true;
		}
	}
	return false;
}

bool isnan(std::vector<Mat>& v){
	for(auto& m : v){
		if(isnan(m))
			return true;
	}
	return false;
}

float sigmoid(float x){
	//if(isnan(x))
	//	throw "SISNAN!!!!!!!";
	return  1.0/(1.0 + exp(-x));
	//cout << val;
}

float sigmoidPrime(float x){
	x = sigmoid(x);
	return x * (1-x);
}

float softplus(float x){
	return	log(1+exp(x));
}

float softplusPrime(float x){
	return sigmoid(x); 
}
float ReLU(float x){
	return x>0?x:0;
}
float ReLUPrime(float x){
	return x>0?1:0;
}

//float tanh(float x){
//	return tanh(x);
//}
float tanhPrime(float x){
	x = tanh(x);
	return 1 - x*x;
	//return x * (1-x);
}


void softMax(Mat& src, Mat& dst){
	if(&dst != &src){
		src.copyTo(dst);
	}
	double m=0;
	//m = *std::max_element(src.begin<float>(),src.end<float>());
	cv::minMaxLoc(src,nullptr,&m,nullptr,nullptr);
	exp(src-m,dst); //subtract by maximum to prevent overflow

	auto s = cv::sum(dst);
	cv::divide(dst,s,dst);
}


void correlate(Mat& I, Mat& O, Mat& W, bool flip){
	Mat K;
	if(flip)
		cv::flip(W,K,-1); //convolution
	else
		K=W; //correlation


	cv::filter2D(I,O,-1,K,Point(-1,-1),0.0,BORDER_CONSTANT);
	
	//std::cout << "I" << I << endl;
	//std::cout << "O" << O << endl;

	//if(isnan(O))
	//	throw "OISNAN!";
	return;
}

Mat ave_pool(Mat& m, Size s){
	Mat res;
	cv::pyrDown(m,res,s,cv::BORDER_REPLICATE);
	return res;
}

int argmax(Mat& m){
	auto i = std::max_element(m.begin<float>(),m.end<float>());
	return std::distance(m.begin<float>(),i);
}

float constrain(float x, float l, float h){
	return x<l?l:x>h?h:x;
}

float ETA = 0.6;
float DECAY = 0.001;
