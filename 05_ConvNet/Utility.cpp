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
	return x * (1-x);
}


void softMax(Mat& src, Mat& dst){
	if(&dst != &src){
		src.copyTo(dst);
	}

	double m = 0;
	cv::minMaxIdx(src,nullptr,&m,nullptr,nullptr);
	exp(src-m,dst); //subtract by maximum to prevent overflow

	auto s = cv::sum(dst);
	cv::divide(dst,s,dst);
}


void correlate(cv::InputArray I, cv::OutputArray O,cv::InputArray W, bool flip){
	if(flip){
		Mat K;
		cv::flip(W,K,-1);
		return cv::filter2D(I,O,-1,K,Point(-1,-1),0.0,BORDER_CONSTANT);//convolution
	}else{
		return cv::filter2D(I,O,-1,W,Point(-1,-1),0.0,BORDER_CONSTANT);//correlation
	}
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


