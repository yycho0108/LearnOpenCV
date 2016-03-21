#include "ActivationLayer.h"

void activate(Mat& src, Mat& dst, float (*f)(float)){
	if(&dst != &src){
		src.copyTo(dst);
	}
	/*for(auto i = dst.begin<float>();i != dst.end<float>(); ++i){
		auto& e = *i;
		e = f(e);
	}*/
	parallel_for_(Range(0,dst.rows*dst.cols),ForEach(dst.data,f));

	//if(isnan(dst))
	//		throw "DISNAN";
}

ActivationLayer::ActivationLayer(std::string _f){
	for(auto& c : _f){
		c = std::tolower(c);
	}

	if(_f == "sigmoid"){
		f = sigmoid;
		f_d = sigmoidPrime;
	}else if (_f == "softplus"){
		f = softplus;
		f_d = softplusPrime;
	}else if (_f == "relu"){
		f = ReLU;
		f_d = ReLUPrime;
	}else if (_f == "tanh"){
		f = tanh;	
		f_d = tanhPrime;
	}else{
		throw "WRONG ACTIVATION FUNCTION!!";
	}
}

std::vector<Mat>& ActivationLayer::FF(std::vector<Mat> _I){
	d = _I.size();
	O.resize(d);
	G.resize(d);
	//assert same size
	I.swap(_I);
	for(int i=0;i<d;++i){
		activate(I[i],O[i],f);
	}
	return O;
}
std::vector<Mat>& ActivationLayer::BP(std::vector<Mat> _G){
	Mat tmp;
	for(int i=0;i<d;++i){
		activate(I[i],tmp,f_d);
		G[i] = _G[i].mul(tmp);
	}
	return G;
}
void ActivationLayer::setup(Size s){
	this->s = s; //doesn't really matter, but necessary to transfer size to next layer
}
Size ActivationLayer::outputSize(){
	return s;
}
