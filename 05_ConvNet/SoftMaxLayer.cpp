#include "SoftMaxLayer.h"

SoftMaxLayer::SoftMaxLayer(){
	d=0;
}

std::vector<Mat>& SoftMaxLayer::FF(std::vector<Mat> _I){
	d = _I.size();
	O.resize(d);
	G.resize(d);
	I.swap(_I);
	for(int i=0;i<d;++i){
		softMax(I[i],O[i]);
	}
	//cout << "O" << endl << O[0] << endl;
	return O;
}
std::vector<Mat>& SoftMaxLayer::BP(std::vector<Mat> _G){
	throw "SoftMAX : BP Not Implemented";
	G.swap(_G);
	return G;
}
double SoftMaxLayer::cost(){
	Mat m;
	for(int i=0;i<d;++i){
		O[i].copyTo(m);
		parallel_for_(Range(0,m.rows*m.cols),ForEach(m.data,[](float a){return 0.5 * a*a;}));
	}
	return cv::sum(m)[0];
}
void SoftMaxLayer::setup(Size s){
	this->s=s;
}
Size SoftMaxLayer::outputSize(){
	return s;
}


