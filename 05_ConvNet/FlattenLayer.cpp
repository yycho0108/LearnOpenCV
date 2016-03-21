#include "FlattenLayer.h"

FlattenLayer::FlattenLayer(int d):d(d){
	O.push_back(Mat());
}

std::vector<Mat>& FlattenLayer::FF(std::vector<Mat> _I){
	I.swap(_I);
	int n = I.size();
	s = I[0].size(); //will be unnecessary soon
	O[0] = I[0].reshape(0,s.width*s.height);
	for(int i=1;i<n;++i){
		cv::vconcat(O[0],I[i].reshape(0,s.width*s.height),O[0]);
	}
	return O;
}

std::vector<Mat>& FlattenLayer::BP(std::vector<Mat> _G){
	G.resize(I.size());
	int n = _G[0].size().height;

	int l = s.width*s.height;

	for(int i=0;i<n/l;++i){
		G[i] = Mat(_G[0](cv::Rect(0,i*l,1,l)).reshape(0,s.height));
	}
	return G;
}

void FlattenLayer::setup(Size s){
	this->s = s;
}
Size FlattenLayer::outputSize(){
	return Size(1,d*s.width*s.height);
}
