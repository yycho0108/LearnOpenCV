#include "DropoutLayer.h"


DropoutLayer::DropoutLayer(int d, double p=0.5):d(d),p(p){

}

std::vector<Mat>& DropoutLayer::FF(std::vector<Mat> _I){
	d = _I.size();
	O.resize(d);
	G.resize(d);
	//assert same size
	I.swap(_I);
	for(int i=0;i<d;++i){
		if(TEST_STATE){
			O[i] = I[i];
		}else{
			// create mask
			cv::randu(Mask[i],0,1.0);
			cv::threshold(Mask[i],Mask[i],p,1.0,cv::THRESH_BINARY);
			// apply mask
			O[i] = I[i].mul(Mask[i]);
		}
	}
	return O;
}

std::vector<Mat>& DropoutLayer::BP(std::vector<Mat> _G){
	for(int i=0;i<d;++i){
		G[i] = _G[i].mul(Mask[i]) / p;
		// divide by p, to ensure same mag.
	}
	return G;
}

void DropoutLayer::setup(Size s){
	this->s = s;
	for(int i=0;i<d;++i){
		Mask.push_back(Mat(s,cv::DataType<float>::type));
	}
}

Size DropoutLayer::outputSize(){
	return s;
}

void DropoutLayer::save(FileStorage&, int){
	
}

void DropoutLayer::load(FileStorage&, int){

}
