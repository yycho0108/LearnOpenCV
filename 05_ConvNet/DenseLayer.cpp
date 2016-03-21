#include "DenseLayer.h"

DenseLayer::DenseLayer(int d, int s_o)
	:d(d),s_o(s_o),W(d),b(d),dW(d),db(d),I(d),O(d),G(d){
}
void DenseLayer::setup(Size s){
	this->s_i = s.height;

	for(int i=0;i<d;++i){
		W[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		b[i] = Mat::zeros(Size(1,s_o),DataType<float>::type);
		dW[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		db[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		cv::randn(W[i],cv::Scalar::all(0),cv::Scalar::all(0.1));
	}

}

std::vector<Mat>& DenseLayer::FF(std::vector<Mat> _I){
	I.swap(_I);
	for(size_t i=0;i<I.size();++i){
		O[i] = W[i]*I[i]+b[i];
	}
	if(isnan(O)){
		cout << "I : " << endl;
		for(auto& i : I){
			cout << i << endl;
		}
		cout << "W : " << endl;
		for(auto& w : W){
			cout << w << endl;
		}
		cout << "b : " << endl;
		for(auto& bb : b){
			cout << bb << endl;
		}
		cout << "O : " << endl;
		for(auto& o : O){
			cout << o << endl;
		}
		throw "OISNAN-2";
	}
	return O;
}
std::vector<Mat>& DenseLayer::BP(std::vector<Mat> _G){
	for(int i=0;i<d;++i){
		G[i] = W[i].t() * _G[i];
		dW[i] = _G[i]*I[i].t(); //bit iffy in here, but I guess... since no sigmoid.
		db[i] = _G[i];
		dW[i] -= DECAY * W[i]; //weight decay
	}
	return G;
}
void DenseLayer::update(){
	for(int i=0;i<d;++i){
		W[i] += ETA * dW[i];
		b[i] += ETA * db[i];
	}	
}
Size DenseLayer::outputSize(){
	return Size(1,s_o); //Size(width,height);
}
