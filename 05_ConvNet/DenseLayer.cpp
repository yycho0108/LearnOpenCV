#include "DenseLayer.h"
#include <algorithm>

DenseLayer::DenseLayer(int d, int s_o)
	:d(d),s_o(s_o),W(d),b(d),dW(d),db(d),g(d),dW_p(d),db_p(d),I(d),O(d),G(d){
		m = 0.3; //momentum
}
void DenseLayer::setup(Size s){
	this->s_i = s.height;

	for(int i=0;i<d;++i){

		W[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		b[i] = Mat::zeros(Size(1,s_o),DataType<float>::type);

		dW[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		db[i] = Mat::zeros(Size(1,s_o),DataType<float>::type);

		dW_p[i] = Mat::zeros(s_o,s_i,DataType<float>::type);
		db_p[i] = Mat::zeros(Size(1,s_o),DataType<float>::type);

		g[i] = Mat::ones(s_o,s_i,DataType<float>::type);

		cv::randn(W[i],cv::Scalar::all(0),cv::Scalar::all(0.1));
	}

}

std::vector<Mat>& DenseLayer::FF(std::vector<Mat> _I){
	I.swap(_I);
	for(size_t i=0;i<I.size();++i){
		O[i] = W[i]*I[i]+b[i];
	}
	/*if(isnan(O)){
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
	}*/
	return O;
}
std::vector<Mat>& DenseLayer::BP(std::vector<Mat> _G){
	for(int i=0;i<d;++i){
		G[i] = W[i].t() * _G[i];

		dW[i] = m*dW_p[i] // momentum * previous dW
				+ ETA * g[i].mul( _G[i]*I[i].t()) // learning rate * weight error
				- DECAY * W[i]; //decay * weight

		db[i] = m*db_p[i]
				+ ETA * _G[i];

//		auto sgn = cv::sum(dW[i].mul(dW_p[i]))[0];
//		std::cout << sgn << std::endl;
//
//		if(sgn>0){
//			g *= 1.05;
//		}else if (sgn<0){
//			g *= 0.7;
//		}
//
//		g = constrain(g,0.1,10.0);
		//std::cout << "G : " << g << std::endl;

		dW_p[i] = dW[i];
		db_p[i] = db[i];
	}
	return G;
}
void DenseLayer::update(){
	for(int i=0;i<d;++i){
		W[i] += dW[i];
		b[i] += db[i];
	}	
}
Size DenseLayer::outputSize(){
	return Size(1,s_o); //Size(width,height);
}


void DenseLayer::save(FileStorage& fs, int i){
	auto prefix = "D" + std::to_string(i);

	for(size_t i=0;i<W.size();++i){
		fs << (prefix + "_" + "W" + std::to_string(i)) << W[i];
	}

	for(size_t i=0;i < b.size();++i){
		fs << (prefix + "_" + "b" + std::to_string(i)) << b[i];
	}
}

void DenseLayer::load(FileStorage& fs, int i){
	auto prefix = "D" + std::to_string(i);
	for(size_t i=0; i<W.size(); ++i){
		fs[prefix+"_"+"W"+std::to_string(i)] >> W[i];
		fs[prefix+"_"+"b"+std::to_string(i)] >> b[i];
	}
}
