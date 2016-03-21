#include "ConvNet.h"

ConvNet::ConvNet(){
	cv::theRNG().state = time(0);
}
ConvNet::~ConvNet(){
	for(auto& l : L){
		delete l;
	}
}

std::vector<Mat> ConvNet::FF(std::vector<Mat> _X){
	auto& X = _X;
	for(auto& l : L){
		X = l->FF(X);

		/*if(isnan(X)){
			cout << "X : " << endl << X[0] << endl;
			throw ("XISNAN!");
		}*/

	}
	return X;
}

void ConvNet::BP(std::vector<Mat> Yp, std::vector<Mat> Y){
	//sample ..
	std::vector<Mat> G(Yp.size());
	for(size_t i=0;i<G.size();++i){
		cv::subtract(Y[i],Yp[i],G[i]); //G[i] = O[i]Y[i] - O[i];
	}
	//cout << "Y" << Y[0] << endl;
	for(auto i = L.rbegin()+1; i != L.rend(); ++i){
		auto& l = (*i);

		/*if(isnan(G)){
			cout << "G" << " : "<< endl << G[0] << endl;
			throw ("GISNAN!");
		}*/

		G = l->BP(G);
	}

	for(auto& l : L){
		l->update();
	}

	//cout << "WTF" << endl << WTF << endl;
}

void ConvNet::setup(Size s){
	for(auto& l : L){
		l->setup(s);
		s = l->outputSize();
	}
}

void ConvNet::push_back(Layer* l){
	L.push_back(l);
}

std::vector<Layer*> ConvNet::getL(){
	return L;
}


