#include "ConvNet.h"
#include <iostream>
#include <cstdio>

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

void ConvNet::load(std::string f){
	//has to be EXACTLY the same architecture... to facilitate loading
	//may change later
	//not implemented
	FileStorage fs(f,FileStorage::READ);
	if(fs.isOpened()){
		for(size_t i=0; i<L.size(); ++i){
			L[i]->load(fs,i);
		}
	}else{
		std::cerr << "WASN'T ABLE TO LOAD FROM FILE" << std::endl;
	}

}

void ConvNet::save(std::string f){
	//dir = directory name
	std::remove(f.c_str()); //const char* filename
	FileStorage fs(f, FileStorage::WRITE);
	if(fs.isOpened()){
		for(size_t i=0; i<L.size(); ++i){
			L[i]->save(fs,i);
		}
	}else{
		std::cerr << "WASN'T ABLE TO LOAD FROM FILE" << std::endl;
	}
}
