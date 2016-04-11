#ifndef __CONVNET_H__
#define __CONVNET_H__

#include "Layer.h"
#include "ActivationLayer.h"
#include "DenseLayer.h"
#include "FlattenLayer.h"
#include "ConvolutionLayer.h"
#include "PoolLayer.h"
#include "SoftMaxLayer.h"


class ConvNet{

private:
	std::vector<Layer*> L;
public:
	
	ConvNet();
	~ConvNet();

	std::vector<Mat> FF(std::vector<Mat> _X);
	void BP(std::vector<Mat> Yp, std::vector<Mat> Y);
	void setup(Size s);
	void push_back(Layer* l);
	std::vector<Layer*> getL();//for debugging.
	void save(std::string dir);//save directory
	void load(std::string dir);

};


#endif
