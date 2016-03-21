#ifndef __DENSELAYER_H__
#define __DENSELAYER_H__
#include "Layer.h"

class DenseLayer : public Layer{
private:
	int d,s_i,s_o;
	std::vector<Mat> W;
	std::vector<Mat> b;
	std::vector<Mat> dW;
	std::vector<Mat> db;
	
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G; //maybe not necessary? idk...
	//
public:
	DenseLayer(int d, int s_o);

	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	virtual void setup(Size s);
	virtual Size outputSize();
	virtual void update();
};

#endif
