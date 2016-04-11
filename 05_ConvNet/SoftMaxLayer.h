#ifndef __SOFTMAXLAYER_H__
#define __SOFTMAXLAYER_H__
#include "Layer.h"

class SoftMaxLayer: public Layer{
private:
	int d;
	Size s;
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G;
public:
	SoftMaxLayer();

	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	virtual void setup(Size s);
	virtual Size outputSize();
	double cost();
	//no need to update since to trainable parameter
	//
	virtual void save(FileStorage& f, int i);
	virtual void load(FileStorage& f, int i);
};

#endif
