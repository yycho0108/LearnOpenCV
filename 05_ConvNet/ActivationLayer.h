#ifndef __ACTIVATIONLAYER_H__
#define __ACTIVATIONLAYER_H__

#include "Layer.h"

extern void activate(Mat& src, Mat& dst, float (*f)(float));

class ActivationLayer : public Layer{
private:
	int d;
	Size s;
	float (*f)(float);
	float (*f_d)(float);
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G; //maybe not necessary? idk...
public:
	ActivationLayer(std::string _f);
	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	virtual void setup(Size s);
	virtual Size outputSize();
	virtual void save(std::string dir);
	//no need to update since to trainable parameter
};


#endif

