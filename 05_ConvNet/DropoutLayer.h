#ifndef __DROPOUTLAYER_H__
#define __DROPOUTLAYER_H__

#include "Layer.h"

class DropoutLayer : public Layer{
private:
	int d;
	Size s;
	double p;
	float m; //momentum
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G; //maybe not necessary? idk...
	std::vector<Mat> Mask;
public:

	DropoutLayer(int d, double p);

	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	virtual void setup(Size s);
	virtual Size outputSize();

	virtual void save(FileStorage& f, int i);
	virtual void load(FileStorage& f, int i);
};

#endif
