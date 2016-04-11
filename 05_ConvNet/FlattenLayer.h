#ifndef __FLATTENLAYER_H__
#define __FLATTENLAYER_H__

#include "Layer.h"

class FlattenLayer : public Layer{
private:
	int d;
	Size s;
	std::vector<Mat> I;
	std::vector<Mat> O;
	std::vector<Mat> G;
public:
	FlattenLayer(int d);

	virtual std::vector<Mat>& FF(std::vector<Mat> I);
	virtual std::vector<Mat>& BP(std::vector<Mat> G);
	virtual void setup(Size);
	virtual Size outputSize();
	virtual void save(std::string dir);
};


#endif
