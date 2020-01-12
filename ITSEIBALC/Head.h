#pragma once
#include<fstream>  
#include<iostream>
#include "string" 
#include<cmath>
#include <vector>
#include <stdio.h>
#include <tchar.h>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "highgui.h"
#include "Math.h"
#include <iostream>
#include <fstream> 
#include <iomanip>
#include <windows.h>
#include "io.h"
#include "./libsvm/svm.h"
using namespace std;
using namespace cv;
//内存分配
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* 封装svm                                                                     */
/************************************************************************/
class CxLibSVM
{
private:

	struct svm_model*	     model_;
	struct svm_parameter	 param;
	struct svm_problem		 prob;
	struct svm_node *		 x_space;
public:
	//************************************
	// Description: Constructor
	// Method: CxLibSVM
	// File name: CxLibSVM :: CxLibSVM
	// Access: public
	// return value: 
	// Qualifier:
	//************************************
	CxLibSVM()
	{
		model_ = NULL;
	}

	//************************************
	// Description: Destructor
	// Method: ~ CxLibSVM
	// File name: CxLibSVM :: ~ CxLibSVM
	// Access: public
	// return value:
	//************************************
	~CxLibSVM()
	{
		free_model();
	}

	//************************************
	// Description: Training model
	// Method: train
	// File name: CxLibSVM :: train
	// Access: public
	// Parameter: const vector<vector<double>> & x
	// Parameter: const vector<double> & y
	// Parameter: const int & alg_type
	// return value: void
	// Qualifier:
	//************************************
	void train(const vector<vector<double>>& x, const vector<double>& y)
	{
		if (x.size() == 0)return;

		//Release previous model
		free_model();

		/*初始化*/
		long	len = x.size();
		long	dim = x[0].size();
		long	elements = len * dim;

		//Parameter initialization, parameter adjustment part can be modified here
		// Default parameter
		param.svm_type = C_SVC;		//Algorithm type
		param.kernel_type = LINEAR;	//Kernel Function Type
		param.degree = 3;	//Parameter of polynomial kernel function degree
		param.coef0 = 0;	//Parameters of polynomial kernel function coef0
		param.gamma = 0.5;	//1/num_features，rbfKernel parameter
		param.nu = 0.5;		//nu-svc
		param.C = 10;		//Penalty coefficient for regular terms10
		param.eps = 1e-3;	//Convergence accuracy
		param.cache_size = 100;	//Solved memory buffer 100MB
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 1;	//The probability used to predict the category to which the sample belongs
		param.nr_weight = 0;	//Category weight
		param.weight = NULL;	//Sample weight
		param.weight_label = NULL;	//Category weight


									//Convert data to libsvm format
		prob.l = len;
		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(struct svm_node *, prob.l);
		x_space = Malloc(struct svm_node, elements + len);
		int j = 0;
		for (int l = 0; l < len; l++)
		{
			prob.x[l] = &x_space[j];
			for (int d = 0; d < dim; d++)
			{
				x_space[j].index = d + 1;
				x_space[j].value = x[l][d];
				j++;
			}
			x_space[j++].index = -1;
			prob.y[l] = y[l];
		}
		/*training*/
		model_ = svm_train(&prob, &param);
	}
	//************************************
	// Description: The category and probability of the prediction test sample
	// Method: predict
	// file name: CxLibSVM::predict
	// access permission: public 
	// Parameter: const vector<double> & x	sample
	// Parameter: double & prob_est			Category estimated probability
	// return value: double						Predicted category
	// Qualifier:
	//************************************
	int predict(const vector<double>& x, double& prob_est)
	{
		//Data conversion
		svm_node* x_test = Malloc(struct svm_node, x.size() + 1);
		for (unsigned int i = 0; i<x.size(); i++)
		{
			x_test[i].index = i + 1;
			x_test[i].value = x[i];
		}
		x_test[x.size()].index = -1;
		double *probs = new double[model_->nr_class];//Probabilities stored for all categories
													 //Prediction categories and probabilities
		int value = (int)svm_predict_probability(model_, x_test, probs);
		for (int k = 0; k < model_->nr_class; k++)
		{
			//Find the corresponding probability of a category
			if (model_->label[k] == value)
			{
				prob_est = probs[k];
				break;
			}
		}
		delete[] probs;
		return value;
	}

	int load_model(string model_path)
	{
		//free_model();
		model_ = svm_load_model(model_path.c_str());
		if (model_ == NULL)return -1;
		return 0;
	}
	
	int save_model(string model_path)
	{
		int flag = svm_save_model(model_path.c_str(), model_);
		return flag;
	}

	void free_model()
	{
		if (model_ != NULL)
		{
			svm_free_and_destroy_model(&model_);
			svm_destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
		}
	}
};
///////////////////////////////////////////Structures and functions defined in this experiment////////////////////////////////////////////////
struct ClassifiedImage
{
	Mat ClassMap;//Create classified images to receive classified images
};
struct USO      //Follow up with new calculations for each participating sample expansion point, the seed point checked
{
	vector<vector<CvPoint>>UpdateExtSamplePoints;
	vector<vector<CvPoint>>UpdateCheckPoint;
	int TotalPoints = 0;
	int QuaSeedPointNum = 0;
	float PassRate = 0;
};
void AbcssFrame(IplImage* Originalimage, IplImage* Greayimage,IplImage* out, string Path, int T1, int T2, int ClassNum, int IterNum, float Stability);
float Get_Mean(IplImage* Greayimage, vector<CvPoint>KeyPointRegion);
float Get_Var(IplImage* Greayimage, vector<CvPoint>KeyPointRegion, int Means);
int ReturnCoordinatePosition(vector<float>Primendata, float Sortdata);
void SVMClassification(string Trainingpath1, string Testpath2, string Reslutpath3);
vector<int>ReadClassificationTxt(string path, int Class_Num, int IterNum);
Mat ClassficationMap(string Path, IplImage* Originalimage, int ClassNum, int IterNum);
void SplitString_1(const string& s, string v[], const string& c);
void SplitString(const string& s, vector<string>& v, const string& c);
vector<vector<CvPoint>>Read_TXT(string path, int ClassNum);
vector<CvPoint> Connected_comp(IplImage* CMI, IplImage* out, CvPoint Keypoints, int pixel_diff, int Connected_num);
vector<vector<CvPoint>>ExtendCorrectSamples(vector<vector<CvPoint>>CSPoints, IplImage* Originalimage, IplImage* out, IplImage* Greayimage, int T1, int T2, int ClassNum);
ClassifiedImage ClassIMG(IplImage* Originalimage, IplImage* out, string Path, int ClassNum, int IterNum);
vector<float>SaturatedCategory(string Path, int ClassNum, int IterNum);
void Training_TXT(IplImage* Originalimage,vector<CvPoint>AllTrainSamplePoints[15], string path, int IterNum, int ClassNum);
vector<CvPoint>ExtenKPPoints(IplImage* Originalimage, IplImage* Greayimage, IplImage* out, CvPoint Keypoint, int T1, int T2);
vector<int>RSC(vector<vector<float>>IterInformation, int IterNum, float Stability);