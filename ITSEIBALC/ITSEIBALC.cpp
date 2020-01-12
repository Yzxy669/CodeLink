#include"Head.h"
int main()
{
	////////////////////////////////////////////////parameter settings//////////////////////////////////////////////////////////////////
	IplImage* Originalimage = cvLoadImage("E:\\Papers\\Paper_12_2019_0902\\Data\\ZH3\\zh3.tif");//Load original image path
	IplImage* Greayimage = cvLoadImage("E:\\Papers\\Paper_12_2019_0902\\Data\\ZH3\\zh3Grayscale.tif");//Load the grayscale image path corresponding to the original image
	string Path = "E:\\Papers\\Paper_12_2019_0902\\Experiment1\\ZH3\\DrawingDebugging";//All Txt text paths used in this experiment
	int ClassNum = 7;//Number of image categories
	float Stability = 0.003;//Iteration condition
	int T1 = 5;//Difference between two pixel neighborhoods
	int T2 = 100;//Number of pixels for adaptive area expansion
	////////////////////////////////////////////////parameter settings////////////////////////////////////////////////////////
	IplImage* out = cvCreateImage(cvGetSize(Originalimage), Originalimage->depth, 1);//Adaptive area label image
	int IterNum = 0;//Record the number of iterations, iteration initialization
	AbcssFrame(Originalimage, Greayimage,out,  Path, T1, T2, ClassNum, IterNum, Stability);
	return 0;
}

void AbcssFrame(IplImage* Originalimage, IplImage* Greayimage, IplImage* out,string Path, int T1, int T2, int ClassNum, int IterNum, float Stability)
{
	ClassifiedImage CI;//Receive the generated image
	vector<float>SCRate;//Store two images with the same coordinates in the same category rate
	vector<vector<CvPoint>>Ittspoints;//Store sample points by category
	vector<vector<CvPoint>>CSPoints;//Store sample points for each expansion by category
	vector<CvPoint>FinalTrainSample[15];//Receive all training samples and iterate once, output training samples once
	vector<vector<float>>IterInformation;
	vector<int>SIC;//Stop iteration category
	int IterCon = ClassNum+1;//Determine iteration conditions
	IterNum++;
	Ittspoints = Read_TXT(Path, ClassNum);//Take the initial sample seed point
	CSPoints = ExtendCorrectSamples(Ittspoints, Originalimage, out, Greayimage, T1, T2, ClassNum);
	for (int k = 1; k <ClassNum + 1; k++)
	{
		for (int i = 0; i < Ittspoints[k].size(); i++)
		{
			FinalTrainSample[k].push_back(Ittspoints[k][i]);
		}
		for (int  j = 0; j < CSPoints[k].size(); j++)
		{
			FinalTrainSample[k].push_back(CSPoints[k][j]);
		}
	}
  	Ittspoints.clear(); Ittspoints.shrink_to_fit();
	Training_TXT(Originalimage,FinalTrainSample, Path,IterNum, ClassNum);
	CI = ClassIMG(Originalimage, out, Path, ClassNum, IterNum);
	char adr[256] = { 0 };//Write image address
	sprintf(adr, "%s\\Proposed-SVM-%d.bmp", Path.data(), IterNum);//Image naming
	imwrite(adr, CI.ClassMap);//Save image
	SCRate = SaturatedCategory(Path, ClassNum, IterNum);
	IterInformation.push_back(SCRate);
	while (IterCon !=0 )
	{
		IterNum++;
		CSPoints = ExtendCorrectSamples(CSPoints, Originalimage, out, Greayimage, T1, T2, ClassNum);
		for (int k = 1; k <ClassNum + 1; k++)
		{
			for (int i = 0; i < CSPoints[k].size(); i++)
			{
				FinalTrainSample[k].push_back(CSPoints[k][i]);
			}
		}
		Training_TXT(Originalimage, FinalTrainSample, Path, IterNum, ClassNum);
		CI = ClassIMG(Originalimage, out, Path, ClassNum, IterNum);
		char adr[256] = { 0 };
		sprintf(adr, "%s\\Proposed-SVM-%d.bmp", Path.data(), IterNum);
		imwrite(adr, CI.ClassMap);
		SCRate = SaturatedCategory(Path, ClassNum, IterNum);//Returns the same coordinates and pixel value rate of two images
		IterInformation.push_back(SCRate);
		SIC = RSC(IterInformation, IterNum, Stability);//Returns categories that are no longer participating in the extension
		if (SIC.size() != 0)
		{
			IterCon = ClassNum+1;
			for (int  a = 0; a < SIC.size(); a++)
			{
				CSPoints[SIC[a]].clear(); CSPoints[SIC[a]].shrink_to_fit();
			}
			for (int  b = 0; b < CSPoints.size(); b++)
			{
				if (CSPoints[b].size() == 0)
				{
					IterCon--;
					if (b!=0)
					{
						cout << "当前迭代第" << IterNum << "次" << "达到饱和的类别为:" << "第" << b << "类" << endl;
					}
					
				}	
			}
		}
	}
}
//Returns saturated categories
vector<int>RSC(vector<vector<float>>IterInformation, int IterNum, float Stability)
{
	vector<int>SaturatedCategory;
	for (int  j = 0; j < IterInformation[IterNum-1].size(); j++)
	{
		if (abs((IterInformation[IterNum - 1][j]- IterInformation[IterNum - 2][j])) <= Stability)
		{
			SaturatedCategory.push_back(j + 1);
		}
	}
	return SaturatedCategory;
}
//Returns the saturation of each category of the image
vector<float>SaturatedCategory(string Path, int ClassNum, int IterNum)
{
	char adr1[256] = { 0 };//Image address
	char adr2[256] = { 0 };
	sprintf(adr1, "%s\\Proposed-SVM-%d.bmp", Path.data(), IterNum-1);//Image name
	sprintf(adr2, "%s\\Proposed-SVM-%d.bmp", Path.data(), IterNum);
	IplImage* ClassMap1 = cvLoadImage(adr1);//Load image path 1
	IplImage* ClassMap2 = cvLoadImage(adr2);//Load image path 2
	int Map1label = 0;//Category of image 1
	int Map2label = 0;//Category of image 2
	int SameCategoryNum = 0;//Record the number of pixels with the same coordinate category in two classified images
	float SaturationRate = 0;//Record the saturation rate of the current category
	vector<float>SRate;//Returns the current saturation rate of each class
	for (int  k = 1; k < ClassNum+1; k++)
	{
		SameCategoryNum = 0;
		for (int i = 0; i < ClassMap1->height; i++)
		{
			for (int j = 0; j < ClassMap1->width; j++)
			{
				Map1label = *cvPtr2D(ClassMap1, i, j, NULL);
				Map2label = *cvPtr2D(ClassMap2, i, j, NULL);
				if (Map1label == Map2label && Map2label ==k)
				{
					SameCategoryNum++;
				}
			}
	    }
		SaturationRate = float(SameCategoryNum) / (ClassMap1->height * ClassMap1->width);
		SRate.push_back(SaturationRate);
	}
	cvReleaseImage(&ClassMap1);
	cvReleaseImage(&ClassMap2);
	return SRate;
}
//Classify training samples and convert them into images
ClassifiedImage ClassIMG(IplImage* Originalimage, IplImage* out, string Path, int ClassNum, int IterNum)
{
	ClassifiedImage CI;
	char adr1[256] = { 0 };
	char adr2[256] = { 0 };
	char adr3[256] = { 0 };
	sprintf(adr1, "%s\\Training-%d.txt", Path.data(), IterNum);
	sprintf(adr2, "%s\\Test.txt", Path.data());
	sprintf(adr3, "%s\\Proposed-SVM-%d.txt", Path.data(), IterNum);
	SVMClassification(adr1, adr2, adr3);
	CI.ClassMap = ClassficationMap(Path, Originalimage, ClassNum, IterNum);
	return CI;
}
//Read the initially selected sample points
vector<vector<CvPoint>>Read_TXT(string path,int ClassNum)
{
	vector<CvPoint>ITTSPoints[15];
	vector<vector<CvPoint>>ITTSP;
	int label = 0;
	string str;
	string Feature[64];
	CvPoint Keypoint;
	fstream dataFile;
	char adr[256] = { 0 };
	sprintf(adr, "%s\\Training-0.txt", path.data());
	dataFile.open(adr, ios::in);
	if (!dataFile)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	//cout << "打开文件成功";
	while (!dataFile.eof())
	{
		getline(dataFile, str);
		if (str == "")
		{
			break;
		}
		SplitString_1(str, Feature, ",");
		label= atoi(Feature[0].c_str());
		Keypoint.x = atoi(Feature[1].c_str());
		Keypoint.y = atoi(Feature[2].c_str());
		ITTSPoints[label].push_back(Keypoint);
	}
	for (int  i = 0; i < ClassNum+1; i++)
	{
		ITTSP.push_back(ITTSPoints[i]);
	}
	dataFile.close();
	return ITTSP;
}
//Cyclic expansion of the sample to generate the final sample
vector<vector<CvPoint>>ExtendCorrectSamples(vector<vector<CvPoint>>CSPoints, IplImage* Originalimage, IplImage* out, IplImage* Greayimage, int T1, int T2, int ClassNum)
{
	vector<CvPoint>EKPPoints;//Receive the key points extended by each point
	vector<CvPoint>CategorySample[15];
	vector<vector<CvPoint>>CS;
	for (int label = 1; label <ClassNum + 1; label++)
	{
		for (int i = 0; i <CSPoints[label].size(); i++)
		{
			EKPPoints = ExtenKPPoints(Originalimage, Greayimage, out, CSPoints[label][i], T1, T2);
			for (int j = 0; j < EKPPoints.size(); j++)
			{
				if (EKPPoints[j].x != CSPoints[label][i].x && EKPPoints[j].y != CSPoints[label][i].y)
				{
					CategorySample[label].push_back(EKPPoints[j]);
				}
			}
		}
	}
	for (int i = 0; i < ClassNum + 1; i++)
	{
		CS.push_back(CategorySample[i]);
	}
	return CS;
}
//Returns the key points of each sample that are expanded out of the sample
vector<CvPoint>ExtenKPPoints(IplImage* Originalimage, IplImage* Greayimage, IplImage* out, CvPoint Keypoint, int T1, int T2)
{
	vector<CvPoint>OExpandKeyPoints;//Extended sample point
	vector<CvPoint>AddExtendedSample;//Generate samples for classification at each iteration
	vector<CvPoint>CPWR;//The set of points in the area is used to calculate the variance of each pixel
	float Mean_Pixel = 0;//Pixels average by area
	float Var_Pixel = 0;//Pixels calculate variance by region
	vector<float>VAR；
	vector<float>TeamVAR;
	int PointNum = 0;/
	AddExtendedSample = Connected_comp(Originalimage, out, Keypoint, T1, T2);
	if (AddExtendedSample.size() > 3)
	{
		VAR.clear(); VAR.shrink_to_fit();
		TeamVAR.clear(); TeamVAR.shrink_to_fit();
		for (int j = 0; j < AddExtendedSample.size(); j++)
		{
			CPWR.clear(); CPWR.shrink_to_fit();
			CPWR = Connected_comp(Originalimage, out, AddExtendedSample[j], T1, T2);
			Mean_Pixel = Get_Mean(Greayimage, CPWR);
			Var_Pixel = Get_Var(Greayimage, CPWR, Mean_Pixel);
			VAR.push_back(Var_Pixel);
			TeamVAR.push_back(Var_Pixel);
		}
		sort(VAR.begin(), VAR.end());
		int Quartile[3] = {(VAR.size() + 1) * 0.25, (VAR.size() + 1) * 0.5, (VAR.size() + 1) * 0.75};
		for (int a = 0; a < 3; a++)
		{
			PointNum = ReturnCoordinatePosition(TeamVAR, VAR[Quartile[a]]);
			OExpandKeyPoints.push_back(AddExtendedSample[PointNum]);
		}
	}
	return OExpandKeyPoints;
}
//Adaptive Region Extension Sample
vector<CvPoint> Connected_comp(IplImage* CMI, IplImage* out, CvPoint Keypoints, int pixel_diff, int Connected_num)
{
	int DIR[8][12] = { { -1,-1 },{ -1,0 },{ -1,1 },{ 0,-1 },{ 0,1 },{ 1,-1 },{ 1,0 },{ 1,1 } };
	cvZero(out);
	int x = Keypoints.x;
	int y = Keypoints.y;
	int center_point = *cvPtr2D(CMI, Keypoints.x, Keypoints.y, NULL);//Seed point gray value
	int num = 0;//Count the number of expanded pixels
	vector<CvPoint> points;
	vector<CvPoint> Connected_points;
	int pixel = 0;
	vector<int> Connected_pixel;
	Connected_pixel.clear();
	Connected_pixel.shrink_to_fit();
	points.clear();
	Connected_points.clear();
	CvPoint point_1, point_2;
	Connected_points.push_back(Keypoints);
	Connected_pixel.push_back(center_point);
	*(cvPtr2D(out, x, y, NULL)) = 255;
	for (int pos = -1; Connected_points.size() < Connected_num; pos++) {//num
		if (pos != -1) {
			if (num == 0)
				return Connected_points;
			else if (pos < points.size()) {
				x = points[pos].x;
				y = points[pos].y;
			}
			else {
				return Connected_points;
			}
		}
		for (int iNum = 0; iNum < 8; iNum++) {
			int iCurPosX = x + DIR[iNum][0];
			int iCurPosY = y + DIR[iNum][1];
			if (iCurPosX >= 0 && iCurPosX < (CMI->height) && iCurPosY >= 0 && iCurPosY < (CMI->width)) {
				if (*(cvPtr2D(out, iCurPosX, iCurPosY, NULL)) != 255) {
					if (abs(center_point - *(cvPtr2D(CMI, iCurPosX, iCurPosY, NULL))) <= pixel_diff) {//生长条件
						*(cvPtr2D(out, iCurPosX, iCurPosY, NULL)) = 255;
						point_2.x = iCurPosX;
						point_2.y = iCurPosY;
						pixel = *(cvPtr2D(CMI, iCurPosX, iCurPosY, NULL));
						Connected_pixel.push_back(pixel);
						Connected_points.push_back(point_2);
						num += 1;
						point_1.x = iCurPosX;
						point_1.y = iCurPosY;
						points.push_back(point_1);
						if (Connected_points.size() >= Connected_num)
							return Connected_points;
					}
				}
			}
		}
	}
	return Connected_points;
}
//Write all training samples into a txt text
void Training_TXT(IplImage* Originalimage, vector<CvPoint>AllTrainSamplePoints[15], string path, int IterNum, int ClassNum)
{
	IplImage* ImageSample = cvCreateImage(cvGetSize(Originalimage), Originalimage->depth,1);
	CvPoint TrainPoint;
	CvScalar C;
	char adr[256] = { 0 };
	sprintf(adr, "%s\\Training-%d.txt", path.data(), IterNum);
	fstream dataFile;//Read file stream
	dataFile.open(adr, ios::app);
	if (!dataFile)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	//cout << "打开文件成功";
	
	for (int i = 1; i <ClassNum + 1; i++)
	{
		for (int j = 0; j <AllTrainSamplePoints[i].size(); j++)
		{
			TrainPoint.x = AllTrainSamplePoints[i][j].x;
			TrainPoint.y = AllTrainSamplePoints[i][j].y;
			if (*(cvPtr2D(ImageSample, TrainPoint.x, TrainPoint.y, NULL)) != 255)
			{
				C = cvGet2D(Originalimage, TrainPoint.x, TrainPoint.y);
				dataFile << i << "," << TrainPoint.x << "," << TrainPoint.y << ","
						 << C.val[0] << "," << C.val[1] << "," << C.val[2]  <<endl;
				*(cvPtr2D(ImageSample, TrainPoint.x, TrainPoint.y, NULL)) = 255;
			}
		}
	}
	cvReleaseImage(&ImageSample);
	dataFile.close();
}
//Calculate the mean of pixels in a region
float Get_Mean(IplImage* Greayimage, vector<CvPoint>KeyPointRegion)
{
	int PV = 0;
	int Sum_PV = 0;
	float Mean_PV = 0;
	for (int i = 0; i < KeyPointRegion.size(); i++)
	{
		PV = *cvPtr2D(Greayimage, KeyPointRegion[i].x, KeyPointRegion[i].y, NULL);
		Sum_PV += PV;
	}
	Mean_PV = Sum_PV / KeyPointRegion.size();
	return Mean_PV;
}
//Calculate the variance of pixels in a region
float Get_Var(IplImage* Greayimage, vector<CvPoint>KeyPointRegion, int Means)
{
	float PV = 0;
	float Var = 0;
	for (int i = 0; i <KeyPointRegion.size(); i++)
	{
		PV = *cvPtr2D(Greayimage, KeyPointRegion[i].x, KeyPointRegion[i].y, NULL);
		Var += pow(PV - Means, 2) / KeyPointRegion.size();
	}
	return Var;
}
int ReturnCoordinatePosition(vector<float>Primendata, float Sortdata)
{
	for (int i = 0; i < Primendata.size(); i++)
	{
		if (Primendata[i] == Sortdata)
		{
			return i;
		}
	}
}
////////////////////////////////////////////////*Libsvm classification*/////////////////////////////////////////////////////
void SVMClassification(string Trainingpath1, string Testpath2, string Reslutpath3)
{
	CxLibSVM svm;
	string str;
	vector <string> Feature;
	vector<double> rx;
	vector<vector<double>>x;	
	vector<double>y;			
	double prob_est;
	double value;
	fstream dataFileTraining;
	fstream dataFileTest;
	fstream dataFileReslut;
	dataFileTraining.open(Trainingpath1, ios::in);
	if (!dataFileTraining)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	//cout << "打开文件成功";
	while (!dataFileTraining.eof())
	{
		getline(dataFileTraining, str);
		if (str == "")
		{
			break;
		}
		Feature.clear(); Feature.shrink_to_fit();
		SplitString(str, Feature, ",");
		rx.clear(); rx.shrink_to_fit();
		for (int i = 3; i < Feature.size(); i++)
		{
			rx.push_back(atof(Feature[i].c_str()));
		}
		x.push_back(rx);
		y.push_back(atof(Feature[0].c_str()));
	}
	svm.train(x, y);

	/****************************************************************************************/
	x.clear(); x.shrink_to_fit();
	y.clear(); y.shrink_to_fit();
	/****************************************************************************************/

	string model_path = ".\\svm_model.txt";
	svm.save_model(model_path);

	string model_path_p = ".\\svm_model.txt";
	svm.load_model(model_path_p);

	vector<double> x_test;
	dataFileTest.open(Testpath2, ios::in);
	if (!dataFileTest)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	dataFileReslut.open(Reslutpath3, ios::out);
	if (!dataFileReslut)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	dataFileReslut << "label" << "," << "i" << "," << "j" << "," << "F1" << "," << "F2" << "......" << endl;
	while (!dataFileTest.eof())
	{
		getline(dataFileTest, str);
		if (str == "")
		{
			break;
		}
		Feature.clear(); Feature.shrink_to_fit();
		SplitString(str, Feature, ",");
		x_test.clear(); x_test.shrink_to_fit();
		for (int i = 2; i < Feature.size(); i++)
		{
			x_test.push_back(atof(Feature[i].c_str()));
		}
		 value = svm.predict(x_test, prob_est);

		dataFileReslut << value;
		for (int i = 0; i < Feature.size(); i++)
		{
			dataFileReslut << "," << Feature[i];
			if (i == Feature.size() - 1)
			{
				dataFileReslut << endl;
			}
		}
	}
	dataFileTraining.close();
	dataFileTest.close();
	dataFileReslut.close();
}
/////////////////////////////////////////////////*Libsvm classification*////////////////////////////////////////////////////*/
///////////////////////////////////////////////*Generate each iteration into an image*////////////////////////////////////////*/
Mat ClassficationMap(string Path, IplImage* Originalimage,int ClassNum, int IterNum)
{
	vector<int>MapLabel;
	MapLabel = ReadClassificationTxt(Path, ClassNum, IterNum);
	int num = 0;
	Mat ClassMap(Originalimage->height, Originalimage->width, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < Originalimage->height; i++)
	{
		for (int j = 0; j < Originalimage->width; j++)
		{
			ClassMap.at<Vec3b>(i, j) [0] = MapLabel[num];//B  
			ClassMap.at<Vec3b >(i, j)[1] = MapLabel[num];//G  
			ClassMap.at<Vec3b >(i, j)[2] = MapLabel[num];//R 
			num++;
		}
	}
	MapLabel.clear(); MapLabel.shrink_to_fit();
	return ClassMap;
}
///////////////////////////////////////////////*Generate each iteration into an image*////////////////////////////////////////*/
vector<int>ReadClassificationTxt(string path, int Class_Num, int IterNum)
{
	int label = 0;
	CvScalar s;
	vector<int>Get_label;
	string DataFeatures[64];
	Get_label.clear();
	Get_label.shrink_to_fit();
	char adr[256] = { 0 };
	fstream dataFile;
	string str;
	sprintf(adr, "%s\\Proposed-SVM-%d.txt", path.data(), IterNum);
	dataFile.open(adr, ios::in || ios::out);
	if (!dataFile)
	{
		cerr << "打开文件失败" << endl;
		exit(0);
	}
	int num = 0;
	cout << "打开文件成功"<<endl;
	while (!dataFile.eof())
	{
		if (num == 0)
		{
			getline(dataFile, str);
			num += 1;
		}
		else
		{
			getline(dataFile, str);
			SplitString_1(str, DataFeatures, ",");
			label = atoi(DataFeatures[0].c_str());
			if (label == 0)
			{
				//cout << label << endl;
				label = 1;
				Get_label.push_back(label);
			}
			else if (label > Class_Num)
			{
				//cout << label << endl;
				label = Class_Num;
				Get_label.push_back(label);
			}
			else
			{
				Get_label.push_back(label);
			}
		}
	}
	dataFile.close();
	return Get_label;
}
//Character segmentation function1
void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	int i = 0;
	while (string::npos != pos2)
	{
		//v[i++] = s.substr(pos1, pos2 - pos1);
		v.push_back(s.substr(pos1, pos2 - pos1));
		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1, pos2 - pos1));
	//v[i] = s.substr(pos1, pos2 - pos1);
}
//Character segmentation function2
void SplitString_1(const string& s, string v[], const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	int i = 0;
	while (string::npos != pos2)
	{
		v[i++] = s.substr(pos1, pos2 - pos1);

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v[i] = s.substr(pos1, pos2 - pos1);
}
