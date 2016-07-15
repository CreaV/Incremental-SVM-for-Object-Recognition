#include "incsvm.h"
#include <vector>
using namespace std;

#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
#endif

static void readDirectory(const string& directoryName, vector<string>& filenames, bool addDirectoryName = true)
{
	filenames.clear();

#if defined(WIN32) | defined(_WIN32)
	struct _finddata_t s_file;
	string str = directoryName + "\\*.*";

	intptr_t h_file = _findfirst(str.c_str(), &s_file);
	if (h_file != static_cast<intptr_t>(-1.0))
	{
		do
		{
			if (addDirectoryName)
				filenames.push_back(directoryName + "\\" + s_file.name);
			else
				filenames.push_back((string)s_file.name);
		} while (_findnext(h_file, &s_file) == 0);
	}
	_findclose(h_file);
#else
	DIR* dir = opendir(directoryName.c_str());
	if (dir != NULL)
	{
		struct dirent* dent;
		while ((dent = readdir(dir)) != NULL)
		{
			if (addDirectoryName)
				filenames.push_back(directoryName + "/" + string(dent->d_name));
			else
				filenames.push_back(string(dent->d_name));
		}

		closedir(dir);
	}
#endif

	sort(filenames.begin(), filenames.end());
}
vector<vector<float> > kernel_values,kernel_values_45,kernel_values_90;   //核
vector<int> labels, labels_45, label_90;                                  //Y的值
IncSVM svm,svm_45,svm_90;
double C;                                                                //过拟合的罚项
int ImgWeight = 80;                                                      //图片宽
int ImgHeight = 224;                                                     //图片高
int features_numbers = 8748;                                             //特征数量
int sample_numbers = 0, sample_numbers_45 = 0, sample_numbers_90 = 0;    //样本数量
Mat data_mat=Mat::zeros(sample_numbers,features_numbers,CV_32FC1);       //hog数据
double scale = 0.025;													//  1/sigma^2

int pos_samples = 82;													//正例数量
int pos_samples_45 = 8;
int pos_samples_90;

vector<string> img_path,img_path_45,img_path_90;  //图像路径

Mat my_guasskernel2(Mat , Mat, vector<vector<float>>&);

void save_model(string filename)
{
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "C" << C;
	fs << "ImgWeiht" << ImgWeight;
	fs << "features_numbers" << features_numbers;
	fs << "sample_numbers" << sample_numbers;
	fs << "data_mat" << data_mat;
	fs << "kernel_values" << kernel_values;
	fs << "labels" << labels;
	fs << "scale" << scale;
	//vector<double> A = svm.get_a();
	fs << "svm-a" <<svm.a;
	fs << "svm-b" << svm.b;
	fs << "svm-c" << svm.c;
	fs << "svm-deps" << svm.deps;
	fs << "svm-ind0" << svm.ind[0];
	fs << "svm-ind1" << svm.ind[1];
	fs << "svm-ind2" << svm.ind[2];
	fs << "svm-ind3" << svm.ind[3];
	fs << "svm-scale" << svm.scale;
	fs << "svm-type" << svm.type;
	fs << "svm-y" << svm.y;
	fs << "svm-perturbations" << svm.perturbations;
	fs << "svm-kernel_evals" << svm.kernel_evals;
	fs << "svm-max_reserve_vectors" << svm.max_reserve_vectors;
	fs << "svm-g" << svm.g;

	fs << "svm-_Q-n" << svm._Q.n;
	fs << "svm-_Q-m" << svm._Q.m;
	fs << "svm-_Q-array" << "[";
	for (int i = 0; i < svm._Q.m; i++)
	{
		for (int j = 0; j < svm._Q.n; j++)
		{
			fs << svm._Q.array[i][j];
		}
	}
	fs << "]";
	fs << "svm-_Q-size" << svm._Q.size;
	fs << "svm-_Q-row_size" << "[";
	for (int i=0; i < svm._Q.m; i++)
	{
		fs << svm._Q.row_size[i];
	}
	fs << "]";
	fs << "svm-_Q-index_to_row" << "[";
	for (map<int, int>::iterator iter = svm._Q.index_to_row.begin(); iter != svm._Q.index_to_row.end(); iter++)
	{
		fs << iter->first;
		fs << iter->second;
	}
	fs << "]";
	fs << "svm-_Q-row_to_index" << "[";
	for (map<int, int>::iterator iter = svm._Q.row_to_index.begin(); iter != svm._Q.row_to_index.end(); iter++)
	{
		fs << iter->first;
		fs << iter->second;
	}
	fs << "]";

	fs << "svm-Rs-m" << svm._Rs.m;
	fs << "svm-Rs-n" << svm._Rs.n;
	fs << "svm-Rs-array" << "[";
	for (int i = 0; i < svm._Rs.m; i++)
	{
		for (int j = 0; j < svm._Rs.n; j++)
			fs << svm._Rs.array[i][j];
	}
	fs << "]";
	fs << "svm-RS-rowsize" << "[";
	for (int i = 0; i < svm._Rs.n; i++)
	{
		fs << svm._Rs.row_size[i];
	}
	fs << "]";
	fs << "svm-RS-size" << svm._Rs.size;


	fs.release();

}





double kernel(int i, int j, void *kparam){
  return kernel_values[i][j];
}

void process_instance(IncSVM& svm, int idx, int label){
  svm.setY(idx,label);
  svm.learn(idx,1);
}

void unlearn_instance(IncSVM& svm, int idx, int label){
  svm.unlearn(idx);
  svm.setY(idx,label);
}

double predict_value(IncSVM& svm, int idx){
    return svm.svmeval(idx,NULL);
}
void my_gaussKernel()
{
	Mat X_ = data_mat.clone();
	Mat K = Mat::ones(X_.rows, X_.rows, CV_32FC1);

	mulTransposed(X_, K, false);
	int Lx = X_.rows;
	int Ly = X_.rows;
	K = K * 2;
	pow(X_, 2, X_);
	Mat sum_X = Mat(Lx, 1, CV_32FC1);
	for (int i = 0; i < X_.rows; i++)
	{
		sum_X.at<float>(i, 0) = 0;
		for (int j = 0; j < X_.cols; j++)
			sum_X.at<float>(i, 0) += X_.at<float>(i, j);
	}
	Mat one_by_Ly = Mat::ones(1, Ly, CV_32FC1);
	K = K - sum_X*one_by_Ly;
	Mat Lx_by_one = Mat::ones(Lx, 1, CV_32FC1), sum_Y = Mat(1, Ly, CV_32FC1);
	sum_Y = sum_X.t();
	K = K - Lx_by_one*sum_Y;
	exp(K*scale, K);


	kernel_values.resize(sample_numbers);
	for (int i = 0; i < sample_numbers; i++)
	{
		kernel_values[i].resize(sample_numbers);
		for (int j = 0; j < sample_numbers; j++)
		{
			kernel_values[i][j] = K.at<float>(i, j);
		}
	}
}
/*
int gaussKernel()
{
	mwArray mwX(sample_numbers, features_numbers, mxDOUBLE_CLASS); 
	mwArray mwK(sample_numbers, sample_numbers, mxDOUBLE_CLASS); 
	mwArray mwScale(1, 1, mxDOUBLE_CLASS);
	double scale[1] = { 0.025 }; // 核的scale = 1/sigma^2
	mwScale.SetData(scale, 1);
	double *temp = new double[features_numbers*sample_numbers];
	int sum = 0;
	for (int i = 0; i < features_numbers; i++)
	{
		for (int j = 0; j < sample_numbers; j++)
		{
			temp[sum] = data_mat.at<float>(j, i); //std::cout << temp[sum] << " ";
			sum++;
		}
	}
	mwX.SetData(temp, sample_numbers*features_numbers);
	mykernel(1, mwK, mwX, mwX, mwScale);
	kernel_values.resize(sample_numbers);
	for (int i = 0; i < sample_numbers; i++)
	{	
		kernel_values[i].resize(sample_numbers);
		for (int j = 0; j < sample_numbers; j++)
		{
			kernel_values[i][j] = mwK.Get(2, i + 1, j + 1);
			//std::cout << kernel_values[i][j]<<" ";
		}
	}
	delete[]temp;
	return 0;
}
*/


void Inc(string filename)
{
	stopwatch sw;
	sw.reset();
	string test_path = filename;
	sample_numbers++;	
	Mat src, temp = Mat::zeros(ImgHeight, ImgWeight, CV_8UC3), current_feature = Mat::zeros(1, features_numbers, CV_32FC1);
	src = imread(test_path.c_str());									 //读入图片
	resize(src, temp, cv::Size(ImgWeight, ImgHeight), 0, 0, INTER_CUBIC);//重新调整大小
	HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWeight, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9); 
	vector<float>current_img_hog;										//当前hog图像特征数组
	hog->compute(temp, current_img_hog, Size(1, 1), Size(0, 0));
	std::cout << "当前图像hog维数：" << current_img_hog.size() << endl;
	int n = 0;
	for (vector<float>::iterator iter = current_img_hog.begin(); iter != current_img_hog.end(); iter++)
	{
		//data_mat.at<float>(0, n);
		current_feature.at<float>(0, n) = *iter;
		n++;
	}
	my_guasskernel2(data_mat, current_feature, kernel_values);
	vconcat(data_mat, current_feature, data_mat);				//合并当前矩阵到data_mat
																//my_gaussKernel();
	svm.AddOne();
																
}



void read()                          //读取labels 和 图片路径 
{
	ifstream svm_data("E:\svm.txt");
	string buf;
	while (svm_data){
		if (getline(svm_data, buf)){
			sample_numbers++;
			if (sample_numbers < pos_samples){
				labels.push_back(1);
				img_path.push_back(buf);//图像路径 
			}
			else{
				labels.push_back(-1);
				img_path.push_back(buf);//图像路径 
			}
		}
	}
	svm_data.close();                    //关闭文件  
	/*ifstream svm_data1("E:\SVM_TARIN_45.txt");
	string buf1;
	while (svm_data1){
		if (getline(svm_data1, buf1)){
			sample_numbers++;
			if (sample_numbers <= pos_samples){
				labels.push_back(1);
				img_path.push_back(buf1);//图像路径 
			}
			else{
				labels.push_back(-1);
				img_path.push_back(buf1);//图像路径 
			}
		}
	}
	svm_data1.close();
	ifstream svm_data2("E:\SVM_TARIN_90.txt");
	string buf2;
	while (svm_data2){
		if (getline(svm_data2, buf2)){
			sample_numbers++;
			if (sample_numbers <= pos_samples){
				labels.push_back(1);
				img_path.push_back(buf2);//图像路径 
			}
			else{
				labels.push_back(-1);
				img_path.push_back(buf2);//图像路径 
			}
		}
	}
	svm_data2.close();*/
}

void GetHogdetection()
{
	Mat src;
	Mat trainImg = Mat::zeros(ImgHeight, ImgWeight, CV_8UC3);//需要分析的图片
	for (string::size_type i = 0; i!= img_path.size(); i++)
	{
		src = imread(img_path[i].c_str(), 1); //读入图片
		std::cout << "processing " << img_path[i].c_str() << endl;
		resize(src, trainImg, cv::Size(ImgWeight, ImgHeight), 0, 0, INTER_CUBIC);
		HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWeight, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float>current_img_hog; //当前hog图像特征数组
		hog->compute(trainImg, current_img_hog, Size(1, 1), Size(0, 0));
		if (i == 0){
			data_mat = Mat::zeros(sample_numbers, current_img_hog.size(), CV_32FC1);
		}
		//开辟训练数组
		std::cout << "hog dims：" << features_numbers << endl;
		int n = 0;
		for (vector<float>::iterator iter = current_img_hog.begin(); iter != current_img_hog.end(); iter++){
			data_mat.at<float>(i, n) = *iter;
			n++;
		}
		std::cout << "end processing:" << img_path[i].c_str() << " " << labels[i] << endl;
	}
}

Mat my_guasskernel2(Mat X, Mat Y, vector<vector<float>>&kernel_values)
{
	Mat Y_ = Y.t();
	Mat K;
	int Lx = X.rows;
	int Ly = Y.rows;

	Mat sum_X = Mat(Lx, 1, CV_32FC1), sum_Y = Mat(Ly, 1, CV_32FC1);
	Mat one_by_Ly = Mat::ones(1, Ly, CV_32FC1);
	Mat Lx_by_one = Mat::ones(Lx, 1, CV_32FC1);
	K = X*Y_;
	K = K * 2;
	float temp = 0;
	for (int i = 0; i < X.rows; i++)
	{
		sum_X.at<float>(i, 0) = 0;
		for (int j = 0; j < X.cols; j++)
		{
			temp = X.at<float>(i, j);
			temp = temp*temp;
			sum_X.at<float>(i, 0) += temp;
		}
	}

	K = K - sum_X*one_by_Ly;
	temp = 0;
	for (int i = 0; i < Y.rows; i++)
	{
		sum_Y.at<float>(i, 0) = 0;
		for (int j = 0; j < Y.cols; j++)
		{
			temp = Y.at<float>(i, j);
			temp = temp*temp;
			sum_Y.at<float>(i, 0) += temp;
		}
	}
	Mat T_Mul = Lx_by_one*sum_Y.t();
	K = K - T_Mul;
	exp(K*scale, K);

	if (kernel_values.size() == 0)
	{
		kernel_values.resize(K.rows);
		for (int i = 0; i < K.rows; i++)
		{
			kernel_values[i].resize(K.rows);
			for (int j = 0; j < K.rows; j++)
			{
				kernel_values[i][j] = K.at<float>(i, j);
			}
		}
	}
	else
	{
		vector<float> a;
		for (int i = 0; i < K.rows; i++)
		{
			kernel_values[i].push_back(K.at<float>(i, 0));
			a.push_back(K.at<float>(i, 0));
		}
		a.push_back(1);
		kernel_values.push_back(a);
	}
	return	K;
}


int main(){
	/*if (!libmykernelInitialize())
	{
		std::cout << "Could not initialize libMyAdd!" << std::endl;
		return -1;
	}
	*/
	std::cout << "输入C:" << endl;
	C=10;
	read();
	stopwatch sw;
	GetHogdetection();
	std::cout << "计算hog特征：" << sw.get_time() << "s" << endl;
	sw.reset();
	my_gaussKernel();
	//load_kernel_from_file("data_kernelnew.txt");
	std::cout<<"核载入： " << sw.get_time() << " s" <<endl;
	sw.reset();

	svm.initialize(kernel_values.size(),1000);
	for (int i = 0; i<kernel_values.size(); i++){
		svm.adddata(i, labels[i], C);
	}


  //training loop
	for (int i = 0; i < kernel_values.size(); i++){
			process_instance(svm, i, labels[i]);
	}
	 std::cout<<"训练完成： " << sw.get_time() << " secs" <<endl;
	 sw.reset();
	int num_correct = 0;
	for(int i = 0; i< kernel_values.size(); i++){
		double out = predict_value(svm, i);
		if (out * labels[i] > 0)
		{
			num_correct++;
		}

  }
	std::cout << "正确分类数量: " << num_correct  << endl;
	save_model("model.xml");
	string images_folder;
	vector<string> images_filenames;
	readDirectory(images_folder, images_filenames);

	for (;;)
	{
		std::cout << "开始预测：" << endl;
		string filename;
		std::cout << "输入图像路径" << endl;
		std::cin >> filename;
		Mat src = imread(filename);
		if (src.empty())
		{
			std::cout << "未找到图片、重新输入" << endl;
			continue;
		}
		sw.reset();

		Inc(filename);
		std::cout << "sum Inc time:" << sw.get_time() << "s" << endl;

		double out = predict_value(svm, kernel_values.size() - 1);
		if (out < 0)
			std::cout << "这是-1类" << endl;
		else
			std::cout << "这是1类" << endl ;
		cout << out << endl;
		std::cout << "判断正确或者错误Y/N" << endl;
		char ans;
		cin >> ans;
		if (ans == 'n' || ans == 'N')
		{
			if (out > 0)
				labels.push_back(-1);
			else
				labels.push_back(1);
		}
		else if (ans == 'Y' || ans == 'y')
		{
			if (out > 0)
				labels.push_back(1);
			else
				labels.push_back(-1);
		}
		else if (ans == 'E' || ans == 'e')
			break;
		svm.adddata(kernel_values.size() - 1, labels[kernel_values.size() - 1], C);
		process_instance(svm, kernel_values.size() - 1, labels[kernel_values.size() - 1]);//增强学习
		std::cout << "学习之后的结果" << predict_value(svm, kernel_values.size() - 1) << endl;
		
		std::cout << endl;
	}
	
  return 0;
}



