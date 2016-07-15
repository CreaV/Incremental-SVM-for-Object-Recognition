#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "screenShotWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <inc_svm.h>
#include <vector>

using namespace std;

Mat draw_img,new_object;
int draw_x1, draw_y1;
bool select_flag = false;
Point P1, P2;
string windowname = "crop";

vector<vector<float>> kernel_values, kernel_values_temp;						//核
vector<int> labels;																//Y的值
IncSVM svm;
double C = 10;                                                                 //过拟合的罚项
int ImgWidth = 80;                                                            //图片宽
int ImgHeight = 224;                                                           //图片高
int features_numbers = 8748;                                                   //特征数量
int sample_numbers = 0;															//样本数量
Mat data_mat = Mat::zeros(sample_numbers, features_numbers, CV_32FC1);         //hog数据
double scale = 0.025;                                                          //  1/sigma^2

static string Wstr2Str(wstring wstr)
{
	if (wstr.length() == 0)
		return "";

	std::string str;
	str.assign(wstr.begin(), wstr.end());
	return str;
}

wstring stringToWstring(const std::string& str)
{
	LPCSTR pszSrc = str.c_str();
	int nLen = MultiByteToWideChar(CP_ACP, 0, pszSrc, -1, NULL, 0);
	if (nLen == 0)
		return std::wstring(L"");

	wchar_t* pwszDst = new wchar_t[nLen];
	if (!pwszDst)
		return std::wstring(L"");

	MultiByteToWideChar(CP_ACP, 0, pszSrc, -1, pwszDst, nLen);
	std::wstring wstr(pwszDst);
	delete[] pwszDst;
	pwszDst = NULL;
	return wstr;
}
void save_model(string filename, IncSVM &svm, int &sample_numbers, Mat &data_mat,
    vector<vector<float> >&kernel_values, vector<int> &labels)
{
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "C" << C;
    fs << "ImgWeiht" << ImgWidth;
    fs << "features_numbers" << features_numbers;
    fs << "scale" << scale;

    fs << "sample_numbers" << sample_numbers;
    fs << "data_mat" << data_mat;
    fs << "kernel_values" << kernel_values;
    fs << "labels" << labels;
    fs << "svm-a" << svm.a;
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
    for (int i = 0; i < svm._Q.m; i++)
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

int read_model(string filename, IncSVM &svm, int &sample_numbers, Mat &data_mat,
    vector<vector<float>>&kernel_values, vector<int>& labels)
{
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    sample_numbers = (int)fs["sample_numbers"];
    fs["data_mat"] >> data_mat;
    vector<vector<float>> temp;
	fs["kernel_values"] >> temp;
    kernel_values.resize(sample_numbers);
    for (int i = 0; i < sample_numbers; i++)
    {
        kernel_values[i].resize(sample_numbers);
        for (int j = 0; j < sample_numbers; j++)
        {
            kernel_values[i][j] = temp[i*sample_numbers + j][0];
        }
    }


    fs["labels"] >> labels;
    fs["svm-a"] >> svm.a;
    fs["svm-b"] >> svm.b;
    fs["svm-c"] >> svm.c;
    fs["svm-deps"] >> svm.deps;
    fs["svm-ind0"] >> svm.ind[0];
    fs["svm-ind1"] >> svm.ind[1];
    fs["svm-ind2"] >> svm.ind[2];
    fs["svm-ind3"] >> svm.ind[3];
    fs["svm-scale"] >> svm.scale;
    fs["svm-type"] >> svm.type;
    fs["svm-y"] >> svm.y;
    fs["svm-perturbations"] >> svm.perturbations;
    fs["svm-kernel_evals"] >> svm.kernel_evals;
    fs["svm-max_reserve_vectors"] >> svm.max_reserve_vectors;
    fs["svm-g"] >> svm.g;


    fs["svm-_Q-n"] >> svm._Q.n;
    fs["svm-_Q-m"] >> svm._Q.m;
    fs["svm-_Q-size"] >> svm._Q.size;
    FileNode n = fs["svm-_Q-array"];
    if (n.type() != FileNode::SEQ){
        cerr << "strings is not a sequence! FAIL" << endl;
        return 1;
    }
    FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
    //for (; it != it_end; ++it)
    svm._Q.load_initialize();
    for (int i = 0; i < svm._Q.m; i++){
        for (int j = 0; j < svm._Q.n; j++){
            svm._Q.array[i][j] = (double)*it;
            it++;//cout << (string)*it << endl;
        }
    }
    n = fs["svm-_Q-row_size"];
    if (n.type() != FileNode::SEQ){
        cerr << "strings is not a sequence! FAIL" << endl;
        return 1;
    }
    it = n.begin(), it_end = n.end();
    for (int i = 0; i < svm._Q.m; i++){
        svm._Q.row_size[i] = *it;
        it++;
    }
    n = fs["svm-_Q-index_to_row"];
    it = n.begin(), it_end = n.end();
    //map<int, int>::iterator iter = svm._Q.index_to_row.begin();
    for (; it != n.end(); it++){
        int first = *it; it++;
        int second= *it;
        svm._Q.index_to_row.insert(map<int, int>::value_type(first, second));
    }
    n = fs["svm-_Q-row_to_index"];
    it = n.begin(), it_end = n.end();
    for (; it != n.end(); it++){
        int first = *it; it++;
        int second = *it;
        svm._Q.row_to_index.insert(map<int, int>::value_type(first, second));
    }


    fs["svm-Rs-m"] >> svm._Rs.m;
    fs["svm-Rs-n"] >> svm._Rs.n;
    svm._Rs.load_initialize();
    n = fs["svm-Rs-array"];
    it = n.begin(), it_end = n.end();
    for (int i = 0; i < svm._Rs.m; i++){
        for (int j = 0; j < svm._Rs.n; j++){
            svm._Rs.array[i][j] = (double)*it;
            it++;//cout << (string)*it << endl;
        }
    }
    n = fs["svm-RS-rowsize"];
    it = n.begin(), it_end = n.end();
    for (int i = 0; i< svm._Rs.m; i++){
        svm._Rs.row_size[i] = *it;
        it++;
    }
    fs["svm-RS-size"] >> svm._Rs.size;
    fs.release();
}

double kernel(int i, int j, void * params){
    return kernel_values[i][j];
}

void process_instance(IncSVM& svm, int idx, int label){
    svm.setY(idx, label);
    svm.learn(idx, 1);
}

void unlearn_instance(IncSVM& svm, int idx, int label){
    svm.unlearn(idx);
    svm.setY(idx, label);
}

double predict_value(IncSVM& svm, int idx){
    return svm.svmeval(idx, NULL);
}
Mat My_resize(Mat src)
{
	double alpha = 0.99;
	int size_w = src.cols, size_h = src.rows;
	while (size_w*size_h >= 117000)
	{
		size_w*=alpha;
		size_h*=alpha;
	}
	resize(src, src, cv::Size(size_w, size_h), 0, 0, INTER_CUBIC);
	return src;
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

void Inc()
{
	sample_numbers++;
	Mat src, temp = Mat::zeros(ImgHeight, ImgWidth, CV_8UC3), current_feature = Mat::zeros(1, features_numbers, CV_32FC1);
	src = new_object;
	resize(src, temp, cv::Size(ImgWidth, ImgHeight), 0, 0, INTER_CUBIC);//重新调整大小
	HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWidth, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float>current_img_hog;										//当前hog图像特征数组
	hog->compute(temp, current_img_hog, Size(1, 1), Size(0, 0));
	//std::cout << "当前图像hog维数：" << current_img_hog.size() << endl;
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

double predict(vector<vector<float>> kernel, double b, vector<int>* ind, vector<double>a, vector<double> y, int indc)
{
    double fx = b;
    int sets[] = { 0, 1, 2 };
    for (int j = 0; j < 3; j++){
        int set = sets[j];
        for (unsigned int i = 0; i<ind[set].size(); i++)
        {
            int index = ind[set][i];
            double k = kernel[indc][index];
            if (set == 0 || set == 1 || a[index] >0)
                fx = fx + a[index] * y[index] * k;
        }
    }
    return fx;
}

void gamma(Mat &src)
{
    CV_Assert(src.data);  //若括号中的表达式为false，则返回一个错误的信息。
    double fGamma = 1 / 2.2;
    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));
    // build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = pow((float)(i / 255.0), fGamma) * 255.0;
    }
    //先归一化，i/255,然后进行预补偿(i/255)^fGamma,最后进行反归一化(i/255)^fGamma*25
    const int channels = src.channels();
    switch (channels)
    {
    case 1:
    {
              //运用迭代器访问矩阵元素
              MatIterator_<uchar> it, end;
              for (it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++)
                  //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
                  *it = lut[(*it)];
              break;
    }
    case 3:
    {
              MatIterator_<Vec3b> it, end;
              for (it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; it++)
              {
                  (*it)[0] = lut[((*it)[0])];
                  (*it)[1] = lut[((*it)[1])];
                  (*it)[2] = lut[((*it)[2])];
              }
              break;
    }
    }
}

Mat Multiple(int indc, Mat src, vector<vector<float>> kernel_values, double b, vector<int>* ind, vector<double>a,
    vector<double> y,Mat data_mat,vector<Mat>& Return_Roi)
    //多尺度识别 并在图片中画出
    //Return_Roi 则为识别出来的图像
    //src 输入图片
    //kernel_values data_mat 不能改变原值
{
	src = My_resize(src);

    int ImgH = ImgHeight;
    int ImgW = ImgWidth;
    bool isfound = false;
    double Times = 1.10;										// 模板宽高之比*Times(倍数)得到最小分辨图片尺寸
    int src_W = src.cols, src_H = src.rows ;				// 原图片宽 高
    int resize_W = src_W * Times, resize_H = src_H * Times;
    double resize_a = 0.90;									//每次缩放比例
    Mat temp;												//图片缓存
    stopwatch sw;
    Mat current_feature = Mat::zeros(1, features_numbers, CV_32FC1);							//当前特征矩阵
    vector<Rect> draw_in;
    sw.reset();
    cvtColor(src, temp, CV_BGR2GRAY);
    gamma(temp);                                                                         //gamma滤波
    resize(src, temp, cv::Size(resize_W, resize_H), 0, 0, INTER_CUBIC);					 //缩放到的最小尺度图片 在最小尺度能找到最大的物体
    while (resize_H >= ImgH&&resize_W >= ImgW)
    {
        resize(temp, temp, cv::Size(resize_W, resize_H), 0, 0, INTER_CUBIC);                     //缩放
        int temp_size_W = temp.cols;
        int temp_size_H = temp.rows;
        int step = 8;                                                                           //检测窗口移动步长
        Mat temp_data_mat = data_mat.clone();
        vector<float>current_img_hog;

        for (int j = 0; j + step + ImgH< temp_size_H; j = j + step)
        {
            for (int k = 0; k + step + ImgW < temp_size_W; k = k + step)
            {
                if (temp.at<Vec3b>(j, k) == Vec3b(0, 0, 0)){
                    k = k + ImgW;
                    continue;
                }

                Mat roi = temp(Rect(k, j, ImgW, ImgH));
                HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgW, ImgH), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);                                                                        //当前hog图像特征数组
                hog->compute(roi, current_img_hog, Size(1, 1), Size(0, 0));
                int n = 0;
                for (vector<float>::iterator iter = current_img_hog.begin(); iter != current_img_hog.end(); iter++)
                {
                    current_feature.at<float>(0, n) = *iter;
                    n++;
                }
                my_guasskernel2(temp_data_mat, current_feature, kernel_values);

                if (predict(kernel_values,b,ind,a,y,indc) > 0)
                {
                    Mat clone = current_feature.clone();
                   // cout << "out:" << predict(kernel_values, b, ind, a, y, indc) << endl;
                    isfound = true;
                    Return_Roi.push_back(clone);                              //返回找到物体的特征
                    double S = (double)src_W / (double)resize_W;
                    Rect R((int)k*S, (int)j*S, (int)ImgW*S, (int)ImgH*S);               //计算找到的矩形框在原图像中的位置 并保存
                    draw_in.push_back(R);
                    Mat temp_roi(temp, Rect(k, j, ImgW, ImgH));
                    temp_roi = Scalar(0, 0, 0);											//把当前的区域涂黑、防止重复检测
                    k = k + ImgW;
                }
                kernel_values.pop_back();
                for (size_t I = 0; I < kernel_values.size(); I++)
                {
                    kernel_values[I].pop_back();
                }
            }
        }
        sw.reset();
        resize_W = resize_a*resize_W;                                                    //变小
        resize_H = resize_a*resize_H;
    }

    Scalar color(0, 0, 255);
    for (size_t size = 0; size < draw_in.size(); size++)
    {
        rectangle(src, draw_in[size], color);
        string num;
        char t[256];
        sprintf(t, "%d", size + 1);
        num = t;
        putText(src, num, Point(draw_in[size].x + 2, draw_in[size].y + 13), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 0), 2);
    }
    return src;
}


void MainWindow::INI()
{
	HRESULT hr = m_SREngine.InitializeSapi(this->winId(), WM_RECOEVENT);  //初始化SAPI
	if (FAILED(hr))
	{
		QMessageBox::information(NULL, "Error", "FAIL TO GET WinID", MB_OK);
		return;
	}

	hr = m_SREngine.LoadCmdFromFile(grammarFileName);   //创建语法规则
	if (FAILED(hr))
	{
		QMessageBox::information(NULL, "Error", "FAIL TO LOAD GRAMMER", MB_OK);
		return;
	}

}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	soundActive = false;
	m_bSoundStart = false;
	m_bSoundEnd = false;
	m_bGotReco = false;
	m_tPreTime = 0;
	m_tDelay = 1000;
	sum_count = 0;
	grammarFileName = "SpeechGrammar.xml";
	INI();

	//TTS初始化
	::CoInitialize(NULL);         // COM初始化
	// 获取ISpVoice接口
	CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_INPROC_SERVER, IID_ISpVoice, (void**)&pSpVoice);
	pSpVoice->SetVolume(200);
	static const WCHAR helloString[] = L"您好，欢迎使用本系统。";
	pSpVoice->Speak(helloString, SPF_DEFAULT, NULL);
}

MainWindow::~MainWindow()
{
	pSpVoice->Release();
    delete ui;
}

void OnMouse(int event, int x, int y, int, void*)
{

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		draw_x1 = x;
		draw_y1 = y;
		select_flag = true;
	}
	
	else if (select_flag&&event == CV_EVENT_MOUSEMOVE)
	{
		P1 = Point(draw_x1, draw_y1);
		P2 = Point(x, y);
		Mat temp = draw_img.clone();
		rectangle(temp, P1, P2, Scalar(0, 255, 0), 2);
		imshow(windowname, temp);
		temp = draw_img.clone();
	}

	else if (select_flag&&event == CV_EVENT_LBUTTONUP)
	{
		P1 = Point(draw_x1, draw_y1);
		P2 = Point(x, y);
		Mat temp = draw_img.clone();
		if (P2.x > temp.cols||P2.y > temp.rows)
		{
			P2.x = temp.cols;
			P2.y = temp.rows;
		}
		rectangle(temp, P1, P2, Scalar(0, 255, 0), 2);
		imshow(windowname, temp);
		temp = draw_img.clone();
		new_object = draw_img(Rect(P1, P2));
		select_flag = false;
		return;
	}
}

void MainWindow::on_pushButton_4_clicked()
{
	draw_img = image.clone();
	if (!image.empty())
	{
		static const WCHAR wtip[] = L"请在图中截取图片，按Y确定";
		pSpVoice->Speak(wtip, SPF_DEFAULT, NULL);
		namedWindow(windowname, WINDOW_AUTOSIZE);
		imshow(windowname, draw_img);
		setMouseCallback(windowname, OnMouse, 0);
		char c = waitKey(0);
		while (true)
		{
			if (c == 'y'||c=='Y')
				break;
		}

		namedWindow("DST", WINDOW_AUTOSIZE);
		imshow("DST", new_object);
		labels.push_back(1);
		Inc();

		svm.adddata(kernel_values.size() - 1, labels[kernel_values.size() - 1], C);
		process_instance(svm, kernel_values.size() - 1, labels[kernel_values.size() - 1]);			  //增强学习
		static const WCHAR learnsuccess[] = L"截取的图片学习成功。关闭剪切图片窗口。点击保存模型保存所学习的新模型。";
		pSpVoice->Speak(learnsuccess, SPF_DEFAULT, NULL);
	}
	else
	{
		QMessageBox::information(NULL,"ERROR","please load img first!",MB_OK);
	}
}

void MainWindow::on_pushButton_clicked()
{
	if(sample_numbers==0){
		static const WCHAR error[] = L"请先点击导入模型，再点击导入图片进行识别";
		pSpVoice->Speak(error, SPF_DEFAULT, NULL);
		return;
	}
    filename = QFileDialog::getOpenFileName(this,
                                          tr("Open Image File"), ".",
                                          tr("Image Files (*.jpg *.png *.bmp *.jpeg *.tiff *.tif *.dib *.jp2 *.jpe *.ppm *.pgm *.pbm *.ras *.sr)"));
	if (!filename.isEmpty()){

		//加载图片并读取文件
		image = imread(filename.toStdString(), CV_LOAD_IMAGE_COLOR);
		
		bool flag = false;
		//检查图片的输入
		if (image.empty())
		{
			QMessageBox msgBox;
			msgBox.setText("The selected image could not be opened!");
			msgBox.show();
			msgBox.exec();
		}
		//如果图片已加载,则在label上显示像素图
		else
		{
			static const WCHAR helloString[] = L"图片已读入，正在进行识别，请耐心等待。";
			pSpVoice->Speak(helloString, SPF_DEFAULT, NULL);
			Return_Roi.clear();
			Mat temp = Multiple(kernel_values.size(), image, kernel_values, svm.b, svm.ind,
				svm.a, svm.y, data_mat, Return_Roi);
			int resize_w = temp.cols, resize_h = temp.rows;
			while (resize_w <= 540 && resize_h <= 400)
			{
				resize_w *= 1.05;
				resize_h *= 1.05;
			}
			cv::resize(temp, temp, cv::Size(resize_w, resize_h), 0, 0, INTER_CUBIC);
			QImage qigm = MatToQImage(temp);
			QPixmap qpix = QPixmap::fromImage(qigm);
			//int resize_w = qpix.width(), resize_h = qpix.height();
	
			
			ui->label->setPixmap(qpix);
			ui->label->resize(resize_w, resize_h);
			//ui->label->setPixmap(QPixmap::fromImage(qigm).scaledToWidth(ui->label->size().width(), Qt::FastTransformation));
		}

		if (Return_Roi.size() == 0)
		{
			sum_count = 0;
			static const WCHAR cannotfind[] = L"图中未找到物体，如存在物体，请点击截取和学习";
			pSpVoice->Speak(cannotfind, SPF_DEFAULT, NULL);
		}
		else
		{

			if (soundActive == false){
				sum_count = 1;
				static const WCHAR find[] = L"找到物体，请回答图中第1个判断是否正确?请回答：是，或，否。";
				pSpVoice->Speak(find, SPF_DEFAULT, NULL);
				hr = m_SREngine.SetRuleState(NULL, NULL, SPRS_ACTIVE);
				if (FAILED(hr))
				{
					QMessageBox::information(NULL, "Error", "SetRuleState Active Error!", MB_OK);
				}
				soundActive == true;
			}//this->nativeEvent;
			else{
				m_SREngine.release();
				INI();
				m_SREngine.SetRuleState(NULL, NULL, SPRS_ACTIVE);
			}
			ui->pushButton->setEnabled(false);
		}
	}
}

bool MainWindow::nativeEvent(const QByteArray &eventType, void *message, long *result)
{
	setWindowTitle("Object Detection");
	Q_UNUSED(eventType);
	MSG* msg = reinterpret_cast<MSG*>(message);
	if (msg->message == WM_RECOEVENT)
	{
		*result = this->OnRecoEvent();
	}
	return false;
}

QImage MainWindow::MatToQImage(const Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1)
    {
        //设置颜色表(used to translate colour indexes to qRgb values)
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
            colorTable.push_back(qRgb(i,i,i));
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    // 8-bits unsigned, NO. OF CHANNELS=3
    else if(mat.type()==CV_8UC3)
    {
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    else
    {
        return QImage();
    }
}

void MainWindow::on_pushButton_2_clicked()
{
    read_model("model_1.xml", svm, sample_numbers, data_mat, kernel_values, labels);
	QMessageBox::information(NULL, "success!", "Load model success!",MB_OK);
	ui->pushButton_2->setEnabled(false);
}

void MainWindow::on_pushButton_3_clicked()
{
	if (sample_numbers > 0)
	{
		save_model("model_1.xml", svm, sample_numbers, data_mat, kernel_values, labels);
		QMessageBox::information(NULL, "success", "model saved", MB_OK);
	}
	else
	{
		QMessageBox::information(NULL, "error", "model empty", MB_OK);
	}
}

// Speech Recognition Event Process
LRESULT MainWindow::OnRecoEvent()
{
	/*
	QString S1 = QStringLiteral("其他");
	QString S2 = QStringLiteral("成功");
	USES_CONVERSION;
	CSpEvent event;
	HRESULT hr = S_OK;
	if (m_SREngine.m_cpRecoContext)
	{
	while (event.GetFrom(m_SREngine.m_cpRecoContext) == S_OK)
	{
	switch (event.eEventId)
	{
	case SPEI_REQUEST_UI:
	case SPEI_INTERFERENCE:
	case SPEI_PROPERTY_NUM_CHANGE:
	case SPEI_PROPERTY_STRING_CHANGE:
	case SPEI_RECO_STATE_CHANGE:
	case SPEI_RECO_OTHER_CONTEXT:
	setWindowTitle(S1);
	break;

	case SPEI_RECOGNITION:
	{
	setWindowTitle(S2);
	m_bGotReco = TRUE;
	static const WCHAR wszURCGNZD[] = L"<Unrecongnized>";
	static const WCHAR SHI[] = L"是";
	wstring wshi(SHI);
	static const WCHAR FOU[] = L"否";
	wstring wfou(FOU);
	CSpDynamicString dstrText;
	wstring ws(dstrText);
	if (FAILED(event.RecoResult()->GetText(SP_GETWHOLEPHRASE, SP_GETWHOLEPHRASE, TRUE, &dstrText, NULL)))
	{
	dstrText = wszURCGNZD;
	}

	if (ws == wshi)
	{
	QString output = QStringLiteral("是\n");
	ui->textEdit->insertPlainText(output);
	}
	else if (ws == wfou)
	{
	QString output = QStringLiteral("否\n");
	ui->textEdit->insertPlainText(output);
	}
	else
	{


	string strstr = Wstr2Str(ws);
	char *cstr = const_cast<char*>(strstr.c_str());
	QString strResult(QString::fromLocal8Bit(cstr));


	QString strResult;
	strResult = W2T(dstrText);
	ui->textEdit->insertPlainText(strResult + "OK\n");
	event.RecoResult()->Release();
	m_bSoundStart = false;
	m_bSoundEnd = false;
	}

	ISpVoice *pSpVoice;        // 重要COM接口
	::CoInitialize(NULL);         // COM初始化
	// 获取ISpVoice接口
	HRESULT my_hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_INPROC_SERVER, IID_ISpVoice, (void**)&pSpVoice);
	pSpVoice->SetVolume(200);
	pSpVoice->Speak(dstrText, SPF_DEFAULT, NULL);
	pSpVoice->Release();

	break;

	}
	case SPEI_FALSE_RECOGNITION:
	setWindowTitle(" FAIL");
	break;
	case SPEI_SOUND_END:
	m_bSoundEnd = true;
	setWindowTitle("Sound End");
	break;
	default:
	setWindowTitle("FUCK");
	break;
	}
	}
	}
	*/

	USES_CONVERSION;
	CSpEvent event;
	HRESULT hr = S_OK;
	bool flag = true;
	if (m_SREngine.m_cpRecoContext)
	{
		while (event.GetFrom(m_SREngine.m_cpRecoContext) == S_OK && flag)
		{
			//Get the ID
			setWindowTitle("Sound Active");
			switch (event.eEventId)
			{
			case SPEI_REQUEST_UI:
			case SPEI_INTERFERENCE:
			case SPEI_PROPERTY_NUM_CHANGE:
			case SPEI_PROPERTY_STRING_CHANGE:
			case SPEI_RECO_STATE_CHANGE:
			case SPEI_RECO_OTHER_CONTEXT:
				break;
			case SPEI_SOUND_START:
				m_bSoundStart = true;
				setWindowTitle("Sound Start");
				break;
			case SPEI_SOUND_END:
				m_bSoundEnd = true;
				setWindowTitle("Sound End");
				break;
			case SPEI_FALSE_RECOGNITION:
				setWindowTitle("False Sound Recognition");
				break;
			case SPEI_HYPOTHESIS:
				setWindowTitle("Hypothesis");
				//break;
			case SPEI_RECOGNITION:
			{
									 setWindowTitle("Sound Recognition");
									 CComPtr <ISpRecoResult> cpResult;
									 CSpDynamicString dstrText;
									 cpResult = event.RecoResult();
									 cpResult->GetText(SP_GETWHOLEPHRASE, SP_GETWHOLEPHRASE, TRUE, &dstrText, NULL);
									 wstring wsResult(dstrText);
									 static const WCHAR yes[] = L"是";
									 wstring wyes(yes);
									 // static const WCHAR right[] = L"正确";
									 //wstring wright(right);
									 //static const WCHAR correct[] = L"对";
									 //wstring wcorrect(correct);
									 //static const WCHAR _correct[] = L"对的";
									 //wstring w_correct(_correct);
									 static const WCHAR no[] = L"否";
									 wstring wno(no);
									 // static const WCHAR fal[] = L"错误";
									 // wstring wfal(fal);
									 //static const WCHAR wrong[] = L"错";
									 //wstring wwrong(wrong);
									 //static const WCHAR _wrong[] = L"不对";
									 //wstring w_wrong(_wrong);

									 if (m_bSoundStart && m_bSoundEnd)
										 //if(1)
									 {
										 int dt;
										 QString strLogTxt;
										 SYSTEMTIME st;
										 GetLocalTime(&st);

										 if (!m_tPreTime)
										 {
											 m_tPreTime = (st.wHour * 3600 + st.wMinute * 60 + st.wSecond) * 1000 + st.wMilliseconds;
											 m_tDelay = 1000;
										 }
										 else
										 {
											 dt = (st.wHour * 3600 + st.wMinute * 60 + st.wSecond) * 1000 + st.wMilliseconds;
											 m_tDelay = dt - m_tPreTime;
											 if (m_tDelay < 0)m_tDelay += 24 * 3600 * 1000;
											 m_tPreTime = dt;
										 }

										 if (m_tDelay>500){
											 if (wsResult == yes){//wsResult==right||wsResult==correct||wsResult==_correct)
												 sum_count++;
												 flag = false;
												 m_SREngine.release();
												 INI();
												 m_SREngine.SetRuleState(NULL, NULL, SPRS_ACTIVE);
												 hr = m_SREngine.m_cpRecoContext->Pause(NULL);

												 setWindowTitle("got yes");
												 if (sum_count <= Return_Roi.size()){
													 char c_num[256];
													 sprintf(c_num, "%d", sum_count);
													 string pre = "图中第";
													 string num(c_num);
													 string rear = "个判断是否正确?";
													 wstring w_num = stringToWstring(pre + num + rear);
													 pSpVoice->Speak(w_num.c_str(), SPF_DEFAULT, NULL);
													 if (SUCCEEDED(hr)){
														 hr = m_SREngine.m_cpRecoContext->Resume(NULL);
													 }
												 }
												 else{
													 m_SREngine.m_cpRecoContext->Pause(NULL);
													 //m_SREngine.SetRuleState(NULL, NULL, SPRS_INACTIVE);
													 static const WCHAR wString[] = L"学习已经完成。如果图中还存在未找到的物体，请按截取和学习键来学习新模型。";
													 pSpVoice->Speak(wString, SPF_DEFAULT, NULL);													
													 setWindowTitle("Got Yes - Sound Recognition End");
												 }
											 }
											 else if (wsResult == no) //sResult == fal || wsResult == wrong||wsResult==_wrong)
											 {
												 sum_count++;
												 flag = false;
												 m_SREngine.release();
												 INI();
												 m_SREngine.SetRuleState(NULL, NULL, SPRS_ACTIVE);
												 hr = m_SREngine.m_cpRecoContext->Pause(NULL);

												 setWindowTitle("got no");

												 sample_numbers++;
												 my_guasskernel2(data_mat, Return_Roi[sum_count - 2], kernel_values);
												 vconcat(data_mat, Return_Roi[sum_count - 2], data_mat);
												 svm.AddOne();
												 labels.push_back(-1);
												 svm.adddata(kernel_values.size() - 1, labels[kernel_values.size() - 1], C);
												 process_instance(svm, kernel_values.size() - 1, labels[kernel_values.size() - 1]);			  //增强学习

												 if (sum_count <= Return_Roi.size()){
													 char c_num[256];
													 sprintf(c_num, "%d", sum_count);
													 string pre = "图中第";
													 string num(c_num);
													 string rear = "个判断是否正确";
													 wstring w_num = stringToWstring(pre + num + rear);
													 pSpVoice->Speak(w_num.c_str(), SPF_DEFAULT, NULL);
													 if (SUCCEEDED(hr)){
														 hr = m_SREngine.m_cpRecoContext->Resume(NULL);
													 }
												 }
												 else{
													 //m_SREngine.SetRuleState(NULL, NULL, SPRS_INACTIVE);
													 m_SREngine.m_cpRecoContext->Pause(NULL);
													 static const WCHAR wString[] = L"学习已经完成。如果图中还存在未找到的物体，请按截取和学习键来学习新模型。";
													 pSpVoice->Speak(wString, SPF_DEFAULT, NULL);

													 setWindowTitle("Got No - Sound Recognition End");
												 }
											 }
											 cpResult.Release();
											 m_bSoundStart = false;
											 m_bSoundEnd = false;
											 Sleep(10);
										 }
										 break;
									 }
			}
			default:
				break;
			}
		}
	}

	return true;
}
