#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "SREngine.h"
//OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// OpenCV namespace for defining out image matrix
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
	QString grammarFileName;
	bool soundActive;
	
public:						//C&C
	int m_tDelay;
	int m_tPreTime;
	bool m_bSoundEnd;
	bool m_bSoundStart;
	bool m_bGotReco;
	vector<Mat> Return_Roi;  //返回的感兴趣的MAT向量
	size_t sum_count;
	SREngine m_SREngine;
	LRESULT OnRecoEvent();
	HRESULT hr;
	//bool atctiveReco();
	//bool atctiveRead();
	void INI();

public:							//TTS
	ISpVoice *pSpVoice;        // 重要COM接口

protected:
	virtual bool nativeEvent(const QByteArray &eventType, void *message, long *result);
signals:
	int SigRecEvent();



public:
    explicit MainWindow(QWidget *parent = 0);
    QImage MatToQImage(const Mat& mat);
    Mat image;
    QString filename;
	
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

	void on_pushButton_4_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
