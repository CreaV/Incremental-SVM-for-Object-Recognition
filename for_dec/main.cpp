#include "mainwindow.h"
#include "fullScreenWidget.h"
#include "screenShotWindow.h"
#include <QApplication>
#include <qmessagebox.h>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    QTextCodec *codec = QTextCodec::codecForName("GB2312");
 //   QTextCodec::setCodecForCStrings(codec);
    QTextCodec::setCodecForLocale(codec);
//    QTextCodec::setCodecForTr(codec);
//    screenShotWindow screenMain;
//    screenMain.show();

    return a.exec();
}
