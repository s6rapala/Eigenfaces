/*!
 *  \brief     Computing Eigenfaces from a collection of images.
 *  \details   Please refer \ref intro_sec for further details.
 *  \author    Raghunandan Palakodety
 *  \author    Himanshu Thakur
 *  \version   1.0
 *  \date      05-29-2015
 *  \pre       First configure the system by following section \ref config_page.
 *  \bug       Not all memory is freed after exiting the project. Stack smashing possible.
 *  \bug	   CMakesList not provided. Can cause manually to configure shared library qcustomplot.so
 *  \warning   Improper use can crash your application.
 */

/*! \mainpage Description of the project.
 *  \tableofcontents
 * \section intro_sec Introduction
 *
 * Computing eigenfaces from collection of face images
 * each of size 19 by 19 pixels. Each of these images can be thought of
 * as a vector of size 361. We can now talk about the vector space in
 * which each of these vectors reside. By treating the images as samples
 * of data, we can find k eigen vectors under a condition. Obtain those
 * eigen vectors which make up the basis of the vector space.
 * In our case, k = 20.
 *
 * \section results Results from the project
 *
 * The plots of eigenvalues in descending order, eigenvector visualizations and the best
 * number of predictors out of \f$ \mathbb{R}^{361} \f$  are described in the section
 *  \ref results_page.
 */

 /*! \page config_page Configuration
  *  \tableofcontents
  *  \section req_sec Required
 * -# Install OpenCV and QCustomPlot.
 * -# \see http://www.qcustomplot.com/index.php/tutorials/settingup  to use the shared library.
 * -# Configure the IDE to include appropriate directories.
 * -# Configure the linker to link the following shared (.so) libraries
 *   - qcustomplot \n
 *   - opencv_core \n
 *   - opencv_calib3d \n
 *   - opencv_contrib \n
 *   - opencv_highgui \n
 *   - opencv_imgproc \n
 *   - opencv_ml
 */
#include <iostream>
#include <vector>
#include <istream>
#include <fstream>
#include <random>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <qapplication.h>
#include <qmainwindow.h>
#include "qcustomplot.h"

/*!
  \def NO_OF_IMAGES
  Number of images in the collection.
*/

#define NO_OF_IMAGES 2429

/*!
   \namespace std
   \namespace cv
   \namespace QCP
  Number of images in the collection.
*/
using namespace std;
using namespace cv;
using namespace QCP;

/*!
 * \static int colSize
  Static integer.
*/

static int colSize = 0;

/*! \fn const vector<Mat> read_faces()
 *  \brief This function reads images from the collection.
 *
 *   This function reads images from the collection and stores in a vector of Mat.
 *
 *  \return A vector of Mat.
 */

vector<Mat> read_faces() {
	vector<Mat> training_images;
	string images_path = "images/train/face";
	string suffix = ".pgm";
	Mat img(19, 19, CV_8UC1);
	for (int i = 0; i < NO_OF_IMAGES; i++) {
		img = imread(cv::format("%s%05d.pgm", images_path.c_str(), i), 0);
		training_images.push_back(img);

	}
	return training_images;
}

/*! \fn extract_train_test_set(vector<Mat> faces, vector<Mat>& test_set)
*  \param faces A vector of Mat.
*  \param test_set A reference to a vector of Mat.
*
*  \brief  Randomly select 90% of images.
*
*
*  Randomly select 90% of these images and collect
*  them into a set training_set and the rest 10% in test_set.
*
*  \return A vector of Mat.
*/

vector<Mat> extract_train_test_set(
		vector<Mat> faces/**< [in] vector of faces or matrices*/,
		vector<Mat> &test_set /**< [out] 10% of images*/) {

	int index;
	int percentage_train = (0.9f * NO_OF_IMAGES);
	vector<Mat> training_set;

	for (int i = 0; i < percentage_train; i++) {
		index = i; //rng.uniform(0, percentage_train);
		Mat img = faces[i];
		//assert(img.empty() == false);
		training_set.push_back(img);
	}

	for (int i = percentage_train; i < NO_OF_IMAGES; i++) {
		index = i; //rng.uniform(percentage_train, NO_OF_IMAGES);
		Mat img = faces[index];
		test_set.push_back(img);
	}

	return training_set;
}
static int matIndex;

/*! \fn visualizeEigenVectors(Mat viz, const Mat& V,
		ofstream& file_eigen_vector_first_column, int index,
		vector<Mat>& eigenVectorViz)
*  \param viz Eigenvector Mat of size 19x19 to be visualized.
*  \param V A reference to a vector of Mat.
*  \param file_eigen_vector_first_column ofstream object for debugging purposes.
*  \param index Row index of the eigenvectors matrix V. V is having dimensions 361x361.
*  \param eigenVectorViz A vector of Mat, in which all the required eigenvectors are stored.
*  \brief  A function for visualizing the k best eigen vectors.
*  \return void.
*/

void minDistComputation(map<int, vector<double>>::iterator iter, vector<Point>& kNeighbours, int k, Point& minLoc, Point& maxLoc){
	int index = (*iter).first;
	vector<double> values = (*iter).second;
	double minValue;
	double maxValue;
	minMaxLoc(values, &minValue, &maxValue, &minLoc, &maxLoc);

	//sort(values.begin(), values.end(), LessThan<float>());
	for(int i = 0; i < k; i++)
		kNeighbours.push_back(minLoc);
//	return values[0];
}
void calcNearestNeighbours(Mat& XTest, Mat& XTrain,
		vector<Mat>& nearestNeighbour, vector<int> testIndices,
		int training_examples_count, int kNeighbours, string space) {

	map<int, vector<double>> allDistances;

	pair<int, vector<double> > highest;
	int indexer = 0;

	for (int i = 0; i < testIndices.size(); i++) {
		for (int j = 0; j < training_examples_count; j++) {
			Scalar distance = norm(XTrain.col(j), XTest.col(testIndices[i]));
			allDistances[testIndices[i]].push_back(distance.val[0]);
		}
	}


	//vector<double> minVals;
	vector<Point> kNNeighbours;
	Point minLoc;
	Point maxLoc;
	auto iter = allDistances.begin();

	while(iter != allDistances.end()) //for 10 test vectors
	{
		//cout<<(*iter).first<<"=>"<<(*iter).second.size();
		minDistComputation(iter, kNNeighbours, kNeighbours, minLoc, maxLoc);
//		minVals.push_back(value);
		cout<<endl;
		cout<<"Nearest training vector index"
				" to test vector for "+ space + " is "<<minLoc.x;
		iter++;
		cout<<endl;
	}

/*
	auto iterVals = kNNeighbours.begin();
	while(iterVals != kNNeighbours.end()){
		cout<<"Min values are ";
		cout<<(*iterVals);
		cout<<endl;
		iterVals++;
	}
*/

}


void visualizeEigenVectors(Mat viz, const Mat& V,
		ofstream& file_eigen_vector_first_column, int index,
		vector<Mat>& eigenVectorViz) {

	viz = V.row(index);
	/**
	 *  Note that clone() ensures viz will be continuous, so
	 *  we can treat it like an array, otherwise we can't reshape it to a rectangle.
	 */
	if (!viz.isContinuous()) {
		viz = viz.clone();
	}

	Mat rectangularMat = viz.reshape(1, 19);
	Mat dst;
	normalize(rectangularMat, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	resize(dst, dst, Size(200, 200));

	/*
	 viz = V.col(index);
	 file_eigen_vector_first_column << viz;
	 Mat M = Mat(19, 19, CV_32FC1);
	 if (!viz.isContinuous()) {
	 viz = viz.clone();
	 }
	 int ind;
	 for (int i = 0; i < M.rows; i++) {
	 for (int j = 0; j < M.cols; j++) {
	 M.at<float>(i, j) = viz.at<float>(ind, 0);
	 ind++;
	 }
	 }

	 eigenVectorViz.push_back(M);

	 resize(M, M, Size(200, 200));
	 */
	eigenVectorViz.push_back(dst);

	//imshow("Eigen vector "+to_string(index), dst);
	imwrite("images/train/eigenVecVis/eigenVecViz" + to_string(index) + ".png",
			dst);
}
/*! \fn plotDistances(QCustomPlot &customPlot1, const QVector<double>& column1,
		QVector<double>& column2, QMainWindow &window1, int plotNumber)
*  \param customPlot1 An object central class of the library. This is the QWidget which displays the plot and interacts with the user.
*  \param column1 A reference to a vector of doubles. It holds indices of the training images.
*  \param column2 A reference to a vector of doubles. It holds Euclidean distances from a test image to all training images.
*  \param window1 The QMainWindow class provides a main application window.
*  \param plotNumber Used for indexing the plots.
*  \brief  A function for plotting Euclidean distances between randomly chosen test vector and all the training data.
*  \return void.
*/

void plotDistances(QCustomPlot &customPlot1, const QVector<double>& column1,
		QVector<double>& column2, QMainWindow &window1, int plotNumber) {
	//create graph and assign data to it
	customPlot1.addGraph();
	string plotNumberIndexing = to_string(plotNumber);
	QString plotNumberIndex = plotNumberIndexing.c_str();
	QString indexing = "Distance from test data " + plotNumberIndex + " to 2186 number of training data";
	window1.setWindowTitle(indexing);
	customPlot1.graph(0)->setData(column1, column2); //setData(column1, column2); //setData(column1,column2);
	//give the axes some labels
	// set title of plot:
	customPlot1.plotLayout()->insertRow(0);
	QCPPlotTitle plotTitle(&customPlot1, indexing);
	customPlot1.plotLayout()->addElement(0, 0, &plotTitle);
	customPlot1.xAxis->setLabel("x");
	customPlot1.yAxis->setLabel("y");
	//set the axes ranges so we see all data
	customPlot1.xAxis->setRange(0, column1.size());
	auto maxYvalue = max_element(column2.begin(), column2.end());
	customPlot1.yAxis->setRange(0, *maxYvalue);
	customPlot1.setInteraction(iRangeDrag, true);
	customPlot1.setInteraction(iRangeZoom, true);
	customPlot1.setNoAntialiasingOnDrag(true);
	//customPlot1.rescaleAxes();
	window1.setGeometry(100, 100, 500, 400);
	window1.show();
	QString outputDir = "images/EuclideandistancesPlots/";
	string prefix = to_string(plotNumber)+to_string(int(*maxYvalue))+".png";
	QString fileName = prefix.c_str();
	QFile file(outputDir+"/"+fileName);
	customPlot1.saveJpg(outputDir+"/"+fileName, 0, 0, 1.0, -1);
}


int main(int argc, char **argv) {

	QApplication a(argc, argv);
	QMainWindow window1[10];

	/** \brief setup an array of customPlots, each being a central widget of window */
	QCustomPlot customPlot1[10];
	for (int i = 0; i < 10; i++)
		window1[i].setCentralWidget(&customPlot1[i]);

	QCustomPlot customPlot2[10];
	QMainWindow window2[10];
	for (int i = 0; i < 10; i++)
		window2[i].setCentralWidget(&customPlot2[i]);

	vector<Mat> eigenVectorViz;

	/** \brief Reading faces into a vector of matrices. */
	vector<Mat> faces = read_faces();

	random_shuffle(faces.begin(), faces.end()); /** \brief Shuffle the faces vector for creating a training set*/
	cout << faces.size() << endl; /** \brief Size of the vector of faces is 2429*/

	vector<Mat> training_set; /** \brief 90% images i.e 2186 are test images. */
	vector<Mat> test_set; /** \brief 10% images i.e 243 are test images. */

	training_set = extract_train_test_set(faces, test_set);

	cout << " Training set size " << training_set.size() << endl;
	cout << " Test set size " << test_set.size() << endl;

	int dim = training_set[0].rows * training_set[0].cols; /** \brief 361 dimension vector. */

	Mat X_train1(2186, 361, CV_8UC1);
	Mat img(19, 19, CV_8UC1);

	for (int index = 0; index < training_set.size(); index++) {
		img = training_set.at(index);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				X_train1.at<uchar>(index, colSize) = img.at<uchar>(i, j);

				if (colSize < dim - 1)

					colSize++;
			}
		}
		colSize = 0;
	}

	cout << "Height " << X_train1.rows << " "
			"Width " << X_train1.cols << endl;

	Mat XTrain_trans = X_train1.t();

	cout << "Rows " << XTrain_trans.rows << " Cols " << XTrain_trans.cols
			<< endl;

	Mat sum_train = Mat::zeros(dim, 1, CV_32FC1);

	Mat example_col(dim, 1, CV_32FC1);

	Mat mean_train;

	for (int index = 0; index < training_set.size(); index++) {
		XTrain_trans.col(index).convertTo(example_col, CV_32FC1);
		sum_train += example_col;
	}

	divide(sum_train, training_set.size(), mean_train);

	Mat samples = XTrain_trans.clone();

	XTrain_trans.convertTo(XTrain_trans, CV_32FC1);

	/*!
	 * \brief Centering the training matrix.
	 */
	for (int index = 0; index < training_set.size(); index++) {
		XTrain_trans.col(index) = XTrain_trans.col(index) - mean_train;
	}

	string file_path_center = "images/train/X_center.dat";

	ofstream file_handle_center(file_path_center.c_str(), ios::trunc);

	file_handle_center << XTrain_trans;
	cout << "Dimensions " << " Rows " << XTrain_trans.rows << " "
			"Columns " << XTrain_trans.cols << endl;


	Mat distance_matrix = Mat::zeros(dim, dim, CV_32FC1);
	Mat covariance_matrix = Mat::zeros(dim, dim, CV_32FC1);

	distance_matrix = XTrain_trans * XTrain_trans.t();

	covariance_matrix = distance_matrix / training_set.size();

	string file_path_covariance = "images/train/X_covariance.dat";

	ofstream file_handle_covariance(file_path_covariance.c_str(), ios::trunc);

	file_handle_covariance << covariance_matrix;

	cout << "**CO-VARIANCE COMPUTED**" << endl;
	cout << "Co-variance dimensions rows " << covariance_matrix.rows << " "
			<< "columns " << covariance_matrix.cols << endl;

	Mat E, V;
	/*! \brief Compute the co-variance matrix \f$C\f$.
	 *
	 *  as well as its eigenvectors \f$v_i\f$
	 * and eigenvalues \f$\lambda_i\f$.
	 *
	 * */
	eigen(covariance_matrix, E, V);

	Mat meanImg(19, 19, CV_32FC1);

	for (int i = 0; i < meanImg.rows; i++) {
		for (int j = 0; j < meanImg.cols; j++) {
			meanImg.at<float>(i, j) = floorf(mean_train.at<float>(matIndex));
			matIndex++;
		}
	}

	meanImg = meanImg / 255;

	string file_path_eigen_values = "images/train/X_eigen_values.dat";

	ofstream file_handle_eigen_values(file_path_eigen_values.c_str(),
			ios::trunc);

	file_handle_eigen_values << E;

	QVector<double> eigenValues;
	QVector<double> basis;

	for(int i = 0; i < E.rows; i++){
		double value = E.at<float>(i);
		eigenValues.push_back(value);
		basis.push_back(i);
	}

	QCustomPlot spectrumPlot;
	QMainWindow window;

	QString indexing1 = "Spectrum of co-variance";
	string indexing2 = " Eigenvalues represent the variance of the data along the eigenvector directions";
	window.setCentralWidget(&spectrumPlot);
	window.setWindowTitle(indexing1);
	spectrumPlot.addGraph();
	spectrumPlot.graph(0)->setData(basis, eigenValues);
	//give the axes some labels
	spectrumPlot.plotLayout()->insertRow(0);
	QCPPlotTitle plotTitle(&spectrumPlot, indexing1);
	spectrumPlot.plotLayout()->addElement(0, 0, &plotTitle);
	spectrumPlot.xAxis->setLabel("x");
	spectrumPlot.yAxis->setLabel("y");
	spectrumPlot.setInteraction(iRangeDrag, true);
	spectrumPlot.setInteraction(iRangeZoom, true);
	spectrumPlot.setNoAntialiasingOnDrag(true);
	//set the axes ranges so we see all data
	spectrumPlot.xAxis->setRange(0, 2186);
	spectrumPlot.yAxis->setRange(1, 900000);
	//customPlot1.rescaleAxes();
	window.setGeometry(100, 100, 500, 400);
	window.show();
	QString fileName = "Spectrum-of-co-variance.png";
	QString outPutDir = "images/";
	spectrumPlot.saveJpg(outPutDir+fileName, 0, 0, 1.0, -1);


	Mat coVarMat;
	Mat meanMat;	//(361,1,CV_32FC1);

	string file_path_covar = "images/coVarMat.dat";
	ofstream file_handle_coVar(file_path_covar.c_str(), ios::trunc);

	calcCovarMatrix(XTrain_trans, coVarMat, meanMat, CV_COVAR_ROWS);
	coVarMat = coVarMat / training_set.size();
	file_handle_coVar << coVarMat;

	file_handle_coVar << endl;
	file_handle_coVar << endl;
	file_handle_coVar << endl;
	file_handle_coVar
			<< "*******************MEAN matrix**************************\n";
	file_handle_coVar << "Columns ";
	file_handle_coVar << meanMat.cols;
	file_handle_coVar << '\t';
	file_handle_coVar << "Rows ";
	file_handle_coVar << meanMat.rows;
	file_handle_coVar << endl;
	file_handle_coVar << endl;
	file_handle_coVar << endl;
	file_handle_coVar << meanMat;
	file_handle_coVar << endl;

	resize(meanImg, meanImg, Size(200, 200));

	cout << "Eigen vector dimensions rows " << V.rows << " columns " << V.cols
			<< endl;

	float sumOfEigenValues;

	for (int i = 0; i < E.rows; i++) {
		for (int j = 0; j < E.cols; j++) {
			sumOfEigenValues += E.at<float>(i);
		}
	}
	cout << "Sum of all " << dim << " eigen values is " << sumOfEigenValues
			<< endl;

	vector<float> sumOfKEigenValues(dim, 0.f); //361 elements initialized to zero
	for (int j = 0; j < dim; j++) {
		for (int k = 0; k <= j; k++) {
			sumOfKEigenValues[j] += E.at<float>(k);
		}
	}

	cout << "Size of sum of k eigen values array " << sumOfKEigenValues.size()
			<< endl;

	vector<float> smallestKEigenValues;
	auto iterK = sumOfKEigenValues.begin();
	 /*! \brief  Determining the smallest \f$ k \f$ eigenvalues.
	  * Determining the smallest \f$ k \f$ eigenvalues, \f$ \lambda_k \f$ such that
	  * \f$ \frac{\sum_{i=1}^{k}\lambda_{i}}{\sum_{j=1}^{d}\lambda_{j}} \ge 0.9 \f$.\n
	  * In our case we got  \f$ k = 20 \f$.
	  *
      */
	while (iterK != sumOfKEigenValues.end()
			&& (float) (*(iterK) / sumOfEigenValues) < 0.9 ) {
		cout << (float) (*iterK) / sumOfEigenValues;
		smallestKEigenValues.push_back((float) (*iterK) / sumOfEigenValues);
		iterK++;
		cout << endl;
	}

	Mat viz(19, 19, CV_32FC1);
	string file_eigen_vector_first_column_path =
			"images/train/X_eigen_vector_first_col.dat";
	ofstream file_eigen_vector_first_column(
			file_eigen_vector_first_column_path.c_str(), ios::trunc);

	int ind = 0;

	while (ind < smallestKEigenValues.size()) {
		visualizeEigenVectors(viz, V, file_eigen_vector_first_column, ind,
				eigenVectorViz);
		ind++;
	}

	string file_test_file_path = "images/test/X_test.dat";
	ofstream file_handle_test_file_stream(file_test_file_path, ios::trunc);
	colSize = 0;

	Mat X_test1(243, 361, CV_8UC1); // 243 examples

	for (int index = 0; index < test_set.size(); index++) {
		img = test_set.at(index);
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				X_test1.at<uchar>(index, colSize) = img.at<uchar>(i, j);
				file_handle_test_file_stream << to_string(img.at<uchar>(i, j));
				if (colSize < dim - 1)
					file_handle_test_file_stream << ",";
				colSize++;
			}
		}
		file_handle_test_file_stream << endl;
		colSize = 0;
	}

	Mat XTest_trans = X_test1.t(); // 361 x 243

	XTest_trans.convertTo(XTest_trans, CV_32FC1);

	for (int index = 0; index < test_set.size(); index++) {
		XTest_trans.col(index) = XTest_trans.col(index) - mean_train;
	}

	RNG rng;
	int randIndex;
	Mat X_rand_test_samples(10, 361, CV_8UC1); // 10 examples
	vector<int> testSamplesIndices;

	for (int i = 0; i < 10; i++) {
		randIndex = rng.uniform(0, 243);
		testSamplesIndices.push_back(randIndex);
		X_rand_test_samples.row(i) = X_test1.row(randIndex);
	}
	string file_X_rand_test_samples = "images/test/X_rand_test_samples_trans";
	ofstream file_X_rand_test_samples_handle(file_X_rand_test_samples.c_str(),
			ios::trunc);

	Mat X_rand_test_samples_trans = X_rand_test_samples;
	X_rand_test_samples_trans = X_rand_test_samples_trans.t(); //check this!!.

	X_rand_test_samples_trans.convertTo(X_rand_test_samples_trans, CV_32FC1);

	string file_test_train_file_path = "images/test/X_test_train_distance.dat";
	ofstream file_handle_test_train_file_stream(file_test_train_file_path,
			ios::trunc);
	vector<float> EucDistToTrain_Data;

	QVector<double> column1;
	QVector<double> column2;

	for (int i = 0; i < 10; i++) {
		file_X_rand_test_samples_handle << X_rand_test_samples_trans.col(i);
		file_X_rand_test_samples_handle << endl;
//		column1.push_back(i);

		for (int j = 0; j < 2186; j++) {
			Mat col1 = XTrain_trans.col(j); // one training sample from the training sample matrix.
			Mat col2 = XTest_trans.col(testSamplesIndices[i]); // test sample 0 to 9 index. Each test sample[column vector] yields a single distance with a single column vector of training sample
			Scalar value = norm(col2, col1, NORM_L2); // Euclidean distance, a single value, between two column vectors.
			EucDistToTrain_Data.push_back(value.val[0]); // magnitude stored for each of 2186 samples with each test vector
			column1.push_back(j);
			column2.push_back(value.val[0]);
		}
		//create graph and assign data to it
		sort(column2.begin(),
				column2.end(), greater<float>());
		plotDistances(customPlot1[i], column1, column2, window1[i], i);
		column1.clear();
		column2.clear();
	}

	sort(EucDistToTrain_Data.begin(), EucDistToTrain_Data.end(),
			greater<float>());

	auto iter = EucDistToTrain_Data.begin();
	while (iter != EucDistToTrain_Data.end()) {
		file_handle_test_train_file_stream << *(iter);
		file_handle_test_train_file_stream << endl;
		iter++;
	}


	/*
	 *  Projecting all training samples into the PCA subspace.
	 */
	Mat projectionMatrix_trainingData(20, 2186, CV_32FC1);
	Mat smallestKEigenVectors = V.rowRange(0, 20); //20x361

	projectionMatrix_trainingData = smallestKEigenVectors * XTrain_trans; //(20x361)x(361x2186)

	string file_projection_matrix = "images/projection_matrix_trainingData.dat";
	ofstream projection_matrix_file_handle(file_projection_matrix, ios::trunc);
	cout << "Projection matrix for training data"
			"rows " << projectionMatrix_trainingData.rows << " and columns "
			<< projectionMatrix_trainingData.cols << endl;
	projection_matrix_file_handle << projectionMatrix_trainingData;
	projection_matrix_file_handle << endl;

	string file_projection_matrix_testData =
			"images/projection_matrix_testData.dat";
	ofstream projection_matrix_file_handle_testData(
			file_projection_matrix_testData, ios::trunc);

	projection_matrix_file_handle_testData
			<< "****Projection of test data onto subspace spanned by k eigen vectors****";
	projection_matrix_file_handle_testData << endl;
	projection_matrix_file_handle_testData << endl;

	Mat projectionMatrix_testData(20, 243, CV_32FC1); //(20x361)x(361x243)
	projectionMatrix_testData = smallestKEigenVectors * XTest_trans;
	projection_matrix_file_handle_testData << projectionMatrix_testData;
	projection_matrix_file_handle_testData << endl;
	projection_matrix_file_handle_testData << endl;
	projection_matrix_file_handle_testData << endl;
	projection_matrix_file_handle_testData << endl;
	projection_matrix_file_handle_testData << endl;

	cout << "Projection matrix for test data "
			"rows " << projectionMatrix_testData.rows << " and columns "
			<< projectionMatrix_testData.cols << endl;

	QVector<double> column_test_set_indices;
	QVector<double> column_distances;
	vector<float> EucDistToTrain_Data_after_pca;

	for (int i = 0; i < testSamplesIndices.size(); i++) {
		//column_test_set_indices.push_back(i);
		for (int j = 0; j < training_set.size(); j++) {
			Mat col1 = projectionMatrix_trainingData.col(j); // one training sample from the training sample matrix.
			Mat col2 = projectionMatrix_testData.col(testSamplesIndices[i]); // test sample 0 to 9 index. Each test sample[column vector] yields a single distance with a single column vector of training sample
			Scalar value = norm(col2, col1, NORM_L2); // Euclidean distance, a single value, between two column vectors.
			EucDistToTrain_Data_after_pca.push_back(value.val[0]); // magnitude stored for each of 2186 samples with each test vector
			column_test_set_indices.push_back(j);
			column_distances.push_back(value.val[0]);
		}
		//create graph and assign data to it
		sort(column_distances.begin(),
				column_distances.end(), greater<float>());

		plotDistances(customPlot2[i], column_test_set_indices, column_distances,
				window2[i], i);
		column_distances.clear();
		column_test_set_indices.clear();
	}
	sort(EucDistToTrain_Data_after_pca.begin(),
			EucDistToTrain_Data_after_pca.end(), greater<float>());

	string file_test_train_file_after_pca_path =
			"images/test/X_test_train_distance_after_pca.dat";
	ofstream file_handle_test_train_file_after_pca_stream(
			file_test_train_file_after_pca_path, ios::trunc);
	auto itera = EucDistToTrain_Data_after_pca.begin();

	while (itera != EucDistToTrain_Data_after_pca.end()) {
		//projection_matrix_file_handle_testData<<*(itera);
		file_handle_test_train_file_after_pca_stream << *(itera);
		itera++;
		file_handle_test_train_file_after_pca_stream << endl;
	}
	cout << "Size of euclidean distance to train data after pca "
			<< EucDistToTrain_Data_after_pca.size() << endl;
	//Nearest neighbours among the training images in the original space.
	vector<Mat> nearestNeighbours_orig_space;
	calcNearestNeighbours(XTest_trans, XTrain_trans, nearestNeighbours_orig_space, testSamplesIndices, training_set.size(), 1, "Original Space");
	calcNearestNeighbours(projectionMatrix_testData, projectionMatrix_trainingData, nearestNeighbours_orig_space, testSamplesIndices, training_set.size(), 1, "Sub-space");

	imshow("Mean Img", meanImg);

	waitKey(0);

//	file_handle_center.close();
//	file_handle_mean.close();
//	file_handle_sum.close();
//	file_handle_stream.close();
//	file_handle_eigen_vectors.close();
//	file_handle_eigen_values.close();
//	file_handle_covariance.close();
	file_eigen_vector_first_column.close();
	file_handle_test_file_stream.close();
	file_X_rand_test_samples_handle.close();
	file_handle_test_train_file_stream.close();
	projection_matrix_file_handle.close();

	X_train1.release();
	XTrain_trans.release();
	covariance_matrix.release();
	viz.release();
	E.release();
	V.release();
	return 0;
}

/*! \page results_page Results
 *  \tableofcontents
 *
 * List of results:
 *  - Plot set of eigenvalues in descending order.
 *  	- \image html Spectrum-of-co-variance.png
 *
 *  - Determine the smallest \f$ k \f$ eigenvalues, \f$ \lambda_k \f$ such that  \f$ \frac{\sum_{i=1}^{k}\lambda_{i}}{\sum_{j=1}^{d}\lambda_{j}} \ge 0.9 \f$
 *		- In our case we got  \f$ k = 20 \f$
 *
 *	- Visualize the first \f$ k \f$ eigenvectors \f$ v_i \f$ \f$\in\f$ \f$ \mathbb{R}^{361} \f$ as \f$ 19\f$ × \f$ 19 \f$ images
 *		- Consider the first visualization. \image html eigenVecVis/eigenVecViz0.png
 *
 *	- Randomly select 10 test images, compute their Euclidean distances to all training images, sort (in descending order) and plot the distances.
 *		- \image html EuclideandistancesPlots/02120.png
 *	- Consider the same 10 test images as above; in the lower dimensional space, compute their Euclidean
 *	distances to all the training images, sort the set of distances in descending order and plot them.
 *		- \image html EuclideandistancesPlots/02157.png
 *
 */

/*!\page check Nearest Neighbours
 * \brief Nearest neighbours to 10 randomly chosen test vectors.
 *
 * Check for nearest neighbours both in original and subspace.
 * Compare whether the training instances (indices) are same for both original space and subspace.
 * Interpretation should be there exists some vector \f$x\f$  whose nearest neighbours in original and subspace are same. Also, there exists
 * some vector \f$y\f$ whose nearest neighbours in original and subspace are not same.
 *  \tableofcontents
 * For k = 1 and 10 randomly chosen test vectors :
 * - Nearest training vector index to test vector for Original Space is 755
 * - Nearest training vector index to test vector for Original Space is 527
 * - Nearest training vector index to test vector for Original Space is 1577
 * - Nearest training vector index to test vector for Original Space is \b\f$319\f$
 * - Nearest training vector index to test vector for Original Space is 1831
 * - Nearest training vector index to test vector for Original Space is 1507
 * - Nearest training vector index to test vector for Original Space is 638
 * - Nearest training vector index to test vector for Original Space is 1785
 * - Nearest training vector index to test vector for Original Space is 232
 * - Nearest training vector index to test vector for Original Space is 1992
 * - Nearest training vector index to test vector for Sub-space is 755
 * - Nearest training vector index to test vector for Sub-space is 527
 * - Nearest training vector index to test vector for Sub-space is 1577
 * - Nearest training vector index to test vector for Sub-space is \b\f$2158\f$
 * - Nearest training vector index to test vector for Sub-space is 1831
 * - Nearest training vector index to test vector for Sub-space is 1507
 * - Nearest training vector index to test vector for Sub-space is 638
 * - Nearest training vector index to test vector for Sub-space is 1785
 * - Nearest training vector index to test vector for Sub-space is 232
 * - Nearest training vector index to test vector for Sub-space is 1992
 *
 *  */
