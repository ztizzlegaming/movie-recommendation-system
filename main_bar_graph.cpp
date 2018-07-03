//Netflix Prize
//Jordan Turley and Michael Lamar
//
//This implementation is using a sphere
//For a data point, the user and movie rating vectors are attracted together
//Then, the user and movie rating are repelled away from random movie ratings
//and users, independently.
//Then, the user is repelled from a random user and the movie-rating from a random movie-rating
//Then, the movie-rating is repelled from the four other movie-ratings for the same movie.
//
//Eta is the strength of the attraction and repulsion
//Phi is used to slow down attraction and repulsion as a vector is seen over and over
//
//The current z is printed out 100 times every iteration (every %), as well as
//the average distance between two movie-ratings, two users, a random user and
//movie-rating, and a user movie-rating pair from a random data point. The
//likelihood value is also printed out.
//
//The RMSE is printed after each iteration.
//
//The data is randomly split into three separate sets. One for training, one
//for validation, and one for test. The model is trained on the training set.
//Then, after each iteration, the RMSE for the model is calculated on the
//validation set. After we have everything configured, the model will be
//evaluated on the test set.
//
//To compile: g++ main.cpp -o netflix -std=c++11 -O3
//To run on Linux: ./netflix input_file.txt data_file.bin 100480507
//To run on Windows: netflix input_file.txt data_file.bin 100480507
//The number at the end can be any number, as long as it is greater than the sample sizes.
//For example, 1000000 to just run it on the first 1000000 data points
//Running the program on the full data set takes almost 4.5 GB of RAM

#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <algorithm>

using namespace std;

//Struct to hold settings from file
//Returned from readSettings
struct Settings {
	int dimensions;
	double eta;
	int phiUser;
	int phiMR;
	int iterations;
	int zSampleSize;
	int scoreSampleSize;
	int repelNum;
};

//Struct to hold data read from binary file
//Returned from readData
struct Data {
	int** data;
	int* dataIndices;
	int maxUserId;
	int maxMovieId;
};

//Struct to hold vector arrays for generating vectors.
//Returned from generateVectors
struct Vectors {
	double** userVectors;
	int *userCounts;
	int *userCumulativeCounts;
	int totalUsers;

	double*** movieRatingVectors;
	int **movieRatingCounts;
	int **movieRatingCumulativeCounts;
	int totalMovies;
};

//Struct to hold three sets: training, validation, and test
//Returned from splitDatasets
struct Datasets {
	int* trainIndices;
	int trainSize;

	int* validationIndices;
	int validationSize;

	int* testIndices;
	int testSize;
};

//Struct to hold initial value of z and array of sample values of z
//Returned from calculateInitialZ
struct ZValues {
	double z;
	double* zValues;
};

struct Settings readSettings(char* file);
struct Data readData(char* file, int numDataPoints);
struct Vectors generateVectors(
	int** data,
	int numDataPoints,
	int maxUserId,
	int maxMovieId,
	int dimensions);
struct Datasets splitDatasets(int* dataIndices, int numDataPoints);
int* generateSet(int* dataIndices, int startIdx, int endIdx);
struct ZValues calculateInitialZ(
	int* trainIndices,
	int trainSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	mt19937 random,
	uniform_int_distribution<int> randomDataPoint,
	int sampleSize,
	int dimensions);
double getDistanceSquared(double a, double b);
double getDistanceSquared(double *a, double *b, int dimensions);
double norm2(double *arr, int length);
double calculateEta(double etaInitial, double phi, int count);
void moveVectors(
	double* user,
	double* mr,
	double movieRating,
	double etaInitial,
	double userEta,
	double mrEta,
	double** userVectorsRepel,
	double** mrVectorsRepel,
	int numRepel,
	double* randomUser,
	double* randomMR,
	int dimensions,
	double z);
double attract(double a, double b, double eta);
double repel(double a, double other, double eta, double z);
double normalize(double a, double initialEta, double delta, double aNorm2);
double normalizeUnitLength(double a, double aNorm2);
double calculateRMSE(
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	int** movieRatingCounts,
	int dimensions);
void writeBarGraphValues(
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	int** movieRatingCounts,
	int dimensions,
	int barGraphIndices[]);

const int NUM_PARTS = 3; //How many parts are there to each triple? 3 - user id, movie id, rating
const int USER_ID_IDX = 0; //The index of the user id in the array storing the triple
const int MOVIE_ID_IDX = 1; //The index of the movie id in the array storing the triple
const int MOVIE_RATING_IDX = 2; //The index of the movie rating in the array storing the triple

const int MAX_STARS = 5;

//Sample size when calculating average distances for debugging
//Ex. average distance between two random users, two random movie-ratings, ...
const int AVERAGE_SAMPLE_SIZE = 10000;

//Sizes for each set
const double TRAIN_SIZE = 0.8;
const double VALIDATION_SIZE = 0.1;
const double TEST_SIZE = 1 - TRAIN_SIZE - VALIDATION_SIZE;

//The input file has:
//dimensions
//initial eta
//phi for users
//phi for movie ratings
//number of iterations
//z sample size
//score sample size

const int BAR_GRAPH_COUNT = 1;

int main(int argc, char *argv[]) {
	//Seed general random number generator
	srand(time(0));

	//Get the input file from command line
	char *settingsFile = (char *) "C:\\input_sphere.txt";
	if (argc > 1) { //The first command-line argument is the name of the program
		settingsFile = argv[1];
	}

	//Get the data file from command line
	char *dataFile = (char *) "C:\\netflix_data_c.bin";
	if (argc > 2) {
		dataFile = argv[2];
	}

	//Get the number of data points from command line
	int numDataPoints = 1000000;
	if (argc > 3) {
		numDataPoints = strtol(argv[3], NULL, 10);
	}

	//Read in settings from the input file
	struct Settings settings = readSettings(settingsFile);

	//Pull a few settings out of the struct since it is used a lot
	int dimensions = settings.dimensions;
	int zSampleSize = settings.zSampleSize;
	int scoreSampleSize = settings.scoreSampleSize;

	//Calculate delta based on eta
	double delta = 1 / (4 * settings.eta);

	cout << "Reading in data" << endl;

	//Read in the data points from the binary file
	struct Data dataStruct = readData(dataFile, numDataPoints);
	int** data = dataStruct.data;
	int* dataIndices = dataStruct.dataIndices;
	int maxUserId = dataStruct.maxUserId;
	int maxMovieId = dataStruct.maxMovieId;

	cout << "Max user id: " << maxUserId << endl;
	cout << "Max movie id: " << maxMovieId << endl;
	cout << "Initializing vectors" << endl;

	//Generate the vectors
	struct Vectors vectors = generateVectors(data, numDataPoints, maxUserId, maxMovieId, dimensions);
	
	//Get the vector and count arrays from the struct
	double** userVectors = vectors.userVectors;
	int *userCounts = vectors.userCounts;
	int *userCumulativeCounts = vectors.userCumulativeCounts;

	double*** movieRatingVectors = vectors.movieRatingVectors;
	int **movieRatingCounts = vectors.movieRatingCounts;
	int **movieRatingCumulativeCounts = vectors.movieRatingCumulativeCounts;

	cout << "Number of users: " << vectors.totalUsers << endl;
	cout << "Number of movies: " << vectors.totalMovies << endl;

	//Split the data into three datasets: training, validation, and test
	struct Datasets datasets = splitDatasets(dataIndices, numDataPoints);
	int* trainIndices = datasets.trainIndices;
	int trainSize = datasets.trainSize;

	int* validationIndices = datasets.validationIndices;
	int validationSize = datasets.validationSize;

	int* testIndices = datasets.testIndices;
	int testSize = datasets.testSize;

	//Pick a few of the validation points to look at the bar graph for
	//For now randomly select them. Later pick user and movie combinations
	//that appear more often.
	random_shuffle(&validationIndices[0], &validationIndices[validationSize - 1]);
	int barGraphIndices[BAR_GRAPH_COUNT];
	for (int i1 = 0; i1 < BAR_GRAPH_COUNT; i1++) {
		int idx = validationIndices[i1];
		barGraphIndices[i1] = idx;

		//Calculate the probability of giving each rating based on the empirical probabilities
		int* dataPt = data[idx];
		int userId = dataPt[USER_ID_IDX];
		int movieId = dataPt[MOVIE_ID_IDX];
		int movieRating = dataPt[MOVIE_RATING_IDX];

		//Initialize output file for empirical probabilities
		ofstream file;
		string name = "empirical_" + to_string(i1 + 1) + "_movieid=" + to_string(movieId) + ".csv";
		file.open(name);

		double sum = 0;
		for (int star = 0; star < MAX_STARS; star++) {
			int c = movieRatingCounts[movieId - 1][star];
			sum += c;
		}

		for (int star = 0; star < MAX_STARS; star++) {
			double p = movieRatingCounts[movieId - 1][star] / sum;

			file << p;
			if (star != MAX_STARS - 1) {
				file << ",";
			}
		}
		file.close();
	}

	//Init random data point generator from training set
	mt19937 random(time(0));
	uniform_int_distribution<int> randomDataPoint(0, trainSize - 1);

	//Print out the size of each set
	cout << "Set sizes:" << endl;
	cout << "Training set: " << trainSize << endl;
	cout << "Validation set: " << validationSize << endl;
	cout << "Test set: " << testSize << endl;

	//Clear out the original array of data indices after it's split up
	delete[] dataIndices;

	cout << "Calculating initial value of Z" << endl;

	//Calculate the initial value of z
	struct ZValues zStruct = calculateInitialZ(
		trainIndices,
		trainSize,
		data,
		userVectors,
		movieRatingVectors,
		random,
		randomDataPoint,
		zSampleSize,
		dimensions);
	double z = zStruct.z;
	double* zValues = zStruct.zValues;
	int oldestIdx = 0;

	//Save the initial z in case we need to use it later, and print it out
	double initialZ = z;
	cout << "Initial z: " << z << endl;

	//Write the initial values for the bar graphs before moving any vectors
	writeBarGraphValues(
		data,
		userVectors,
		movieRatingVectors,
		movieRatingCounts,
		dimensions,
		barGraphIndices);

	//Go through number of iterations to move vectors
	for (int iteration = 0; iteration < settings.iterations; iteration++) {
		random_shuffle(&trainIndices[0], &trainIndices[trainSize - 1]);

		int reportNum = trainSize / 100;

		cout << "Starting iteration " << iteration + 1 << endl;

		//Go through each data point in the training set
		for (int dataIdx = 0; dataIdx < trainSize; dataIdx++) {
			int idx = trainIndices[dataIdx];
			int* dataPt = data[idx];

			//Get the user id, movie id, and movie rating from the data point
			int userId = dataPt[USER_ID_IDX];
			int movieId = dataPt[MOVIE_ID_IDX];
			int movieRating = dataPt[MOVIE_RATING_IDX];

			//Update cumulative counts
			userCumulativeCounts[userId - 1]++;
			movieRatingCumulativeCounts[movieId - 1][movieRating - 1]++;

			//Get the vectors and calculate eta for the user and movie rating vectors
			double* userVec = userVectors[userId - 1];
			double userEta = calculateEta(settings.eta, settings.phiUser, userCumulativeCounts[userId - 1]);

			double* movieRatingVec = movieRatingVectors[movieId - 1][movieRating - 1];
			double mrEta = calculateEta(settings.eta, settings.phiMR, movieRatingCumulativeCounts[movieId - 1][movieRating - 1]);

			double** userVectorsRepel = new double*[settings.repelNum];
			double** mrVectorsRepel = new double*[settings.repelNum];

			//Get small sample of vectors to repel from
			for (int i1 = 0; i1 < settings.repelNum; i1++) {
				//Get a user vector
				int idxTemp = randomDataPoint(random);
				int* dataPtTemp = data[trainIndices[idxTemp]];
				int userIdTemp = dataPtTemp[USER_ID_IDX];
				double* userVecTemp = userVectors[userIdTemp - 1];
				userVectorsRepel[i1] = userVecTemp;

				idxTemp = randomDataPoint(random);
				dataPtTemp = data[trainIndices[idxTemp]];
				int movieIdTemp = dataPtTemp[MOVIE_ID_IDX];
				int movieRatingTemp = dataPtTemp[MOVIE_RATING_IDX];
				double* mrVecTemp = movieRatingVectors[movieIdTemp - 1][movieRatingTemp - 1];
				mrVectorsRepel[i1] = mrVecTemp;
			}

			//Get more random vectors for user-user and mr-mr repulsion
			int randomUserDataIdx = randomDataPoint(random);
			idx = trainIndices[randomUserDataIdx];
			dataPt = data[idx];
			int randomUserId = dataPt[USER_ID_IDX];
			double* randomUserVec = userVectors[randomUserId - 1];

			int randomMRDataIdx = randomDataPoint(random);
			idx = trainIndices[randomMRDataIdx];
			dataPt = data[idx];
			int randomMovieId = dataPt[MOVIE_ID_IDX];
			int randomMovieRating = dataPt[MOVIE_RATING_IDX];
			double* randomMRVec = movieRatingVectors[randomMovieId - 1][randomMovieRating - 1];

			//Move the vectors toward each other, and away from the randomly chosen vectors
			moveVectors(
				userVec,
				movieRatingVec,
				movieRating,
				settings.eta,
				userEta,
				mrEta,
				userVectorsRepel,
				mrVectorsRepel,
				settings.repelNum,
				randomUserVec,
				randomMRVec,
				dimensions,
				z);

			//Deallocate the vector repel arrays
			delete[] userVectorsRepel;
			delete[] mrVectorsRepel;

			//Get random new vectors to update z with
			randomUserDataIdx = randomDataPoint(random);
			idx = trainIndices[randomUserDataIdx];
			dataPt = data[idx];
			randomUserId = dataPt[USER_ID_IDX];
			randomUserVec = userVectors[randomUserId - 1];

			randomMRDataIdx = randomDataPoint(random);
			idx = trainIndices[randomMRDataIdx];
			dataPt = data[idx];
			randomMovieId = dataPt[MOVIE_ID_IDX];
			randomMovieRating = dataPt[MOVIE_RATING_IDX];
			randomMRVec = movieRatingVectors[randomMovieId - 1][randomMovieRating - 1];

			//Actually update the value of z
			double oldZVal = zValues[oldestIdx];
			double newZVal = exp(-getDistanceSquared(randomUserVec, randomMRVec, dimensions));

			zValues[oldestIdx] = newZVal;

			z += (newZVal - oldZVal) / zSampleSize;

			oldestIdx++;
			oldestIdx %= zSampleSize;

			//Print out logging information 100 times an iterations
			if (dataIdx % reportNum == 0) { //Print out Z and the percentage completed of the iteration
				double perc = (double) dataIdx / trainSize * 100;
				cout << perc << "%, Z: " << z << endl;

				//Calculate averages for data collection:
				//Two random movie ratings, two random users, a random user and random movie rating, and a user and movie rating from a random data point
				//First, two random movie ratings
				double mrmrAvg = 0;
				for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
					//Pick two random movie ratings
					int index1 = randomDataPoint(random);
					int index2 = randomDataPoint(random);

					index1 = trainIndices[index1];
					index2 = trainIndices[index2];

					int* dataPt1 = data[index1];
					int movieId1 = dataPt1[MOVIE_ID_IDX];
					int movieRating1 = dataPt1[MOVIE_RATING_IDX];
					double *movieRatingVec1 = movieRatingVectors[movieId1 - 1][movieRating1 - 1];

					int* dataPt2 = data[index2];
					int movieId2 = dataPt2[MOVIE_ID_IDX];
					int movieRating2 = dataPt2[MOVIE_RATING_IDX];
					double *movieRatingVec2 = movieRatingVectors[movieId2 - 1][movieRating2 - 1];

					//Calculate distance between these two
					double distance = sqrt(getDistanceSquared(movieRatingVec1, movieRatingVec2, dimensions));
					mrmrAvg += distance;
				}
				mrmrAvg /= AVERAGE_SAMPLE_SIZE;
				cout << "MRMR: " << mrmrAvg << endl;

				//Then, two random users
				double useruserAvg = 0;
				for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
					int index1 = randomDataPoint(random);
					int index2 = randomDataPoint(random);

					index1 = trainIndices[index1];
					index2 = trainIndices[index2];

					int *dataPt1 = data[index1];
					int userId1 = dataPt1[USER_ID_IDX];
					double *userVec1 = userVectors[userId1 - 1];

					int *dataPt2 = data[index2];
					int userId2 = dataPt2[USER_ID_IDX];
					double *userVec2 = userVectors[userId2 - 1];

					double distance = sqrt(getDistanceSquared(userVec1, userVec2, dimensions));
					useruserAvg += distance;
				}
				useruserAvg /= AVERAGE_SAMPLE_SIZE;
				cout << "User_User: " << useruserAvg << endl;

				//Then, a random user and random movie rating
				double randUserMrAvg = 0;
				for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
					int userIndex = randomDataPoint(random);
					int mrIndex = randomDataPoint(random);

					userIndex = trainIndices[userIndex];
					mrIndex = trainIndices[mrIndex];

					int *userDataPt = data[userIndex];
					int userId = userDataPt[USER_ID_IDX];
					double *userVec = userVectors[userId - 1];

					int *mrDataPt = data[mrIndex];
					int movieId = mrDataPt[MOVIE_ID_IDX];
					int movieRating = mrDataPt[MOVIE_RATING_IDX];
					double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

					double distance = sqrt(getDistanceSquared(userVec, mrVec, dimensions));
					randUserMrAvg += distance;
				}
				randUserMrAvg /= AVERAGE_SAMPLE_SIZE;
				cout << "Rand_User_MR: " << randUserMrAvg << endl;

				//Finally, distance between user and movie rating for a random data point
				double usermrAvg = 0;
				for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
					int dataIdx = randomDataPoint(random);
					dataIdx = trainIndices[dataIdx];
					
					int *dataPt = data[dataIdx];
					
					int userId = dataPt[USER_ID_IDX];
					double *userVec = userVectors[userId - 1];

					int movieId = dataPt[MOVIE_ID_IDX];
					int movieRating = dataPt[MOVIE_RATING_IDX];
					double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

					double distance = sqrt(getDistanceSquared(userVec, mrVec, dimensions));
					usermrAvg += distance;
				}
				usermrAvg /= AVERAGE_SAMPLE_SIZE;
				cout << "User_MR: " << usermrAvg << endl;

				//Calculate the likelihood
				double likelihoodAvg = 0;
				for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
					//Get a random data index
					int dataIdx = randomDataPoint(random);
					dataIdx = trainIndices[dataIdx];

					//Get the data point: the user id, movie id, and rating
					int *dataPt = data[dataIdx];

					//Get the user vector
					int userId = dataPt[USER_ID_IDX];
					double *userVec = userVectors[userId - 1];

					//Get the movie rating vector
					int movieId = dataPt[MOVIE_ID_IDX];
					int movieRating = dataPt[MOVIE_RATING_IDX];
					double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

					//Calculate pbar for the user and movie rating
					double userPBar = (double) userCounts[userId - 1] / numDataPoints;
					double mrPBar = (double) movieRatingCounts[movieId - 1][movieRating - 1] / numDataPoints;

					double likelihood = userPBar * mrPBar * exp(-getDistanceSquared(userVec, mrVec, dimensions)) / z;
					likelihoodAvg += likelihood;
				}
				likelihoodAvg /= AVERAGE_SAMPLE_SIZE;
				cout << "Likelihood: " << likelihoodAvg << endl;
			}
		}

		//Find the RMSE after each iteration to see if it is improving
		double rmse = calculateRMSE(
			validationIndices,
			validationSize,
			data,
			userVectors,
			movieRatingVectors,
			movieRatingCounts,
			dimensions);
		cout << "RMSE: " << rmse << endl;

		//Write the bar graph values after every iteration
		writeBarGraphValues(
			data,
			userVectors,
			movieRatingVectors,
			movieRatingCounts,
			dimensions,
			barGraphIndices);
	}

	//Clear out all of the dynamic arrays before stopping
	for (int i1 = 0; i1 < numDataPoints; i1++) {
		delete[] data[i1];
	}
	delete[] data;

	for (int i1 = 0; i1 < maxUserId; i1++) {
		delete[] userVectors[i1];
	}
	delete[] userVectors;
	delete[] userCounts;
	delete[] userCumulativeCounts;

	for (int i1 = 0; i1 < maxMovieId; i1++) {
		for (int i2 = 0; i2 < MAX_STARS; i2++) {
			delete[] movieRatingVectors[i1][i2];
		}
		delete[] movieRatingVectors[i1];
		delete[] movieRatingCounts[i1];
		delete[] movieRatingCumulativeCounts[i1];
	}
	delete[] movieRatingVectors;
	delete[] movieRatingCounts;
	delete[] movieRatingCumulativeCounts;

	delete[] trainIndices;
	delete[] validationIndices;
	delete[] testIndices;

	delete[] zValues;

	return 0;
}

/**
 * Read in the settings for the run from a given input file
 * The input file contains, on each line in this order:
 * number of dimensions (integer)
 * initial value of eta (double)
 * user phi value (integer)
 * movie rating phi value (integer)
 * number of iterations to run for (integer)
 * sample size for calculating z (integer)
 * sample size for calculating RMSE score (integer)
 * @param  file The file to read the settings from
 * @return Settings struct containing all settings values
 */
struct Settings readSettings(char* file) {
	ifstream settingsInput(file, ios::in);

	//Initialize variables to hold all the settings
	int dimensions;
	double eta; //The initial value of eta to use in calculating eta
	int phiUser;
	int phiMR;
	int iterations;
	int zSampleSize;
	int scoreSampleSize;
	int repelNum;

	settingsInput >> dimensions;
	settingsInput >> eta;
	settingsInput >> phiUser;
	settingsInput >> phiMR;
	settingsInput >> iterations;
	settingsInput >> zSampleSize;
	settingsInput >> scoreSampleSize;
	settingsInput >> repelNum;

	settingsInput.close();

	struct Settings settings;
	settings.dimensions = dimensions;
	settings.eta = eta;
	settings.phiUser = phiUser;
	settings.phiMR = phiMR;
	settings.iterations = iterations;
	settings.zSampleSize = zSampleSize;
	settings.scoreSampleSize = scoreSampleSize;
	settings.repelNum = repelNum;

	return settings;
}

/**
 * Reads in all of the data points from the given file. The file must be binary
 * and each data point is stored sequentially (user id, movie id, movie rating)
 * @param file The binary file to read the data from
 * @param numDataPoints The number of data points to read from the file
 * @return The array of data points, a vector of indices used for shuffling,
 * and the maximum user and movie ids for vector generation, in a struct.
 */
struct Data readData(char* file, int numDataPoints) {
	ifstream dataFile(file, ios::in | ios::binary);

	//Initialize array to hold all data points
	int **data = new int*[numDataPoints];

	//Init array to hold Indices of all data points
	//Just holds the numbers 0, 1, ... 100480506
	//This is used to shuffle to be able to go through the data in a random order
	int* dataIndices = new int[numDataPoints];

	for (int i1 = 0; i1 < numDataPoints; i1++) {
		data[i1] = new int[NUM_PARTS];
		dataIndices[i1] = i1;
		//dataIndices.push_back(i1);
	}

	int maxUserId = 0;
	int maxMovieId = 0;

	//Go through and read in all the data
	for (int triple = 0; triple < numDataPoints; triple++) {
		for (int part = 0; part < NUM_PARTS; part++) {
			int in;
			dataFile.read((char *)&in, sizeof(int));

			data[triple][part] = in;
		}

		int userId = data[triple][USER_ID_IDX];
		int movieId = data[triple][MOVIE_ID_IDX];

		//Find max user and movie ids
		if (userId > maxUserId) {
			maxUserId = userId;
		}

		if (movieId > maxMovieId) {
			maxMovieId = movieId;
		}
	}

	dataFile.close();

	//Put everything in the struct and return it
	struct Data dataStruct;
	dataStruct.data = data;
	dataStruct.maxUserId = maxUserId;
	dataStruct.maxMovieId = maxMovieId;
	dataStruct.dataIndices = dataIndices;

	return dataStruct;
}

/**
 * Generates vectors for each user id and movie id in the dataset. Each user
 * gets a vector and each movie gets five; one for each rating it can receive.
 * @param data The array of data points to generate for
 * @param numDataPoints The number of data points in the array
 * @param maxUserId The maximum user id, used for initializing the user vector
 * array
 * @param maxMovieId The maximum movie-rating id, used for initializing the
 * movie-rating vector array
 * @param dimensions The number of dimensions each vector should have
 * @return The arrays of vectors for users and movie-ratings, the counts of
 * each user and movie-rating for calculating empirical probabilities, and the
 * number of distinct users and movies in the dataset, in a struct.
 */
struct Vectors generateVectors(
	int** data,
	int numDataPoints,
	int maxUserId,
	int maxMovieId,
	int dimensions) {

	//Initialize random number generators
	mt19937 random(time(0));
	normal_distribution<float> randomNormal;

	//Init array to hold user vectors and to hold user counts
	double **userVectors = new double*[maxUserId];
	int *userCounts = new int[maxUserId]; //To calculate the empirical probability
	int *userCumulativeCounts = new int[maxUserId]; //To calculate eta

	//Init each element of arrays
	for (int i1 = 0; i1 < maxUserId; i1++) {
		userVectors[i1] = NULL;
		userCounts[i1] = 0;
		userCumulativeCounts[i1] = 0;
	}

	//Init array to hold movie rating vectors
	double ***movieRatingVectors = new double**[maxMovieId];
	int **movieRatingCounts = new int*[maxMovieId]; //To calculate the empirical probability
	int **movieRatingCumulativeCounts = new int*[maxMovieId]; //To calculate eta

	//Init each element of arrays
	for (int i1 = 0; i1 < maxMovieId; i1++) {
		movieRatingVectors[i1] = NULL;

		movieRatingCounts[i1] = new int[MAX_STARS];
		movieRatingCumulativeCounts[i1] = new int[MAX_STARS];
		for (int i2 = 0; i2 < MAX_STARS; i2++) {
			movieRatingCounts[i1][i2] = 0;
			movieRatingCumulativeCounts[i1][i2] = 0;
		}
	}

	//Init number of users and movies
	int numUsers = 0;
	int numMovies = 0;

	//Go through the data and generate the vectors
	for (int i1 = 0; i1 < numDataPoints; i1++) {
		int *dataPt = data[i1];
		int userId = dataPt[USER_ID_IDX];
		int movieId = dataPt[MOVIE_ID_IDX];
		int movieRating = dataPt[MOVIE_RATING_IDX];

		userCounts[userId - 1]++;
		movieRatingCounts[movieId - 1][movieRating - 1]++;

		//Only generate a new vector if this user doesn't have one yet
		if (userVectors[userId - 1] == NULL) {
			numUsers++; //Increment number of users

			userVectors[userId - 1] = new double[dimensions];
			double length = 0;
			for (int dimension = 0; dimension < dimensions; dimension++) {
				double component = randomNormal(random);
				userVectors[userId - 1][dimension] = component;
				length += pow(component, 2);
			}

			length = sqrt(length);

			for (int dimension = 0; dimension < dimensions; dimension++) {
				userVectors[userId - 1][dimension] /= length;
			}
		}

		//Only generate new vectors of the movie doesn't have vectors yet
		if (movieRatingVectors[movieId - 1] == NULL) {
			numMovies++;

			movieRatingVectors[movieId - 1] = new double*[MAX_STARS];
			for (int star = 0; star < MAX_STARS; star++) {
				movieRatingVectors[movieId - 1][star] = new double[dimensions];

				double length = 0;
				for (int dimension = 0; dimension < dimensions; dimension++) {
					double component = randomNormal(random);
					movieRatingVectors[movieId - 1][star][dimension] = component;
					length += pow(component, 2);
				}

				length = sqrt(length);

				for (int dimension = 0; dimension < dimensions; dimension++) {
					movieRatingVectors[movieId - 1][star][dimension] /= length;
				}
			}
		}
	}

	//Stick everything in the struct and return it
	struct Vectors vectors;
	vectors.userVectors = userVectors;
	vectors.userCounts = userCounts;
	vectors.userCumulativeCounts = userCumulativeCounts;
	vectors.totalUsers = numUsers;
	vectors.movieRatingVectors = movieRatingVectors;
	vectors.movieRatingCounts = movieRatingCounts;
	vectors.movieRatingCumulativeCounts = movieRatingCumulativeCounts;
	vectors.totalMovies = numMovies;

	return vectors;
}

/**
 * Splits the original dataset into three separate, unique datasets.
 * @param dataIndices The full vector of all indices of data points
 * @return Vectors holding the indices of the data points in the training,
 * validation, and test datasets, in a struct.
 */
struct Datasets splitDatasets(int* dataIndices, int numDataPoints) {
	//Shuffle the data indices
	random_shuffle(&dataIndices[0], &dataIndices[numDataPoints - 1]); //dataIndices.begin(), dataIndices.end());

	//Split up the data into training, validation, and test sets
	int trainIdxStart = 0;
	int trainIdxEnd = TRAIN_SIZE * numDataPoints;

	int validationIdxStart = trainIdxEnd + 1;
	int validationIdxEnd = validationIdxStart + VALIDATION_SIZE * numDataPoints;

	int testIdxStart = validationIdxEnd + 1;
	int testIdxEnd = numDataPoints - 1;

	int* trainIndices = generateSet(dataIndices, trainIdxStart, trainIdxEnd);
	int* validationIndices = generateSet(dataIndices, validationIdxStart, validationIdxEnd);
	int* testIndices = generateSet(dataIndices, testIdxStart, testIdxEnd);

	struct Datasets datasets;
	datasets.trainIndices = trainIndices;
	datasets.trainSize = trainIdxEnd - trainIdxStart + 1;
	
	datasets.validationIndices = validationIndices;
	datasets.validationSize = validationIdxEnd - validationIdxStart + 1;

	datasets.testIndices = testIndices;
	datasets.testSize = testIdxEnd - testIdxStart + 1;

	return datasets;
}

/**
 * Generates a set from the original set of data indices with a given starting
 * and ending index.
 * @param dataIndices The original set of data indices
 * @param startIdx The starting index of the resulting set
 * @param endIdx The ending index of the resulting set
 * @return The resulting set of indices
 */
int* generateSet(int* dataIndices, int startIdx, int endIdx) {
	int* indices = new int[endIdx - startIdx + 1];
	int c = 0;
	for (int i1 = startIdx; i1 <= endIdx; i1++) {
		indices[c] = dataIndices[i1];
		c++;
	}
	return indices;
}

/**
 * Samples from the training data and calculates the initial value of z.
 * @param trainIndices The array of indices of data points in the training set
 * @param data The array of all data points
 * @param userVectors The array of all user vectors
 * @param movieRatingVectors The array of all movie-rating vectors
 * @param random The Mersenne Twister 19937 generator for random numbers
 * @param randomDataPoint The uniform int distribution random number generator
 * @param sampleSize The number of data points to sample in calculating z
 * @param dimensions The dimensionality of the vectors
 * @return The initial value of z, as well as each value of z sampled, for
 * updating the average later when we remove a data point, in a struct.
 */
struct ZValues calculateInitialZ(
	int* trainIndices,
	int trainSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	mt19937 random,
	uniform_int_distribution<int> randomDataPoint,
	int sampleSize,
	int dimensions) {

	double z = 0;
	double *zValues = new double[sampleSize];

	//Random shuffle for sample for dist2
	//random_shuffle(trainIndices.begin(), trainIndices.end());
	random_shuffle(&trainIndices[0], &trainIndices[trainSize - 1]);

	//Go through samples and calculate z and dist2
	for (int i1 = 0; i1 < sampleSize; i1++) {
		int userIdx = trainIndices[randomDataPoint(random)];
		int *userSampleDataPt = data[userIdx];
		int userId = userSampleDataPt[USER_ID_IDX];
		double *userVec = userVectors[userId - 1];

		int mrIdx = trainIndices[randomDataPoint(random)];
		int *mrSampleDataPt = data[mrIdx];
		int movieId = mrSampleDataPt[MOVIE_ID_IDX];
		int movieRating = mrSampleDataPt[MOVIE_RATING_IDX];
		double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

		double zVal = exp(-getDistanceSquared(userVec, mrVec, dimensions));

		z += zVal;
		zValues[i1] = zVal;
	}

	//Average z and dist^2
	z /= sampleSize;

	struct ZValues zStruct;
	zStruct.z = z;
	zStruct.zValues = zValues;

	return zStruct;
}

/**
 * Find the squared distance between two given vectors
 * @param a The first vector
 * @param b The second vector
 * @param dimensions The length of the vectors
 * @return The squared distance between a and b
 */
double getDistanceSquared(double *a, double *b, int dimensions) {
	double sum = 0.0;

	for (int i1 = 0; i1 < dimensions; i1++) {
		double aPt = a[i1];
		double bPt = b[i1];

		sum += getDistanceSquared(aPt, bPt);
	}

	return sum;
}

/**
 * Find the squared distance between two given numbers
 * @param a First number
 * @param b Second number
 * @return (a - b)^2
 */
double getDistanceSquared(double a, double b) {
	double x = a - b;
	return x * x;
}

/**
 * Calculate eta (the weight of the other vector when moving) based on a given
 * phi and the number of times this vector has been moved.
 * @param etaInitial The initial value of eta, given in the input file
 * @param phi The value of phi, given in the input file
 * @param count The number of times the vector has been moved
 * @return The adjusted value of eta
 */
double calculateEta(double etaInitial, double phi, int count) {
	return etaInitial * (phi / (phi + count));
}

/**
 * Executes the move operations for a specific data point.
 * Sequence of moves:
 * 1. Attract the user and movie-rating vectors toward each other
 * 2a. Repel the user vector away from a random movie-rating vector
 * 2b. Repel the movie-rating vector away from a random user vector
 * 3a. Repel the user away from a random user vector
 * 3b. Repel the movie-rating away from a random movie-rating
 * 4. Repel the movie-rating away from the four other movie-ratings for the
 * same movie.
 * 
 * @param user The user vector of the data point
 * @param mr The movie-rating vector of the data point
 * @param movieRating The actual rating that was given (1, 2, 3, 4, 5)
 * @param etaInitial The initial version of eta given in the settings file
 * @param userEta The calculated value of eta based on user phi and count
 * @param mrEta The calculated value of eta based on movie-rating phi and count
 * @param userVectorsRepel The vectors to repel the user from
 * @param randomUser2 A random user that the user is repelled from
 * @param mrVectorsRepel The vectors to repel the movie-rating from
 * @param numRepel The number of vectors in the repel arrays
 * @param randomMR2 A random movie-rating that the movie-rating is repelled from
 * @param dimensions The length of the vectors, the number of dimensions given
 * in settings
 * @param z The value of z
 */
void moveVectors(
	double* user,
	double* mr,
	double movieRating,
	double etaInitial,
	double userEta,
	double mrEta,
	double** userVectorsRepel,
	double** mrVectorsRepel,
	int numRepel,
	double* randomUser,
	double* randomMR,
	int dimensions,
	double z) {

	//Attract and repel the vectors
	for (int dimension = 0; dimension < dimensions; dimension++) {
		double userComponent = user[dimension];
		double mrComponent = mr[dimension];

		//These are for moving user away from user or mr away from mr
		double randomUserComponent2 = randomUser[dimension];
		double randomMRComponent2 = randomMR[dimension];

		//Attract the user and movie rating vectors
		double newUserComponent = attract(userComponent, mrComponent, userEta);
		double newMRComponent = attract(mrComponent, userComponent, mrEta);
		
		//Repel the user away from a few random movie ratings
		double userAmnt = 0;
		double mrAmnt = 0;
		for (int i1 = 0; i1 < numRepel; i1++) {
			double userComponentTemp = userVectorsRepel[i1][dimension];
			userAmnt += (userEta / numRepel * exp(-getDistanceSquared(newUserComponent, userComponentTemp)) / z) * (newUserComponent - userComponentTemp);

			double mrComponentTemp = mrVectorsRepel[i1][dimension];
			mrAmnt += (mrEta / numRepel * exp(-getDistanceSquared(newMRComponent, mrComponentTemp)) / z) * (newMRComponent - mrComponentTemp);
		}
		newUserComponent += userAmnt;
		newMRComponent += mrAmnt;

		//Repel the user away from a random user and the movie rating away from a random movie rating
		newUserComponent = repel(newUserComponent, randomUserComponent2, userEta, z);
		newMRComponent = repel(newMRComponent, randomMRComponent2, mrEta, z);

		//Set the updated components back into the array
		user[dimension] = newUserComponent;
		mr[dimension] = newMRComponent;
	}

	//Calculate norm^2 for both vectors
	double userNorm2 = norm2(user, dimensions);
	double mrNorm2 = norm2(mr, dimensions);
	
	//Normalize the vectors to unit length
	for (int dimension = 0; dimension < dimensions; dimension++) {
		user[dimension] = normalizeUnitLength(user[dimension], userNorm2);
		mr[dimension] = normalizeUnitLength(mr[dimension], mrNorm2);
	}
}

/**
 * Attracts a towards b by taking a weighted average of a and b.
 * The bigger weight goes to a, the original vector
 * @param a The vector component to be attracted
 * @param b The vector component to attract to
 * @param eta The eta value to scale by, aka the weight of the average
 * @return The resulting value of a
 */
double attract(double a, double b, double eta) {
	return ((1 - eta) * a) + (eta * b);
}

/**
 * Repels a away from other
 * @param a The vector to be repeled
 * @param other The vector to repel away from
 * @param eta The eta value to scale by
 * @param z The current z value
 * @return The resulting value of a
 */
double repel(double a, double other, double eta, double z) {
	return a + (eta * exp(-getDistanceSquared(a, other)) / z) * (a - other);
}

/**
 * Find the squared length of a vector
 * @param arr The vector
 * @param length The length of the vector (dimensions)
 * @return The squared length of the vector
 */
double norm2(double *arr, int length) {
	double sum = 0.0;

	for (int i1 = 0; i1 < length; i1++) {
		double component = arr[i1];
		sum += component * component;
	}

	return sum;
}

/**
 * Normalizes a to be about unit length
 * @param a The vector component to normalize
 * @param eta The value of eta to use
 * @param delta The value of delta
 * @param aNorm2 The value of the norm^2 of the vector a
 * @return The resulting value of a
 */
double normalize(double a, double eta, double delta, double aNorm2) {
	return eta * delta * (aNorm2 - 1) * a;
}

/**
 * Normalize a to be exactly unit length
 * @param a The component of the vector to normalize
 * @param aNorm2 The total length of a, squared
 * @return The normalized component of a
 */
double normalizeUnitLength(double a, double aNorm2) {
	return a / sqrt(aNorm2);
}

/**
 * Calculates the RMSE for the model on given data points.
 * @param evaluationIndices The data indices to evaluate the model on
 * @param data The array of data points
 * @param userVectors The array of user vectors
 * @param movieRatingVectors The array of movie-rating vectors
 * @param movieRatingCounts The array of counts of movie-rating vectors
 * @param dimensions The dimensionality of the vectors
 * @return The RMSE of the model
 */
double calculateRMSE(
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	int** movieRatingCounts,
	int dimensions) {

	//Calculate the score on the validation set
	random_shuffle(&evaluationIndices[0], &evaluationIndices[evaluationSize - 1]);
	double* error = new double[evaluationSize];

	for (int i1 = 0; i1 < evaluationSize; i1++) {
		//Get a random data point
		int idx = evaluationIndices[i1];
		int* triple = data[idx];

		//Get the info from it
		int userId = triple[USER_ID_IDX];
		int movieId = triple[MOVIE_ID_IDX];
		int movieRating = triple[MOVIE_RATING_IDX];

		//Get the user vector and all movie rating vectors
		double* userVector = userVectors[userId - 1];
		double** movieVectors = movieRatingVectors[movieId - 1];
		int* movieCounts = movieRatingCounts[movieId - 1];

		double avgStar = 0;
		double pTotal = 0;

		//Go through each star, calculate the probability of the user giving that rating
		for (int star = 0; star < MAX_STARS; star++) {
			double* movieRatingVector = movieVectors[star];
			double d2 = getDistanceSquared(userVector, movieRatingVector, dimensions);

			double p = exp(-d2) * movieCounts[star];

			avgStar += (star + 1) * p;
			pTotal += p;
		}

		//Find the average star rating
		avgStar /= pTotal;

		//Calculate the error between our prediction and the actual
		error[i1] = avgStar - movieRating;
	}

	//Calculate the root mean squared error
	double mse = 0;
	for (int i1 = 0; i1 < evaluationSize; i1++) {
		double err = error[i1];
		mse += err * err;
	}
	mse /= evaluationSize;
	double rmse = sqrt(mse);

	//Clear out the error array
	delete[] error;

	return rmse;
}

void writeBarGraphValues(
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	int** movieRatingCounts,
	int dimensions,
	int barGraphIndices[]) {

	//Go through the data points we want to generate bar graphs for
	//Calculate the probabilities of each star and write them to a file
	for (int i1 = 0; i1 < BAR_GRAPH_COUNT; i1++) {
		int idx = barGraphIndices[i1];
		int* triple = data[idx];

		//Get the info from it
		int userId = triple[USER_ID_IDX];
		int movieId = triple[MOVIE_ID_IDX];
		int movieRating = triple[MOVIE_RATING_IDX];

		//Get the user vector and all movie rating vectors
		double* userVector = userVectors[userId - 1];
		double** movieVectors = movieRatingVectors[movieId - 1];
		int* movieCounts = movieRatingCounts[movieId - 1];

		double pTotal = 0;

		//Go through each star and calculate the sum for normalization
		for (int star = 0; star < MAX_STARS; star++) {
			double* movieRatingVector = movieVectors[star];
			double d2 = getDistanceSquared(userVector, movieRatingVector, dimensions);

			pTotal += exp(-d2) * movieCounts[star];
		}

		//Create the file, in comma separated value format
		//Each row is an iteration, each column is a probability
		//first column: probability of rating with a 1
		//second column: probability of rating with a 2
		//so on...
		ofstream file;
		string name = "bar_graph_" + to_string(i1 + 1) + "_movieid=" + to_string(movieId) + ".csv";
		file.open(name, ios::app);

		//Go back through the stars and calculate the probability of each, and
		//write it out to the output file
		for (int star = 0; star < MAX_STARS; star++) {
			double* movieRatingVector = movieVectors[star];
			double d2 = getDistanceSquared(userVector, movieRatingVector, dimensions);

			//This is the conditional probability
			//P(rating = k | user = i, movie = j)
			double p = exp(-d2) * movieCounts[star] / pTotal;

			//Write the probability to the file
			file << p;
			if (star != MAX_STARS - 1) {
				//Don't write a comma after the last number
				file << ",";
			}
		}
		file << "\n";
		file.close();
	}
}