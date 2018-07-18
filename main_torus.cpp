//Netflix Prize
//Jordan Turley and Michael Lamar

//This implementation is using a torus
//For a data point, the user and movie rating vectors are attracted together
//Then, the user is repelled away from the average movie rating, and the movie rating from the average user
//The current z is printed out 100 times every iteration (every %)

//To compile: g++ main.cpp -o netflix -std=c++11 -O3
//To run on Linux: ./netflix input_file.txt data_file.bin 100480507
//To run on Windows: netflix input_file.txt data_file.bin 100480507
//The number at the end can be any number, as long as it is greater than the sample sizes.
//For example, 1000000 to just run it on the first 1000000 data points
//Running the program on the full data set takes almost 4.5 GB of RAM

//scp main.cpp zeyang@dragon.centre.edu:~/ upload to the server using command line

#include <iostream>
#include <random>
#include <ctime>
#include <fstream>
#include <cmath>
#include <vector>
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
	//int zSampleSize;
	int repulsionSampleSize;
	int scoreSampleSize;
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
	int dimensions,
	double scalingFactor);
void moveVectors(
	double *userVector,
	double *movieRatingVector,
	double *newUserVector,
	double *newMovieRatingVector,
	double *randomUserVec2, //new
	double *randomMRVec2, //new
	int dimensions,
	double etaInitial,
	double userEta,
	double mrEta,
	double z,
	double scalingFactor);
double attract(double a, double b, double c, double scalingFactor);
double sign(double num);
double repel(double a, double b, double c, double z, double scalingFactor);
double mod(double a, double b);
double getDistanceSquared(double a, double b);
double getDistanceSquared(double *a, double *b, int dimensions);
double *averageTorusSampleToXY(double *sample, int sampleSize);
double getAvgAngForRepulsion(double *sample, int sampleSize);
double getAvgMagnitudeForRepulsion(double *sample, int sampleSize);
double calculateEta(double etaInitial, double phi, int count);
double calculateRMSE(
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	double** userVectors,
	double*** movieRatingVectors,
	int** movieRatingCounts,
	int dimensions,
	double scalingFactor);
double calculateRMSEEmpirical(
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	int** movieRatingCounts);

//Total data points: 100480507

const int NUM_PARTS = 3; //How many parts are there to each triple? 3 - user id, movie id, rating
const int USER_ID_IDX = 0; //The index of the user id in the array storing the triple
const int MOVIE_ID_IDX = 1; //The index of the movie id in the array storing the triple
const int MOVIE_RATING_IDX = 2; //The index of the movie rating in the array storing the triple

//Total number of users: 480189
//Max user id: 2649429
//Number of movies: 17700

//The input file has:
//dimensions
//initial eta
//phi for users
//phi for movie ratings
//number of iterations
//repulsion sample size
//score sample size

const int MAX_STARS = 5;

const double VECTOR_MAX_SIZE = 1; //If we ever decided to make the vectors bigger than 1 we can change this
const double VECTOR_HALF_SIZE = VECTOR_MAX_SIZE / 2;

const double PI = 3.14159;
const double TWO_PI = PI * 2;

//Sizes for each set
const double TRAIN_SIZE = 0.8;
const double VALIDATION_SIZE = 0.1;
const double TEST_SIZE = 1 - TRAIN_SIZE - VALIDATION_SIZE;

//Sample size to calculate distances, likelihood, etc.
const int AVERAGE_SAMPLE_SIZE = 10000;

int main(int argc, char *argv[]) {
	//Seed general random number generator
	srand(time(0));

	char *settingsFile = (char *) "C:\\input.txt";
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

	//Read in the settings into a struct from the file given on command line
	struct Settings settings = readSettings(settingsFile);

	//Pull out a few settings out of the struct since they are used a lot
	int dimensions = settings.dimensions;
	double etaInitial = settings.eta;
	int repulsionSampleSize = settings.repulsionSampleSize;
	int scoreSampleSize = settings.scoreSampleSize;

	//TODO calculate this scaling factor based on the number of dimensions
	//Something like 1 / sqrt(dimensions)
	double scalingFactor = 50 / (double) dimensions;
	scalingFactor = scalingFactor * scalingFactor; //squared

	cout << "Scaling: " << scalingFactor << endl;

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
		repulsionSampleSize, //TODO for now just use repulsion sample size for z sample size too
		dimensions,
		scalingFactor);
	double z = zStruct.z;
	double* zValues = zStruct.zValues;
	int oldestIdx = 0;

	//Save the initial z in case we need to use it later, and print it out
	double initialZ = z;
	cout << "Initial z: " << z << endl;

	//Calculate the RMSE using only the empirical probabilities w/o the model
	double rmseEmpirical = calculateRMSEEmpirical(validationIndices, validationSize, data, movieRatingCounts);
	cout << "Empirical_RMSE: " << rmseEmpirical << endl;

	//Go through the number of iterations to move the vectors
	for (int iteration = 0; iteration < settings.iterations; iteration++) {
		random_shuffle(&trainIndices[0], &trainIndices[trainSize - 1]);

		int reportNum = trainSize / 100;

		cout << "Starting iteration " << iteration + 1 << endl;

		//Go through each data point in the training set
		for (int dataIdx = 0; dataIdx < trainSize; dataIdx++) {
			int idx = trainIndices[dataIdx];
			int *triple = data[idx];

			//Get the individual parts of it
			int userId = triple[USER_ID_IDX];
			int movieId = triple[MOVIE_ID_IDX];
			int movieRating = triple[MOVIE_RATING_IDX];

			//Update the cumulative counts for the user and movie rating
			userCumulativeCounts[userId - 1]++;
			movieRatingCumulativeCounts[movieId - 1][movieRating - 1]++;

			//Get the vectors and calculate eta for the user and movie rating vectors
			double *userVector = userVectors[userId - 1];
			double userEta = calculateEta(etaInitial, settings.phiUser, userCumulativeCounts[userId - 1]);

			double *movieRatingVector = movieRatingVectors[movieId - 1][movieRating - 1];
			double movieRatingEta = calculateEta(etaInitial, settings.phiMR, movieRatingCumulativeCounts[movieId - 1][movieRating - 1]);
			
			//Get a random user vector
			int newUserDataIdx = randomDataPoint(random);
			int *dataPt = data[newUserDataIdx];
			int randomUserId = dataPt[USER_ID_IDX];
			double *newUserVector = userVectors[randomUserId - 1];

			//Get a random movie rating vector
			int newMovieRatingDataIdx = randomDataPoint(random);
			dataPt = data[newMovieRatingDataIdx];
			int randomMovieId = dataPt[MOVIE_ID_IDX];
			int randomMovieRating = dataPt[MOVIE_RATING_IDX];
			double *newMovieRatingVector = movieRatingVectors[randomMovieId - 1][randomMovieRating - 1];

			//----------new---
			//repel a user from a random user and 
			//repel a movierating from a random movierating
			//get two new vector before moveVectors is called
			//pass them into movevectors
			//then repel in the loop

			//get a random user vector
			int randomUserDataIdx2 = trainIndices[randomDataPoint(random)]; //fixed the random seed
			int *dataPt2 = data[randomUserDataIdx2]; //pull a random data index from the dataset
			int randomUserId2 = dataPt2[USER_ID_IDX]; //user_id_index a constant 0
			double *randomUserVec2 = userVectors[randomUserId2 - 1];

			//a random movie rating vector
			int randomMRDataIdx2 = trainIndices[randomDataPoint(random)]; //fixed the random seed
			dataPt2 = data[randomMRDataIdx2]; //pull a random data index from the dataset
			int randomMovieId2 = dataPt2[MOVIE_ID_IDX]; //user_id_index a constant 0
			int randomMovieRating2 = dataPt2[MOVIE_RATING_IDX];
			double *randomMRVec2 = movieRatingVectors[randomMovieId2 - 1][randomMovieRating2 - 1];
			//-----------------

			moveVectors(
				userVector,
				movieRatingVector,
				newUserVector,
				newMovieRatingVector,
				randomUserVec2, //new for user-random-user repel
				randomMRVec2, //new for mr-random-mr repel
				dimensions,
				etaInitial,
				userEta,
				movieRatingEta,
				z,
				scalingFactor);

			//Select new random user and mr vectors for the z calculation
			newUserDataIdx = randomDataPoint(random);
			dataPt = data[newUserDataIdx];
			newUserVector = userVectors[dataPt[USER_ID_IDX] - 1];

			newMovieRatingDataIdx = randomDataPoint(random);
			dataPt = data[newMovieRatingDataIdx];
			newMovieRatingVector = movieRatingVectors[dataPt[MOVIE_ID_IDX] - 1][dataPt[MOVIE_RATING_IDX] - 1];

			//Recalculate z based on the average
			double oldestZVal = zValues[oldestIdx];
			double newZVal = exp(-scalingFactor * getDistanceSquared(newUserVector, newMovieRatingVector, dimensions));
			z = z + (newZVal - oldestZVal) / repulsionSampleSize;
			zValues[oldestIdx] = newZVal;

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

					double likelihood = userPBar * mrPBar * exp(-scalingFactor * getDistanceSquared(userVec, mrVec, dimensions)) / z;
					likelihoodAvg += likelihood;
				}
				likelihoodAvg /= AVERAGE_SAMPLE_SIZE;
				cout << "Likelihood: " << likelihoodAvg << endl;
			}

			//Move to the next oldest for the samples
			oldestIdx++;
			oldestIdx %= repulsionSampleSize;
		}

		//Find the RMSE after each iteration to see if it is improving
		double rmse = calculateRMSE(
			validationIndices,
			validationSize,
			data,
			userVectors,
			movieRatingVectors,
			movieRatingCounts,
			dimensions,
			scalingFactor);
		cout << "Model_RMSE: " << rmse << endl;
	}

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
 * sample size for repulsion and calculating z (integer)
 * TODO this will be split up later but for now this is for both repulsion and z
 * sample size for calculating RMSE score (integer)
 * @param  file The file to read the settings from
 * @return Settings struct containing all settings values
 */
struct Settings readSettings(char* file) {
	//Read in settings from input file
	ifstream settingsInput(file, ios::in);

	int dimensions;
	double etaInitial;
	double phiUser;
	double phiMR;
	int iterations;
	int repulsionSampleSize;
	int scoreSampleSize;

	settingsInput >> dimensions;
	settingsInput >> etaInitial;
	settingsInput >> phiUser;
	settingsInput >> phiMR;
	settingsInput >> iterations;
	settingsInput >> repulsionSampleSize;
	settingsInput >> scoreSampleSize;

	settingsInput.close();

	struct Settings settings;
	settings.dimensions = dimensions;
	settings.eta = etaInitial;
	settings.phiUser = phiUser;
	settings.phiMR = phiMR;
	settings.iterations = iterations;
	settings.repulsionSampleSize = repulsionSampleSize;
	settings.scoreSampleSize = scoreSampleSize;

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
 * each user and movie-rating for calculating emperical probabilities, and the
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
	uniform_real_distribution<float> randomDouble(0.0, VECTOR_MAX_SIZE);

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

		//If the user vector hasn't been generated, generate it
		if (userVectors[userId - 1] == NULL) {
			numUsers++;

			userVectors[userId - 1] = new double[dimensions];
			for (int dimension = 0; dimension < dimensions; dimension++) {
				double d = randomDouble(random);
				userVectors[userId- 1][dimension] = d;
			}
		}

		//If the movie rating vectors haven't been generated yet, generate them
		if (movieRatingVectors[movieId - 1] == NULL) {
			numMovies++;

			movieRatingVectors[movieId - 1] = new double*[MAX_STARS];

			//Generate a vector for each rating, 1 through 5
			for (int star = 0; star < MAX_STARS; star++) {
				movieRatingVectors[movieId - 1][star] = new double[dimensions];
				for (int dimension = 0; dimension < dimensions; dimension++) {
					double d = randomDouble(random);
					movieRatingVectors[movieId - 1][star][dimension] = d;
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
	int dimensions,
	double scalingFactor) {

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

		double zVal = exp(-scalingFactor * getDistanceSquared(userVec, mrVec, dimensions));

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

void moveVectors(
	double *userVector,
	double *movieRatingVector,
	double *newUserVector,
	double *newMovieRatingVector,
	double *randomUserVec2, //new param for user-random-user repel
	double *randomMRVec2, //new param for MR-random-MR repel
	int dimensions,
	double etaInitial,
	double userEta,
	double movieRatingEta,
	double z,
	double scalingFactor) {
	
	//Go through each dimension of the vector
	for (int dimension = 0; dimension < dimensions; dimension++) {
		//Get this component of the user and movie rating vectors
		double userPt = userVector[dimension];
		double movieRatingPt = movieRatingVector[dimension];

		//Get the components for the user-mr and mr-user repulsion
		double newUserComponent = newUserVector[dimension];
		double newMovieRatingComponent = newMovieRatingVector[dimension];

		//Get the components for the user-user and mr-mr repulsion
		double newUserComponent2 = randomUserVec2[dimension];
		double newMRComponent2 = randomMRVec2[dimension];

		//Move the user towards the mr, away from a random mr, and away from a random user
		double newUserPt = attract(userPt, movieRatingPt, userEta, scalingFactor);
		newUserPt = repel(newUserPt, newMovieRatingComponent, userEta, z, scalingFactor);
		newUserPt = repel(newUserPt, newUserComponent2, userEta, z, scalingFactor); //new user-randomuser repel
		// ---- > randomUserEta, userEta

		//Move the mr towards the user, away from a random user, and away from a random mr
		double newMovieRatingPt = attract(movieRatingPt, userPt, movieRatingEta, scalingFactor);
		newMovieRatingPt = repel(newMovieRatingPt, newUserComponent, movieRatingEta, z, scalingFactor);
		newMovieRatingPt = repel(newMovieRatingPt, newMRComponent2, movieRatingEta, z, scalingFactor);//new movie-random repel
		// ----> movieRatingEta ? randomMReta??

		//Set the components back into their vectors
		userVector[dimension] = newUserPt;
		movieRatingVector[dimension] = newMovieRatingPt;
	}
}

/**
* @return The value of a after being attracted to b
*/
double attract(double a, double b, double c, double scalingFactor) {
	//Multiply scaling factor into the step size
	c *= scalingFactor;

	double r = a - c * (a - b);
	if (abs(a - b) > VECTOR_HALF_SIZE) {
		r += c * sign(a - b);
	}

	return mod(r, VECTOR_MAX_SIZE);
}

/**
 * Repel just attracts the vector to the opposite of the other vector
 * @return the value of a after being repelled from b
 */
double repel(double a, double b, double c, double z, double scalingFactor) {
	c = c * exp(-scalingFactor * getDistanceSquared(a, b) / z);
	return attract(a, mod(b + VECTOR_HALF_SIZE, VECTOR_MAX_SIZE), c, scalingFactor);
}

/**
 * Returns -1 if num is negative, 0 if num is 0, and 1 if num is positive.
 */
double sign(double num) {
	return (num > 0) ? 1 : num == 0 ? 0 : -1;
}

double mod(double a, double b) {
	double r = fmod(a, b);
	if (r < 0) {
		r += b;
	}
	return r;
}

/**
* @return The squared distance between two points (scalars) on the torus
*/
double getDistanceSquared(double a, double b) {
	double diff = abs(a - b);

	double diff2 = pow(diff, 2);
	if (diff > 0.5) {
		diff2 += 1 - 2 * diff;
	}

	return diff2;
}

/**
 * @return The squared distance between two vectors on the torus
 */
double getDistanceSquared(double *a, double *b, int dimensions) {
	double sum = 0;

	for (int i1 = 0; i1 < dimensions; i1++) {
		double aPt = a[i1];
		double bPt = b[i1];

		sum += getDistanceSquared(aPt, bPt);
	}

	return sum;
}

/**
 * Helper function to generate average (x, y) value for a sample of points on the torus
 */
double* averageTorusSampleToXY(double *sample, int sampleSize) {
	double x = 0, y = 0;

	for (int i1 = 0; i1 < sampleSize; i1++) {
		double pt = sample[i1];
		x += cos(TWO_PI * pt);
		y += sin(TWO_PI * pt);
	}

	x /= sampleSize;
	y /= sampleSize;

	double *result = new double[2];
	result[0] = x;
	result[1] = y;

	return result;
}

/**
 * @return The angle of the vector of the average of a sample of points on the torus. Between 0 and 1.
 */
double getAvgAngForRepulsion(double *sample, int sampleSize) {
	double *avg = averageTorusSampleToXY(sample, sampleSize);
	double x = avg[0], y = avg[1];

	double result = atan2(y, x) / TWO_PI;
	result = fmod(result + VECTOR_MAX_SIZE, VECTOR_MAX_SIZE);

	return result;
}

/**
 * @return The magnitude of the vector of the average of a sample of points on the torus. Between 0 and 1.
 */
double getAvgMagnitudeForRepulsion(double *sample, int sampleSize) {
	double *avg = averageTorusSampleToXY(sample, sampleSize);
	double x = avg[0], y = avg[1];

	double result = sqrt(x * x + y * y);

	if (result != result) {
		return 0;
	}

	return result;
}

/**
 * Calculates eta using a learning rate thta decreases as vectors are seen more often
 * @return Eta, based on the initial eta, a given phi, and a given count of vectors
 */
double calculateEta(double etaInitial, double phi, int count) {
	return etaInitial * (phi / (phi + count));
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
	int dimensions,
	double scalingFactor) {

	//Calculate the score on the validation set
	random_shuffle(&evaluationIndices[0], &evaluationIndices[evaluationSize - 1]);
	double *error = new double[evaluationSize];

	for (int i1 = 0; i1 < evaluationSize; i1++) {
		//Get a random data point
		int idx = evaluationIndices[i1];
		int *triple = data[idx];

		//Get the info from it
		int userId = triple[USER_ID_IDX];
		int movieId = triple[MOVIE_ID_IDX];
		int movieRating = triple[MOVIE_RATING_IDX];

		//Get the user vector and all movie rating vectors
		double *userVector = userVectors[userId - 1];
		double **movieVectors = movieRatingVectors[movieId - 1];

		double avgStar = 0;
		double pTotal = 0;

		//Go through each star, calculate the probability of the user giving that rating
		for (int star = 0; star < MAX_STARS; star++) {
			double *movieRatingVector = movieVectors[star];
			double d2 = getDistanceSquared(userVector, movieRatingVector, dimensions);

			double p = exp(-scalingFactor * d2) * movieRatingCounts[movieId - 1][star];

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

/**
 * Calculate the RMSE based on only the empirical probabilities
 * @param evaluationIndices The data indices to evaluate the model on
 * @param evaluationSize The size of the evaluation set
 * @param data The array of data points
 * @param movieRatingCounts The array of counts of movie-rating vectors
 * @return The RMSE of the model
 */
double calculateRMSEEmpirical(
	int* evaluationIndices,
	int evaluationSize,
	int** data,
	int** movieRatingCounts) {

	double* error = new double[evaluationSize];
	for (int i1 = 0; i1 < evaluationSize; i1++) {
		//Get a random data point
		int idx = evaluationIndices[i1];
		int* triple = data[idx];

		//Get the info from it
		int userId = triple[USER_ID_IDX];
		int movieId = triple[MOVIE_ID_IDX];
		int movieRating = triple[MOVIE_RATING_IDX];

		int* movieCounts = movieRatingCounts[movieId - 1];

		double avgStar = 0;
		double pTotal = 0;

		//Go through each star, calculate the probability of the user giving that rating
		for (int star = 0; star < MAX_STARS; star++) {

			double p = movieCounts[star];

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