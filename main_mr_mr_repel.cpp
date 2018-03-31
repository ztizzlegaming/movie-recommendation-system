//Netflix Prize
//Jordan Turley and Michael Lamar

//This implementation is using a sphere
//For a data point, the user and movie rating vectors are attracted together
//Then, the user and movie rating are repelled away from random movie ratings and users, independently
//Then, the movie rating is repelled away from the other four movie ratings for that movie

//The current z is printed out 100 times every iteration (every %)

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
#include <set>

using namespace std;

double getDistanceSquared(double a, double b);
double getDistanceSquared(double *a, double *b, int dimensions);
double norm2(double *arr, int length);
double calculateEta(double etaInitial, double phi, int count);
void moveVectors(double *user, double *mr, double movieRating, double **movieRatingsArr, double etaInitial, double userEta, double mrEta, double *randomUser1, double *randomUser2, double *randomMR1, double *randomMR2, int dimensions, double z);
double attract(double a, double b, double eta);
double repel(double a, double other, double eta, double z);
double normalize(double a, double initialEta, double delta, double aNorm2);
double normalizeUnitLength(double a, double aNorm2);

const int NUM_PARTS = 3; //How many parts are there to each triple? 3 - user id, movie id, rating
const int USER_ID_IDX = 0; //The index of the user id in the array storing the triple
const int MOVIE_ID_IDX = 1; //The index of the movie id in the array storing the triple
const int MOVIE_RATING_IDX = 2; //The index of the movie rating in the array storing the triple

const int MAX_STARS = 5;

const int AVERAGE_SAMPLE_SIZE = 10000;

//The input file has:
//dimensions
//initial eta
//phi for users
//phi for movie ratings
//number of iterations
//z sample size
//score sample size

int main(int argc, char *argv[]) {
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

	ifstream settingsInput(settingsFile, ios::in);

	int dimensions;
	double eta;
	int phiUser;
	int phiMR;
	int iterations;
	int zSampleSize;
	int scoreSampleSize;

	settingsInput >> dimensions;
	settingsInput >> eta;
	settingsInput >> phiUser;
	settingsInput >> phiMR;
	settingsInput >> iterations;
	settingsInput >> zSampleSize;
	settingsInput >> scoreSampleSize;

	settingsInput.close();

	//Calculate delta based on eta
	double delta = 1 / (4 * eta);

	mt19937 random(time(0));
	uniform_int_distribution<int> randomDataPoint(0, numDataPoints - 1);
	normal_distribution<float> randomNormal;

	ifstream file(dataFile, ios::in | ios::binary);

	//Initialize array to hold all data points
	int **data = new int*[numDataPoints];
	for (int i1 = 0; i1 < numDataPoints; i1++) {
		data[i1] = new int[NUM_PARTS];
	}

	//Init vector to hold indexes of all data points
	//This is used to shuffle to be able to go through the data in a random order
	vector<int> dataIndexes;

	int maxUserId = 0;
	int maxMovieId = 0;

	cout << "Reading in data" << endl;

	//Go through and read in all the data
	for (int triple = 0; triple < numDataPoints; triple++) {
		for (int part = 0; part < NUM_PARTS; part++) {
			int in;
			file.read((char *)&in, sizeof(int));

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

	file.close();

	cout << "Max user id: " << maxUserId << endl;
	cout << "Max movie id: " << maxMovieId << endl;

	cout << "Initializing vectors" << endl;

	//Init array to hold user vectors and to hold user counts
	double **userVectors = new double*[maxUserId];
	int *userCounts = new int[maxUserId];
	int *userCumulativeCounts = new int[maxUserId];
	for (int i1 = 0; i1 < maxUserId; i1++) {
		userVectors[i1] = NULL;
		userCounts[i1] = 0;
		userCumulativeCounts[i1] = 0;
	}

	//Init array to hold movie rating vectors
	double ***movieRatingVectors = new double**[maxMovieId];
	int **movieRatingCounts = new int*[maxMovieId];
	int **movieRatingCumulativeCounts = new int*[maxMovieId];
	for (int i1 = 0; i1 < maxMovieId; i1++) {
		movieRatingVectors[i1] = NULL;

		movieRatingCounts[i1] = new int[MAX_STARS];
		movieRatingCumulativeCounts[i1] = new int[MAX_STARS];
		for (int i2 = 0; i2 < MAX_STARS; i2++) {
			movieRatingCounts[i1][i2] = 0;
			movieRatingCumulativeCounts[i1][i2] = 0;
		}
	}

	set<int> userIds;
	set<int> movieIds;

	//Go through the data and generate the vectors
	for (int i1 = 0; i1 < numDataPoints; i1++) {
		dataIndexes.push_back(i1);

		int *dataPt = data[i1];
		int userId = dataPt[USER_ID_IDX];
		int movieId = dataPt[MOVIE_ID_IDX];
		int movieRating = dataPt[MOVIE_RATING_IDX];

		userIds.insert(userId);
		movieIds.insert(movieId);

		userCounts[userId - 1]++;
		movieRatingCounts[movieId - 1][movieRating - 1]++;

		if (userVectors[userId - 1] == NULL) {
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

		if (movieRatingVectors[movieId - 1] == NULL) {
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

	int totalUsers = userIds.size();
	int totalMovies = movieIds.size();

	cout << "Number of users: " << totalUsers << endl;
	cout << "Number of movies: " << totalMovies << endl;

	cout << "Generating sample for z" << endl;

	int *zUserSample = new int[zSampleSize];
	int *zMRSample = new int[zSampleSize];

	double z = 0;
	double *zValues = new double[zSampleSize];

	double *dist2Values = new double[zSampleSize];
	double averageDist2 = 0;

	int oldestIdx = 0;

	//Generate user sample
	random_shuffle(dataIndexes.begin(), dataIndexes.end());
	for (int i1 = 0; i1 < zSampleSize; i1++) {
		zUserSample[i1] = dataIndexes[i1];
	}

	//Generate movie rating sample
	random_shuffle(dataIndexes.begin(), dataIndexes.end());
	for (int i1 = 0; i1 < zSampleSize; i1++) {
		zMRSample[i1] = dataIndexes[i1];
	}

	//Random shuffle for sample for dist2
	random_shuffle(dataIndexes.begin(), dataIndexes.end());

	//Go through samples and calculate z and dist2
	for (int i1 = 0; i1 < zSampleSize; i1++) {
		int userIdx = zUserSample[i1];
		int *userSampleDataPt = data[userIdx];
		int userId = userSampleDataPt[USER_ID_IDX];
		double *userVec = userVectors[userId - 1];

		int mrIdx = zMRSample[i1];
		int *mrSampleDataPt = data[mrIdx];
		int movieId = mrSampleDataPt[MOVIE_ID_IDX];
		int movieRating = mrSampleDataPt[MOVIE_RATING_IDX];
		double *mrVec = movieRatingVectors[movieId - 1][movieRating - 1];

		double zVal = exp(-getDistanceSquared(userVec, mrVec, dimensions));

		z += zVal;
		zValues[i1] = zVal;

		//Calculate dist2
		int *dist2DataPt = data[dataIndexes[i1]];
		int dist2UserId = dist2DataPt[USER_ID_IDX];
		int dist2MovieId = dist2DataPt[MOVIE_ID_IDX];
		int dist2MovieRating = dist2DataPt[MOVIE_RATING_IDX];

		double *dist2UserVector = userVectors[dist2UserId - 1];
		double *dist2MRVector = movieRatingVectors[movieId - 1][movieRating - 1];

		double dist2 = getDistanceSquared(dist2UserVector, dist2MRVector, dimensions);
		averageDist2 += dist2;
		dist2Values[i1] = dist2;
	}

	z /= zSampleSize;
	averageDist2 /= zSampleSize;

	double initialZ = z;

	cout << "Initial z: " << z << endl;

	//Go through number of iterations to move vectors
	for (int iteration = 0; iteration < iterations; iteration++) {
		random_shuffle(dataIndexes.begin(), dataIndexes.end());

		int reportNum = numDataPoints / 100;

		cout << "Starting iteration " << iteration + 1 << endl;

		for (int dataIdx = 0; dataIdx < numDataPoints; dataIdx++) {
			int idx = dataIndexes[dataIdx];

			int *dataPt = data[idx];

			int userId = dataPt[USER_ID_IDX];
			int movieId = dataPt[MOVIE_ID_IDX];
			int movieRating = dataPt[MOVIE_RATING_IDX];

			//Update cumulative counts
			userCumulativeCounts[userId - 1]++;
			movieRatingCumulativeCounts[movieId - 1][movieRating - 1]++;

			//Get the vectors and calculate norm^2 and eta for this data point
			double *userVec = userVectors[userId - 1];
			double userVecNorm2 = norm2(userVec, dimensions);
			double userEta = calculateEta(eta, phiUser, userCumulativeCounts[userId - 1]);

			double *movieRatingVec = movieRatingVectors[movieId - 1][movieRating - 1];
			double movieRatingVecNorm2 = norm2(movieRatingVec, dimensions);
			double mrEta = calculateEta(eta, phiMR, movieRatingCumulativeCounts[movieId - 1][movieRating - 1]);

			//Update the average dist2 value
			double newDist2 = getDistanceSquared(userVec, movieRatingVec, dimensions);
			double oldDist2 = dist2Values[oldestIdx];
			averageDist2 += (newDist2 - oldDist2) / zSampleSize;

			dist2Values[oldestIdx] = newDist2;

			//Get random new vectors to update z with
			int randomUserDataIdx = randomDataPoint(random);
			dataPt = data[randomUserDataIdx];
			int randomUserId = dataPt[USER_ID_IDX];
			double *randomUserVec = userVectors[randomUserId - 1];

			int randomMRDataIdx = randomDataPoint(random);
			dataPt = data[randomMRDataIdx];
			int randomMovieId = dataPt[MOVIE_ID_IDX];
			int randomMovieRating = dataPt[MOVIE_RATING_IDX];
			double *randomMRVec = movieRatingVectors[randomMovieId - 1][randomMovieRating - 1];

			//Get more random vectors for user-user and mr-mr repulsion
			int randomUserDataIdx2 = randomDataPoint(random);
			dataPt = data[randomUserDataIdx2];
			int randomUserId2 = dataPt[USER_ID_IDX];
			double *randomUserVec2 = userVectors[randomUserId - 1];

			int randomMRDataIdx2 = randomDataPoint(random);
			dataPt = data[randomMRDataIdx2];
			int randomMovieId2 = dataPt[MOVIE_ID_IDX];
			int randomMovieRating2 = dataPt[MOVIE_RATING_IDX];
			double *randomMRVec2 = movieRatingVectors[randomMovieId2 - 1][randomMovieRating2 - 1];

			double **movieRatingsArr = movieRatingVectors[movieRating];

			//Move the vectors toward each other, and away from the randomly chosen vectors
			moveVectors(userVec, movieRatingVec, movieRating, movieRatingsArr, eta, userEta, mrEta, randomUserVec, randomUserVec2, randomMRVec, randomMRVec2, dimensions, initialZ);

			//Update z using the random vectors from earlier
			zUserSample[oldestIdx] = randomUserDataIdx;
			zMRSample[oldestIdx] = randomMRDataIdx;

			double oldZVal = zValues[oldestIdx];
			double newZVal = exp(-getDistanceSquared(randomUserVec, randomMRVec, dimensions));

			zValues[oldestIdx] = newZVal;

			z += (newZVal - oldZVal) / zSampleSize;

			oldestIdx++;
			oldestIdx %= zSampleSize;

			if (dataIdx % reportNum == 0) { //Print out Z and the percentage completed of the iteration
				double perc = (double)dataIdx / numDataPoints * 100;
				cout << perc << "%, Z: " << z << endl;

				//Calculate averages for data collection:
				//Two random movie ratings, two random users, a random user and random movie rating, and a user and movie rating from a random data point
				//First, two random movie ratings
				double mrmrAvg = 0;
				for (int i1 = 0; i1 < AVERAGE_SAMPLE_SIZE; i1++) {
					//Pick two random movie ratings
					int index1 = randomDataPoint(random);
					int index2 = randomDataPoint(random);

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

		random_shuffle(dataIndexes.begin(), dataIndexes.end());
		double *error1 = new double[scoreSampleSize];
		double *error2 = new double[scoreSampleSize];

		for (int i1 = 0; i1 < scoreSampleSize; i1++) {
			//Get a random data point
			int idx = dataIndexes[i1];
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

			double mostLikelyP = 0;
			double mostLikelyStar = 0;

			//Go through each star, calculate the probability of the user giving that rating
			for (int star = 0; star < MAX_STARS; star++) {
				double *movieRatingVector = movieVectors[star];
				double d2 = getDistanceSquared(userVector, movieRatingVector, dimensions);

				double p = exp(-d2) * movieRatingCounts[movieId - 1][star];

				if (p > mostLikelyP) {
					mostLikelyP = p;
					mostLikelyStar = star + 1;
				}

				avgStar += (star + 1) * p;
				pTotal += p;
			}

			//Find the average star rating
			avgStar /= pTotal;

			//Calculate the error between our prediction and the actual
			error1[i1] = avgStar - movieRating;
			error2[i1] = mostLikelyStar - movieRating;
		}

		//Calculate the root mean squared error
		double meanSquaredErrorAvg = 0;
		double meanSquaredErrorML = 0;
		for (int i1 = 0; i1 < scoreSampleSize; i1++) {
			double err1 = error1[i1];
			meanSquaredErrorAvg += err1 * err1;

			double err2 = error2[i1];
			meanSquaredErrorML += err2 * err2;
		}
		meanSquaredErrorAvg /= scoreSampleSize;
		meanSquaredErrorML /= scoreSampleSize;

		double scoreAvg = sqrt(meanSquaredErrorAvg);
		double scoreML = sqrt(meanSquaredErrorML);

		cout << "RMS Error for average star: " << scoreAvg << endl;
		cout << "RMS Error for most likely star: " << scoreML << endl;
	}

	return 0;
}

double getDistanceSquared(double a, double b) {
	return pow(a - b, 2);
}

double getDistanceSquared(double *a, double *b, int dimensions) {
	double sum = 0.0;

	for (int i1 = 0; i1 < dimensions; i1++) {
		double aPt = a[i1];
		double bPt = b[i1];

		sum += getDistanceSquared(aPt, bPt);
	}

	return sum;
}

double norm2(double *arr, int length) {
	double sum = 0.0;

	for (int i1 = 0; i1 < length; i1++) {
		double component = arr[i1];
		sum += pow(component, 2.0);
	}

	return sum;
}

double calculateEta(double etaInitial, double phi, int count) {
	return etaInitial * (phi / (phi + count));
}

void moveVectors(double *user, double *mr, double movieRating, double **movieRatingsArr, double etaInitial, double userEta, double mrEta, double *randomUser1, double *randomUser2, double *randomMR1, double *randomMR2, int dimensions, double z) {
	//Attract and repel the vectors
	for (int dimension = 0; dimension < dimensions; dimension++) {
		double userComponent = user[dimension];
		double mrComponent = mr[dimension];
		double randomUserComponent1 = randomUser1[dimension];
		double randomMRComponent1 = randomMR1[dimension];

		//These are for moving user away from user or mr away from mr
		double randomUserComponent2 = randomUser2[dimension];
		double randomMRComponent2 = randomMR2[dimension];

		//Attract the user and movie rating vectors
		double newUserComponent = attract(userComponent, mrComponent, etaInitial);
		double newMRComponent = attract(mrComponent, userComponent, etaInitial);

		//Repel the user away from a random movie rating and the movie rating away from a random user
		newUserComponent = repel(newUserComponent, randomMRComponent1, userEta, z);
		newMRComponent = repel(newMRComponent, randomUserComponent1, mrEta, z);

		//Repel the user away from a random user and the movie rating away from a random movie rating
		//newUserComponent = repel(newUserComponent, randomUserComponent2, userEta, z);
		//newMRComponent = repel(newMRComponent, randomMRComponent2, mrEta, z);

		//Repel this movie rating away from the other movie ratings
		for (int i1 = 0; i1 < MAX_STARS; i1++) {
			//Don't repel it from itself
			if (i1 == movieRating - 1) {
				continue;
			}

			double curMRComponent = movieRatingsArr[i1][dimension];

			newMRComponent = repel(newMRComponent, curMRComponent, mrEta, z);
		}

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
 * Attracts a towards b
 * @param a The vector component to be attracted
 * @param b The vector component to attract to
 * @param eta The eta value to scale by
 * @return The resulting value of a
 */
double attract(double a, double b, double eta) {
	//return ((1 - eta) * a) + (eta * b);
	return a - eta * (a - b);
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

double normalizeUnitLength(double a, double aNorm2) {
	return a / sqrt(aNorm2);
}