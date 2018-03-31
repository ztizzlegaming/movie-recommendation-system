# Netflix Prize 
## Dr. Michael Lamar, Jordan Turley, Centre College
### Overview

An implementation of our solution to the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize), a competition by Netflix from 2006 to 2009 allowing anyone to develop an algorithm for predicting user ratings for films.

Our implementation takes a genetic or biological approach to solve this problem. We begin by creating a random vector for each user and each movie rating (each movie has five possible ratings, so each movie gets five vectors). We put these vectors on a high dimensional unit sphere. We go through each user-movie-rating pair and attract the two closer together, as well as repelling these vectors away from random vectors. Depending on the implementation, we either repel away from single random vectors or an average of all of the vectors, and in one implementation, the movie rating is repelled away from the four other movie ratings.

The probability is calculated for each movie rating, and the mean squared error is calculated to see how well our algorithm is doing. The best scores we have achieved are as follows:

Regular: 0.936458  
MR-MR Repel: 0.927208  
Average plus MR-MR Repel: 0.975033

## How to use
First, clone this repository using the following code:
```
git clone https://github.com/ztizzlegaming/netflix-prize
```
One must also have the Netflix Prize dataset, which can be found [http](here). For ease, put this file in the same netflix-prize folder.

To compile the code, use the following command and replace main_base.cpp with whatever version you want to use (main_base.cpp, main_mr_mr_repel.cpp, main_average.cpp):
```
g++ main_base.cpp -o netflix -std=c++11 -O3
```

To run, use the following command:
```
./netflix input_base.txt netflix_data_c.bin 100480507
```
The first argument is the input file. The second is the dataset. The third is the number of data points to consider. 100,480,507 is the total number of data points. Fewer data points can be used for the algorithm to run more quickly.