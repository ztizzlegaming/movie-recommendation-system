# Netflix Prize 
## Dr. Michael Lamar, Jordan Turley
## Centre College

An implementation of our solution to the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize), a competition by Netflix from 2006 to 2009 allowing anyone to develop an algorithm for predicting user ratings for films.

Our implementation takes a genetic or biological approach to solve this problem. We begin by creating a random vector for each user and each movie rating (each movie has five possible ratings, so each movie gets five vectors). We put these vectors on a high dimensional unit sphere. We go through each user-movie-rating pair and attract the two closer together, as well as repelling these vectors away from random vectors. Depending on the implementation, we either repel away from single random vectors or an average of all of the vectors, and in one implementation, the movie rating is repelled away from the four other movie ratings.

The probability is calculated for each movie rating, and the mean squared error is calculated to see how well our algorithm is doing. The best scores we have achieved are as follows:

Regular: 0.936458  
MR-MR Repel: 0.927208  
Average plus MR-MR Repel: 0.975033