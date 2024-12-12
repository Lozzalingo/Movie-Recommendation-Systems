# Load the configuration file.
source("config.R")

# Use the variable from the config file.
#remove all # (comments)

if (!dir.exists(workingDirectory)) {
  cat("Directory does not exist. Creating directory...\n")
  dir.create(workingDirectory, recursive = TRUE)
  knitr::opts_knit$set(root.dir = workingDirectory)
} else {
  cat("Directory exists.\n")
}

install.packages(c(
  "tidyverse", 
  "caret", 
  "data.table", 
  "kableExtra", 
  "lubridate", 
  "Matrix", 
  "DT", 
  "wordcloud", 
  "RColorBrewer", 
  "ggthemes", 
  "irlba", 
  "recommenderlab", 
  "reticulate", 
  "h2o", 
  "plyr", 
  "dplyr", 
  "reshape2", 
  "stringr", 
  "ggplot2", 
  "keras", 
  "glue", 
  "mlbench", 
  "microbenchmark", 
  "magrittr", 
  "neuralnet", 
  "sparklyr", 
  "keras",
  "tictoc"
))

# Load libraries.
library(tidyverse)
library(caret)
library(data.table)
library(kableExtra)
library(lubridate)
library(Matrix)
library(DT)
library(wordcloud)
library(RColorBrewer)
library(ggthemes)
library(irlba)
library(recommenderlab)
library(recosystem)
library(reticulate)
library(h2o)
require(data.table)
require(foreach)
library(neuralnet)
# Load 'plyr' first, then 'dplyr' to prevent function conflicts.
# This ensures that 'dplyr' functions (like summarise, mutate, etc.) override those in 'plyr'.
library(plyr); library(dplyr)
require(reshape2)
library(stringr)
library(ggplot2)
library(keras)
library(tidyverse)
library(glue)
library(mlbench)
library(microbenchmark)
library(magrittr)
library(neuralnet)
library(sparklyr)
library(tictoc)
source(SlopeOne)

# Load Movielens File.
ratings_file <- file.path(workingDirectory, "ml-10M100K", "ratings.dat")
movies_file <- file.path(workingDirectory, "ml-10M100K", "movies.dat")

# Read the lines from the ratings data file, split them on ::, and convert the data into a dataframe.
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
  stringsAsFactors = FALSE
)

# Lable the columns of the ratings dataframe.
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

# Convert the ratings data into numeric values.
ratings <- ratings %>%
  mutate(
    userId = as.integer(userId),
    movieId = as.integer(movieId),
    rating = as.numeric(rating),
    timestamp = as.integer(timestamp)
  )

# Read the lines from the movies data file, split them on ::, and convert the data into a dataframe.
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
  stringsAsFactors = FALSE
)

# Lable the columns of the movies dataframe.
colnames(movies) <- c("movieId", "title", "genres")

# Convert the movies data into numeric values.
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Combine the dataframes using the movieID.
movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data.
set.seed(42, sample.kind = "Rounding")

# Create a data partition using 10% of the data and return a vector.
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)

# Create the training set using rows where the indices are not in test_index.
edx <- movielens[-test_index, ]

# Create the test set using rows where the indices are in test_index.
temp <- movielens[test_index, ]

# Use semi_join to create a variable with the rows from the original temp dataframe and those that have matching values in both the "movieId" and "userId" columns from the edx dataframe.
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set.
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Remove objects from the global environment.
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Use glimpse to explore rows, columns, and classes

#Test Set

'training data'
glimpse(edx)

###### Validation Set

'testing data'
glimpse(final_holdout_test)

## 2.Exploring The Data 

# Create a variable that groups ratings by whole or half-star values.
ratings_group <- ifelse(edx$rating %in% 1:5, "Whole Star", "Half Star")

# Create a dataframe of ratings and rating groups.
ratings_df <- data.frame(rating = edx$rating, ratings_group)

# Plot a histogram of whole and half ratings against their volume.
ggplot(ratings_df, aes(x = rating, fill = ratings_group)) +
  geom_histogram(binwidth = .5, color = "white", alpha = 0.7) +
  scale_x_continuous(breaks = seq(0, 5, by = 0.5)) +
  scale_fill_manual(values = c("Half Star" = "#533dfc", "Whole Star" = "#3ddffc")) +
  labs(x = "Rating", y = "Number of Ratings", caption = "Source Data: edx set") +
  ggtitle("Rating by Number of Ratings") +
  theme_minimal() +
  theme(
    legend.position = "top",
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_blank(),
    legend.text = element_text(size = 10)
  ) +
  labs(caption = "Source Data: testing data") 

# Split movie genres using regular expression and arrange them by their count.
top_genres <- edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  dplyr::summarize(count = n()) %>%
  dplyr::arrange(desc(count))

# Separate the movie year from the movie title.
edx_title_year <- edx %>%
  mutate(
    year = gsub("\\(|\\)", "", regmatches(title, regexpr("\\(\\d{4}\\)", title))),
    title = sub("\\s*\\(\\d{4}\\)", "", title)
  )

final_holdout_year <- final_holdout_test %>%
  mutate(
    year = gsub("\\(|\\)", "", regmatches(title, regexpr("\\(\\d{4}\\)", title))),
    title = sub("\\s*\\(\\d{4}\\)", "", title)
  )

# Group the movies by their release year and arrange by the highest count.
top_years <- edx_title_year %>%
  group_by(year) %>%
  dplyr::summarize(count = n()) %>%
  dplyr::arrange(desc(count))

# Group the movies by their title and arrange by the highest count.
top_titles <- edx_title_year %>%
  group_by(title) %>%
  dplyr::summarize(count = n()) %>%
  dplyr::arrange(desc(count))

#Create interactive HTML tables to explore the data for top genres, years, and titles

# Top Genres

datatable(
  top_genres,
  rownames = FALSE, filter = "top", options = list(pageLength = 5, scrollX = TRUE)
) %>%
  formatRound("count", digits = 0, interval = 3, mark = ",")

# Top Years

datatable(
  top_years,
  rownames = FALSE, filter = "top", options = list(pageLength = 5, scrollX = TRUE)
) %>%
  formatRound("count", digits = 0, interval = 3, mark = ",")

# Top Titles

datatable(
  top_titles,
  rownames = FALSE, filter = "top", options = list(pageLength = 5, scrollX = TRUE)
) %>%
  formatRound("count", digits = 0, interval = 3, mark = ",")

# Extract the top 20 movies based on the number of ratings.
top_title <- edx %>%
  group_by(title) %>%
  dplyr::summarize(count = n()) %>%
  top_n(20, count) %>%
  dplyr::arrange(desc(count))

# Create a bar plot using ggplot2 to visualise the top 20 movies based on ratings.
top_title %>%
  ggplot(aes(x = reorder(title, count), y = count)) +
  geom_bar(stat = "identity", fill = "#3ddffc") +
  geom_text(aes(label = str_sub(title, start = 1)), hjust = 1, size = 2.1, color = "black") +
  coord_flip(y = c(0, 35000)) +
  labs(x = "", y = "Number of Ratings") +
  labs(title = "Top 20 Movies Based on Number of Ratings", caption = "Source Data: edx set") +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.title.y = element_text(size = 12)) +
  theme(axis.title.x = element_text(size = 12))

# Calculate summary statistics for genres with a minimum of 100,000 ratings.
# Each point on the plot represents a genre, and the y-axis represents the average rating.
# Error bars indicate the range within which the true average rating is likely to fall (mean ± 2 * standard error).
# Genres with higher average ratings and larger error bars may have more variability in audience opinions.
edx_title_year %>%
  group_by(genres) %>%
  dplyr::summarise(
    N = n(),
    avg = mean(rating),
    SE = sd(rating) / sqrt(N)
  ) %>%
  filter(N >= 50000) %>%
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = reorder(genres, avg), y = avg, ymin = avg - 2 * SE, ymax = avg + 2 * SE)) +
  geom_point() +
  geom_errorbar(color = "grey", linewidth = 0.7) +
  scale_fill_brewer(palette = "Set3") +
  theme_minimal() +
  labs(title = "Error Bar Plots by Genres", caption = "Source Data: edX Set", x = "Genres", y = "Avg") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        aspect.ratio = 0.6)

# Calculate summary statistics for movie years with a minimum of 100,000 ratings.
edx_title_year %>%
  group_by(year) %>%
  dplyr::summarize(n = n(), avg = mean(rating), se = sd(rating) / sqrt(n())) %>%
  filter(n >= 20000) %>%
  mutate(year = reorder(year, avg)) %>%
  ggplot(aes(x = year, y = avg, ymin = avg - 2 * se, ymax = avg + 2 * se)) +
  geom_point() +
  geom_errorbar(color = "grey", linewidth = 0.7) +
  scale_fill_brewer(palette = "Set3") +
  theme_minimal() +
  labs(title = "Error Bar Plots by Year", caption = "Source Data: edX Set", x = "Year", y = "Avg") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Calculate the number of distinct users and movies in the edX dataset.
edx %>%
  dplyr::summarise(
    n_users = n_distinct(userId),
    n_movies = n_distinct(movieId)
  )

# Distribution of ratings by movie

# Generate a histogram depicting the distribution of ratings by movie.
edx %>%
  dplyr::count(movieId) %>%
  filter(n < 10000) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "skyblue", fill = "lightblue", alpha = 0.7) +
  ggtitle("Distribution of Ratings by Movie") +
  labs(
    subtitle = "Number of Ratings by MovieId",
    x = "Number of Ratings < 10000",
    y = "Frequency",
    caption = "Source Data: edX Set"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA))

# Generate a histogram depicting the distribution of ratings by movie.
edx %>%
  dplyr::count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "skyblue", fill = "lightblue", alpha = 0.7) +
  scale_x_log10() +
  ggtitle("Distribution of Ratings by Movie") +
  labs(
    subtitle = "Number of Ratings by MovieId",
    x = "Number of Ratings (log scale)",
    y = "Frequency",
    caption = "Source Data: edX Set"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA))

# Distribution of ratings by user

# Generate a histogram depicting the distribution of ratings by user.
edx %>%
  dplyr::count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "skyblue", fill = "lightblue", alpha = 0.7) +
  ggtitle("Distribution of Ratings by User") +
  labs(
    subtitle = "Number of Ratings by UserId",
    x = "Number of Ratings",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA))

# Generate a histogram depicting the distribution of ratings by user.
edx %>%
  dplyr::count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "skyblue", fill = "lightblue", alpha = 0.7) +
  scale_x_log10() +
  ggtitle("Distribution of Ratings by User") +
  labs(
    subtitle = "Number of Ratings by UserId",
    x = "Number of Ratings (log scale)",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA))

# Analyse average ratings over time by extracting weekly timestamps and calculating the mean rating for each week.
edx %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  dplyr::summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point(color = "#4287f5", size = 1) +
  geom_smooth(formula = y ~ x, method = "loess", se = TRUE, color = "#ff6199", linetype = "solid") +
  ggtitle("Average Ratings Over Time") +
  labs(
    subtitle = "Timestamp, Time Unit: Week",
    x = "Date",
    y = "Average Ratings",
    caption = "Source Data: edX Set"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA))

# Filter genres with under 100 ratings.
filtered_genres <- edx %>%
  group_by(genres) %>%
  dplyr::summarize(num_ratings = n()) %>%
  filter(num_ratings > 1)

# Filter the original dataset based on the filtered genres.
filtered_edx <- edx %>%
  filter(genres %in% filtered_genres$genres)

# Plot the histogram with genre names on the x-axis.
ggplot(filtered_genres, aes(x = reorder(genres, desc(num_ratings)), y = num_ratings)) +
  geom_point(color = "skyblue", size = 1) +
  ggtitle("Number of Ratings by Genre") +
  labs(
    subtitle = "Distribution of Ratings by Genre",
    x = "Genre",
    y = "Number of Ratings",
    caption = "Source Data: edX Set"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA)) +
  theme(axis.text.x = element_blank(),
        axis.title.x = element_text())

# Summarise average ratings by genre.
average_ratings <- edx_title_year %>%
  group_by(genres) %>%
  summarize(avg_rating = mean(rating, na.rm = TRUE))

edx_title_year %>%
  group_by(genres) %>%
  dplyr::summarize(rating = mean(rating)) %>%
  ggplot(aes(genres, rating)) +
  geom_point(color = "#4287f5", size = 1) +
  geom_smooth(formula = y ~ x, method = "loess", se = TRUE, color = "#ff6199", linetype = "solid") +
  ggtitle("Average Ratings by Genre") +
  labs(
    subtitle = "Timestamp, Time Unit: Week",
    x = "Genre",
    y = "Average Ratings",
    caption = "Source Data: edX Set"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA)) +
  theme(axis.text.x = element_blank(),
        axis.title.x = element_text())
  
# Analyse average ratings over years by calculating the mean rating for each year.
# Calculate the average rating for each year.
edx_title_year$year <- as.numeric(edx_title_year$year)

# Summarise average ratings by year.
average_ratings <- edx_title_year %>%
  group_by(year) %>%
  summarize(avg_rating = mean(rating, na.rm = TRUE))

edx_title_year %>%
  group_by(year) %>%
  dplyr::summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point(color = "#4287f5", size = 1) +
  geom_smooth(formula = y ~ x, method = "loess", se = TRUE, color = "#ff6199", linetype = "solid") +
  ggtitle("Average Ratings by Release Date") +
  labs(
    subtitle = "Timestamp, Time Unit: Week",
    x = "Date",
    y = "Average Ratings",
    caption = "Source Data: edX Set"
  ) +
  theme_minimal() +
  theme(panel.border = element_rect(colour = "black", fill = NA))

# III. Data Preprocessing

## 1. Matrix Transformation 
# First I created a ratings matrix to start to help us reduce the sparsity problem. 

# Create a copy of the edx_title_year data frame for manipulation without altering the original data.
edx_title_year.copy <- edx_title_year

# Convert the userId and movieId columns to factors for categorical representation.
edx_title_year.copy$userId <- as.factor(edx_title_year.copy$userId)
edx_title_year.copy$movieId <- as.factor(edx_title_year.copy$movieId)

# Convert userId and movieId back to numeric representation.
edx_title_year.copy$userId <- as.numeric(edx_title_year.copy$userId)
edx_title_year.copy$movieId <- as.numeric(edx_title_year.copy$movieId)

# Create a sparse matrix representation of ratings.
sparse_ratings <- sparseMatrix(
  i = edx_title_year.copy$userId,
  j = edx_title_year.copy$movieId,
  x = edx_title_year.copy$rating,
  dims = c(
    length(unique(edx_title_year.copy$userId)),
    length(unique(edx_title_year.copy$movieId))
  ),
)

# Remove the copied data frame to free up memory.
rm(edx_title_year.copy)

# Define constants.
num_users <- 50
num_movies <- 50
num_display_users <- 10

# Suppress row and column names in sparse matrix display.
options(Matrix.print.rownames = FALSE)
options(Matrix.print.colnames = FALSE)

# Display a subset of the sparse ratings.
sparse_ratings[1:num_display_users, 1:num_display_users]

# Convert rating matrix into a recommenderlab sparse matrix.
ratingMat <- new("realRatingMatrix", data = sparse_ratings)
ratingMat

# Compute user similarity using cosine similarity for the first 200 users.
similarity_users <- similarity(ratingMat[1:200,], 
                               method = "cosine", 
                               which = "users")

# Visualise user similarity using an image plot.
image(as.matrix(similarity_users), main = "User Similarity")

# Compute movie similarity using cosine similarity for the first 50 movies.
similarity_movies <- similarity(ratingMat[,1:50], 
                                method = "cosine", 
                                which = "items")

# Visualise movie similarity using an image plot.
image(as.matrix(similarity_movies), main = "Movies Similarity")
  
# 2. Dimension Reduction
# In the next stage of my data processing, I need to tighten up the sparsity of the data. I can do this using techniques that reduce the dimentionality of the Euclean spaces. I can use Principal Component Analysis (PCA) to reduce the linear dimnetionality of the data, and Singular Value Decomposition to factorise, rotate and rescale the data in order to extrapolate some of the more important patterns, with lower dimentionality. 

# Set seed for reproducibility.
set.seed(42)

# Perform incremental randomised SVD on the sparse_ratings matrix.
suppressMessages({
  svd_result <- irlba(sparse_ratings, tol=1e-4, verbose=TRUE, nv = 100, maxit = 1000)
})

# Plot singular values for the User-Movie Matrix.
plot(svd_result$d, pch=20, col = "blue", cex = 1.5, xlab='Singular Value', ylab='Magnitude', 
     main = "Singular Values for User-Movie Matrix")

# Calculate the percentage of total sum of squares for the first 6, 12, and 20 singular values.
all_sing_sq <- sum(svd_result$d^2)
first_6 <- sum(svd_result$d[1:6]^2)
print(first_6/all_sing_sq)

first_12 <- sum(svd_result$d[1:12]^2)
print(first_12/all_sing_sq)

first_20 <- sum(svd_result$d[1:20]^2)
print(first_20/all_sing_sq)

# Calculate the cumulative percentage of total sum of squares for each singular value.
perc_vec <- NULL
for (i in 1:length(svd_result$d)) {
  perc_vec[i] <- sum(svd_result$d[1:i]^2) / all_sing_sq
}

# Plot the cumulative percentage against singular values and a horizontal line at 90%.
plot(perc_vec, pch=20, col = "blue", cex = 1.5, xlab='Singular Value', 
     ylab='% of Sum of Squares of Singular Values', main = "Choosing k for Dimensionality Reduction")
lines(x = c(0,100), y = c(.90, .90))

# Determine the optimal value for k:
# To find k, calculate the length of the vector derived from the cumulative sum of squares.
# The chosen k corresponds to the number of singular values needed to capture 90% of the total sum of squares,
# excluding any values that exceed the 0.90 threshold.

#Find the optimal k value.
k = length(perc_vec[perc_vec <= .90])
cat("Optimal k Value:", k, "\n")

# Decompose Y into matrices U, D, and V.
U_k <- svd_result$u[, 1:k]
D_k <- Diagonal(x = svd_result$d[1:k])
V_k <- t(svd_result$v)[1:k, ]

# Display dimensions.
cat("Dimensions of U_k:", dim(U_k), "\n")
cat("Dimensions of D_k:", dim(D_k), "\n")
cat("Dimensions of V_k:", dim(V_k), "\n")

# 3. Relevant Data
# During the data analysis process, I observed significant left skewness in both the user rating counts and movie ratings, indicating that a substantial portion of the data holds limited predictive value. To mitigate computational load without sacrificing predictive accuracy, I propose implementing a threshold for the minimum number of ratings required for inclusion in my models, both for users and movies.

# Determine the minimum number of movies and users.
min_n_movies <- round(quantile(rowCounts(ratingMat), 0.90))
min_n_users <- round(quantile(colCounts(ratingMat), 0.75))

cat("Minimum number of movies (90th percentile):", min_n_movies, "\n")
cat("Minimum number of users (90th percentile):", min_n_users, "\n")

# Extract ratings for movies and users meeting the criteria.
ratings_movies <- ratingMat[
  rowCounts(ratingMat) > min_n_movies,
  colCounts(ratingMat) > min_n_users
]

# Display the resulting ratings matrix.
ratings_movies

# IV. Models and Results

#Define the RMSE function.
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }

## 1. Linear Regression

# Define the RMSE function.
calculate_RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Calculate the global average rating.
mu <- mean(edx$rating)
cat("Mu:", mu, "\n")

# Calculate training time and predict ratings.
training_time_mu <- system.time({
predicted_ratings <- final_holdout_test %>%
  mutate(pred = mu) %>%
  pull(pred)
})

# Calculate RMSE.
rmse_mu <- calculate_RMSE(final_holdout_test$rating, predicted_ratings)

# Calculate model size.
model_size_mu <- round(
  sum(
    object.size(mu),
    object.size(predicted_ratings)
  ) / (1024^2),  # Convert to MB
  4
)

# Save the results of Mu model.
saveRDS(list(rmse = rmse_mu, time = training_time_mu["elapsed"], size = model_size_mu), file = "mu_model.rds")

# Load model results.
mu_model <- readRDS(file.path(workingDirectory, "mu_model.rds"))

# Print results.
cat("RMSE for Mu:", mu_model$rmse, "\n")
cat("Training Time:", round(mu_model$time["elapsed"], 4), "sec\n")
cat("Model Size:", mu_model$size, "MB")

# Movie effect.
movie_avgs <- edx %>%
  group_by(movieId) %>%
  dplyr::summarize(b_i = mean(rating - mu))

# Calculate training time and predict ratings.
training_time_movie <- system.time({
  predicted_ratings_bi <- final_holdout_test %>%
    left_join(movie_avgs, by = "movieId") %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
})

# Calculate RMSE for movie effect.
rmse_model_movie <- calculate_RMSE(final_holdout_test$rating, predicted_ratings_bi)

# Calculate model size.
model_size_movie_effect <- round(
  sum(
    object.size(movie_avgs),
    object.size(predicted_ratings_bi)
  ) / (1024^2),  # Convert to MB.
  4
)

# Save the results of Movie effect model.
saveRDS(list(rmse = rmse_model_movie, time = training_time_movie["elapsed"], size = model_size_movie_effect), file = "movie_effect_model.rds")

# Load model results.
movie_effect_model <- readRDS(file.path(workingDirectory, "movie_effect_model.rds"))

# Print results.
cat("RMSE for Movie Effect:", movie_effect_model$rmse, "\n")
cat("Training Time:", round(movie_effect_model$time["elapsed"], 4), "sec\n")
cat("Model Size:", movie_effect_model$size, "MB")

# Movie + User effect.
user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  dplyr::summarize(b_u = mean(rating - mu - b_i))

# Calculate training time and predict ratings.
training_time_movie_user <- system.time({
  predicted_ratings_bu <- final_holdout_test %>%
    left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
})

# Calculate RMSE for user effect.
rmse_model_movie_user <- calculate_RMSE(final_holdout_test$rating, predicted_ratings_bu)

# Calculate model size.
model_size_movie_user <- round(
  sum(
    object.size(movie_avgs),
    object.size(user_avgs),
    object.size(predicted_ratings_bu)
  ) / (1024^2),  # Convert to MB.
  4
)

# Save the results of Movie + User effect model.
saveRDS(list(rmse = rmse_model_movie_user, time = training_time_movie_user["elapsed"], size = model_size_movie_user), file = "movie_user_effect_model.rds")

# Load model results.
movie_user_effect_model <- readRDS(file.path(workingDirectory, "movie_user_effect_model.rds"))

# Print results.
cat("RMSE for Movie + User Effect:", movie_user_effect_model$rmse, "\n")
cat("Training Time:", round(movie_user_effect_model$time["elapsed"], 4), "sec\n")
cat("Model Size:", movie_user_effect_model$size, "MB")

# Movie + User + Time effect.
final_holdout_year <- final_holdout_year %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

# Calculate time averages.
time_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  dplyr::summarize(b_t = mean(rating - mu - b_i - b_u))

# Calculate training time and predict ratings with time effect.
training_time_movie_user_time <- system.time({
  predicted_ratings_bt <- final_holdout_year %>%
    left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    left_join(time_avgs, by = "date") %>%
    mutate(pred = mu + b_i + b_u + b_t) %>%
    pull(pred)
})

# Calculate RMSE for time effect.
rmse_model_movie_user_time <- calculate_RMSE(final_holdout_test$rating, predicted_ratings_bt)

# Calculate model size.
model_size_movie_user_time <- round(
  sum(
    object.size(movie_avgs),
    object.size(user_avgs),
    object.size(time_avgs),
    object.size(predicted_ratings_bt)
  ) / (1024^2),  # Convert to MB.
  4
)

# Save the results of Movie + User + Time effect model.
saveRDS(list(rmse = rmse_model_movie_user_time, time = training_time_movie_user_time["elapsed"], size = model_size_movie_user_time), file = "movie_user_time_effect_model.rds")

# Load model results.
movie_user_time_effect_model <- readRDS(file.path(workingDirectory, "movie_user_time_effect_model.rds"))

# Print results.
cat("RMSE for Movie + User + Time Effect:", movie_user_time_effect_model$rmse, "\n")
cat("Training Time:", round(movie_user_time_effect_model$time["elapsed"], 4), "sec\n")
cat("Model Size:", movie_user_time_effect_model$size, "MB")

# Calculate genre averages.
genre_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  dplyr::summarize(b_g = mean(rating - mu - b_i - b_u))

# Calculate training time and predict ratings with genre effect.
training_time_movie_user_genre <- system.time({
  predicted_ratings_bg <- final_holdout_test %>%
    left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    left_join(genre_avgs, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
})

# Calculate RMSE for genre effect.
rmse_model_movie_user_genre <- calculate_RMSE(final_holdout_test$rating, predicted_ratings_bg)

# Calculate model size.
model_size_movie_user_genre <- round(
  sum(
    object.size(movie_avgs),
    object.size(user_avgs),
    object.size(genre_avgs),
    object.size(predicted_ratings_bg)
  ) / (1024^2),  # Convert to MB.
  4
)

# Save the results of Movie + User + Genre effect model.
saveRDS(list(rmse = rmse_model_movie_user_genre, time = training_time_movie_user_genre["elapsed"], size = model_size_movie_user_genre), file = "movie_user_genre_effect_model.rds")

# Load model results.
movie_user_genre_effect_model <- readRDS(file.path(workingDirectory, "movie_user_genre_effect_model.rds"))

# Print results.
cat("RMSE for Movie + User + Genre Effect: ", movie_user_genre_effect_model$rmse, "\n")
cat("Training Time:", round(movie_user_genre_effect_model$time["elapsed"], 4), "sec\n")
cat("Model Size:", movie_user_genre_effect_model$size, "MB")

# Calculate genre averages.
year_avgs <- edx_title_year %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(year) %>%
  dplyr::summarize(b_y = mean(rating - mu - b_i - b_u))

year_avgs <- year_avgs %>%
  mutate(year = as.character(year))

# Calculate training time and predict ratings with genre, time, and year effects.
training_time_movie_user_year <- system.time({
  predicted_ratings_bg_bt_yr <- final_holdout_year %>%
    left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    left_join(year_avgs, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_y) %>%
    pull(pred)
})

# Calculate RMSE for genre, time, and year effects.
rmse_model_movie_user_year <- calculate_RMSE(final_holdout_test$rating, predicted_ratings_bg_bt_yr)

# Calculate model size.
model_size_movie_user_year <- round(
  sum(
    object.size(movie_avgs),
    object.size(user_avgs),
    object.size(year_avgs),
    object.size(predicted_ratings_bg_bt_yr)
  ) / (1024^2),  # Convert to MB.
  4
)

# Save the results of Movie + User + Year effect model.
saveRDS(list(rmse = rmse_model_movie_user_year, time = training_time_movie_user_year["elapsed"], size = model_size_movie_user_year), file = "movie_user_year_effect_model.rds")

# Load model results.
movie_user_year_effect_model <- readRDS(file.path(workingDirectory, "movie_user_year_effect_model.rds"))

# Print results.
cat("RMSE for Movie + User + Year Effect: ", movie_user_year_effect_model$rmse, "\n")
cat("Training Time:", round(movie_user_year_effect_model$time["elapsed"], 4), "sec\n")
cat("Model Size:", movie_user_year_effect_model$size, "MB")

# Calculate training time and predict ratings with genre, time, and year effects.
training_time_movie_user_genre_time_year <- system.time({
  predicted_ratings_bg_bt_yr_dt_gr <- final_holdout_year %>%
    left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    left_join(genre_avgs, by = "genres") %>%
    left_join(time_avgs, by = "date") %>%
    left_join(year_avgs, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
    pull(pred)
})

# Calculate RMSE for genre, time, and year effects.
rmse_model_movie_user_genre_time_year <- calculate_RMSE(final_holdout_test$rating, predicted_ratings_bg_bt_yr_dt_gr)

# Calculate model size.
model_size_movie_user_genre_time_year <- round(
  sum(
    object.size(movie_avgs),
    object.size(user_avgs),
    object.size(time_avgs),
    object.size(genre_avgs),
    object.size(year_avgs),
    object.size(predicted_ratings_bg_bt_yr_dt_gr)
  ) / (1024^2),  # Convert to MB.
  4
)

# Save the results of Movie + User + Time + Genre + Year effect model.
saveRDS(list(rmse = rmse_model_movie_user_genre_time_year, time = training_time_movie_user_genre_time_year["elapsed"], size = model_size_movie_user_genre_time_year), file = "movie_user_time_genre_year_effect_model.rds")

# Load model results.
movie_user_time_genre_year_effect_model <- readRDS(file.path(workingDirectory, "movie_user_time_genre_year_effect_model.rds"))

# Print results.
cat("RMSE for Movie + User + Genre + Time + Year Effect: ", movie_user_time_genre_year_effect_model$rmse, "\n")
cat("Training Time:", round(movie_user_time_genre_year_effect_model$time["elapsed"], 4), "sec\n")
cat("Model Size:", movie_user_time_genre_year_effect_model$size, "MB")

movieID: $$\hat{b}_i(\lambda) = \frac{1}{\lambda + n_i} \sum_{u=1}^{n_i} \left(Y_{u,i} - \hat{\mu}\right)$$
userID: $$\hat{b}_u(\lambda) = \frac{1}{\lambda + n_i} \sum_{u=1}^{n_i} \left(Y_{u,i} - \hat{\mu} - \hat{b}_i \right)$$. 

# Regularisation.
# Set up lambda values for cross-validation.
lambdas <- seq(0, 10, 0.25)

# Initialise vectors to store RMSE and model sizes
rmses_movieID <- numeric(length(lambdas))
model_size_movie_user_reg <- numeric(length(lambdas))

# Function to calculate RMSE with regularisation.
calculate_RMSE_reg <- function(edx, final_holdout_year, lambda) {
  
  mu_reg <- mean(edx$rating)
  
  b_i_reg <- edx %>%
    group_by(movieId) %>%
    dplyr::summarize(b_i_reg = sum(rating - mu_reg) / (n() + lambda))
  
  b_u_reg <- edx %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(userId) %>%
    dplyr::summarize(b_u_reg = sum(rating - b_i_reg - mu_reg) / (n() + lambda))
  
  predicted_ratings_b_i_u <- final_holdout_test %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg) %>%
    pull(pred)
  
  model_size_movie_user_reg <<- round(  # Using <<- to assign it globally outside of the function. 
    sum(
      object.size(mu_reg),
      object.size(b_i_reg),
      object.size(b_u_reg),
      object.size(predicted_ratings_b_i_u)
    ) / (1024^2),  # Convert to MB.
    4
  )
  
  return(list(RMSE = RMSE(final_holdout_test$rating, predicted_ratings_b_i_u), Model_Size = model_size_movie_user_reg))
}

# Calculate training time and RMSE for different lambdas using sapply.
training_time_movie_user_reg <- system.time({
  for (i in seq_along(lambdas)) {
    result <- calculate_RMSE_reg(edx, final_holdout_test, lambdas[i])
    rmses_movieID[i] <- result$RMSE
    model_size_movie_user_reg[i] <- result$Model_Size
  }
})

# Plot RMSE values for different lambdas.
ggplot(data = data.frame(lambdas, rmses_movieID), aes(x = lambdas, y = rmses_movieID)) +
  geom_point(color = "blue") +
  geom_line(color = "red") +
  labs(
    title = "RMSE vs. Lambda",
    x = "Lambda",
    y = "RMSE"
  ) +
  theme_minimal()

rmses_userID <- numeric(length(lambdas))
model_size_user_movie_reg <- numeric(length(lambdas))

# Function to calculate RMSE with regularisation.
calculate_RMSE_reg <- function(edx, final_holdout_year, lambda) {
  
  mu_reg <- mean(edx$rating)
  
  b_i_reg <- edx %>%
    group_by(userId) %>%
    dplyr::summarize(b_i_reg = sum(rating - mu_reg) / (n() + lambda))
  
  b_u_reg <- edx %>%
    left_join(b_i_reg, by = "userId") %>%
    group_by(movieId) %>%
    dplyr::summarize(b_u_reg = sum(rating - b_i_reg - mu_reg) / (n() + lambda))
  
  predicted_ratings_b_i_u <- final_holdout_test %>%
    left_join(b_i_reg, by = "userId") %>%
    left_join(b_u_reg, by = "movieId") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg) %>%
    pull(pred)
  
  model_size_user_movie_reg <<- round(  # Using <<- to assign it globally outside of the function. 
    sum(
      object.size(mu_reg),
      object.size(b_i_reg),
      object.size(b_u_reg),
      object.size(predicted_ratings_b_i_u)
    ) / (1024^2),  # Convert to MB
    4
  )
  
  return(list(RMSE = RMSE(final_holdout_test$rating, predicted_ratings_b_i_u), Model_Size = model_size_user_movie_reg))
}

# Calculate training time and RMSE for different lambdas using sapply.
training_time_user_movie_reg <- system.time({
  for (i in seq_along(lambdas)) {
    result <- calculate_RMSE_reg(edx, final_holdout_test, lambdas[i])
    rmses_userID[i] <- result$RMSE
    model_size_user_movie_reg[i] <- result$Model_Size
  }
})

# Plot RMSE values for different lambdas.
ggplot(data = data.frame(lambdas, rmses_userID), aes(x = lambdas, y = rmses_userID)) +
  geom_point(color = "blue") +
  geom_line(color = "red") +
  labs(
    title = "RMSE vs. Lambda",
    x = "Lambda",
    y = "RMSE"
  ) +
  theme_minimal()

# Find the optimal lambda for movieID.
optimal_lambda_all <- lambdas[which.min(rmses_movieID)]
cat("Optimal Lambda: ", optimal_lambda_all, "\n")

# Find the optimal lambda for UserID.
optimal_lambda_all <- lambdas[which.min(rmses_userID)]
cat("Optimal Lambda: ", optimal_lambda_all, "\n")

# Calculate RMSE for the full model with the optimal lambda.
rmse_regularised_movieID <- min(rmses_movieID)

# Save the results of Regularised Movie + User Effect model
saveRDS(list(rmse = rmse_regularised_movieID, time = training_time_movie_user_reg["elapsed"], size = model_size_movie_user_reg[1]), file = "regularised_movie_user_effect_model.rds")

# Calculate RMSE for the full model with the optimal lambda.
rmse_regularised_userID <- min(rmses_userID)

# Save the results of Regularised User + Movie effect model
saveRDS(list(rmse = rmse_regularised_userID, time = training_time_user_movie_reg["elapsed"], size = model_size_user_movie_reg[1]), file = "regularised_user_movie_effect_model.rds")

# Load model results.
regularised_movie_user_effect_model <- readRDS(file.path(workingDirectory, "regularised_movie_user_effect_model.rds"))

# Print results.
cat("RMSE for movieId with Regularisation:", regularised_movie_user_effect_model$rmse, "\n")
cat("Training Time:", round(regularised_movie_user_effect_model$time["elapsed"], 4), "sec\n")
cat("Model size:", regularised_movie_user_effect_model$size, "MB\n\n")

# Load model results.
regularised_user_movie_effect_model <- readRDS(file.path(workingDirectory, "regularised_user_movie_effect_model.rds"))

# Print results.
cat("RMSE for userId with Regularisation: ", regularised_user_movie_effect_model$rmse, "\n")
cat("Training Time:", round(regularised_user_movie_effect_model$time["elapsed"], 4), "sec\n")
cat("Model size:", regularised_user_movie_effect_model$size, "MB", "\n")

final_holdout_time <- final_holdout_year %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

edx_title_time <- edx_title_year %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

# Initialise vectors to store RMSE and model sizes.
rmses_all <- numeric(length(lambdas))
model_size_all_reg <- numeric(length(lambdas))

# Convert year column to character type in both data frames
edx_title_time <- edx_title_time %>%
  mutate(year = as.character(year))

# Function to calculate RMSE with regularisation.
calculate_RMSE_reg_all <- function(edx_title_time, final_holdout_year, lambda) {
  
  mu_reg <- mean(edx_title_time$rating)
  
  b_i_reg <- edx_title_time %>%
    group_by(movieId) %>%
    dplyr::summarize(b_i_reg = sum(rating - mu_reg) / (n() + lambda))
  
  b_u_reg <- edx_title_time %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(userId) %>%
    dplyr::summarize(b_u_reg = sum(rating - b_i_reg - mu_reg) / (n() + lambda))
  
  b_g_reg <- edx_title_time %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(genres) %>%
    dplyr::summarize(b_g_reg = sum(rating - b_i_reg - mu_reg) / (n() + lambda))
  
  b_y_reg <- edx_title_time %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(year) %>%
    dplyr::summarize(b_y_reg = sum(rating - b_i_reg - mu_reg) / (n() + lambda))
  
  b_t_reg <- edx_title_time %>%
    left_join(b_i_reg, by = "movieId") %>%
    group_by(date) %>%
    dplyr::summarize(b_t_reg= sum(rating - b_i_reg - mu_reg) / (n() + lambda))
  
  predicted_ratings_b_i_u <- final_holdout_time %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_g_reg, by = "genres") %>%
    left_join(b_y_reg, by = "year") %>%
    left_join(b_t_reg, by = "date") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg + b_g_reg+ b_y_reg + b_t_reg) %>%
    pull(pred)
  
  model_size_all_reg <<- round(  # Using <<- to assign it globally outside of the function.
    sum(
      object.size(mu_reg),
      object.size(b_i_reg),
      object.size(b_u_reg),
      object.size(b_g_reg),
      object.size(b_y_reg),
      object.size(b_t_reg),
      object.size(predicted_ratings_b_i_u)
    ) / (1024^2),  # Convert to MB
    4
  )
  
  return(list(RMSE = RMSE(final_holdout_test$rating, predicted_ratings_b_i_u), Model_Size = model_size_all_reg))
}

# Calculate training time and RMSE for different lambdas using sapply.
training_time_all_reg <- system.time({
  for (i in seq_along(lambdas)) {
    result <- calculate_RMSE_reg_all(edx_title_time, final_holdout_test, lambdas[i])
    rmses_all[i] <- result$RMSE
    model_size_all_reg[i] <- result$Model_Size
  }
})

# Plot RMSE values for different lambdas.
ggplot(data = data.frame(lambdas, rmses_all), aes(x = lambdas, y = rmses_all)) +
  geom_point(color = "blue") +
  geom_line(color = "red") +
  labs(
    title = "RMSE vs. Lambda",
    x = "Lambda",
    y = "RMSE"
  ) +
  theme_minimal()

# Find the optimal lambda.
optimal_lambda_all <- lambdas[which.min(rmses_all)]
cat("Optimal Lambda: ", optimal_lambda_all, "\n")

rmse_regularised_all <- min(rmses_all)
# Save the results of Regularised Movie + User + Time + Genre + Year effect model
saveRDS(list(rmse = rmse_regularised_all, time = training_time_all_reg["elapsed"], size = model_size_all_reg[1]), file = "regularised_all_effect_model.rds")

# Load model results.
regularised_all_effect_model <- readRDS(file.path(workingDirectory, "regularised_all_effect_model.rds"))

# Calculate results for full model with the optimal lambda.
rmse_regularised_userID <- min(rmses_userID)
cat("RMSE for Full Model with Regularisation: ", regularised_all_effect_model$rmse, "\n")
cat("Training Time:", round(regularised_all_effect_model$time["elapsed"], 4), "sec\n")
cat("Model size:", regularised_all_effect_model$size, "MB", "\n")

# Summarise the RMSE values on the validation set for the linear regression models.
rmse_results <- data.frame(
  Method = c("Mu", "Movie Effect", "Movie + User Effects", "Movie + User + Time Effects", "Movie + User + Genre Effects", "Movie + User + Year Effects", "Movie + User + Time + Genre + Year Effects", "Regularised Movie + User Effect", "Regularised User + Movie Effect", "Regularised Movie + User + Time + Genre + Year Effect"),
  
  RMSE = c(mu_model$rmse, movie_effect_model$rmse, movie_user_effect_model$rmse, movie_user_time_effect_model$rmse, movie_user_genre_effect_model$rmse, movie_user_year_effect_model$rmse, movie_user_time_genre_year_effect_model$rmse, regularised_movie_user_effect_model$rmse, regularised_user_movie_effect_model$rmse, regularised_all_effect_model$rmse),
  
  Time = c(mu_model$time, movie_effect_model$time, movie_user_effect_model$time, movie_user_time_effect_model$time, movie_user_genre_effect_model$time, movie_user_year_effect_model$time, movie_user_time_genre_year_effect_model$time, regularised_movie_user_effect_model$time, regularised_user_movie_effect_model$time, regularised_all_effect_model$time),
  
  Size = c(mu_model$size, movie_effect_model$size, movie_user_effect_model$size, movie_user_time_effect_model$size, movie_user_genre_effect_model$size, movie_user_year_effect_model$size, movie_user_time_genre_year_effect_model$size, regularised_movie_user_effect_model$size, regularised_user_movie_effect_model$size, regularised_all_effect_model$size)
)

# Rename the columns to replace full stops with spaces.
colnames(rmse_results) <- gsub("Time", "Time (sec)", colnames(rmse_results))
colnames(rmse_results) <- gsub("Size", "Size (MB)", colnames(rmse_results))

# Display the results in an HTML table using the kable function.
kable(rmse_results, "html") %>%
  kable_styling(
    bootstrap_options = c("striped", "bordered", "hover"),
    full_width = FALSE,
    position = "center"
  ) %>%
  column_spec(1, bold = TRUE, color = "black") %>%
  column_spec(2, bold = TRUE, color = "white", background = "#ff2b76") %>%
  column_spec(3, bold = TRUE, color = "black", background = "#affaf9") %>%
  column_spec(4, bold = TRUE, color = "black", background = "#ffce8f") %>%
  row_spec(0, extra_css = "text-align: left;") %>%
  add_header_above(c("Linear Regression" = 4)) 

## 2. Recommender Engines

# Create a POPULAR recommender model.
model_popular <- Recommender(ratings_movies, method = "POPULAR", param = list(normalize = "center"))

# Example prediction on the first 10 users.
predictions_popular <- predict(model_popular, ratings_movies[1:10], type = "ratings")
as(predictions_popular, "matrix")[, 1:10]

# Set the seed for reproducibility.
set.seed(42)

# Create an evaluation scheme with a 60-40 train-test split.
evaluation_scheme <- evaluationScheme(ratings_movies, method = "split", train = 0.6, given = -2)

# Exclude 5 ratings of 30% of users for testing.
model_popular <- Recommender(getData(evaluation_scheme, "train"), 
                             method = "POPULAR")

# Make predictions on the test set.
predictions_popular <- predict(model_popular, getData(evaluation_scheme, "known"), type = "ratings")

# Calculate RMSE for the POPULAR algorithm.
rmse_popular <- calcPredictionAccuracy(predictions_popular, getData(evaluation_scheme, "unknown"))[1]

# Save the POPULAR RMSE
saveRDS(rmse_popular, file = "rmse_popular.rds")

# Load the POPULAR RMSE.
rmse_popular <- readRDS(file.path(workingDirectory, "rmse_popular.rds"))

# Print results.
rmse_popular

# Create a user-based collaborative filtering (UBCF) recommender model using Cosine similarity and 50 neighbors based on cross-validation.
set.seed(42)

user_based_collaborative_filtering_model <- Recommender(getData(evaluation_scheme, "train"),
                          method = "UBCF",
                          param = list(normalize = "center", method = "Cosine", nn = 350, shrink = 10, lambda = 0.01)

)

ubcf_prediction <- predict(user_based_collaborative_filtering_model, getData(evaluation_scheme, "known"), type = "ratings")

rmse_ubcf <- calcPredictionAccuracy(ubcf_prediction, getData(evaluation_scheme, "unknown"))[1]

# Save the UBCF RMSE
saveRDS(rmse_ubcf, file = "rmse_ubcf.rds")

# Load the UBCF RMSE
rmse_ubcf <- readRDS(file.path(workingDirectory, "rmse_ubcf.rds"))

# Print results.
rmse_ubcf

# Create an item-based collaborative filtering (IBCF) recommender model using Cosine similarity and 350 neighbors based on cross-validation.
item_based_collaborative_filtering_model <- Recommender(getData(evaluation_scheme, "train"),
                          method = "IBCF",
                          param = list(normalize = "center", method = "Cosine", k = 350)
)

ibcf_prediction <- predict(item_based_collaborative_filtering_model, getData(evaluation_scheme, "known"), type = "ratings")

rmse_ibcf <- calcPredictionAccuracy(ibcf_prediction, getData(evaluation_scheme, "unknown"))[1]

saveRDS(rmse_ibcf, file = "rmse_ibcf.rds")

# Load the IBCF RMSE.
rmse_ibcf <- readRDS(file.path(workingDirectory, "rmse_ibcf.rds"))

# Print results.
rmse_ibcf

Finally, I explored the item-based collaborative filtering algorithm with the same set of tuning parameters. Given my earlier exploration of linear regression models using the difference in movie ratings based on users compared to user ratings based on movies, I congruently obtained a higher RMSE score of 0.801.

## 3. Slope One

# Clear unused memory.
invisible(gc())

# Slope One Recommender
# Create copies of training (edx) and validation sets, retaining only essential columns.
# - "genres," "title," and "timestamp" are excluded.
edx.copy <- edx_title_year %>%
  select(-c("genres", "title", "timestamp"))

valid.copy <- final_holdout_test %>%
  select(-c("genres", "title", "timestamp"))

# Rename columns in valid.copy to "user_id," "item_id," and "rating".
names(edx.copy) <- c("user_id", "item_id", "rating")
names(valid.copy) <- c("user_id", "item_id", "rating")

# Convert to a data.tables.
edx.copy <- data.table(edx.copy)
valid.copy <- data.table(valid.copy)

# Convert user_id and item_id columns to character in edx.copy.
edx.copy[, user_id := as.character(user_id)]
edx.copy[, item_id := as.character(item_id)]
valid.copy[, user_id := as.character(user_id)]
valid.copy[, item_id := as.character(item_id)]

# Set key to sort data.tables and mark them as sorted for efficient memory usage.
setkey(edx.copy, user_id, item_id)
setkey(valid.copy, user_id, item_id)

# Split data to create a small training sample to address RAM memory issues.
idx <- createDataPartition(y = edx.copy$rating, times = 1, p = 0.5, list = FALSE)
edx.copy_train <- edx.copy[idx, ]

# Normalise ratings in the training set.
ratings_train_norm <- normalize_ratings(edx.copy_train)

# Build a Slope One model using the training set with normalised ratings.
# Calculate training time.
training_time_slope_one <- system.time({
  
  model <- build_slopeone(ratings_train_norm$ratings)
  
  # Clear unused memory.
  invisible(gc())
  
  # Make predictions using the Slope One model on the validation set.
  predictions <- predict_slopeone(
    model,
    valid.copy[, c(1, 2), with = FALSE],
    ratings_train_norm$ratings
  )
  
  # Unnormalise the predictions using the original rating scale.
  unnormalised_predictions <- unnormalize_ratings(
    normalized = ratings_train_norm,
    ratings = predictions
  )
  
  # Calculate Root Mean Squared Error (RMSE) for the Slope One model.
  rmse_slopeone <- RMSE(valid.copy$rating, unnormalised_predictions$predicted_rating)
  
  model_size_slope_one <<- round(  # Using <<- to assign it globally outside of the function.
    sum(
      object.size(edx.copy),
      object.size(valid.copy),
      object.size(idx),
      object.size(edx.copy_train),
      object.size(ratings_train_norm),
      object.size(model),
      object.size(predictions),
      object.size(unnormalised_predictions),
      object.size(rmse_slopeone)
    ) / (1024^2),  # Convert to MB
    4
  )
  
  # Remove the created copies of sets to free up memory.
  rm(edx.copy, valid.copy, edx.copy_train)
  
  
})

# Save the SlopeOne results
saveRDS(list(rmse = rmse_slopeone, time = training_time_slope_one, size = model_size_slope_one), file = "slopeone_model_results.rds")

# Load the SlopeOne results.
slopeone_results <- readRDS(file.path(workingDirectory, "slopeone_model_results.rds"))

# Display the results for the Slope One model.
cat("RMSE for Slope One: ", slopeone_results$rmse, "\n")
cat("Training Time:", round(slopeone_results$time["elapsed"], 4), "sec\n")
cat("Model size:", slopeone_results$size, "MB", "\n")

## 4. Matrix Factorisation
set.seed(42)

# Calculate training time.
training_time_recosystem_matrix_factorisation <- system.time({
  
  train_recosystem <- with(edx, data_memory(user_index = userId, 
                                                  item_index = movieId,
                                                  rating     = rating))
  
  test_recosystem <- with(final_holdout_test, data_memory(user_index = userId, 
                                                item_index = movieId, 
                                                rating     = rating))
  
  recommendation_system <- Reco()
  
  tuning <- recommendation_system$tune(train_recosystem, 
                                       opts = list(dim = c(10, 20, 30),
                                                   lrate = c(0.1, 0.2),
                                                   nthread  = 1,
                                                   niter = 10))
  
  recommendation_system$train(train_recosystem, 
                              opts = c(tuning$min,
                                       nthread = 1,
                                       niter = 20))
  
  predicted_ratings_MF <-  recommendation_system$predict(test_recosystem, out_memory())
  
  rmse_recosystem_matrix_factorisation <- RMSE(final_holdout_test$rating, predicted_ratings_MF)

  model_size_recosystem_matrix_factorisation <<- round(  # Using <<- to assign it globally outside of the function.
    sum(
      object.size(train_recosystem),
      object.size(test_recosystem),
      object.size(recommendation_system),
      object.size(tuning),
      object.size(predicted_ratings_MF),
      object.size(rmse_recosystem_matrix_factorisation)
    ) / (1024^2),  # Convert to MB
    4
  )
})

# Save the Matrix Factorization (RAM) results
saveRDS(list(rmse = rmse_recosystem_matrix_factorisation, time = training_time_recosystem_matrix_factorisation, size = model_size_recosystem_matrix_factorisation), file = "matrix_factorization_ram_results.rds")

# Load the Matrix Factorisation (RAM) results.
mf_ram_results <- readRDS(file.path(workingDirectory, "matrix_factorization_ram_results.rds"))

# Display the RMSE for the RAM Matrix Factorisation using RAM.
cat("RMSE for RAM Matrix Factorisation:", mf_ram_results$rmse, "\n")
cat("Training Time:", round(mf_ram_results$time["elapsed"], 4), "sec\n")
cat("Model size:", mf_ram_results$size, "MB", "\n")

## Before performing Matrix Factorisation (MF) method, clear unused memory.
invisible(gc())

# Matrix Factorisation with parallel stochastic gradient descent.
# Calculate training time.
# Create copies of training test and validation sets, retaining only essential columns.
# - "genres," "title," and "timestamp" are excluded.
training_time_disk_matrix_factorisation <- system.time({
  
  edx.copy <- edx %>%
    select(-c("genres", "title", "timestamp"))
  names(edx.copy) <- c("user", "item", "rating")
  
  valid.copy <- final_holdout_test %>%
    select(-c("genres", "title", "timestamp"))
  names(valid.copy) <- c("user", "item", "rating")
  
  # Convert edx.copy and valid.copy to matrices.
  edx.copy <- as.matrix(edx.copy)
  valid.copy <- as.matrix(valid.copy)
  
  # Write edx.copy and valid.copy tables to disk.
  write.table(edx.copy, file = "trainset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
  write.table(valid.copy, file = "validset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
  
  # Specify data sets from files on the hard disk using data_file().
  train_set <- file.path(workingDirectory, "trainset.txt")
  valid_set <- file.path(workingDirectory, "validset.txt")
  
  # Build a Recommender object for Matrix Factorisation.
  recommender <- Reco()
  
  # Matrix Factorisation: Tune hyperparameters on the training set.
  opts <- recommender$tune(
    train_set,
    opts = list(
      dim = c(10, 20, 30),
      lrate = c(0.1, 0.2),
      costp_l1 = 0,
      costq_l1 = 0,
      nthread = 1,
      niter = 10
    )
  )
  
  # Matrix Factorisation: Train the recommender model.
  recommender$train(train_set, opts = c(opts$min, nthread = 1, niter = 20))
  
  # Making predictions on the validation set and calculating RMSE.
  pred_file <- tempfile()
  recommender$predict(valid_set, out_file(pred_file))
  
  # Load the true ratings from the validation set.
  scores_real <- read.table("validset.txt", header = FALSE, sep = " ")$V3
  
  # Load predicted ratings from the temporary prediction file.
  scores_pred <- scan(pred_file)
  
  # Calculate RMSE for Matrix Factorisation.
  rmse_to_disk_matrix_factorisation <- RMSE(scores_real, scores_pred)
  
  model_size_disk_matrix_factorisation <<- round(  # Using <<- to assign it globally outside of the function.
    sum(
      object.size(edx.copy),
      object.size(valid.copy),
      object.size(train_set),
      object.size(valid_set),
      object.size(recommender),
      object.size(pred_file),
      object.size(scores_real),
      object.size(scores_pred),
      object.size(rmse_to_disk_matrix_factorisation)
    ) / (1024^2),  # Convert to MB
    4
  )
  
  # Remove copies of training and validation sets to free up memory.
  rm(edx.copy, valid.copy)
  
})

# Save the trained model using saveRDS
saveRDS(recommender, "matrix_factorization_model_recommender.rds")

# Save the Matrix Factorization (Disk) results
saveRDS(list(rmse = rmse_to_disk_matrix_factorisation, time = training_time_disk_matrix_factorisation, size = model_size_disk_matrix_factorisation), file = "matrix_factorization_disk_results.rds")

# Load the Matrix Factorization (Disk) results.
mf_disk_results <- readRDS(file.path(workingDirectory, "matrix_factorization_disk_results.rds"))

# Display the RMSE for Disk Matrix Factorisation model.
cat("RMSE for Disk Matrix Factorisation: ", mf_disk_results$rmse, "\n")
cat("Training Time:", round(mf_disk_results$time["elapsed"], 4), "sec\n")
cat("Model size:", mf_disk_results$size, "MB", "\n")

# Before performing Matrix Factorisation (MF) method, clear unused memory.
invisible(gc())

small_edx <- edx[1:2000000, ]  # Adjust the number of rows as needed
small_final_holdout_test <- final_holdout_test[1:2000000, ]  # Adjust the number of rows as needed

# Calculate training time.
training_time_als_spark <- system.time({
  # Set Spark configurations with increased memory and additional JVM options
  config <- spark_config()  # Initialize a Spark configuration object.
  
  config$spark.executor.memory <- "16g"  # Increase executor memory.
  config$spark.executor.cores <- 4  # Allocate 4 CPU cores to each executor.
  config$spark.executor.instances <- 4  # Set the number of executor instances to 4.
  config$spark.sql.shuffle.partitions <- 400  # Specify the number of partitions for shuffle operations.
  config$spark.driver.memory <- "16g"  # Increase driver memory.
  config$spark.driver.maxResultSize <- "8g"  # Increase max result size.
  config$spark.driver.extraJavaOptions <- "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp"  # Add Java options to the driver for improved garbage collection and heap dump handling.
  config$spark.executor.extraJavaOptions <- "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp"  # Add Java options to the executors for similar benefits.
  
  # Function to handle null values and convert to integer.
  prepare_data <- function(df) {
    df %>%
      mutate(
        user = ifelse(is.na(userId), -1L, as.integer(userId)),   # Convert userId to integer and handle NAs.
        item = ifelse(is.na(movieId), -1L, as.integer(movieId))  # Convert movieId to integer and handle NAs.
      ) %>%
      filter(user != -1 & item != -1)  # Filter out rows with NAs
  }
  
  copy_data_in_batches <- function(df, df_name, batch_size = 500000) {
    # Function to copy data into Spark in batches to manage large datasets. df: The input data frame to be copied. df_name: The base name for the Spark data frame(s). batch_size: The number of rows per batch (default is 500,000).
    
    total_rows <- nrow(df) # Get the total number of rows in the input data frame.
    
    combined_sdf <- NULL # Initialise a variable to store the combined Spark data frame.
    
    for (start_row in seq(1, total_rows, by = batch_size)) { # Loop through the data frame in increments of `batch_size`.
      
      end_row <- min(start_row + batch_size - 1, total_rows) # Calculate the ending row for the current batch, ensuring it doesn't exceed total rows.
      
      batch <- df[start_row:end_row, ] # Extract a batch of rows from the input data frame.
      
      sdf_batch <- sdf_copy_to(sc, batch, name = paste0(df_name, "_batch_", start_row), overwrite = TRUE) %>% # Copy the batch to Spark, creating a temporary Spark data frame with a unique name.
        sdf_repartition(8, partition_by = "user") # Repartition the Spark data frame into 8 partitions, grouped by the "user" column.
      
      if (is.null(combined_sdf)) { # If this is the first batch, initialize the combined Spark data frame.
        combined_sdf <- sdf_batch
      } else { # Otherwise, append the current batch to the combined Spark data frame.
        combined_sdf <- sdf_bind_rows(combined_sdf, sdf_batch)
      }
    }
    
    combined_sdf <- sdf_register(combined_sdf, name = df_name) %>% # Register the combined Spark data frame with the given name for further use.
      sdf_persist(storage.level = "MEMORY_AND_DISK") # Persist the Spark data frame in memory and disk to optimise performance.
    
    return(combined_sdf) # Return the combined Spark data frame.
  }
  
  sc <- spark_connect(master = "local[*]", config = config)
  
  # Prepare training data.
  tic("Prepare training data")
  sdf_MovieLense_Train <- small_edx %>% prepare_data()
  toc()
  
  # Prepare testing data.
  tic("Prepare testing data")
  sdf_MovieLense_Test <- small_final_holdout_test %>% prepare_data()
  toc()
  
  # Copy and combine training data to Spark.
  tic("Copy training data to Spark")
  edx_spark <- copy_data_in_batches(sdf_MovieLense_Train, "small_edx")
  toc()
  
  # Copy and combine testing data to Spark.
  tic("Copy testing data to Spark")
  final_holdout_test_spark <- copy_data_in_batches(sdf_MovieLense_Test, "small_final_holdout_test")
  toc()
  
  # Training.
  tic("ALS Model Training - sparklyr")
  sdf_als_model <- ml_als(edx_spark, rating_col = "rating", user_col = "user", item_col = "item", rank = 10, reg_param = 0.1, max_iter = 5)
  toc()
  
  # Predicting.
  tic("ALS Model Predicting - sparklyr")
  prediction <- ml_transform(sdf_als_model, final_holdout_test_spark) %>% collect()
  toc()
  
  # Calculate model size.
  model_size_als <<- round(
    sum(
      object.size(sdf_MovieLense_Train),
      object.size(sdf_MovieLense_Test),
      object.size(edx_spark),
      object.size(final_holdout_test_spark),
      object.size(sdf_als_model),
      object.size(prediction)
    ) / (1024^2),  # Convert to MB
    4
  )
  
  # Disconnect from Spark.
  spark_disconnect(sc)
})

#Calculate RMSE.
rmse_als_spark <- RMSE(prediction$rating, prediction$prediction)

# Save the ALS (Spark) results.
saveRDS(list(rmse = rmse_als_spark, time = training_time_als_spark, size = model_size_als), file = "als_spark_results.rds")

# Load the ALS (Spark) results.
als_spark_results <- readRDS(file.path(workingDirectory, "als_spark_results.rds"))

# Print reults.
cat("RMSE for Alternating Least Square Means: ", als_spark_results$rmse, "\n")
cat("Training Time:", round(als_spark_results$time["elapsed"], 4), "sec\n")
cat("Model size:", als_spark_results$size, "MB", "\n")

# Sample the data
small_edx <- edx[1:2000000, ]
small_final_holdout_test <- final_holdout_test[1:2000000, ]

# Calculate training time.
training_time_als_recommenderlab <- system.time({
  # Convert data frames to data.tables.
  small_edx <- as.data.table(small_edx)
  small_final_holdout_test <- as.data.table(small_final_holdout_test)
  
  # Prepare data function to handle NAs and convert to integers.
  prepare_data <- function(df) {
    df[, user := as.integer(factor(userId))]
    df[, item := as.integer(factor(movieId))]
    df <- df[complete.cases(df), ]  # Remove rows with NA values.
    df
  }
  
  # Prepare training and testing data.
  small_edx <- prepare_data(small_edx)
  small_final_holdout_test <- prepare_data(small_final_holdout_test)
  
  # Create a sparse matrix for training.
  rating_matrix <- as(small_edx, "realRatingMatrix")
  
  # ALS model training with hyperparameters.
  als_model <- Recommender(rating_matrix, method = "ALS", parameter = list(
    n_factors = 20,      # Number of latent factors.
    lambda = 0.1,        # Regularisation parameter.
    n_iterations = 10    # Number of iterations.
  ))
  
  # Prediction.
  predictions <- predict(als_model, rating_matrix)
  
  # Extract actual ratings from the testing data.
  actual_ratings <- as(small_final_holdout_test, "realRatingMatrix")
  
  # Convert predictions to a regular matrix.
  predictions_matrix <- as(predictions, "matrix")
  actual_ratings_matrix <- as(actual_ratings, "matrix")
  
  # Get common users and items.
  common_users <- intersect(rownames(predictions_matrix), rownames(actual_ratings_matrix))
  common_items <- intersect(colnames(predictions_matrix), colnames(actual_ratings_matrix))
  
  # Subset predictions and actual ratings.
  predictions_subset <- predictions_matrix[common_users, common_items]
  actual_ratings_subset <- actual_ratings_matrix[common_users, common_items]
  
  # Calculate RMSE.
  rmse_als_recommenderlab <- RMSE(actual_ratings_subset, predictions_subset)
  
  # Calculate model size.
  model_size_als_recommenderlab <<- round(
    sum(
      object.size(predictions_matrix),
      object.size(prepare_data),
      object.size(small_edx),
      object.size(small_final_holdout_test),
      object.size(rating_matrix),
      object.size(als_model),
      object.size(predictions),
      object.size(actual_ratings_matrix),
      object.size(common_users),
      object.size(common_items),
      object.size(predictions_subset),
      object.size(actual_ratings_subset),
      object.size(rmse_als_recommenderlab)
    ) / (1024^2),  # Convert to MB
    4
  )
  
})

# Save the ALS (RecommenderLab) results.
saveRDS(list(rmse = rmse_als_recommenderlab, time = training_time_als_recommenderlab, size = model_size_als_recommenderlab), file = "als_recommenderlab_results.rds")

# Load the ALS (RecommenderLab) results.
als_recommenderlab_results <- readRDS(file.path(workingDirectory, "als_recommenderlab_results.rds"))

#Print results. 
cat("RMSE for Alternating Least Square Means: ", als_recommenderlab_results$rmse, "\n")
cat("Training Time:", round(als_recommenderlab_results$time["elapsed"], 4), "sec\n")
cat("Model size:", als_recommenderlab_results$size, "MB", "\n")

# Summarise the RMSE values on the validation set for the linear regression models.
rmse_results <- data.frame(
  Method = c("SlopeOne", "Matrix factorisation using RAM", "Matrix factorisation using Disk", "Alternating Least Squares using Spark", "Alternating Least Squares using Recommender"),
  
  RMSE = c(slopeone_results$rmse, mf_ram_results$rmse, mf_disk_results$rmse, als_spark_results$rmse, als_recommenderlab_results$rmse),
  
  Time = c(slopeone_results$time["elapsed"], mf_ram_results$time["elapsed"], mf_disk_results$time["elapsed"], als_spark_results$time["elapsed"], als_recommenderlab_results$time["elapsed"]),
  
  Size = c(slopeone_results$size, mf_ram_results$size, mf_disk_results$size,  als_spark_results$size, als_recommenderlab_results$size)
)

# Rename the columns to replace full stops with spaces.
colnames(rmse_results) <- gsub("Time", "Time (sec)", colnames(rmse_results))
colnames(rmse_results) <- gsub("Size", "Size (MB)", colnames(rmse_results))

# Display RMSE results in an HTML table using the kable function.
kable(rmse_results, "html") %>%
  kable_styling(
    bootstrap_options = c("striped", "bordered", "hover"),
    full_width = FALSE,
    position = "center"
  ) %>%
  column_spec(1, bold = TRUE, color = "black") %>%
  column_spec(2, bold = TRUE, color = "white", background = "#ff2b76") %>%
  column_spec(3, bold = TRUE, color = "black", background = "#affaf9") %>%
  column_spec(4, bold = TRUE, color = "black", background = "#ffce8f") %>%
  row_spec(0, extra_css = "text-align: left;") %>%
  add_header_above(c("Matrix Factorisation" = 4)) 

# Clear unused memory.
invisible(gc())

# Create a copy of the edx set, retaining all features.
edx.copy <- edx

# Add new columns for the number of movies each user rated (n.movies_byUser) and the number of users that rated each movie (n.users_bymovie).
edx.copy <- edx.copy %>%
  dplyr::group_by(userId) %>%
  dplyr::mutate(n.movies_byUser = dplyr::n())

edx.copy <- edx.copy %>%
  dplyr::group_by(movieId) %>%
  dplyr::mutate(n.users_bymovie = n())

# Convert userId and movieId columns to factor vectors.
edx.copy$userId <- as.factor(edx.copy$userId)
edx.copy$movieId <- as.factor(edx.copy$movieId)

# Repeat the same process for the validation set.
valid.copy <- final_holdout_test

valid.copy <- valid.copy %>%
  dplyr::group_by(userId) %>%
  dplyr::mutate(n.movies_byUser = n())

valid.copy <- valid.copy %>%
  dplyr::group_by(movieId) %>%
  dplyr::mutate(n.users_bymovie = n())

valid.copy$userId <- as.factor(valid.copy$userId)
valid.copy$movieId <- as.factor(valid.copy$movieId)

# Attempts to start and/or connect to an H2O instance.
h2o.init(
  nthreads = -1, ## -1: use all available threads.
  max_mem_size = "10G"
)

# Clear all the objects from the H2O cluster
h2o.removeAll()

# Partitioning the data into training and testing sets.
splits <- h2o.splitFrame(as.h2o(edx.copy),
                         ratios = 0.7,
                         seed = 1
)

train <- splits[[1]]
test <- splits[[2]]

# Clear unused memory.
invisible(gc())

# Remove progress bar for H2O operations.
h2o.no_progress()

# Calculate training time.
training_time_gradient_boosted_decision_tree_1 <- system.time({
  
# First Gradient Boosting Machine (GBM) model:
# Parameters: ntrees = 50, max depth = 5, learn rate = 0.1, nfolds = 3.
gradient_boosted_decision_tree_1 <- h2o.gbm(
  x = c("movieId", "userId", "n.movies_byUser", "n.users_bymovie"),
  y = "rating",
  training_frame = train,
  nfolds = 3
)

# Display a summary of the first GBM model.
summary(gradient_boosted_decision_tree_1)

# Second GBM model:
# Parameters: ntrees = 50, max depth = 5, learn rate = 0.1, nfolds = 3.
gradient_boosted_decision_tree_2 <- h2o.gbm(
  x = c("movieId", "userId"),
  y = "rating",
  training_frame = train,
  nfolds = 3,
  seed = 42,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Random"
)

# Display a summary of the third GBM model.
summary(gradient_boosted_decision_tree_2)

# Since gradient_boosted_decision_tree_2 has the lower RMSE on the training set.
# Evaluate performance on the test set.
h2o.performance(gradient_boosted_decision_tree_2, test)

# Predict ratings on the validation set and evaluate RMSE.
pred.ratings.gradient_boosted_decision_tree_2 <- h2o.predict(gradient_boosted_decision_tree_2, as.h2o(valid.copy))

rmse_gbdt <- RMSE(pred.ratings.gradient_boosted_decision_tree_2, as.h2o(valid.copy$rating))

  # Calculate model size.
  model_size_gradient_boosted_decision_tree_1 <<- round(
    sum(
      object.size(gradient_boosted_decision_tree_1),
      object.size(gradient_boosted_decision_tree_2),
      object.size(pred.ratings.gradient_boosted_decision_tree_2),
      object.size(rmse_gbdt)
    ) / (1024^2),  # Convert to MB.
    4
  )

})

# Save Gradient Boosting results.
saveRDS(list(rmse = rmse_gbdt, time = training_time_gradient_boosted_decision_tree_1,size = model_size_gradient_boosted_decision_tree_1[1]), file = "gradient_boosting_results.rds")

# Load Gradient Boosting results
gradient_boosting_results <- readRDS(file.path(workingDirectory, "gradient_boosting_results.rds"))

# Display Gradient Boosting results.
cat("RMSE for Gradient Boosting: ", gradient_boosting_results$rmse, "\n")
cat("Training Time:", round(gradient_boosting_results$time["elapsed"], 4), "sec\n")
cat("Model size:", gradient_boosting_results$size, "MB", "\n")

# Clear unused memory.
invisible(gc())

# Remove progress bar for H2O operations.
h2o.no_progress()

# Calculate training time.
training_time_random_forest <- system.time({

# First Random Forest (RF) model:
# Parameters: ntrees = 50, max depth = 20.
random_forest_1 <- h2o.randomForest(
  training_frame = train,
  x = c("movieId", "userId", "timestamp", "n.movies_byUser", "n.users_bymovie"),
  y = "rating",
  ntrees = 50,
  max_depth = 20
)

# Display a summary of the first RF model.
summary(random_forest_1)

# Third RF model:
# Parameters: ntrees = 50, max depth = 20, nfolds = 3.
random_forest_2 <- h2o.randomForest(
  training_frame = train,
  x = c("movieId", "userId"),
  y = "rating",
  nfolds = 3,
  seed = 42,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Random"
)

# Display a summary of the third RF model.
summary(random_forest_2)

# Since random_forest_2 has the lower RMSE on the training set,
# Evaluate performance on the test set.
h2o.performance(random_forest_2, test)

# Predict ratings on the validation set and evaluate RMSE.
pred.ratings.random_forest_2 <- h2o.predict(random_forest_2, as.h2o(valid.copy))

rmse_rf <- RMSE(pred.ratings.random_forest_2, as.h2o(valid.copy$rating))

  # Calculate model size.
  model_size_random_forest <<- round(
    sum(
      object.size(random_forest_1),
      object.size(random_forest_2),
      object.size(pred.ratings.random_forest_2),
      object.size(rmse_rf)
    ) / (1024^2),  # Convert to MB.
    4
  )

})

# Save Random Forest results.
saveRDS(list(rmse = rmse_rf, time = training_time_random_forest, size = model_size_random_forest[1]), file = "random_forest_results.rds")

# Load Random Forest results.
random_forest_results <- readRDS(file.path(workingDirectory, "random_forest_results.rds"))

# Display Random Forest results.
cat("RMSE for Random Forest: ", random_forest_results$rmse, "\n")
cat("Training Time:", round(random_forest_results$time["elapsed"], 4), "sec\n")
cat("Model size:", random_forest_results$size, "MB", "\n")

# Clear unused memory.
invisible(gc())

# Calculate training time.
training_time_ensemble <- system.time({
  
  # Stacked Ensemble: Using the best two previous models (gradient_boosted_decision_tree_2 and random_forest_2).
ensemble <- h2o.stackedEnsemble(
  x = c("movieId", "userId"),
  y = "rating",
  training_frame = train,
  model_id = "my_ensemble_auto",
  base_models = list(gradient_boosted_decision_tree_2@model_id, random_forest_2@model_id)
)

# Predict ratings on the validation set and evaluate RMSE.
pred.ratings.ensemble <- h2o.predict(ensemble, as.h2o(valid.copy))

# Calculate RMSE.
rmse_ensemble <- RMSE(pred.ratings.ensemble, as.h2o(valid.copy$rating))

  # Calculate model size.
  model_size_ensemble <<- round(
    sum(
      object.size(random_forest_1),
      object.size(ensemble),
      object.size(pred.ratings.ensemble),
      object.size(rmse_ensemble)
    ) / (1024^2),  # Convert to MB.
    4
  )

})

# Save Stacked Ensemble results.
saveRDS(list(rmse = rmse_ensemble, time = training_time_ensemble, size = model_size_ensemble[1]), file = "stacked_ensemble_results.rds")

# Remove unnecessary objects to free up memory.
rm(edx.copy, valid.copy)

# Close the H2O cluster.
h2o.shutdown()

# Load Stacked Ensemble results
stacked_ensemble_results <- readRDS(file.path(workingDirectory, "stacked_ensemble_results.rds"))

# Display Stacked Ensemble results
cat("RMSE for Stacked Ensemble: ", stacked_ensemble_results$rmse, "\n")
cat("Training Time:", round(stacked_ensemble_results$time["elapsed"], 4), "sec\n")
cat("Model size:", stacked_ensemble_results$size, "MB", "\n")

Using the Stacked Ensemble method I was able to achieve a RMSE score of 0.952, which was ever so slighty better than the base models on their own. 

# Summarise the RMSE values on the validation set for the linear regression models.
rmse_results <- data.frame(
  Method = c("Gradient Boosting", "Random Forest", "Stacked Ensemble"),
  
  RMSE = c(gradient_boosting_results$rmse, random_forest_results$rmse, stacked_ensemble_results$rmse),
  
    Time = c(gradient_boosting_results$time["elapsed"], random_forest_results$time["elapsed"], stacked_ensemble_results$time["elapsed"]),
  
    Size = c(gradient_boosting_results$size, gradient_boosting_results$size, stacked_ensemble_results$size)
)

# Rename the columns to replace full stops with spaces.
colnames(rmse_results) <- gsub("Time", "Time (sec)", colnames(rmse_results))
colnames(rmse_results) <- gsub("Size", "Size (MB)", colnames(rmse_results))

# Display RMSE results in an HTML table using the kable function.
kable(rmse_results, "html") %>%
  kable_styling(
    bootstrap_options = c("striped", "bordered", "hover"),
    full_width = FALSE,
    position = "center"
  ) %>%
  column_spec(1, bold = TRUE, color = "black") %>%
  column_spec(2, bold = TRUE, color = "white", background = "#ff2b76") %>%
  column_spec(3, bold = TRUE, color = "black", background = "#affaf9") %>%
  column_spec(4, bold = TRUE, color = "black", background = "#ffce8f") %>%
  row_spec(0, extra_css = "text-align: left;") %>%
  add_header_above(c("Ensemble Methods" = 4)) 

## 6. Neural Networks

# Combine final_holdout_test back into edx.
full_data <- rbind(edx, final_holdout_test)

# Neural Networks .
dense_movies <- full_data %>% select(movieId) %>% distinct() %>% rowid_to_column()
movie_data <- full_data %>% dplyr::inner_join(dense_movies) %>% dplyr::rename(movieIdDense = rowid)
ratings <- movie_data %>% inner_join(full_data) %>% select(userId, movieIdDense, rating, title, genres)

# Write.csv(movie_data).
max_rating <- ratings %>% summarise(max_rating = max(rating)) %>% pull()
min_rating <- ratings %>% summarise(min_rating = min(rating)) %>% pull()

n_movies <- ratings %>% select(movieIdDense) %>% distinct() %>% nrow()
n_users <- ratings %>% select(userId) %>% distinct() %>% nrow()

train_indices <- sample(1:nrow(ratings), 0.9 * nrow(ratings))
train_ratings <- ratings[train_indices,]
valid_ratings <- ratings[-train_indices,]

x_train <- train_ratings %>% select(c(userId, movieIdDense)) %>% as.matrix()
y_train <- train_ratings %>% select(rating) %>% as.matrix()
x_final_holdout_test <- valid_ratings %>% select(c(userId, movieIdDense)) %>% as.matrix()
y_final_holdout_test <- valid_ratings %>% select(rating) %>% as.matrix()

embedding_dim <- 50

# Calculate training time.
training_time_complex_dot <- system.time({
  
complex_dot_model <- function(embedding_dim,
                              n_users,
                              n_movies,
                              max_rating,
                              min_rating,
                              input_dim = 75000,
                              name = "dot_with_bias") {
  keras_model_custom(name = name, function(self) {
    
    self$user_embedding <-
      layer_embedding(input_dim = input_dim,
                      output_dim = embedding_dim,
                      name = "user_embedding")
    self$movie_embedding <-
      layer_embedding(input_dim = input_dim,
                      output_dim = embedding_dim,
                      name = "movie_embedding")
    self$user_bias <-
      layer_embedding(input_dim = input_dim,
                      output_dim = 1,
                      name = "user_bias")
    self$movie_bias <-
      layer_embedding(input_dim = input_dim,
                      output_dim = 1,
                      name = "movie_bias")
    
    self$user_dropout <- layer_dropout(rate = 0.2)
    self$movie_dropout <- layer_dropout(rate = 0.4)
    
    self$dot <-
      layer_lambda(
        f = function(x)
          k_batch_dot(x[[1]], x[[2]], axes = 2),
        name = "dot"
      )
    
    self$dot_bias <-
      layer_lambda(
        f = function(x)
          k_sigmoid(x[[1]] + x[[2]] + x[[3]]),
        name = "dot_bias"
      )
    
    self$pred <- layer_lambda(
      f = function(x)
        x * (self$max_rating - self$min_rating) + self$min_rating,
      name = "pred"
    )
    
    self$max_rating <- max_rating
    self$min_rating <- min_rating
    
    function(x, mask = NULL, training = TRUE) {
      users <- x[, 1]
      movies <- x[, 2]
      user_embedding <- self$user_embedding(users) %>% self$user_dropout()
      movie_embedding <- self$movie_embedding(movies) %>% self$movie_dropout()
      dot <- self$dot(list(user_embedding, movie_embedding))
      dot_bias <- self$dot_bias(list(dot, self$user_bias(users), self$movie_bias(movies)))
      self$pred(dot_bias)
    }
  })
}

model <- complex_dot_model(embedding_dim,
                           n_users,
                           n_movies,
                           max_rating,
                           min_rating)

model %>% compile(
  loss = "mse",
  optimizer = "adam"
)

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = list(x_final_holdout_test, y_final_holdout_test),
  callbacks = list(callback_early_stopping(patience = 2))
)

# Get predicted ratings on the validation set.
predicted_ratings <- model %>% predict(x_final_holdout_test)

rmse_nn_complex_dot <- RMSE(predicted_ratings, y_final_holdout_test)
  
#Calculate model size.
model_size_complex_dot <<- round(
  sum(
    object.size(complex_dot_model),
    object.size(model),
    object.size(history),
    object.size(predicted_ratings),
    object.size(rmse_nn_complex_dot)
  ) / (1024^2),  # Convert to MB.
  4
)

})

# Save Complex Dot Model results.
saveRDS(list(rmse = rmse_nn_complex_dot, time = training_time_complex_dot, size = model_size_complex_dot[1]), file = "complex_dot_model_results.rds")

# Load Complex Dot Model results.
complex_dot_model_results <- readRDS(file.path(workingDirectory, "complex_dot_model_results.rds"))

#Print results. 
cat("RMSE for Complex Dot Model: ", complex_dot_model_results$rmse, "\n")
cat("Training Time:", round(complex_dot_model_results$time["elapsed"], 4), "sec\n")
cat("Model size:", complex_dot_model_results$size, "MB", "\n")

# Clear unused memory.
invisible(gc())

# Calculate training time.
training_time_simplified_dot_model <- system.time({
  
simplified_dot_model <- function(embedding_dim,
                                      n_users,
                                      n_movies,
                                      max_rating,
                                      min_rating,
                                      input_dim = 75000,
                                      name = "dot_with_bias") {
  keras_model_custom(name = name, function(self) {
    
    self$user_embedding <-
      layer_embedding(input_dim = input_dim,
                      output_dim = embedding_dim,
                      name = "user_embedding")
    self$movie_embedding <-
      layer_embedding(input_dim = input_dim,
                      output_dim = embedding_dim,
                      name = "movie_embedding")
    self$user_bias <-
      layer_embedding(input_dim = input_dim,
                      output_dim = 1,
                      name = "user_bias")
    self$movie_bias <-
      layer_embedding(input_dim = input_dim,
                      output_dim = 1,
                      name = "movie_bias")
    
    self$user_dropout <- layer_dropout(rate = 0.2)  
    self$movie_dropout <- layer_dropout(rate = 0.4) 
    self$batch_norm_user <- layer_batch_normalization()
    self$batch_norm_movie <- layer_batch_normalization()
    self$dot <- layer_dot(axes = 1, name = "dot")
    self$dot_bias <-
      layer_add(name = "dot_bias")
    
    self$max_rating <- max_rating
    self$min_rating <- min_rating
    
    function(x, mask = NULL, training = TRUE) {
      users <- x[, 1]
      movies <- x[, 2]
      user_embedding <- self$user_embedding(users) %>% self$batch_norm_user() %>% self$user_dropout()
      movie_embedding <- self$movie_embedding(movies) %>% self$batch_norm_movie() %>% self$movie_dropout()
      dot <- self$dot(list(user_embedding, movie_embedding))
      dot_bias <- self$dot_bias(list(dot, self$user_bias(users), self$movie_bias(movies)))
      dot_bias
      
    }
  })
}

model <- simplified_dot_model(embedding_dim,
                                n_users,
                                n_movies,
                                max_rating,
                                min_rating)

model %>% compile(
  loss = "mse",
  optimizer = "adam"
)

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = list(x_valid, y_valid),
  callbacks = list(callback_early_stopping(patience = 2))
)

# Get predicted ratings on the validation set.
predicted_ratings <- model %>% predict(x_valid)

simplified_dot_model_rmse <- RMSE(predicted_ratings, y_valid)

# Calculate model size.
  model_size_simplified_dot_model <<- round(
    sum(
      object.size(simplified_dot_model),
      object.size(model),
      object.size(history),
      object.size(predicted_ratings),
      object.size(rmse_nn_complex_dot)
    ) / (1024^2),  # Convert to MB
    4
  )

})

# Save Simple Dot Model results.
saveRDS(list(rmse = simplified_dot_model_rmse, time = training_time_simplified_dot_model, size = model_size_simplified_dot_model[1]), file = "simple_dot_model_results.rds")

# Load Simple Dot Model results.
simple_dot_model_results <- readRDS(file.path(workingDirectory, "simple_dot_model_results.rds"))

#Print results. 
cat("RMSE for Simple Dot Model: ", simple_dot_model_results$rmse, "\n")
cat("Training Time:", round(simple_dot_model_results$time["elapsed"], 4), "sec\n")
cat("Model size:", simple_dot_model_results$size, "MB", "\n")

# Clear unused memory.
invisible(gc())

# Summarise the results for the Neural Networks.
rmse_results <- data.frame(
  Method = c("Complex Dot NN", "Simplified Dot NN"),
  
  RMSE = c(complex_dot_model_results$rmse, simple_dot_model_results$rmse),
  
  Time = c(complex_dot_model_results$time["elapsed"], simple_dot_model_results$time["elapsed"]),
  
  Size = c(complex_dot_model_results$size, simple_dot_model_results$size)
  
)

# Rename the columns to replace full stops with spaces.
colnames(rmse_results) <- gsub("Time", "Time (sec)", colnames(rmse_results))
colnames(rmse_results) <- gsub("Size", "Size (MB)", colnames(rmse_results))

# Display RMSE results in an HTML table using the kable function.
kable(rmse_results, "html") %>%
  kable_styling(
    bootstrap_options = c("striped", "bordered", "hover"),
    full_width = FALSE,
    position = "center"
  ) %>%
  column_spec(1, bold = TRUE, color = "black") %>%
  column_spec(2, bold = TRUE, color = "white", background = "#ff2b76") %>%
  column_spec(3, bold = TRUE, color = "black", background = "#affaf9") %>%
  column_spec(4, bold = TRUE, color = "black", background = "#ffce8f") %>%
  row_spec(0, extra_css = "text-align: left;") %>%
  add_header_above(c("Neural Networks" = 4)) 

## Before performing Matrix Factorisation (MF) method, clear unused memory.
invisible(gc())

# Matrix Factorisation with parallel stochastic gradient descent.
# Calculate training time.
# Create copies of training test and validation sets, retaining only essential columns.
# - "genres," "title," and "timestamp" are excluded.
training_time_disk_matrix_factorisation <- system.time({
  
edx.copy <- edx %>%
  select(-c("genres", "title", "timestamp"))
names(edx.copy) <- c("user", "item", "rating")

valid.copy <- final_holdout_test %>%
  select(-c("genres", "title", "timestamp"))
names(valid.copy) <- c("user", "item", "rating")

# Convert edx.copy and valid.copy to matrices.
edx.copy <- as.matrix(edx.copy)
valid.copy <- as.matrix(valid.copy)

# Write edx.copy and valid.copy tables to disk.
write.table(edx.copy, file = "trainset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table(valid.copy, file = "validset.txt", sep = " ", row.names = FALSE, col.names = FALSE)

# Specify data sets from files on the hard disk using data_file().
train_set <- file.path(workingDirectory, "trainset.txt")
valid_set <- file.path(workingDirectory, "validset.txt")

# Build a Recommender object for Matrix Factorisation.
recommender <- Reco()

# Optimise/tune the recommender model.
opts <- recommender$tune(train_set, opts = list(
  dim = c(1:20), lrate = c(0.05),
  nthread = 4, costp_l1 = 0,
  costq_l1 = 0,
  niter = 50, nfold = 20,
  verbose = FALSE
))

# Train the recommender model.
recommender$train(train_set, opts = c(opts$min, nthread = 4, niter = 100, verbose = FALSE))

# Display the training set and optimisation options.
train_set
opts

# Make predictions on the validation set and calculate RMSE.
pred_file <- tempfile()
recommender$predict(valid_set, out_file(pred_file))

# Matrix Factorisation: Display the first 10 predicted values.
print(scan(pred_file, n = 10))

# Read actual ratings from the validation set.
scores_real <- read.table("validset.txt", header = FALSE, sep = " ")$V3

# Read predicted ratings from the saved prediction file.
scores_pred <- scan(pred_file)

# Remove edx.copy and valid.copy objects to free up memory.
rm(edx.copy, valid.copy)

# Calculate the RMSE between actual and predicted ratings.
rmse_mf_opt <- RMSE(scores_real, scores_pred)
rmse_mf_opt

# Train the recommender model with verbose output for the first 30 iterations.
output <- capture.output(recommender$train(train_set, opts = c(opts$min, nthread = 4, niter = 30, verbose = TRUE)))

output <- output[-1]
output <- trimws(output)

# Extract data using regular expressions.
output_df <- do.call(rbind, strsplit(output, "\\s+"))
colnames(output_df) <- c("iter", "tr_rmse", "obj")

# Convert columns to appropriate types.
output_df <- as.data.frame(output_df, stringsAsFactors = FALSE)
output_df$iter <- as.integer(output_df$iter)
output_df$tr_rmse <- as.numeric(output_df$tr_rmse)
output_df$obj <- as.numeric(output_df$obj)

# Save the data frame to an RData file.
save(output_df, file = "trainRmse_MovieLens.RData")

# Load the model.
load("trainRmse_MovieLens.RData")

# Specify the iteration number for analysis.
iter.line <- 15

# Extract the training RMSE at the specified iteration.
training_rmse.line <- output_df$tr_rmse[which(output_df$iter == 15)]

# Plot the training RMSE over iterations.
suppressMessages({
  output_df %>%
    ggplot(aes(x = iter, y = tr_rmse)) +
    geom_point(size = 3, shape = 19) +
    geom_smooth(aes(x = iter, y = tr_rmse), formula = y ~ x, method = "loess") +
    geom_segment(x = 0, xend = iter.line, y = training_rmse.line, yend = training_rmse.line, color = "orange", lty = 2) +
    geom_segment(x = iter.line, xend = iter.line, y = 0, yend = training_rmse.line, color = "orange", lty = 2) +
    annotate(
      geom = "label", x = iter.line, y = 0.8350, color = 5,
      label = paste("x =", round(iter.line, 0), "\ny =", round(training_rmse.line, 4))
    ) +
    labs(
      title = "RMSE for different number of latent factors",
      caption = "Based on the output of r$train(train_set, opts = c(opts$min, nthread = 4, niter = 100), \n show just first 30 iterations)"
    ) +
    ylab("RMSE") +
    xlab("Latent factors")
})

# V. Making Predictions

# Choose a random user from the testing data.
random_user <- sample(unique(final_holdout_test$userId), 1)

# Ensure user is not empty.
if (length(random_user) == 0) {
  stop("No users found in the testing data")
}

# List all unique movie IDs.
all_movies <- unique(edx_title_year$movieId)

# Create a data frame for the user with all movie IDs (the rating is set to NA or 0 for prediction purposes).
user_movies <- data.frame(user = random_user, item = all_movies, rating = NA)

# Write the user_movies data frame to a temporary file for prediction using the recosystem package.
user_file <- tempfile()
write.table(user_movies[, c("user", "item")], file = user_file, sep = " ", row.names = FALSE, col.names = FALSE)

# Create a temporary file to store predictions using the recosystem package.
pred_file <- tempfile()

# Load the best model from the saved RDS file
recommender <- readRDS("matrix_factorization_model_recommender.rds")

# Make predictions using the trained Reco model.
recommender$predict(data_file(user_file), out_file(pred_file))

# Read predicted ratings from the temporary file.
predicted_ratings <- read.table(pred_file, header = FALSE)$V1

# Combine the predicted ratings with movie IDs.
user_predictions <- data.frame(movieId = all_movies, predicted_rating = predicted_ratings)

# Sort by predicted rating in descending order to get top 10 recommendations based on movieId.
top_10_recommendations <- user_predictions %>%
  arrange(desc(predicted_rating)) %>%
  slice(1:10)

# Sort by predicted rating in ascending order to get bottom 10 recommendations based on movieId.
bottom_10_recommendations <- user_predictions %>%
  arrange(predicted_rating) %>%
  slice(1:10)

# Add back movie titles:
# Remove duplicate titles keeping only the first occurrence.
edx_title_year_unique <- edx_title_year %>%
  distinct(movieId, .keep_all = TRUE)

# Perform an inner join on predicted ratings.
top_10_recommendations_title <- inner_join(top_10_recommendations, edx_title_year_unique, by = "movieId") %>% 
  mutate(predicted_rating = round(predicted_rating, 2))

bottom_10_recommendations_title <- inner_join(bottom_10_recommendations, edx_title_year_unique, by = "movieId") %>% 
  mutate(predicted_rating = round(predicted_rating, 2))

# Remove the extra columns from the joined_data.
top_10_recommendations_title <- top_10_recommendations_title %>%
  select(-userId, -rating, -timestamp)

bottom_10_recommendations_title <- bottom_10_recommendations_title %>%
  select(-userId, -rating, -timestamp)

# View results.
top_10_recommendations_title %>%
  kable("html", col.names = c("Movie ID", "Predicted Rating", "Title", "Genre", "Year")) %>%
  add_header_above(setNames(5, paste("Top Picks for User", random_user))) %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = F)

bottom_10_recommendations_title %>%
  kable("html", col.names = c("Movie ID", "Predicted Rating", "Title", "Genre", "Year"))  %>%
  add_header_above(setNames(5, paste("Bottom Picks for User", random_user))) %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = F)