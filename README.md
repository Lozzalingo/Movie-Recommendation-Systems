# Movie Recommendation Systems: An Overview and Analysis

A recommendation system utilises machine learning algorithms to analyse past user behaviour and predict preferences for unobserved or future data. In the context of movie recommendations, such systems rely on user and movie rating data to make predictions about movies a user has not yet rated. The primary approaches in movie recommendation algorithms are **content-based filtering** and **collaborative filtering**.

## Recommendation Techniques

### 1. Content-Based Filtering:
   - This approach suggests items similar to those a user has previously liked or interacted with.
   - It relies on the attributes of items (e.g., genres, actors, or directors in the case of movies) and a user’s historical preferences.

### 2. Collaborative Filtering:
   - This method recommends items that are highly rated by users with similar profiles.
   - Collaborative filtering is further divided into:
     - **User-Based Collaborative Filtering**: Identifies users with similar preferences to the target user and recommends movies based on their ratings.
     - **Item-Based Collaborative Filtering**: Focuses on finding similarities between movies themselves, suggesting movies similar to those the user has rated highly.

---

## Developing a Movie Recommendation System

The creation of a movie recommendation system involves several key stages:

### 1. Data Acquisition:
   - Data can be sourced from public platforms, web scraping, or third-party providers.
   - For this project, the MovieLens 10M and 100K datasets, curated by GroupLens.org, were utilised.

### 2. Data Analysis and Preparation:
   - Comprehensive analysis was conducted to extract meaningful insights and structure the data effectively.
   - This step ensures the data is cleaned and transformed for optimal use in machine learning models.

### 3. Model Training and Evaluation:
   - The focus was on collaborative filtering models, where multiple approaches were trained, evaluated, and compared using metrics such as Root Mean Squared Error (RMSE), training time, and resource efficiency.
   - The best-performing model was selected and optimised to balance accuracy and computational efficiency.

### 4. Prediction:
   - The optimised model was used to predict a randomly selected user's preferences for unseen movies.

---

## Business Applications and Importance

Personalised recommendations are vital for businesses aiming to deliver superior service and drive engagement. By tailoring suggestions to users’ interests and behaviours, companies can:
- Increase customer satisfaction.
- Gain a competitive advantage in the market.

Companies like **Spotify**, **Amazon**, **Disney+**, and **Netflix** exemplify the effectiveness of such systems. Notably, Netflix's 2006 competition to enhance its recommendation system accuracy significantly advanced the field, inspiring innovative algorithms that remain influential in data science today.

---

## Model Insights and Results

This analysis compared various recommendation models using 10 million ratings from the MovieLens dataset. The best-performing models were:
- **Neural Networks**:
  - A simplified dot product model achieved low RMSE on both training and test sets.
- **Matrix Factorisation**:
  - Implemented using stochastic gradient descent, this approach delivered high accuracy.
- **Alternating Least Squares (ALS)** and **Linear Regression**:
  - These models also provided competitive performance, with lower computational requirements compared to neural networks.

---

## Challenges in Recommendation Systems

Recommendation systems face two primary challenges:

### 1. Ranking Problem:
   - The ranking problem involves ranking and recommending items based on known user data, such as past interactions. For instance, factors like actors, release dates, and genres that a user has previously rated favourably are used to generate recommendations.

### 2. Matrix Completion Problem:
   - This challenge involves predicting missing data points within a matrix by analysing rating patterns across observed data. For example, if a group of users have historically shown similar rating patterns for a subset of movies they’ve watched, it’s reasonable to infer that if the group rates a movie that one user hasn’t yet seen, their ratings would likely be similar. By leveraging this insight, we can recommend highly-rated movies from users with comparable profiles to those who haven't rated the movie yet, anticipating a similar rating and potential enjoyment.

---

## Conclusion

My report on recommendation algorithms examines various models, including Linear Regression, Matrix Factorisation, Ensemble Methods, and Neural Networks. I also explored several libraries, infrastructural frameworks, and services to optimise model training.

Model accuracy was assessed using Root Mean Squared Error (RMSE), a common metric for its interpretability and its ability to penalise larger errors more severely. Infrastructural load was analysed based on training time and memory usage. Time reflects computational resources, which are essentially tied to energy consumption. Although all models were tested locally, tools like Spark and H2O create clusters to distribute computational resources across multiple nodes, speeding up model training and data processing. Therefore, time should be adjusted based on the hardware used to provide a more accurate assessment of energy consumption. Moreover, time itself is a valuable resource and can be analysed in isolation as shorter training times are desirable.

---

## Recommendations

Based on the analysis, the following models are recommended:

### 1. Linear Regression with Movie + User + Genre Effects:
   - This model achieved an RMSE of **0.8657**, a training time of **0.209 seconds**, and used **8.6289 MB** of memory. Despite its simplicity, it provides reasonably accurate movie rating predictions with very fast training times and minimal memory usage.

### 2. Matrix Factorisation using Disk Memory:
   - If accuracy is the primary concern, this model is the best choice with the lowest RMSE of **0.7834**. However, it requires a long training time of **2155.302 seconds** and uses **793.4668 MB** of memory.

### 3. Alternating Least Squares (ALS) using Spark:
   - This model offers a balanced option with an RMSE of **0.8446**, a training time of **90.368 seconds**, and a memory usage of **361.5343 MB**. Due to hardware constraints, only a subset of the data was used, and a more accurate RMSE may be achieved with additional resources. Furthermore, there is potential for ALS to outperform Matrix Factorisation in all metrics given its current performance and resource usage.

---

## Limitations

The limitations of this report are primarily related to hardware constraints. The study would benefit from being conducted on a more powerful machine. Additionally, there is a lack of resources to measure computational and energy consumption, which are crucial for large-scale model training. A more detailed analysis of hardware and energy requirements, and how these scale with hardware and data, would further enhance the findings.

Another limitation to consider is the tendency for recommendation systems to overgeneralise with widely admired or widely disliked movie titles. Social norms can influence individual ratings; for instance, if a movie is well-respected, people might rate it more highly due to its reputation, leading to its more frequent recommendation. This can skew recommendations towards popular movies, while potentially overlooking more personalised suggestions.

To address this issue, one approach could be to introduce a subcategory for recommendations, such as "Movies We Think You'll Like, That Everyone Loves." This allows for the inclusion of highly-rated films while also leaving room for further subcategorised recommendations that are more tailored to the user's individual preferences and perceived uniqueness.
