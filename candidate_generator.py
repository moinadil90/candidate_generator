from surprise import Reader, Dataset, SVD, accuracy

# Load your dataset (replace with actual data loading logic)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_folds(filepath='your_ratings.csv', reader=reader, fold_size=1)

# Train a recommender model (replace with your preferred algorithm)
algo = SVD()
algo.fit(data.build_full_trainset())

# Define user ID for whom to generate recommendations
user_id = 10

# Get all rated items by the user
user_ratings = data.ppdict[user_id]
user_rated_items = [rating[0] for rating in user_ratings]  # Extract item IDs

# Generate recommendations (excluding already rated items)
predictions = algo.predict(user_id, data.all_rating_scale[1:])  # All items except already rated
recommended_items = [pred.iid for pred in predictions if pred.iid not in user_rated_items]
top_n = recommended_items[:10]  # Select top 10 recommendations

# Print recommendations
print(f"Top 10 Recommendations for User {user_id}: {top_n}")
