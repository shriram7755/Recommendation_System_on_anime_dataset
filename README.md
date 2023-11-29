# Recommendation_System_on_anime_dataset

There are various approaches to recommendation systems, but the two main types are:

Collaborative Filtering:

User-based Collaborative Filtering: Recommends items based on the preferences of users who are similar to the target user. It assumes that if a user A likes the same items as user B, A is likely to enjoy other items that B likes.
Item-based Collaborative Filtering: Recommends items based on their similarity to items the user has already liked or interacted with. It assumes that if a user likes item A, they are likely to also like items that are similar to A.
Content-Based Filtering:

Recommends items similar to those the user has liked or interacted with in the past. It takes into account the characteristics of items and the preferences expressed by the user for items with similar features.
Hybrid Methods:

Combine both collaborative and content-based filtering to improve the accuracy and overcome the limitations of individual methods.

Content-Based Filtering:
Vectorize item features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings.
Use similarity metrics (cosine similarity, Euclidean distance) to find items similar to the user's preferences.
