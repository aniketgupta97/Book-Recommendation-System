**Book Recommendation System**
**Overview**
The Book Recommendation System is a machine learning project designed to recommend books to users based on the text content and genres of other books. The system utilizes TF-IDF for text vectorization, MultiLabelBinarizer for genre encoding, and a Naive Bayes classifier for making predictions.

**Features**
Text Vectorization: Converts text data into numerical form using TF-IDF.
Genre Encoding: Encodes book genres using MultiLabelBinarizer.
Model Training: Uses Multinomial Naive Bayes for classifying book genres.
Book Recommendations: Provides book recommendations based on cosine similarity of the combined features.

**Setup Prerequisites**
Python 3.x
Required Python libraries: pandas, scikit-learn, numpy, matplotlib, seaborn, pickle
