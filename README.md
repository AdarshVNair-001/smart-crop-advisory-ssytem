# smart-crop-advisory-system

Overview

The Smart Agriculture Advisory System is a web-based intelligent platform designed to assist farmers with data-driven, personalized agricultural recommendations. By integrating multiple computational models and real-time environmental data, the system provides stage-specific guidance throughout the crop lifecycle. It addresses challenges such as unpredictable climate conditions, pest infestations, and nutrient deficiencies, while optimizing resource usage.

Features

Crop Input & Scheduling: Users can input crop type and planting date; the system automatically incorporates historical and real-time weather data using external APIs.
Decision Tree Model: Generates targeted irrigation and nutrient application guidelines under well-defined conditions.
Support Vector Machine (SVM): Improves decision accuracy in cases of overlapping or ambiguous conditions.
LSTM Network: Performs time-series analysis of weather patterns for proactive scheduling of weather-dependent operations.
CNN-based Pest & Disease Detection: Analyzes user-submitted crop images to detect pests and diseases automatically.
Random Forest Model: Provides fertilizer recommendations based on crop and soil conditions.
Cloud & Edge Deployment: Adaptable for small-scale urban gardening to large-scale agricultural production.

Datasets Used

Agricultural Pests Image Dataset
Plant Disease Dataset=https://www.kaggle.com/datasets/emmarex/plantdisease
Technology Stack=https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset

Frontend: Web-based interface (HTML/CSS/JavaScript or framework of choice)
Backend: Python (Flask/Django) for model integration and API handling
Machine Learning Models: Decision Tree, SVM, Random Forest, LSTM, CNN
Data Sources: Real-time weather APIs, historical crop/weather datasets
Deployment: Cloud platforms (AWS/GCP/Azure) and edge computing devices

Benefits

Personalized, stage-specific recommendations for crops
Automated pest and disease detection
Efficient irrigation and nutrient management
Improved productivity and resource optimization
Resilient against environmental variability

Usage

Upload crop images and enter crop type and planting date.
System fetches weather data and processes it using LSTM for predictive analysis.
Machine learning models generate actionable recommendations for irrigation, fertilization, and pest/disease management.
Users receive tailored guidance to optimize crop yield and resource usage.

License
This project uses publicly available datasets from Kaggle. Users should adhere to the respective dataset licenses
