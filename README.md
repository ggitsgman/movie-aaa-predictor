# movie-aaa-predictor
# 🎬 AAA Movie + IMDb & TMDb Movies Dashboard - Data Strategist Project Pretest

This project analyzes IMDb and TMDb movie data to identify trends, define a 'hit movie' metric, and support data-driven decisions for creating the next AAA title.

## 📊 Features
- Interactive Streamlit dashboard for exploratory analysis  
- Key performance metrics (ratings, votes, popularity, profitability)  
- Visualizations and Exploratory Data Analysis on genre trends, actor popularity, AAA index, and more
- A 'hit movie' metric benchmark which has been tested and trained against a Logistic Regression machine learning model

## 📂 Data Requirements

The dataset and cleaned data is **not included** in this repository due to size constraints.
To run the project:
1. **Download the data from [Google Drive](https://drive.google.com/drive/folders/12HL7HRkJhm1drbv7hC9eZDDpTTTIKZP7).** This folder hosts the raw datasets used for this project. Should you wish to also download the cleaned datasets which are the product of Data Strategist Pretest - Read, Collect, Clean.ipynb you can do so here [Google Drive](https://drive.google.com/drive/folders/1rH7Z1C3Io3zywpsV3ITENFcCK-Y6z7UM) 
2. Place the files in the following structure:
```
[For RAW datasets]
├── datasets/
│   ├── name.basics.tsv.gz
│   ├── title.basics.tsv.gz
│   ├── title.principals.tsv.gz
│   ├── title.ratings.tsv.gz
│   └── TMDB_movie_dataset_v11.csv
└── app.py
```
```
[For CLEANED datasets]
├── data/
│   ├── movies_data.csv
│   ├── actor_actress_director.csv
│   ├── genres_data.csv
│   ├── keyword_data.csv
│   └── production_companies_data.csv
└── app.py
```
3. Ensure the filenames match exactly.

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run dashboard/movie_dashboard_streamlit.py
```

## 📌 Notes
- The dashboard is optimized for data filtered through sidebar controls
- Some visualizations may take time to load due to large datasets, but should be cached on first run for better performance

Author: Gabriel K. Manibog
Date: February 3, 2025