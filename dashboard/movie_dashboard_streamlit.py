import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sqlite3
import os
import plotly.express as px
from wordcloud import WordCloud

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

# Page Config
st.set_page_config(
    page_title="IMDb & TMDb Movies Dashboard",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(":movie_camera: IMDb & TMDb Movies Dashboard :clapper:")
st.markdown("_Gabriel K. Manibog | GCash Application Pre-Test | 02/03/2025_")

@st.cache_data
def load_data():
    movie_path = os.path.join(data_dir, 'movies_data.csv')
    cast_path = os.path.join(data_dir, 'actor_actress_director.csv')
    genre_path = os.path.join(data_dir, 'genres_data.csv')
    keyword_path = os.path.join(data_dir, 'keyword_data.csv')
    production_companies_path = os.path.join(data_dir, 'production_companies_data.csv')
    
    movies_df = pd.read_csv(movie_path)
    actor_actress_director_df = pd.read_csv(cast_path)
    genres_df = pd.read_csv(genre_path)
    keywords_df = pd.read_csv(keyword_path)
    prod_comp_df = pd.read_csv(production_companies_path)

    genres_df = genres_df[['tconst', 'genres']]
    actor_actress_director_df = actor_actress_director_df[['tconst', 'nconst', 'primaryName', 'category']]
    keywords_df = keywords_df[['tconst', 'keywords']]
    prod_comp_df = prod_comp_df[['tconst', 'production_companies']]

    return movies_df, actor_actress_director_df, genres_df, keywords_df, prod_comp_df
movies_df, actor_actress_director_df, genres_df, keywords_df, prod_comp_df = load_data()


# Plotting & Design 

with st.sidebar.expander("Filters", expanded=True):
    # Genre Filter 
    genres = genres_df['genres'].unique()
    genre_filter = st.selectbox("Select Genre", ["All"] + list(genres))

    # Filter by movie title (primaryTitle - search bar)
    title_search = st.text_input("Search by Movie Title")

    # Filter by Actor/Actress/Director (search bar)
    cast_search = st.text_input("Search by Actor/Actress/Director")

    # Filter by release year (slider)
    min_year, max_year = movies_df['releaseYear'].min(), movies_df['releaseYear'].max()
    year_filter = st.slider("Select Release Year Range", min_year, max_year, (min_year, max_year))

    # Filter by releaes month (slider)
    month_filter = st.slider("Select Release Month", 1, 12, (1, 12))

    # Filter by runtime (slider)
    min_runtime, max_runtime = movies_df['runtime_consolidated'].min(), movies_df['runtime_consolidated'].max()
    runtime_filter = st.slider("Select Runtime Range (minutes)", min_runtime, max_runtime, (min_runtime, max_runtime))

    # AAA Index Filter (slider from 1 to 10)
    aaa_index_filter = st.slider("Select AAA Index", 0.00, 10.00, (0.00, 10.00))

    # Filtering movies dataframe
    filtered_data = movies_df.copy() 

# Apply Genre filter 
if genre_filter != "All":
    filtered_movies = genres_df[genres_df['genres'] == genre_filter]
    filtered_data = pd.merge(filtered_movies[['tconst']], filtered_data, on='tconst', how='inner')

# Apply Title Fitler
if title_search:
    filtered_data = filtered_data[filtered_data['primaryTitle'].str.contains(title_search, case=False)]

# Apply Year Filter
filtered_data = filtered_data[(filtered_data['releaseYear'] >= year_filter[0]) & 
                              (filtered_data['releaseYear'] <= year_filter[1])]

# Apply Month Filter
filtered_data = filtered_data[(filtered_data['releaseMonth'] >= month_filter[0]) & 
                              (filtered_data['releaseMonth'] <= month_filter[1])]

# Apply Runtime Filter
filtered_data = filtered_data[(filtered_data['runtime_consolidated'] >= runtime_filter[0]) & 
                              (filtered_data['runtime_consolidated'] <= runtime_filter[1])]

# Apply AAA Index Filter
filtered_data = filtered_data[(filtered_data['aaa_index'] >= aaa_index_filter[0]) & 
                              (filtered_data['aaa_index'] <= aaa_index_filter[1])]

# Apply Cast Filter 
if cast_search:
    filtered_data = pd.merge(filtered_data, actor_actress_director_df[actor_actress_director_df['primaryName'].str.contains(cast_search, case=False)], on='tconst', how='inner')


# Display filtered data preview with expander option
filtered_data_sample = filtered_data.head(100)

with st.expander("Filtered Data Preview"):
    styled_data = filtered_data_sample.style.format({"releaseYear": "{:d}"})
    st.dataframe(styled_data)                  


### KEY METRICS SECTION!
st.markdown("---")
st.markdown("### Key Metrics")

# kpi metrics calc
avg_aaa_index = filtered_data['aaa_index'].mean()
avg_weighed_rating = filtered_data['weighted_averageRating'].mean()
total_num_votes = filtered_data['total_numVotes'].sum()

num_movies = len(filtered_data)
perfect_rated_movies = filtered_data[
    (filtered_data['imdb_averageRating'] == 10) | (filtered_data['tmdb_averageRating'] == 10)
].shape[0]
movies_above_benchmark = filtered_data[filtered_data['aaa_index'] >= 7.88].shape[0]

# disp kpi cards
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Overall Avg. AAA Index", f"{avg_aaa_index:.2f}")
col2.metric("Overall Weighed Avg. Rating", f"{avg_weighed_rating:.2f}")
col3.metric("Total Votes", f"{total_num_votes:,}")

col4.metric("Number of Movies", f"{num_movies:,}")
col5.metric("10/10 IMDb/TMDb Movies", f"{perfect_rated_movies:,}")
col6.metric("AAA Index >= 7.88 Movies", f"{movies_above_benchmark:,}")

### TOP MOVIES SECTION!
st.markdown("---")
st.markdown("### Top Movies")

cola, colb, colc = st.columns(3)
cold, cole, colf = st.columns(3)

# plot: top 3 movies by AAA index
top_aaa_movies = filtered_data.nlargest(3, 'aaa_index')

with cola:
    figa = px.bar(
        top_aaa_movies,
        x='primaryTitle',
        y='aaa_index',
        color='aaa_index',
        title='Top 3 Movies by AAA Index',
        labels={"primaryTitle": "Movie Title", "aaa_index": "AAA Index"},
        hover_data=['releaseYear', 'genres', 'popularity']
    )

    figa.update_layout(
        xaxis_title="Movie Title",
        yaxis_title="AAA Index",
        template="plotly_dark"
    )

    st.plotly_chart(figa, use_container_width=True)

#plot: top 3 movies by imdb 
top_imdb_movies = filtered_data.nlargest(3, 'imdb_averageRating')

with colb:
    figb = px.bar(
        top_imdb_movies,
        x='primaryTitle',
        y='imdb_averageRating',
        color='imdb_averageRating',
        title='Top 3 Movies by IMDb Average Rating',
        labels={"primaryTitle": "Movie Title", "imdb_averageRating": "IMDb Rating"},
        hover_data=['releaseYear', 'genres', 'popularity']
    )

    figb.update_layout(
        xaxis_title="Movie Title",
        yaxis_title="IMDb Rating",
        template="plotly_dark"
    )

    st.plotly_chart(figb, use_container_width=True)


# plot: top 3 movies by tmdb
top_tmdb_movies = filtered_data.nlargest(3, 'tmdb_averageRating')

with colc:
    figc = px.bar(
        top_tmdb_movies,
        x='primaryTitle',
        y='tmdb_averageRating',
        color='tmdb_averageRating',
        title='Top 3 Movies by TMDb Average Rating',
        labels={"primaryTitle": "Movie Title", "tmdb_averageRating": "TMDb Rating"},
        hover_data=['releaseYear', 'genres', 'popularity']
    )

    figc.update_layout(
        xaxis_title="Movie Title",
        yaxis_title="TMDb Rating",
        template="plotly_dark"
    )

    st.plotly_chart(figc, use_container_width=True)

#plot: top 3 movies by popularity
top_popular_movies = filtered_data.nlargest(3, 'popularity')

with cold:
    figd = px.bar(
        top_popular_movies,
        x='primaryTitle',
        y='popularity',
        color='popularity',
        title='Top 3 Movies by Popularity',
        labels={"primaryTitle": "Movie Title", "popularity": "Popularity"},
        hover_data=['releaseYear', 'genres', 'aaa_index']
    )

    figd.update_layout(
        xaxis_title="Movie Title",
        yaxis_title="Popularity",
        template="plotly_dark"
    )

    st.plotly_chart(figd, use_container_width=True)

# plot top 3 movies by budget 
top_budget_movies = filtered_data.nlargest(3, 'budget')

with cole:
    fige = px.bar(
        top_budget_movies,
        x='primaryTitle',
        y='budget',
        color='budget',
        title='Top 3 Movies by Budget',
        labels={"primaryTitle": "Movie Title", "budget": "Budget"},
        hover_data=['releaseYear', 'genres', 'revenue']
    )

    fige.update_layout(
        xaxis_title="Movie Title",
        yaxis_title="Budget",
        template="plotly_dark"
    )

    st.plotly_chart(fige, use_container_width=True)

# plot top 3 movies by revenue
top_revenue_movies = filtered_data.nlargest(3, 'revenue')

with colf:
    figf = px.bar(
        top_revenue_movies,
        x='primaryTitle',
        y='revenue',
        color='revenue',
        title='Top 3 Movies by Revenue',
        labels={"primaryTitle": "Movie Title", "budget": "Budget"},
        hover_data=['releaseYear', 'genres', 'revenue']
    )

    figf.update_layout(
        xaxis_title="Movie Title",
        yaxis_title="Budget",
        template="plotly_dark"
    )

    st.plotly_chart(figf, use_container_width=True)

### TRENDS OVER TIME SECTION!! 
st.markdown("---")
st.markdown("### Trend Data")

## Popularity by Release Year
fig = px.scatter(
    filtered_data,
    x="releaseYear",
    y="popularity",
    color="releaseYear",
    hover_name="primaryTitle",
    hover_data=["genres", "runtime_consolidated", "revenue"],
    title="Popularity by Release Year",
    labels={"releaseYear": "Release Year", "popularity": "Popularity"},
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Popularity",
    template="plotly_dark",
)

st.plotly_chart(fig)

## Popularity by Month
average_popularity_by_month = filtered_data.groupby('releaseMonth')['popularity'].mean()

average_popularity_by_month_df = average_popularity_by_month.reset_index()
average_popularity_by_month_df.columns = ['releaseMonth', 'popularity']  # Rename columns

fig_month = px.line(
    average_popularity_by_month_df,
    x="releaseMonth",
    y="popularity",
    title="Average Popularity by Release Month",
    labels={"releaseMonth": "Release Month", "popularity": "Average Popularity"},
)

fig_month.update_layout(
    xaxis_title="Release Month",
    yaxis_title="Average Popularity",
    template="plotly_dark"
)

st.plotly_chart(fig_month)

# plot: IMDb vs. TMDb vs. AAA Index Multi Lin
avg_ratings_by_year = filtered_data.groupby('releaseYear')[['imdb_averageRating', 'tmdb_averageRating', 'aaa_index']].mean().reset_index()

fig = px.line(
    avg_ratings_by_year,
    x='releaseYear',
    y=['imdb_averageRating', 'tmdb_averageRating', 'aaa_index'],
    title="Average IMDb, TMDb, and AAA Index per Year",
    labels={"value": "Average Rating", "releaseYear": "Release Year", "variable": "Rating Type"},
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Average Rating",
    template="plotly_dark"
)

st.plotly_chart(fig)

# plot: Movie Releases Over Time
movies_per_year = filtered_data.groupby('releaseYear').size().reset_index(name='count')

fig = px.line(
    movies_per_year,
    x='releaseYear',
    y='count',
    title="Movie Releases Over Time",
    labels={"releaseYear": "Release Year", "count": "Number of Movies"},
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Number of Movies",
    template="plotly_dark"
)

st.plotly_chart(fig)

# plot: Genre Releases over Time
genre_releases = filtered_data.dropna(subset=["genres"]).copy()
genre_releases['genres'] = genre_releases['genres'].str.split(',')
genre_releases = genre_releases.explode("genres")
genre_releases = genre_releases.groupby(['releaseYear', 'genres']).size().reset_index(name='count')

fig = px.line(
    genre_releases,
    x="releaseYear",
    y="count",
    color="genres",
    title="Genre Releases Over Time",
    labels={"releaseYear": "Release Year", "count": "Number of Releases"},
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Number of Releases",
    template="plotly_dark"
)

st.plotly_chart(fig)

# plot: AAA Index per Genre Over Time
genre_aaa_index = filtered_data.dropna(subset=["genres"]).copy()
genre_aaa_index['genres'] = genre_aaa_index['genres'].str.split(',')
genre_aaa_index = genre_aaa_index.explode("genres")
genre_aaa_index = genre_aaa_index.groupby(['releaseYear', 'genres'])['aaa_index'].mean().reset_index()

fig = px.line(
    genre_aaa_index,
    x="releaseYear",
    y="aaa_index",
    color="genres",
    title="AAA Index Per Genre Over Time",
    labels={"releaseYear": "Release Year", "aaa_index": "Average AAA Index"},
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Average AAA Index",
    template="plotly_dark"
)

st.plotly_chart(fig)

# plot weigthed rating per genre ove time
avg_weighted_rating_genre = filtered_data.dropna(subset=["genres"]).copy()
avg_weighted_rating_genre['genres'] = avg_weighted_rating_genre['genres'].str.split(',')
avg_weighted_rating_genre = avg_weighted_rating_genre.explode("genres")
avg_weighted_rating_genre = avg_weighted_rating_genre.groupby(['releaseYear', 'genres'])['weighted_averageRating'].mean().reset_index()

fig = px.line(
    avg_weighted_rating_genre,
    x='releaseYear',
    y='weighted_averageRating',
    color='genres',
    title="Weighted Rating per Genre Over Time",
    labels={'releaseYear': 'Release Year', 'weighted_averageRating': 'Weighted Rating'}
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Weighted Rating",
    template="plotly_dark"
)

st.plotly_chart(fig)


# plot avg number of votes over time 
avg_votes_genre = filtered_data.dropna(subset=["genres"]).copy()
avg_votes_genre['genres'] = avg_votes_genre['genres'].str.split(',')
avg_votes_genre = avg_votes_genre.explode("genres")
avg_votes_genre = avg_votes_genre.groupby(['releaseYear', 'genres'])['total_numVotes'].mean().reset_index()

fig = px.line(
    avg_votes_genre,
    x='releaseYear',
    y='total_numVotes',
    color='genres',
    title="Average Number of Votes per Genre Over Time",
    labels={'releaseYear': 'Release Year', 'total_numVotes': 'Average Votes'}
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Average Votes",
    template="plotly_dark"
)

st.plotly_chart(fig)


# plot average profitability over time
# Average Profit Ratio Over Time
avg_profit_ratio = filtered_data.groupby('releaseYear')['profit_ratio'].mean().reset_index()

fig = px.line(
    avg_profit_ratio,
    x='releaseYear',
    y='profit_ratio',
    title="Average Profit Ratios Over Time",
    labels={'releaseYear': 'Release Year', 'profit_ratio': 'Average Profit Ratio'}
)

fig.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Average Profit Ratio",
    template="plotly_dark"
)

st.plotly_chart(fig)


# plot stacked bar of total votes per genre per releaseYear

filtered_data_exploded = filtered_data.copy()
filtered_data_exploded['genres'] = filtered_data_exploded['genres'].str.split(',')
filtered_data_exploded = filtered_data_exploded.explode('genres')

votes_per_genre_year = filtered_data_exploded.groupby(['releaseYear', 'genres'])['total_numVotes'].sum().reset_index()

fig_stacked_bar = px.bar(
    votes_per_genre_year,
    x='releaseYear',
    y='total_numVotes',
    color='genres',
    title="Total Number of Votes per Genre Over Release Years",
    labels={'releaseYear': 'Release Year', 'total_numVotes': 'Total Number of Votes'},
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_stacked_bar.update_layout(
    xaxis_title="Release Year",
    yaxis_title="Total Number of Votes",
    barmode='stack',
    template='plotly_dark',
    legend_title="Genres"
)

st.plotly_chart(fig_stacked_bar, use_container_width=True)


### AAA STATISTICS SECTION!

st.markdown("---")
st.markdown('### AAA Index Statistics & Benchmarking')

aaa_col1, aaa_col2 = st.columns(2)

# plot: AAA Index Histogram w/ BM @ UB
with aaa_col1:
    fig_hist = px.histogram(
        filtered_data,
        x="aaa_index",
        nbins=30,
        title="AAA Index Distribution & Benchmark",
        labels={"aaa_index": "AAA Index"},
    )

    fig_hist.add_vline(
        x=7.88,
        line=dict(color="red", dash="dash"),
        annotation_text="Benchmark",
        annotation_position="top right"
    )

    fig_hist.update_layout(
        xaxis_title="AAA Index",
        yaxis_title="Count",
        template="plotly_dark"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

# plot: AAA Index Box PLot
with aaa_col2:
    figbox = px.box(
        filtered_data,
        x="aaa_index",
        title="AAA Index Box Plot",
        labels={"aaa_index": "AAA Index"},
    )

    figbox.update_layout(
        xaxis_title="AAA Index",
        template="plotly_dark"
    )

    st.plotly_chart(figbox, use_container_width=True)

### WORD CLOUD PARKING! 

st.markdown("---")
st.markdown("### Popular Keywords (Top 100 Voted Movies): For Brainstorming Purposes")

# plot: word cloud based on tot num_votes
from wordcloud import WordCloud

top_movies = filtered_data.nlargest(100, 'total_numVotes')

keywords_str = ' '.join(top_movies['keywords'].dropna().astype(str))

wordcloud = WordCloud(width=800, height=400, background_color="black").generate(keywords_str)

st.image(wordcloud.to_array(), use_container_width=True, caption="Top Movie Keywords Word Cloud")