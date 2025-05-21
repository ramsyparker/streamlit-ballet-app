from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pymongo
from pymongo.errors import BulkWriteError
import plotly.express as px
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Koneksi MongoDB
try:
    mongo_uri = st.secrets["mongo"]["uri"]
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    st.success("✔️ Koneksi ke MongoDB Atlas berhasil")
except Exception as e:
    st.error(f"❌ Gagal koneksi ke MongoDB Atlas:\n{e}")
    st.stop()

db = client["bigdata"]
collection = db["ballet"]

# Hapus duplikat berdasarkan link
# def remove_duplicates():
#     pipeline = [
#         {"$group": {"_id": "$link", "ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
#         {"$match": {"count": {"$gt": 1}}}
#     ]
#     duplicates = list(collection.aggregate(pipeline))
#     total_deleted = 0
#     for doc in duplicates:
#         ids_to_delete = doc["ids"][1:]
#         result = collection.delete_many({"_id": {"$in": ids_to_delete}})
#         total_deleted += result.deleted_count
#     return total_deleted

# deleted_count = remove_duplicates()
# if deleted_count > 0:
#     st.info(f"🗑️ Menghapus {deleted_count} data duplikat berdasarkan link")

# # Buat index unik pada field 'link'
# try:
#     idx_name = collection.create_index("link", unique=True)
#     st.info(f"✔️ Index unik pada 'link' berhasil dibuat ({idx_name})")
# except Exception as e:
#     st.warning(f"⚠️ Gagal membuat index unik pada 'link': {e}")

ballet_vocabulary = {
    # (Isi seperti kode sebelumnya)
    'ballet', 'dancer', 'dance', 'performance', 'rehearsal', 'choreography', 'balletic', 'pirouette',
    'ballerina', 'balletschool', 'balletcompany', 'pas', 'tendu', 'arabesque', 'pointe', 'pas de deux',
    'balletdance', 'balletperformance', 'balletclass', 'danceacademy', 'danceday', 'balletshow',
    'balet', 'penari', 'tari', 'pertunjukan', 'latihan', 'koreografi', 'gerakan balet', 'putaran',
    'ballerina', 'sekolah balet', 'perusahaan balet', 'langkah', 'tendu', 'arabesque', 'ujung jari',
    'pas de deux', 'tari balet', 'pertunjukan balet', 'kelas balet', 'akademi tari', 'hari tari', 'pertunjukan balet',
    'ballet klasik', 'pentas balet', 'gerak balet', 'tarian balet', 'komunitas balet', 'pembelajaran balet',
    'pas ballerina', 'karya balet'
}

stop_words = set(nltk.corpus.stopwords.words('indonesian')).union({
    'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau',
    'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 'dia', 'mereka', 'kita', 'akan',
    'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah', 'para', 'seperti', 'saat',
    'bagi', 'serta', 'tapi', 'lain', 'sebuah', 'karena', 'ketika', 'jika', 'apa',
    'seorang', 'tentang', 'dalam', 'bisa', 'sementara', 'dilakukan', 'setelah',
    'yakni', 'menurut', 'hampir', 'dimana', 'bagaimana', 'selama', 'sebelum',
    'hingga', 'kepada', 'sebagai', 'masih', 'hal', 'sempat', 'sedang', 'selain',
    'sembari', 'mendapat', 'sedangkan', 'tetapi', 'membuat', 'namun', 'gimana'
})

def scrape_detik():
    articles = []
    base_url = "https://www.detik.com/tag/balet"
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    try:
        response = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles_container = soup.find_all('div', class_='list-content__item')
        for article in articles_container:
            try:
                title_element = article.find('a', class_='media__title')
                link_element = title_element
                date_element = article.find('span', class_='media__date')
                category_element = article.find('div', class_='media__category')
                if title_element and link_element and date_element:
                    title = title_element.text.strip()
                    link = link_element['href']
                    date = date_element.text.strip()
                    category = category_element.text.strip() if category_element else "Unknown"
                    articles.append({
                        'source': 'Detik',
                        'title': title,
                        'date': date,
                        'link': link,
                        'category': category,
                        'scraped_at': datetime.now()
                    })
            except Exception as e:
                st.write(f"Error parsing article: {e}")
        st.success(f"Found {len(articles)} articles from Detik")
    except Exception as e:
        st.error(f"Error scraping Detik: {e}")
    return articles

def save_to_mongodb(articles):
    if not articles:
        st.warning("No articles to save")
        return
    try:
        for article in articles:
            collection.update_one(
                {"link": article["link"]},
                {"$setOnInsert": article},
                upsert=True
            )
        st.success(f"Saved {len(articles)} articles to MongoDB (avoiding duplicates)")
    except Exception as e:
        st.error(f"Error saving to MongoDB: {e}")

def visualize_data():
    data = list(collection.find())
    df = pd.DataFrame(data)
    if df.empty:
        st.warning("No data available for visualization")
        return
    df['scraped_at'] = pd.to_datetime(df['scraped_at'])
    df['year'] = df['scraped_at'].dt.year.astype(int)
    df['month'] = df['scraped_at'].dt.month
    sources = df['source'].unique()
    selected_source = st.multiselect('Select Sources', sources, default=sources, key="source_selection")
    filtered_df = df[df['source'].isin(selected_source)]
    st.subheader("1. Number of Articles by Source")
    source_counts = filtered_df['source'].value_counts()
    fig1 = px.bar(source_counts, title='Total Articles by Source')
    st.plotly_chart(fig1)
    st.subheader("2. Yearly Trends")
    yearly_data = filtered_df.groupby('year').size().reset_index(name='count')
    yearly_data['year'] = yearly_data['year'].astype(int)
    fig2 = px.bar(yearly_data, x='year', y='count', title='Articles by Year')
    fig2.update_xaxes(tickmode='array', tickvals=yearly_data['year'].unique())
    st.plotly_chart(fig2)
    st.subheader("3. Monthly Trends for Selected Year")
    selected_year = st.selectbox('Select Year for Monthly Breakdown', sorted(df['year'].unique()), key="year_selection")
    monthly_data = filtered_df[filtered_df['year'] == selected_year].groupby(['month', 'source']).size().reset_index(name='count')
    monthly_data['month'] = pd.to_datetime(monthly_data['month'], format='%m').dt.strftime('%B')
    fig3 = px.bar(monthly_data, x='month', y='count', color='source', title=f'Monthly Articles in {selected_year}')
    st.plotly_chart(fig3)
    st.subheader("4. Most Common Words - Bar Chart & Word Cloud")
    try:
        all_words = []
        for title in filtered_df['title']:
            words = title.lower().split()
            words = [word.strip('.,!?()[]{}:;"\'') for word in words]
            words = [word for word in words if word and word not in stop_words and word in ballet_vocabulary]
            all_words.extend(words)
        if all_words:
            wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(all_words))
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            st.subheader("5. Most Common Words - Bar Chart")
            word_freq = pd.Series(all_words).value_counts().head(15)
            fig4 = px.bar(word_freq, title='Most Common Words in Titles', labels={'index':'Word', 'value':'Frequency'})
            st.plotly_chart(fig4)
        else:
            st.warning("No words to analyze after filtering")
    except Exception as e:
        st.error(f"Error in word frequency analysis: {e}")
    st.subheader("5. Article Details")
    for source in selected_source:
        st.write(f"\n**{source} Articles**")
        source_df = filtered_df[filtered_df['source'] == source][['title', 'date', 'link', 'scraped_at']]
        st.dataframe(source_df)

def main():
    st.title("Ballet News Scraper")
    if st.button("Scrape & Save Latest Articles"):
        articles = scrape_detik()
        save_to_mongodb(articles)
    tab1, tab2 = st.tabs(["Recent Articles", "Visualizations"])
    with tab1:
        st.subheader("Recent Articles")
        articles = list(collection.find().sort('scraped_at', -1).limit(10))
        for article in articles:
            st.write(f"**{article['title']}**")
            st.write(f"Date: {article['date']} | Category: {article['category']}")
            st.write(f"[Read more]({article['link']})")
            st.markdown("---")
    with tab2:
        st.subheader("Data Visualization")
        visualize_data()

if __name__ == "__main__":
    main()
