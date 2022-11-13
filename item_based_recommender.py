###########################################
# Item-Based Collaborative Filtering
###########################################
#Item, ürün, film benzerliği üzerinden öneri yapıyoruz.!İçerik temelliden farkımız var
#Item_Based'ın olayı: izlenen filmin beğenilme yapısına benzer film aramak
#yani bunun olayı şöyle: mehmet bey aşkı memnu izlemiş ona ne önerelim fatmagülün suçu ne önerelim çünkü pattern bunu gösteriyor mehmet bey de toplumun bir parçası

#Topluluğun kanatlerini barındırıcak öneriler sunucaz bir önceki gibi matrix'le aynı kelimeleri içerenleri değil


# Veri seti: https://grouplens.org/datasets/movielens/ #veri setimizin adresi sadece movie.csv ve rating.csv dosyaları lazım bize



# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500) #olası tüm değişkenleri görmek için satırları görmek için yaptık bu satırı
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId") #left join olacak şekilde movıeıd ye göre değişkenlerimizi birleştirdik
df.head()


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

#burada sıkıntı matrislerdeki seyreklik sorunu.
#Yani sıkıntı şu adam gelmiş sadece tek bi filme puan vermiş diğerlerine vermemiş user-movie matrisinde bu da yer kaplıyor 
#böyle olunca yapılacak olan hesaplamaları geciktirecek bu adam
#o yüzden bazı indirgemeler yapmamız lazım

df.head()


df.shape
#bu kodun çıktısı kaç yorum olduğunu bize verir out:20m yaklaşık

df["title"].nunique() #eşsiz film sayısı out 20k yaklaşık

df["title"].value_counts().head()
#bu yukarıdaki kodu yazma sebebimiz şu veri setinde filmlere birden çok yorum yapıldı filmler toplamda birden çok gözükücek 
#yorum yapalım 1k'dan az yorum alan filmleri hiç önermeyelim çok yoğun hesaplama maliyetleri ve zaman kaybı

comment_counts = pd.DataFrame(df["title"].value_counts())
#yukarıdaki kodun çıktısıyla her bir filme kaç tane yorum verilmiş ya da puan ona baktık 
#kodun çıktısında title aslında count ifadesi o karışabilir titleın altında filmin kaç tane yorum aldığı yazıyor

rare_movies = comment_counts[comment_counts["title"] <= 1000].index #sonuna index koyduk isimlerini de verdi
#yukarıdaki kodla dedik ki hadi python kardeş 1000 ve aşağısı yorumları göster bakalım gördük ki 24k film 1k altı 
#
common_movies = df[~df["title"].isin(rare_movies)] #isin'le yukarıda yazdığımız rare movies kodunun çıktılarının içinde gez dedik tildayla da bunları seçme kardeş dedik

common_movies.shape #out:17 milyon rate var az önce 20 milyonduu yani aslında az sayıda silmişiz hadi o zaman kaç tane eşsiz sayıda film silmişiz
common_movies["title"].nunique()#burda da 3k küsür film kaldı yaptığımız işlemlerden sonra
df["title"].nunique()#veri setinin ilk hali 27k film varmış
# e fark ettik ki yaptığımız indirgeme mantıklıymışke

#bu pivot table ağır çalışıyor bilgisayarına dikkat et kardeşimmmm!!!!
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
#yukarıda yaptık güzelce satırlarda kullanıcı sütunda filmlerin adı kırılım da verdiği rate olsun dedik güzelce ama 
#çıktı yine biraz sıkıntılı çıktı niye çünkü kullanıcı tek filme yorum yapmış işte aslında kullanıcılara da indirgeme işlemi yapmak lazım

user_movie_df.shape
#out:138493,3159  ilki kullanıcılar ikincisi filmler

user_movie_df.columns # bu da bize tüm titleları verirkeee


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################
#had bakalım şimdi önerilerimizi yapalım 
#korelasyon bakar gibi gidicezke

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]

#bu kadar işlem yaptık şimdi ne var korelasyonumuzz
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
#user movie dfyi al korelasyona bak movie_name ile dedik sonra da büyükten küçüğe sırala ve ilk 10 gözlemi ver dedik
#işbirlikçi filtremizle önerimizi toplumun beğenilerine göre sundukk 
#bu filmi izleyen insanların verdiği ratelere benzer ratelemelere sahip önerilerr


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0] #bunu al istediğin filme bak 
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]
#bu yukarıdaki kardeşte user movie df'in içinde senin yazacağın anahtar kelimeyi arayacak filmin adını unuttuysan ve bir kaç şey hatırında kaldıysa bulmak istiyorsan

check_film("Insomnia", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





