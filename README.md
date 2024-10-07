# Laporan Proyek Machine Learning - Halim Sajidi

## Domain Proyek

Dalam era digital, industri telekomunikasi menghadapi tantangan yang signifikan terkait customer churn (pengurangan pelanggan). Customer churn dapat mengakibatkan kerugian besar, karena lebih mahal untuk mendapatkan pelanggan baru daripada mempertahankan yang sudah ada. Oleh karena itu, perusahaan telekomunikasi perlu mengidentifikasi pelanggan yang cenderung untuk churn sehingga mereka dapat menerapkan strategi pencegahan.

**Mengapa masalah ini penting?**:
- Churn mengurangi pendapatan perusahaan.
- Mengidentifikasi pelanggan yang berpotensi churn dapat mengarahkan perusahaan untuk menawarkan layanan yang lebih sesuai, meningkatkan loyalitas pelanggan.

Customer churn adalah masalah umum di berbagai bisnis di berbagai industri, termasuk keuangan, berita, asuransi, game mobile online, telekomunikasi, dan perjudian online. Churn management adalah konsep untuk mengidentifikasi pelanggan yang berniat untuk memindahkan kebiasaan mereka ke penyedia layanan yang bersaing. Pelanggan mungkin berhenti menggunakan produk atau layanan karena alasan yang berbeda - beberapa alasan yang mungkin tidak dapat dihindari, dan yang lainnya mungkin tidak (Saleh dan Saha 2023). Oleh karena itu, memprediksi pelanggan mana yang cenderung berpindah dan faktor-faktor yang terkait dengan preferensi mereka sangat penting untuk melindungi pendapatan berulang, meningkatkan retensi pelanggan, dan memastikan pertumbuhan.

## Business Understanding
### Problem Statements
- Bagaimana kita bisa mengidentifikasi pelanggan yang akan churn berdasarkan pola penggunaan layanan mereka?
- Fitur-fitur apa saja yang paling memengaruhi keputusan pelanggan untuk churn?

### Goals
- Mengembangkan model machine learning yang dapat memprediksi pelanggan yang berpotensi churn.
- Menganalisis fitur yang berperan penting dalam keputusan churn pelanggan.

### Solution statements
- Menggunakan LightGBM, Random Forest, XGBoost, dan SVM untuk membangun model klasifikasi.
- Melakukan hyperparameter tuning untuk meningkatkan performa model, mengoptimalkan metrik evaluasi seperti accuracy, recall, dan precision.

## Data Understanding
Dataset yang digunakan dalam proyek ini berasal dari Kaggle. Dataset ini mencakup data pelanggan dari perusahaan telekomunikasi di California, dengan 7.043 entri dan 38 kolom yang berisi Customer ID, 36 kolom independen yang digunakan untuk training model, dan  Customer Status sebagai kolom targetnya. Dataset yang digunakan masih terdapat _missing value_ sehigga perlu dilakukan pembersihan yang dilakukan pada saat _data preprocessing_. Dataset dapat di akses melalui link https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics/data 

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Customer ID: Identifikasi unik untuk setiap pelanggan dalam dataset (bertipe objek/karakter).
- Gender: Jenis kelamin pelanggan (bertipe objek "Male" atau "Female").
- Age: Usia pelanggan dalam satuan tahun (bertipe int64).
- Married: Status pernikahan pelanggan, apakah sudah menikah atau belum (bertipe objek, "Yes" atau "No").
- Number of Dependents: Jumlah tanggungan yang dimiliki pelanggan, seperti anak atau anggota keluarga lainnya (bertipe int64).
- City: Nama kota tempat tinggal pelanggan (bertipe objek).
- Zip Code: Kode pos dari tempat tinggal pelanggan (bertipe int64).
- Latitude: Koordinat garis lintang tempat tinggal pelanggan (bertipe float64).
- Longitude: Koordinat garis bujur tempat tinggal pelanggan (bertipe float64).
- Number of Referrals: Jumlah referensi atau rekomendasi yang diberikan pelanggan (bertipe int64).
- Tenure in Months: Lama pelanggan berlangganan dalam bulan (bertipe int64).
- Offer: Penawaran khusus yang diterima pelanggan (bertipe objek, dengan banyak nilai kosong).
- Phone Service: Indikasi apakah pelanggan memiliki layanan telepon atau tidak (bertipe objek, misalnya "Yes" atau "No").
- Avg Monthly Long Distance Charges: Rata-rata biaya panggilan jarak jauh per bulan (bertipe float64, dengan beberapa nilai kosong).
- Multiple Lines: Apakah pelanggan memiliki lebih dari satu jalur telepon (bertipe objek, misalnya "Yes", "No", atau "No phone service").
- Internet Service: Indikasi apakah pelanggan memiliki layanan internet atau tidak (bertipe objek, misalnya "Yes" atau "No").
- Internet Type: Jenis layanan internet yang digunakan pelanggan, seperti "DSL", "Fiber", atau "None" (bertipe objek, dengan beberapa nilai kosong).
- Avg Monthly GB Download: Rata-rata penggunaan data internet dalam GB per bulan (bertipe float64, dengan beberapa nilai kosong).
- Online Security: Apakah pelanggan memiliki layanan keamanan online atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Online Backup: Apakah pelanggan memiliki layanan cadangan data online atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Device Protection Plan: Apakah pelanggan memiliki paket perlindungan perangkat atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Premium Tech Support: Apakah pelanggan memiliki dukungan teknis premium atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Streaming TV: Apakah pelanggan berlangganan layanan streaming TV atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Streaming Movies: Apakah pelanggan berlangganan layanan streaming film atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Streaming Music: Apakah pelanggan berlangganan layanan streaming musik atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Unlimited Data: Apakah pelanggan memiliki paket data tanpa batas atau tidak (bertipe objek, dengan beberapa nilai kosong).
- Contract: Jenis kontrak yang diambil oleh pelanggan, misalnya "Month-to-Month", "One Year", atau "Two Year" (bertipe objek).
- Paperless Billing: Indikasi apakah pelanggan memilih tagihan tanpa kertas atau tidak (bertipe objek, misalnya "Yes" atau "No").
- Payment Method: Metode pembayaran yang digunakan oleh pelanggan, seperti "Bank Transfer", "Credit Card", atau "Electronic Check" (bertipe objek).
- Monthly Charge: Biaya bulanan yang dikenakan kepada pelanggan (bertipe float64).
- Total Charges: Total biaya yang dikenakan kepada pelanggan hingga saat ini (bertipe float64).
- Total Refunds: Jumlah total pengembalian dana yang diterima oleh pelanggan (bertipe float64).
- Total Extra Data Charges: Jumlah biaya tambahan untuk penggunaan data yang melebihi batas (bertipe int64).
- Total Long Distance Charges: Total biaya panggilan jarak jauh yang dikenakan kepada pelanggan (bertipe float64).
- Total Revenue: Total pendapatan yang dihasilkan dari pelanggan ini (bertipe float64).
- Customer Status (data target): Status pelanggan saat ini, apakah masih aktif, berhenti, atau churn (bertipe objek).
- Churn Category: Alasan spesifik kenapa pelanggan berhenti menggunakan layanan (bertipe objek, dengan banyak nilai kosong).
- Churn Reason: Alasan rinci pelanggan melakukan churn, seperti "Competitor Service", "Price", "Dissatisfaction" (bertipe objek, dengan banyak nilai kosong).
  
## Exploratory Data Analysis
Untuk memahami pola dan karakteristik data, beberapa visualisasi data digunakan, antara lain:

- Churn Category by Customer Status

![image](https://github.com/user-attachments/assets/d13d6339-0ef4-4b74-96dc-dcfb4fb1ab64)

Grafik ini menunjukkan bahwa sebagian besar pelanggan tetap setia menggunakan layanan, sementara pelanggan yang churn terutama disebabkan oleh kompetitor, diikuti oleh ketidakpuasan dan sikap pelanggan.

- Distribution of Gender Type

![image](https://github.com/user-attachments/assets/0453d495-0a76-4e02-9f03-7ff63c5daef0)

Berdasarkan visualisasi dapat diamati jumlah kategori gender yang tersedia. Dapat dilihat pada grafik di atas, kategori gender tidak berbeda jauh satu sama lain.

- Average Revenue by Customer Status

![image](https://github.com/user-attachments/assets/715f4e98-a5ee-4730-8c8e-93f536a898d6)

Grafik ini menunjukkan bahwa rata-rata pendapatan pelanggan yang stayed jauh lebih tinggi dibandingkan dengan pelanggan yang churn.

- Average Monthly charge by Customer Status

![image](https://github.com/user-attachments/assets/c68f4e58-d3ab-4e0e-a13b-9370a5ff1ca4)

Grafik di atas menunjukkan bahwa pelanggan dengan biaya bulanan lebih tinggi cenderung mengalami churn. 

- Payment Method vs Churn

![image](https://github.com/user-attachments/assets/8ffc0988-42f9-4694-bcfb-55636a30982d)

Berdasarkan grafik di atas, terlihat bahwa metode pembayaran bank withdrawal, diikuti oleh credit card, merupakan metode pembayaran yang paling diminati. Namun, pada pembayaran credit card selisih pengguna yang stayed dengan churned terbilang besar, sehingga dapat dikatakan bahwa metode ini adalah yang paling efektif.

- Offer vs Churn

![image](https://github.com/user-attachments/assets/4a0edc89-7936-4852-adfd-9e9638ba75d1)

Terlihat bahwa pelanggan yang menerima "Offer B" cenderung lebih banyak memilih untuk tetap bertahan (Stayed) dibandingkan dengan penawaran lainnya.

- Tenure vs Churn

![image](https://github.com/user-attachments/assets/1c0b8a0c-8e85-456d-80d7-a0f41a3d5f24)

Berdasarkan grafik di atas dapat diketahui bahwa pelanggan dengan masa berlangganan yang pendek lebih cenderung churn dibandingkan dengan pelanggan dengan masa berlangganan lebih lama.

## Data Preparation
- Handling Missing Values: Beberapa kolom seperti "Churn Reason" dan "Offer" memiliki missing values yang harus ditangani. Missing value ditangani dengan cara mengganti nilai nan dengan rata-rata untuk data numerik dan modus untuk data kategorik.
- Encoding: Kolom kategorikal seperti "Gender" dan "Contract" diubah menjadi bentuk numerik menggunakan LabelEncoder.
- Splitting Data: Membagi data ke dalam dua bagian utama Training Data dan Test Data dengan proporsi 80% dan 20% yang dilakukan dengan menggunakan fungsi train_test_split dari library Scikit-Learn.

Alasan langkah-langkah ini dilakukan adalah untuk memastikan data yang bersih dan siap digunakan dalam pemodelan, sehingga dapat menghindari bias dan meningkatkan akurasi model.

## Modeling
Algoritma ML dapat digunakan untuk klasifikasi. Beberapa algoritma yang dapat diterapkan yaitu decision trees, random forest, adaptive boosting, dan neural networks (Waiwattana et al. 2022). Dengan demikian digunakan empat algoritma utama yaitu LightGBM (LGBM), Random Forest, XGBoost, dan SVM (Support Vector Machine). Berikut penjelasan tiap algoritma:
- LightGBM (LGBM): LightGBM merupakan algoritma boosting yang bekerja dengan membangun model secara bertahap. Setiap model baru berusaha mengoreksi kesalahan dari model sebelumnya dengan fokus pada data yang sulit diprediksi. Algoritma ini efisien dan cepat dalam menangani dataset besar, dengan kemampuan menangani fitur numerik dan kategorikal serta mengurangi waktu komputasi. Cocok untuk data yang memiliki ketidakseimbangan kelas.
- Random Forest: Random Forest terdiri dari banyak decision trees yang dibangun menggunakan teknik bootstrap sampling dari dataset. Setiap tree menghasilkan prediksi, dan hasil akhir diperoleh melalui voting mayoritas untuk klasifikasi. Algoritma berbasis ensemble yang terdiri dari banyak decision trees, cocok untuk data yang bervariasi dan dapat menangani data dengan banyak fitur. Salah satu kelebihannya adalah feature importance, yang membantu mengidentifikasi fitur yang paling berpengaruh terhadap churn.
- XGBoost: XGBoost membangun model secara bertahap, dengan setiap model baru berfokus pada kesalahan yang dibuat oleh model sebelumnya. Ini membantu dalam meningkatkan akurasi. Algoritma boosting berbasis decision tree yang terkenal dengan kemampuannya dalam meningkatkan akurasi prediksi. XGBoost memiliki kemampuan untuk mengatasi overfitting dan sangat cocok untuk masalah klasifikasi.
- SVM (Support Vector Machine): SVM mencari hyperplane yang memisahkan dua kelas dengan margin maksimum. Model berusaha meminimalkan kesalahan klasifikasi dengan memaksimalkan margin antara kelas. Algoritma klasifikasi yang sangat kuat dalam memisahkan kelas dengan hyperplane terbaik. SVM sangat efektif dalam menangani data yang tidak linier dengan kernel tertentu.

Setiap model hanya diberikan parameter random_state yang didapatkan dari Scikit-learn. Random state digunakan untuk mengontrol pengacakan yang diterapkan pada data sebelum melakukan pemisahan. Tujuan utama pada tahap ini adalah membandingkan basemodel yang terbaik, setelahnya akan dilakukan tuning untuk meningkatkan hasil prediksi.

### Hyprparameter Tuning 
Pada proyek ini, dilakukan optimasi hyperparameter menggunakan Optuna untuk model LightGBM. Proses tuning hyperparameter sangat penting untuk meningkatkan performa model dalam melakukan klasifikasi. Optuna memungkinkan optimasi hyperparameter secara otomatis melalui pendekatan Bayesian Optimization, yang mempercepat proses pencarian kombinasi terbaik dari parameter yang dicoba.

**Hyperparameter yang Dioptimalkan**:
```bash
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_class': len(np.unique(y)),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
```
Berdasarkan parameter di atas terdapat beberapa parameter yang perlu di perhatikan seperti parameter objective yang menggunakan multiclass karena menyesuaikan dengan data target yang memiliki tiga klasifikasi dan num_class yang menyesuaikan data target. Beberapa paramter lainnya dipertimbangkan berdasarkan device yang digunakan agar tidak terlalu kompleks dan juga tidak terlalu sederhana.

## Evaluation
Metrik yang digunakan untuk mengevaluasi model adalah Akurasi, Precision, Recall, dan F1-Score. Metrik ini dipilih karena sesuai dengan karakteristik masalah churn di mana Recall (kemampuan model mendeteksi churn) sangat penting untuk mengurangi potensi kehilangan pelanggan.
- Accuracy: Mengukur persentase prediksi yang benar.
- Precision: Mengukur seberapa baik model menghindari false positives.
- Recall: Mengukur kemampuan model dalam mendeteksi churn (true positives).
- F1-Score: Kombinasi antara precision dan recall, yang memberikan gambaran umum tentang kinerja model pada dataset yang tidak seimbang.

Accuracy=  (TP+TN)/(TP+TN+FP+FN)

Precision=  TP/(TP+FP)

Recall=  TP/(TP+FN)

F1-score=2×(Precision×Recall)/(Precision+Recall)

Keterangan:
TP	=	Jumlah Signal yang diprediksi benar sebagai signal.
TN	=	Jumlah background yang diprediksi benar sebagai background.
FP	=	Jumlah background yang salah diprediksi sebagai signal.
FN	=	Jumlah signal yang salah diprediksi sebagai background.


## Hasil Proyek
Setelah melakukan tuning, model LightGBM memberikan hasil terbaik dengan metrik sebagai berikut:
- Akurasi: 95%
- Precision: 91%
- Recall: 92%
- F1-Score: 91%

Model LightGBM dipilih sebagai model final karena memiliki performa terbaik dalam menyeimbangkan recall dan precision, yang sangat penting dalam kasus prediksi churn. Model ini juga lebih efisien dalam komputasi dibandingkan model lainnya.

## Kesimpulan
Berdasarkan analisis data dan visualisasi yang telah dilakukan, berikut adalah kesimpulan untuk menjawab problem statements:

1. **Bagaimana kita bisa mengidentifikasi pelanggan yang akan churn berdasarkan pola penggunaan layanan mereka?**
   - Pelanggan yang lebih mungkin mengalami churn biasanya memiliki beberapa pola yang dapat diidentifikasi, seperti biaya bulanan yang lebih tinggi, durasi berlangganan yang lebih pendek, dan metode pembayaran tertentu (seperti **credit card**). Model machine learning (LGBM) dengan akurasi 95% dapat membantu memprediksi pelanggan yang akan churn berdasarkan fitur-fitur ini, seperti **biaya bulanan**, **metode pembayaran**, dan **tenure** (masa berlangganan).

2. **Fitur-fitur apa saja yang paling memengaruhi keputusan pelanggan untuk churn?**
   - Fitur-fitur utama yang memengaruhi churn meliputi:
     - **Tenure** (masa berlangganan): Pelanggan dengan durasi berlangganan lebih pendek lebih cenderung churn.
     - **Average Monthly Charges** (biaya bulanan rata-rata): Pelanggan dengan biaya bulanan yang lebih tinggi lebih cenderung churn.
     - **Payment Method** (metode pembayaran): Metode pembayaran melalui **credit card** menunjukkan perbedaan signifikan antara pelanggan yang churn dan yang stayed.
     - **Offer** (penawaran): Pelanggan yang menerima **Offer B** lebih cenderung tetap bertahan, sementara penawaran lain kurang efektif dalam mempertahankan pelanggan.
     - **Customer Satisfaction** (kepuasan pelanggan) juga menjadi faktor penting, di mana sebagian besar pelanggan churn disebabkan oleh kompetitor atau ketidakpuasan dengan layanan.

3. **Churn Category by Customer Status**
   - Mayoritas pelanggan yang churn disebabkan oleh **kompetitor**, diikuti oleh ketidakpuasan pelanggan terhadap layanan. Hal ini menunjukkan bahwa **peningkatan kualitas layanan** dan **loyalty program** dapat menjadi langkah strategis untuk mengurangi churn.

Melalui implementasi **model machine learning LGBM**, prediksi churn dapat dilakukan dengan tingkat akurasi yang tinggi (95%), sehingga memungkinkan perusahaan untuk **mengantisipasi churn lebih dini** dan melakukan **strategi mitigasi** yang tepat.


## Referensi
Saleh S, Saha S. 2023. Customer retention and churn prediction in the telecommunication industry: a case study on a Danish university. SN Appl. Sci. 5(7):undefined-undefined.doi:10.1007/s42452-023-05389-6.

Waiwattana J, Asawatangtrakuldee C, Saksirimontri P, Wachirapusitanand V, Pitakkultorn N. 2022. Application of Machine Learning Algorithms for Searching BSM Higgs Bosons Decaying to a Pair of Bottom Quarks. Trends Sci. 19(19).doi:10.48048/tis.2022.5373.

**---Ini adalah bagian akhir laporan---**

