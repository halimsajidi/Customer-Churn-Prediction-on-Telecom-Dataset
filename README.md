# Laporan Proyek Machine Learning - Halim Sajidi

## Domain Proyek

Dalam era digital, industri telekomunikasi menghadapi tantangan yang signifikan terkait customer churn (pengurangan pelanggan). Customer churn dapat mengakibatkan kerugian besar, karena lebih mahal untuk mendapatkan pelanggan baru daripada mempertahankan yang sudah ada. Oleh karena itu, perusahaan telekomunikasi perlu mengidentifikasi pelanggan yang cenderung untuk churn sehingga mereka dapat menerapkan strategi pencegahan.

**Mengapa masalah ini penting?**:
- Churn mengurangi pendapatan perusahaan.
- Mengidentifikasi pelanggan yang berpotensi churn dapat mengarahkan perusahaan untuk menawarkan layanan yang lebih sesuai, meningkatkan loyalitas pelanggan.

  Riset menunjukkan bahwa 67% pelanggan yang churn menganggap layanan yang mereka terima tidak memenuhi harapan mereka, dan perusahaan yang tidak memiliki upaya pencegahan churn yang memadai dapat kehilangan hingga 25-30% basis pelanggannya per tahun.
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

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
Dataset yang digunakan dalam proyek ini berasal dari Kaggle. Dataset ini mencakup data pelanggan dari perusahaan telekomunikasi di California, dengan 7.043 entri dan 38 kolom. Dataset dapat di akses melalui link https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics/data 

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

## Exploratory Data Analysis
Untuk memahami pola dan karakteristik data, beberapa visualisasi data digunakan, antara lain:

- Distribution of Gender Type

![image](https://github.com/user-attachments/assets/0453d495-0a76-4e02-9f03-7ff63c5daef0)

Berdasarkan visualisasi dapat diamati jumlah kategori gender yang tersedia. Dapat dilihat pada grafik di atas, kategori gender tidak berbeda jauh satu sama lain.

- Payment Method Distribution

![image](https://github.com/user-attachments/assets/206bd770-0e69-4381-b056-c5038b8e28f4)



- Churn Category by Gender

![image](https://github.com/user-attachments/assets/6a16e62d-252a-48ef-9aa4-c107784ecf78)

- Internet Service vs Churn Category

![image](https://github.com/user-attachments/assets/d1067fa2-9816-4113-9f91-133fd81ff6c5)

- Distribution of Internet Type

![image](https://github.com/user-attachments/assets/ec40568b-9679-4ab1-9976-0225ed9a6938)


## Data Preparation
- Handling Missing Values: Beberapa kolom seperti "Churn Reason" dan "Offer" memiliki missing values yang harus ditangani.
- Encoding: Kolom kategorikal seperti "Gender" dan "Contract" diubah menjadi bentuk numerik menggunakan LabelEncoder.
- Splitting Data: Membagi data ke dalam dua bagian utama Training Data dan Test Data dengan proporsi 80% dan 20% yang dilakukan dengan menggunakan fungsi train_test_split dari library Scikit-Learn.

Alasan langkah-langkah ini dilakukan adalah untuk memastikan data yang bersih dan siap digunakan dalam pemodelan, sehingga dapat menghindari bias dan meningkatkan akurasi model.

## Modeling
Pada tahap ini, empat algoritma utama yang digunakan adalah LightGBM (LGBM), Random Forest, XGBoost, dan SVM (Support Vector Machine). Berikut penjelasan tiap algoritma:
- LightGBM (LGBM): Algoritma yang efisien dan cepat dalam menangani dataset besar, dengan kemampuan menangani fitur numerik dan kategorikal serta mengurangi waktu komputasi. Cocok untuk data yang memiliki ketidakseimbangan kelas.
- Random Forest: Algoritma berbasis ensemble yang terdiri dari banyak decision trees, cocok untuk data yang bervariasi dan dapat menangani data dengan banyak fitur. Salah satu kelebihannya adalah feature importance, yang membantu mengidentifikasi fitur yang paling berpengaruh terhadap churn.
- XGBoost: Algoritma boosting berbasis decision tree yang terkenal dengan kemampuannya dalam meningkatkan akurasi prediksi. XGBoost memiliki kemampuan untuk mengatasi overfitting dan sangat cocok untuk masalah klasifikasi.
- SVM (Support Vector Machine): Algoritma klasifikasi yang sangat kuat dalam memisahkan kelas dengan hyperplane terbaik. SVM sangat efektif dalam menangani data yang tidak linier dengan kernel tertentu.

### Hyprparameter Tuning 
Pada proyek ini, dilakukan optimasi hyperparameter menggunakan Optuna untuk model LightGBM. Proses tuning hyperparameter sangat penting untuk meningkatkan performa model dalam melakukan klasifikasi. Optuna memungkinkan optimasi hyperparameter secara otomatis melalui pendekatan Bayesian Optimization, yang mempercepat proses pencarian kombinasi terbaik dari parameter yang dicoba.

**Hyperparameter yang Dioptimalkan**:
Beberapa hyperparameter penting yang dioptimalkan adalah learning_rate, num_leaves, max_depth, n_estimators, min_child_samples, subsample, dan colsample_bytree.

## Evaluation
Metrik yang digunakan untuk mengevaluasi model adalah Akurasi, Precision, Recall, dan F1-Score. Metrik ini dipilih karena sesuai dengan karakteristik masalah churn di mana Recall (kemampuan model mendeteksi churn) sangat penting untuk mengurangi potensi kehilangan pelanggan.
- Accuracy: Mengukur persentase prediksi yang benar.
- Precision: Mengukur seberapa baik model menghindari false positives.
- Recall: Mengukur kemampuan model dalam mendeteksi churn (true positives).
- F1-Score: Kombinasi antara precision dan recall, yang memberikan gambaran umum tentang kinerja model pada dataset yang tidak seimbang.

## Hasil Proyek
Setelah melakukan tuning, model LightGBM memberikan hasil terbaik dengan metrik sebagai berikut:

- Akurasi: 95%
- Precision: 91%
- Recall: 92%
- F1-Score: 91%

Model LightGBM dipilih sebagai model final karena memiliki performa terbaik dalam menyeimbangkan recall dan precision, yang sangat penting dalam kasus prediksi churn. Model ini juga lebih efisien dalam komputasi dibandingkan XGBoost.

**---Ini adalah bagian akhir laporan---**

