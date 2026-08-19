# BuzzGuard

**Framework machine learning hybrid untuk deteksi bot dan indikasi akun buzzer terkoordinasi, berbasis fitur non-teks**

BuzzGuard membedakan tiga hal di media sosial: akun bot otomatis, akun manusia asli, dan indikasi koordinasi buzzer. Bedanya dengan sistem deteksi bot kebanyakan: BuzzGuard nggak nganalisis isi teks tweet sama sekali. Alasannya sederhana, konten teks makin gampang dipalsukan seiring makin canggihnya AI generatif. Jadi fokusnya dipindah ke dua hal yang lebih susah dipalsukan: gimana akun itu berperilaku, dan gimana akun itu berkoordinasi sama akun lain.
![alt text](image-1.png)
## Kenapa 2 tahap, bukan classifier 3 kelas?

Ini bagian penting yang perlu dijelasin di awal. Nggak ada dataset publik manapun (termasuk cresci-2017 yang dipakai di sini) yang punya label "buzzer" beneran. Yang ada cuma label bot dan human. Riset soal *coordinated inauthentic behavior* (CIB) di literatur juga pada dasarnya unsupervised, nyari pola koordinasi, bukan melatih classifier dari label yang memang nggak tersedia.

Makanya BuzzGuard dibikin dua tahap:

1. **Tahap 1, deteksi bot (supervised).** Logistic Regression, Random Forest, dan XGBoost dilatih pakai label bot/human asli dari dataset.
2. **Tahap 2, buzzer flagging (unsupervised).** Akun yang lolos Tahap 1 sebagai "human" dicek lagi pakai Isolation Forest, khusus pada fitur koordinasi antar-akun. Yang polanya mencurigakan ditandai sebagai kandidat buzzer.

Lebih ribet dikit dibanding classifier 3 kelas langsung, tapi jujur secara metodologi. Nggak maksa ada ground truth yang sebenarnya nggak ada.

## Fitur

- Deteksi bertingkat: classifier bot/human (Tahap 1) + flagging koordinasi (Tahap 2)
- Dua kelompok fitur:
  - Individual akun (usia akun, rasio follower/following, frekuensi posting, dll.)
  - Koordinasi antar-akun (kemiripan waktu posting, overlap hashtag, kemiripan konten)
- Ablation: tiap model dites dua kali, pakai fitur individual saja vs individual+koordinasi, buat buktiin fitur koordinasi beneran nambah nilai
- Explainability pakai SHAP (Tahap 1) dan validasi diagnostik (Tahap 2)
- Metrik: Accuracy, Precision, Recall, F1-score, AUC-ROC

## Struktur Proyek

```text
BuzzGuard/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── lr_feature_A.pkl
│   ├── lr_feature_AB.pkl
│   ├── rf_feature_A.pkl
│   ├── rf_feature_AB.pkl
│   ├── xgb_feature_A.pkl
│   └── xgb_feature_AB.pkl
│
├── notebooks/
│   ├── 01_Merge_Dataset.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Logistic_Regression.ipynb
│   ├── 05_Random_Forest.ipynb
│   ├── 06_XGBoost.ipynb
│   ├── 07_Model_Comparison.ipynb
│   ├── 08_SHAP_Analysis.ipynb
│   └── 09_Buzzer_Flagging.ipynb
│
├── results/
├── README.md
└── requirements.txt
```

## Alur Metodologi

```text
Dataset (cresci-2017)
        │
        ▼
Preprocessing & Feature Engineering
        │
        ├── Fitur A: Individual Akun
        └── Fitur B: Koordinasi Antar-Akun
        │
        ▼
Tahap 1: Deteksi Bot (Supervised)
LR / Random Forest / XGBoost
dites 2x, Fitur A saja vs Fitur A+B
        │
        ▼
Akun terprediksi "Human"
        │
        ▼
Tahap 2: Buzzer Flagging (Unsupervised)
Isolation Forest pada fitur koordinasi
        │
        ▼
Output: Bot / Human Asli / Kandidat Buzzer
        │
        ▼
SHAP (Tahap 1) + Validasi Diagnostik (Tahap 2)
```

## Kelompok Fitur

**Fitur Individual.** Menggambarkan satu akun sendirian, nggak dibandingin sama akun lain:
- Usia akun
- Jumlah follower & following
- Rasio follower/following
- Jumlah tweet
- Kelengkapan profil

**Fitur Koordinasi.** Nangkep pola yang cuma kelihatan kalau lihat banyak akun sekaligus:
- Kemiripan waktu posting antar-akun
- Overlap penggunaan hashtag
- Kemiripan konten/teks antar-akun

## Model

| Model | Peran | Dievaluasi dengan |
|---|---|---|
| Logistic Regression | Baseline | Fitur A / Fitur A+B |
| Random Forest | Baseline lebih kuat | Fitur A / Fitur A+B |
| XGBoost | Model utama | Fitur A / Fitur A+B |

Total 6 kombinasi eksperimen (3 model kali 2 skema fitur) buat ablation study Tahap 1.

## Explainability

Tahap 1 pakai SHAP di model XGBoost. Bukan cuma buat laporan angka akurasi, tapi juga buat lihat fitur individual dan fitur koordinasi mana yang paling nentuin prediksi bot/human, termasuk berapa persen kontribusi masing-masing kelompok fitur.

Tahap 2 divalidasi secara diagnostik. Caranya: cek berapa persen akun yang di-flag jadi kandidat buzzer itu ternyata memang berlabel bot di ground truth, artinya sempat lolos Tahap 1 tapi ketangkep di Tahap 2. Angka ini dibandingin sama rata-rata bot di seluruh populasi akun "human", biar kelihatan Tahap 2 nangkep sesuatu yang beneran, bukan noise random.

## Hasil

Repositori ini isinya:
- Tabel perbandingan model (ablation Fitur A vs A+B)
- Confusion matrix
- ROC curve
- SHAP summary plot
- Hasil buzzer flagging plus visualisasi scatter

Contoh visualisasinya ada di folder `results/`.

## Dataset

Dataset utama yang dipakai: **cresci-2017**, dari Bot Repository (Indiana University).

Link download: https://botometer.osome.iu.edu/bot-repository/datasets.html

Dipilih karena profil akun dan data tweet-nya lengkap buat bikin fitur individual maupun fitur koordinasi, dan bisa didownload langsung tanpa proses approval (beda sama TwiBot-20/22 yang harus minta izin ke penulisnya dulu).

Satu catatan penting: pola koordinasi di cresci-2017 itu asalnya dari kampanye promosi komersial, bukan buzzer politik Indonesia. Jadi dataset ini diperlakukan sebagai proxy, bukan representasi langsung dari kasus buzzer Indonesia.

## Instalasi

```bash
git clone https://github.com/askyaya31/BuzzGuard.git
cd BuzzGuard
pip install -r requirements.txt
```

## Menjalankan Proyek

Jalankan notebook urut dari nomor 1 sampai 9:

1. `01_Merge_Dataset.ipynb`
2. `02_Preprocessing.ipynb`
3. `03_Feature_Engineering.ipynb`
4. `04_Logistic_Regression.ipynb`
5. `05_Random_Forest.ipynb`
6. `06_XGBoost.ipynb`
7. `07_Model_Comparison.ipynb`
8. `08_SHAP_Analysis.ipynb`
9. `09_Buzzer_Flagging.ipynb`

## Pengembangan Lanjutan

- Integrasi Graph Neural Network buat fitur relasi jaringan yang lebih kaya
- Studi kasus akun Twitter/X Indonesia beneran
- Pipeline deteksi real-time
- Deployment web sederhana
- Dukungan platform media sosial lain
