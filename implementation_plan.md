# Implementation Plan — Multiclass Network Intrusion Classification on CIC-ToN-IoT (Autoencoder + LightGBM)

Rencana ini memecah proposal menjadi fase implementasi yang bisa dikerjakan berurutan. Setiap fase punya tujuan, tugas konkret, deliverable, dan acceptance criteria supaya progres gampang dicek sebelum lanjut ke fase berikutnya.

Referensi utama: proposal `Multiclass_Network_Intrusion_Classification_on_the_CIC_ToN_IoT_Dataset_Using_Autoencoder_and_LightGBM.pdf` (Bab III — Methodology and System Design, hlm. 24–37).

---

## Struktur project yang disarankan

```
.
├── data/
│   ├── raw/                # CIC-ToN-IoT asli (tidak diubah, tidak di-commit ke git)
│   ├── interim/             # hasil cleaning sebelum split
│   └── processed/           # train/test yang sudah dinormalisasi + latent features
├── configs/
│   └── config.yaml          # semua path & hyperparameter di satu tempat
├── src/
│   ├── data_loading.py      # load & inspeksi awal dataset
│   ├── preprocessing.py     # cleaning, encoding, split, normalisasi
│   ├── autoencoder.py       # arsitektur + training + save/load model
│   ├── latent_extraction.py # encode train/test jadi latent features
│   ├── imbalance.py         # 4 skenario: none, class_weight, upsample, downsample
│   ├── classifier.py        # training & prediksi LightGBM
│   └── evaluation.py        # accuracy, macro P/R/F1, confusion matrix
├── notebooks/                # eksplorasi ad-hoc (EDA, debugging)
├── results/
│   ├── metrics/              # csv/json hasil tiap skenario
│   └── figures/              # confusion matrix, training curve, dst.
├── run_pipeline.py          # entry point orkestrasi end-to-end
└── implementation_plan.md
```

Struktur ini mengikuti pemisahan tahapan yang sudah digambarkan di Gambar 3.1 (System Flowchart) proposal, supaya tiap fungsi Python punya tanggung jawab yang jelas dan gampang ditest terpisah.

---

## Phase 0 — Project Setup & Environment

**Tujuan:** environment siap, dataset tersedia, dependency terpasang.

- [x] Inisialisasi git repo + `.gitignore` (exclude `data/raw`, `data/interim`, model checkpoint besar).
- [x] Buat virtual environment (conda/venv), Python 3.10+.
- [x] Install dependency inti: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `tensorflow` atau `torch` (pilih satu untuk Autoencoder), `imbalanced-learn` (untuk upsampling/downsampling), `matplotlib`/`seaborn`, `pyyaml`.
- [x] Download dataset CIC-ToN-IoT, taruh di `data/raw/`.
- [x] Buat `configs/config.yaml` berisi path, random_state=42, rasio split 80:20, ukuran latent=16, dsb.

**Deliverable:** repo terstruktur, `requirements.txt`/`environment.yml`, dataset ada di `data/raw/`.

**Acceptance:** `python -c "import pandas, lightgbm, tensorflow"` (atau torch) tidak error; file dataset terbaca.

---

## Phase 1 — Data Loading & Inspection

**Tujuan:** memahami struktur dataset sebelum diproses (sesuai proposal 3.1.a & 3.2).

- [x] `src/data_loading.py`: fungsi load CSV dataset CIC-ToN-IoT.
- [x] Cek jumlah baris/kolom (ekspektasi ~5,351,760 baris, 85 kolom).
- [x] Inspeksi tipe data tiap kolom, jumlah missing value, jumlah nilai unik.
- [x] Verifikasi kolom label `Attack` dan daftar 10 kelas (Benign, Backdoor, DDoS, DoS, Injection, MITM, Password, Ransomware, Scanning, XSS).
- [x] Hitung & simpan distribusi kelas (harus mendekati Tabel 3.2 di proposal) → `results/metrics/class_distribution.csv`.
- [x] Identifikasi kolom identifier yang akan dibuang (Flow ID, Src IP, Dst IP, Timestamp, dll).
- [x] Identifikasi kandidat fitur quasi-constant (variance sangat rendah).

**Deliverable:** notebook/report singkat EDA + daftar final kolom yang dipakai (target 69 fitur numerik).

**Acceptance:** distribusi kelas & jumlah fitur yang dihasilkan cocok dengan Tabel 3.1/3.2 proposal (atau terdokumentasi kalau berbeda).

---

## Phase 2 — Data Preprocessing Pipeline

**Tujuan:** dataset bersih, siap dipakai model, tanpa data leakage (proposal 3.3, Tabel 3.3).

- [x] `src/preprocessing.py`:
  - Drop kolom identifier/metadata (Flow ID, Src IP, Dst IP, Timestamp).
  - Drop fitur quasi-constant.
  - Konversi `inf`/`-inf` → NaN, lalu handle missing value (drop baris jika sedikit, drop kolom jika parah).
  - Drop baris dengan label kosong/tidak valid.
  - Label encoding kolom `Attack` → index numerik (ikuti mapping Tabel 2.5).
  - **Stratified split 80:20** (train/test) — split dulu sebelum normalisasi.
  - Fit `MinMaxScaler` HANYA di data train, lalu transform train & test dengan scaler yang sama.
- [x] Simpan hasil: `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, serta scaler (`joblib`) dan label encoder.
- [x] Unit test kecil: pastikan tidak ada NaN/inf tersisa, jumlah fitur akhir = 69, proporsi kelas train/test mendekati proporsi asli (stratifikasi berhasil).

**Deliverable:** pipeline preprocessing reproducible + artefak tersimpan di `data/processed/`.

**Acceptance:** menjalankan ulang preprocessing dari raw data menghasilkan output identik (deterministic, `random_state` tetap); tidak ada leakage (scaler fit hanya di train).

---

## Phase 3 — Autoencoder untuk Dimensionality Reduction

**Tujuan:** bangun & latih Autoencoder sesuai arsitektur Tabel 3.4 / Gambar 3.2.

- [ ] `src/autoencoder.py`:
  - Arsitektur: Input(69) → Dense(64, ReLU) → Dense(32, ReLU) → Latent Dense(16) → Dense(32, ReLU) → Dense(64, ReLU) → Output(69, Sigmoid).
  - Loss: MSE, Optimizer: Adam.
  - Training hanya pakai `X_train` (normalized), dengan validation split internal untuk memantau overfitting.
  - Simpan training history (loss curve) → `results/figures/ae_loss_curve.png`.
  - Simpan model terlatih (`models/autoencoder.h5` atau `.pt`) dan encoder terpisah untuk reuse di Phase 4.
- [ ] Evaluasi kualitas rekonstruksi: reconstruction error rata-rata di train vs test (cek tidak overfit parah).

**Deliverable:** model Autoencoder terlatih + kurva loss + file model tersimpan.

**Acceptance:** reconstruction loss konvergen (turun & stabil), tidak NaN; encoder bisa dipanggil ulang secara independen dari decoder.

---

## Phase 4 — Latent Feature Extraction

**Tujuan:** hasilkan representasi laten 16 dimensi untuk train & test (proposal 3.5, Tabel 3.5).

- [ ] `src/latent_extraction.py`: load encoder dari Phase 3, transform `X_train`/`X_test` → `Z_train`, `Z_test` (masing-masing 16 fitur).
- [ ] Simpan `Z_train.npy`, `Z_test.npy` di `data/processed/`.
- [ ] Sanity check: shape sesuai (n_samples, 16), tidak ada NaN, rentang nilai masuk akal (karena aktivasi ReLU di latent layer → non-negatif, cek konsisten dengan ekspektasi).

**Deliverable:** latent features siap pakai untuk semua skenario di Phase 5–6.

**Acceptance:** `Z_train`/`Z_test` dimensinya benar, dan digunakan (bukan `X_train`/`X_test` asli) di tahap classifier.

---

## Phase 5 — Class Imbalance Handling (4 Skenario)

**Tujuan:** implementasi S1–S4 sesuai proposal 3.6 (Tabel 3.6) — hanya diterapkan ke data **training**.

- [ ] `src/imbalance.py` dengan 4 fungsi/mode:
  - **S1 — No handling:** pakai `Z_train`, `y_train` apa adanya (baseline).
  - **S2 — Class weight:** hitung `w_c = n / (C * n_c)` (persamaan 2.8), dipakai sebagai parameter `class_weight`/`sample_weight` saat training LightGBM (bukan mengubah data).
  - **S3 — Upsampling:** perbanyak sample kelas minoritas di `Z_train`/`y_train` (mis. random oversampling atau SMOTE via `imbalanced-learn`).
  - **S4 — Downsampling:** kurangi sample kelas mayoritas di `Z_train`/`y_train`.
- [ ] Pastikan `Z_test`/`y_test` **tidak pernah** disentuh oleh fungsi-fungsi ini.
- [ ] Log distribusi kelas sebelum/sesudah tiap skenario → `results/metrics/imbalance_distribution_{scenario}.csv`.

**Deliverable:** fungsi reusable yang menerima skenario sebagai parameter dan mengembalikan data training yang sudah diperlakukan (atau sample_weight untuk S2).

**Acceptance:** distribusi kelas test tetap sama di semua skenario; distribusi kelas train berubah sesuai skenario yang diharapkan (S3 lebih seimbang naik, S4 lebih seimbang turun).

---

## Phase 6 — Klasifikasi dengan LightGBM

**Tujuan:** latih LightGBM multiclass di atas latent features, untuk tiap skenario imbalance (proposal 3.7, Tabel 3.7).

- [ ] `src/classifier.py`:
  - Konfigurasi dasar: `objective='multiclass'`, `num_class=10`, `random_state=42`, `learning_rate`, `n_estimators`, `num_leaves=31`, dst. (bisa ditaruh di `configs/config.yaml`).
  - Fungsi `train_lightgbm(Z_train, y_train, scenario, sample_weight=None)`.
  - Fungsi `predict(model, Z_test)` → `y_pred`.
  - Simpan model per skenario → `models/lgbm_{scenario}.txt`.
- [ ] Jalankan training untuk keempat skenario (S1–S4) secara terpisah, simpan model & prediksi masing-masing.

**Deliverable:** 4 model LightGBM terlatih (satu per skenario) + hasil prediksi `y_pred` per skenario tersimpan.

**Acceptance:** training selesai tanpa error untuk semua skenario; prediksi menghasilkan 10 kelas valid.

---

## Phase 7 — Evaluation & Result Analysis

**Tujuan:** evaluasi & bandingkan performa antar skenario (proposal 3.8, Tabel 3.8; Bab II.4).

- [ ] `src/evaluation.py`:
  - Hitung accuracy, macro precision, macro recall, macro F1-score (pakai `sklearn.metrics`).
  - Generate confusion matrix (10x10) + visualisasi heatmap → `results/figures/confusion_matrix_{scenario}.png`.
  - Simpan classification report lengkap per kelas → `results/metrics/report_{scenario}.json`.
- [ ] Buat tabel ringkasan perbandingan S1 vs S2 vs S3 vs S4 (accuracy, macro P/R/F1) → `results/metrics/summary_comparison.csv`.
- [ ] Analisis kualitatif: kelas mana yang paling terbantu/tidak terbantu oleh tiap skenario imbalance handling (fokus ke kelas minoritas: DoS, DDoS, MITM, Ransomware, Backdoor).

**Deliverable:** laporan hasil evaluasi lengkap (metrik + confusion matrix + insight) untuk keempat skenario.

**Acceptance:** semua metrik tersimpan dan bisa direproduksi; ada kesimpulan skenario mana yang terbaik untuk macro recall/F1 di kelas minoritas — ini nanti jadi bahan Bab IV (Hasil dan Pembahasan) skripsi.

---

## Phase 8 — Orkestrasi Pipeline End-to-End

**Tujuan:** satukan Phase 1–7 jadi satu pipeline yang bisa dijalankan ulang penuh.

- [ ] `run_pipeline.py`: entry point yang menjalankan seluruh tahap sesuai `configs/config.yaml`, mengikuti Algorithm 1 di proposal (hlm. 36).
- [ ] Argumen CLI: `--scenario s1|s2|s3|s4|all`, `--skip-preprocessing`, dst. untuk mempercepat iterasi eksperimen.
- [ ] Logging tiap tahap (mis. pakai `logging` module) supaya progres & durasi tiap fase kelihatan (dataset besar, ~5.3 juta baris → perlu diperhatikan waktu training).
- [ ] (Opsional) cek performa/waktu — jika dataset terlalu berat untuk laptop, pertimbangkan subsampling awal untuk iterasi cepat sebelum full run.

**Deliverable:** satu perintah (`python run_pipeline.py --scenario all`) menjalankan semuanya dari raw data sampai hasil evaluasi.

**Acceptance:** pipeline jalan tanpa intervensi manual dan menghasilkan seluruh artefak di `results/`.

---

## Phase 9 — Dokumentasi Hasil untuk Laporan Skripsi

**Tujuan:** menyiapkan bahan Bab IV (Hasil dan Pembahasan) berdasarkan proposal ini.

- [ ] Rangkum hasil eksperimen (tabel + grafik) untuk dimasukkan ke laporan.
- [ ] Bandingkan hipotesis di proposal (1.4) dengan hasil aktual:
  - H1: preprocessing yang benar (split sebelum normalisasi) mencegah data leakage — verifikasi lewat metodologi, bukan angka.
  - H2: latent representation Autoencoder cukup menyimpan pola antar kelas — verifikasi lewat performa LightGBM di atas latent vs (opsional) baseline tanpa Autoencoder.
  - H3: skenario imbalance handling memengaruhi recall/F1 kelas minoritas — verifikasi lewat Tabel perbandingan Phase 7.
- [ ] (Opsional, kalau supervisor minta pembanding) tambahkan baseline: LightGBM langsung di 69 fitur asli (tanpa Autoencoder) untuk membuktikan manfaat reduksi dimensi.

**Deliverable:** draft bagian hasil & pembahasan siap ditulis di laporan final.

---

## Urutan Pengerjaan yang Disarankan

Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 (linear, karena tiap fase bergantung pada output fase sebelumnya). Phase 8 (orkestrasi) sebenarnya bisa mulai dicicil sejak Phase 2, sebagai wrapper yang terus dilengkapi.

## Catatan Risiko/Hal yang Perlu Diwaspadai

- **Ukuran dataset** (~5.3 juta baris) — proses preprocessing dan training Autoencoder bisa berat di laptop biasa; pertimbangkan chunked loading atau downsampling sementara untuk development, baru full run di akhir.
- **Class imbalance ekstrem** (DoS = 145 sampel, Benign = 2.5 juta) — upsampling besar-besaran berisiko overfitting; downsampling besar-besaran berisiko buang informasi kelas mayoritas. Perlu dicatat sebagai limitation di laporan.
- **Reproducibility** — pastikan `random_state=42` dipakai konsisten di semua tahap (split, autoencoder init, LightGBM, upsampling/downsampling).
- **Data leakage** — titik paling kritis: scaler HARUS di-fit hanya di train, dan imbalance handling HARUS hanya di train. Ini nilai jual utama proposal (H1), jangan sampai salah implementasi.
