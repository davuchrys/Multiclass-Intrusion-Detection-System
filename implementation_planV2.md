# Implementation Plan V2 — Penulisan Conference Paper (IEEE, 5–6 halaman)

Rencana ini memandu adaptasi skripsi (Bab I–V) menjadi paper conference format IEEE
(`IEEE-conference-template-062824/`). Fokusnya: apa yang diambil dari proposal, apa
yang dipangkas, aset hasil mana yang ditempel, dan **bagian mana yang perlu
di-screenshot** karena tidak bisa dicomot langsung dari `results/`.

## Sumber & aturan

- **Bab 1–3** diadaptasi dari proposal LaTeX di folder
  `Multiclass_Network_Intrusion_Classification_on_the_CIC_ToN_IoT_Dataset_Using_Autoencoder_and_LightGBM (1)/`:
  - `Pendahuluan.tex` → Section I (Introduction)
  - `Kajian-Pustaka.tex` → Section II (Related Work / Background)
  - `Metodologi.tex` → Section III (Methodology)
- **Bab 4–5** dari hasil eksperimen di `reports/phase9/` dan `results/`.
- **Referensi**: pakai `References.bib` yang **sama persis** dari folder proposal
  (25 entri). Salin file itu apa adanya ke folder template IEEE.
- **Batas keras**: total 5–6 halaman **termasuk** references (kolom ganda IEEE).
  Karena itu Bab 1–3 harus dipadatkan agresif — proposal ~37 halaman tidak muat.

---

## Realita anggaran halaman (kolom ganda IEEE)

Paper 6 halaman kolom ganda itu sempit. Perkiraan realistis:

| Bagian | Target | Catatan |
|---|---|---|
| Abstract + Index Terms | ~0.3 hal | 150–200 kata |
| I. Introduction | ~0.75 hal | dari `Pendahuluan.tex`, dipadatkan |
| II. Related Work | ~0.75 hal | dari `Kajian-Pustaka.tex`, sangat dipadatkan |
| III. Methodology | ~1.7 hal | dari `Metodologi.tex` + 2 diagram + Algorithm 1 |
| IV. Results & Discussion | ~1.75 hal | tabel + gambar dari hasil |
| V. Conclusion | ~0.3 hal | |
| References | ~0.75 hal | 25 entri format IEEE cukup ringkas |

Total ≈ 6.1 halaman → **perlu dipangkas sedikit lagi**. Strategi: buang sub-bagian
teoretis yang panjang di Bab 2 (penjelasan detail PCA, notasi Autoencoder, rumus
per-metrik) — di paper cukup disebut singkat + sitasi.

**Anggaran gambar/tabel yang realistis muat: ~2 gambar + 1 algoritma + 3 tabel.**
Algorithm 1 wajib dipertahankan (permintaan tetap), jadi anggaran tabel sedikit lebih
ketat. Bagian di bawah menandai mana yang WAJIB vs OPSIONAL.

---

## Section I — Introduction (dari `Pendahuluan.tex`)

**Target:** ~0.75 halaman, 3–4 paragraf.

- [x] Paragraf 1: latar IoT + kebutuhan NIDS multiclass (padatkan §1.1 Background).
- [x] Paragraf 2: tiga tantangan CIC-ToN-IoT (fitur banyak, redundansi, imbalance) +
  posisi Autoencoder sebagai feature extractor dan LightGBM sebagai classifier.
- [x] Paragraf 3: kontribusi paper (turunkan dari Objectives §1.3), tulis sebagai
  bullet/kalimat: (1) pipeline AE+LightGBM leakage-free di CIC-ToN-IoT penuh,
  (2) perbandingan 4 skenario imbalance dengan metrik makro, (3) analisis
  representasi laten vs fitur asli.
- [x] Pangkas: Problem Formulation (§1.2), Hypotheses (§1.4), Scope (§1.5),
  Research Plan/Schedule (§1.6–1.7) → **tidak ada di paper**. Hipotesis boleh
  diselipkan 1 kalimat di akhir intro kalau perlu.

**Aset:** tidak ada gambar. Murni teks.
**📸 Screenshot:** tidak perlu.

---

## Section II — Related Work / Background (dari `Kajian-Pustaka.tex`)

**Target:** ~0.75 halaman. Ini bagian yang paling banyak dipangkas.

- [x] Ringkas jadi 2–3 paragraf naratif:
  - IDS/NIDS + signature vs anomaly, binary vs multiclass (1 paragraf, sitasi
    [11][14][17]).
  - Feature reduction: feature selection vs feature extraction, posisi Autoencoder;
    sebut PCA/SAE/SSAE singkat (1 paragraf, sitasi [6][13][5][23]).
  - LightGBM & boosting untuk data tabular NIDS + penanganan imbalance
    (1 paragraf, sitasi [8][9][10][2]).
- [x] **Pangkas total:** Tabel 2.1–2.4 (perbandingan IDS/HIDS-NIDS/binary-multiclass/
  FS-FE), semua persamaan (2.1–2.18), Gambar 2.1 (arsitektur AE), Gambar 2.2
  (ilustrasi confusion matrix), Tabel 2.5 (class index mapping). Konsep-konsep ini
  cukup disebut satu kalimat + sitasi.
- [x] Boleh sisakan **1 persamaan** paling relevan kalau mau: bobot kelas
  Eq. (2.8) `w_c = n/(C·n_c)`, karena dipakai di eksperimen. Sisanya buang.

**Aset:** tidak ada gambar (semua gambar Bab 2 dibuang).
**📸 Screenshot:** tidak perlu.

---

## Section III — Methodology (dari `Metodologi.tex`)

**Target:** ~1.5 halaman. Inti paper — pertahankan yang esensial.

- [ ] **III.A Dataset** — padatkan §3.2. Sebut: CIC-ToN-IoT, 5.351.760 baris,
  85 kolom, kolom `Attack` sebagai label, 10 kelas, imbalance ekstrem.
  - Aset: **Tabel** ringkas gabungan Tabel 3.1 (dataset summary) + Tabel 3.2
    (distribusi kelas). Ambil angka dari `results/metrics/class_distribution.csv`.
    Format ulang jadi 1 tabel kompak (kelas | jumlah | %).
- [ ] **III.B Preprocessing** — padatkan §3.3. Tekankan urutan anti-leakage:
  drop identifier + quasi-constant → 69 fitur, stratified split 80:20, MinMaxScaler
  fit di train saja. Boleh sebut Tabel 3.3 (preprocessing strategy) dalam bentuk
  naratif, bukan tabel penuh.
- [ ] **III.C Autoencoder** — padatkan §3.4. Arsitektur 69→64→32→**16**→32→64→69,
  ReLU, output Sigmoid, loss MSE, optimizer Adam.
  - Aset: **Gambar 3.2** (arsitektur AE+LightGBM) — WAJIB, ini jantung metode.
  - Aset opsional: Tabel 3.4 (spesifikasi layer) — bisa dilebur ke caption gambar
    untuk hemat tempat.
- [ ] **III.D Latent Extraction + Imbalance Scenarios** — padatkan §3.5–3.6.
  Sebut 16 fitur laten; empat skenario S1–S4 (Tabel 3.6).
  - Aset: **Tabel** skenario S1–S4 ringkas (skenario | perlakuan) — WAJIB, karena
    seluruh Bab 4 mengacu ke sini.
- [ ] **III.E LightGBM + Evaluation** — padatkan §3.7–3.8. objective=multiclass,
  num_class=10, metrik: accuracy, macro P/R/F1, confusion matrix.
  - Aset opsional: Tabel 3.7 (config LightGBM) — boleh dibuang, sebut naratif saja.
- [ ] **III.F Algoritma** — **Algorithm 1 WAJIB masuk** (dari §3.9), tampilkan
  utuh dengan `algorithm` + `algpseudocode` (paket ini sudah dipakai di proposal,
  salin lingkungannya). Algoritma makan ~0.3–0.4 hal di kolom ganda; kalau meluber,
  taruh sebagai `algorithm*` (selebar dua kolom) atau kecilkan sedikit dengan
  `\small`. **Jangan dibuang.**
  - Konsekuensi ruang: karena Algorithm 1 + Gambar 3.1 + Gambar 3.2 dipertahankan
    semua, kompensasinya ambil dari tabel opsional Section III (Tabel 3.4 dilebur ke
    caption, Tabel 3.7 dibuang jadi naratif) dan dari pemangkasan Section II.

**Aset wajib di Section III:**
1. **Gambar 3.1 — System flowchart** (dari §3.1)
2. **Gambar 3.2 — Arsitektur AE+LightGBM** (dari §3.4)
3. **Algorithm 1 — Core AE+LightGBM** (dari §3.9) — wajib, lihat III.F

**📸 CATATAN DIAGRAM — Section III (INI YANG PENTING):**
> Gambar 3.1 dan 3.2 di proposal dibuat pakai **TikZ** (ada 4 blok `tikzpicture`
> di `Metodologi.tex`). **Pakai Opsi A: salin kode TikZ langsung** — hasil vektor,
> tajam, dan seragam dengan gaya paper. Bukan screenshot.
>
> **Cara menyesuaikan ukuran di kolom ganda IEEE** (TikZ proposal dibuat untuk
> halaman satu kolom yang lebih lebar, jadi hampir pasti perlu dikecilkan):
> - Salin dulu `\usetikzlibrary{shapes.geometric, arrows, positioning}` dan paket
>   `tikz` ke preamble paper (lihat baris 33–35 `main.tex` proposal).
> - **Cara paling aman:** bungkus `tikzpicture` dengan
>   `\resizebox{\columnwidth}{!}{ ... }` supaya otomatis pas selebar satu kolom.
>   Kalau flowchart terlalu tinggi/gepeng, pakai `\resizebox{\linewidth}{!}{...}`
>   di dalam `figure*` (selebar dua kolom) — cocok untuk Gambar 3.2 yang lebar.
> - **Alternatif lebih rapi:** tambah `[scale=0.7]` (atau nilai lain) pada opsi
>   `tikzpicture`, lalu kecilkan font node dengan `font=\footnotesize` /
>   `\scriptsize` di style node, dan rapatkan `node distance`. Ini menjaga
>   ketebalan garis tetap proporsional (resizebox kadang membuat garis terlalu
>   tebal/tipis).
> - Saran penempatan: **Gambar 3.1 (flowchart)** → satu kolom (`figure[t]`) dengan
>   `scale` diturunkan; **Gambar 3.2 (arsitektur, memanjang horizontal)** → dua
>   kolom (`figure*[t]`) + `\resizebox{\linewidth}{!}{...}`.
> - Kompilasi, lihat hasilnya, lalu setel angka `scale`/`\resizebox` sampai pas.
>   Iteratif — sesuaikan sambil cek luapan kolom.
>
> Kedua diagram ini **tidak ada di folder `results/`** — sumbernya kode TikZ di
> `Metodologi.tex`. (Kalau suatu saat TikZ benar-benar tidak mau kompilasi di
> template IEEE, jalan darurat terakhir baru screenshot dari PDF proposal
> hlm. 25 & 31 pada zoom ≥200%, tapi utamakan TikZ.)

---

## Section IV — Results and Discussion (dari `reports/phase9/`)

**Target:** ~1.75 halaman. Semua angka & gambar SUDAH tersedia — tinggal pilih.
Sumber utama: `reports/phase9/chapter4_results_and_discussion.md` (draft lengkap).

- [ ] **IV.A Hasil per skenario (S1–S4)** — inti hasil.
  - Aset WAJIB: **Tabel 4.3** (`reports/phase9/tables/table_4_3_scenario_metrics.csv`)
    → accuracy, macro P/R/F1 per skenario. Konversi CSV ke tabel LaTeX `booktabs`.
  - Aset WAJIB: **Gambar** confusion matrix S2
    (`reports/phase9/figures/figure_4_4_s2_confusion_matrix.png`) — satu gambar
    hasil paling informatif.
  - Narasi: S1 akurasi tertinggi tapi macro F1 terendah; S2 terbaik macro F1/recall;
    S2≈S3 (jelaskan ekuivalensi bobot); S4 terburuk. (Ambil dari draft §4.3.)
- [ ] **IV.B Efek per kelas & trade-off** — dari draft §4.4.
  - Aset opsional: Tabel 4.4 (minority class) ATAU Gambar 4.3 (minority F1) —
    **pilih salah satu**, jangan dua-duanya (hemat tempat). Saran: Gambar 4.3.
  - Narasi kunci: XSS recall turun 0.9241→0.4898 di S2 (trade-off), Backdoor &
    Ransomware terbantu, DoS/DDoS tetap gagal.
- [ ] **IV.C Representasi laten vs fitur asli (uji H2)** — dari draft §4.5.
  - Aset WAJIB: **Tabel 4.5**
    (`reports/phase9/tables/table_4_5_representation_baseline.csv`) → latent-16 vs
    original-69. Ini bukti kunci: original-69 S2 macro F1 0.5406 > latent-16 0.4249.
  - Narasi: H2 partially supported; AE memberi kompresi 76.8% tapi tidak
    mempertahankan semua info diskriminatif.
- [ ] **IV.D Diskusi cacat dataset DoS/DDoS** — dari draft §4.7 (posisikan sebagai
  **diskusi/limitasi**, bukan kontribusi — sesuai arahan pembimbing).
  - Narasi: 145 grup fitur identik berlabel ganda DoS/DDoS (100% DoS, 71.8% DDoS);
    batas teoretis; menjelaskan kegagalan. Angka dari
    `results/metrics/dataset_label_conflicts.json`.
  - Aset opsional (kalau muat): tabel kecil 2–3 baris contoh Flow ID identik dari
    `results/metrics/dataset_label_conflicts.md` bagian "Examples". Kalau tidak
    muat, cukup naratif.
- [ ] **IV.E (Opsional) Perbandingan penelitian terkait** — dari
  `results/metrics/literature_comparison.md`.
  - Narasi: pipeline (macro F1 0.4249) > baseline no-augmentation Ma et al. [2]
    (0.392); LightGBM fitur penuh (0.5436) > FSLLM [2] (0.506). **Wajib sertakan
    caveat**: protokol beda (split 70:30 vs 80:20, 13 vs 69 fitur) → perbandingan
    indikatif, bukan head-to-head.
  - ⚠️ **Konfirmasi ke pembimbing dulu** apakah subbab ini di Bab IV atau Bab II.

**Aset gambar/tabel Section IV — daftar prioritas (pilih agar muat ~2 gbr + 3 tbl total paper):**

| Aset | File | Prioritas |
|---|---|---|
| Tabel 4.3 skenario metrics | `reports/phase9/tables/table_4_3_scenario_metrics.csv` | WAJIB |
| Tabel 4.5 latent vs original | `reports/phase9/tables/table_4_5_representation_baseline.csv` | WAJIB |
| Gambar confusion matrix S2 | `reports/phase9/figures/figure_4_4_s2_confusion_matrix.png` | WAJIB |
| Gambar minority F1 | `reports/phase9/figures/figure_4_3_minority_f1.png` | opsional |
| Tabel perbandingan literatur | `results/metrics/literature_comparison.md` | opsional (tanya pembimbing) |
| Gambar scenario metrics bar | `reports/phase9/figures/figure_4_2_scenario_metrics.png` | opsional |
| Gambar AE loss curve | `reports/phase9/figures/figure_4_1_autoencoder_loss.png` | buang (kurang penting utk paper) |
| Gambar PCA laten | `results/figures/latent_pca_projection.png` | buang (hemat tempat) |

**📸 CATATAN SCREENSHOT — Section IV:**
> Semua figur Section IV **sudah berupa PNG jadi** di `reports/phase9/figures/` dan
> `results/figures/`. **Tidak perlu screenshot** — tinggal `\includegraphics`.
> Tabel dari CSV perlu **dikonversi manual ke LaTeX** (bukan screenshot — jangan
> pernah screenshot tabel angka untuk paper, ketik ulang pakai `booktabs`).

---

## Section V — Conclusion

**Target:** ~0.3 halaman, 1 paragraf. Ambil dari draft §4.9 + Objectives.

- [ ] Rangkum: pipeline AE+LightGBM leakage-free berhasil; S2 terbaik untuk macro
  metrics; H1 didukung metodologis, H3 didukung, H2 partially supported; catat
  keterbatasan DoS/DDoS. Sebut future work singkat (mis. latent-aware imbalance,
  atau perbaikan kualitas dataset).

**📸 Screenshot:** tidak perlu.

---

## References

- [x] Salin `References.bib` dari folder proposal **apa adanya** ke folder template
  IEEE. 25 entri, tidak diubah.
- [x] Set `\bibliographystyle{IEEEtran}` (template IEEE sudah menyediakan
  `IEEEtran.cls`; untuk bibliografi biasanya `IEEEtran` bst).
- [ ] **Penomoran:** proposal pakai `unsrt` + `\nocite{*}` → semua 25 ref muncul,
  bernomor urut kemunculan. Di paper:
  - Kalau ingin **semua 25 ref tetap tampil** (aman, seperti proposal): pakai
    `\nocite{*}`.
  - Kalau ingin **hanya ref yang disitir** (lebih lazim untuk paper): jangan pakai
    `\nocite{*}`, cukup `\cite{}` yang dipakai. Konsekuensi: nomor akan berbeda
    dari proposal karena urutan sitasi di paper berbeda.
  - ⚠️ Nomor [1]–[25] **tidak dijamin sama** dengan proposal karena urutan
    kemunculan berubah. Kalau pembimbing minta nomor persis sama, urutkan sitasi
    mengikuti urutan proposal atau atur manual. Isi/kunci sitasi tetap identik.
- [ ] Ganti `\citep{}`/`\citet}` (natbib, dipakai proposal) → `\cite{}` (IEEE)
  saat memindah teks.

---

## Checklist diagram & aset — ringkas (yang perlu KAMU siapkan manual)

Mayoritas aset sudah jadi PNG/CSV. Yang benar-benar butuh tindakan manual **cuma dua
diagram metodologi**, dan keduanya via **salin kode TikZ (Opsi A)** — bukan
screenshot:

1. **Gambar 3.1 (System flowchart)** — salin TikZ dari `Metodologi.tex`, sesuaikan
   ukuran (`scale`/`\resizebox` selebar kolom). Tidak ada di `results/`.
2. **Gambar 3.2 (Arsitektur AE+LightGBM)** — salin TikZ dari `Metodologi.tex`, taruh
   dua kolom (`figure*`) + `\resizebox{\linewidth}{!}{...}`. Tidak ada di `results/`.

Detail penyesuaian ukuran ada di "CATATAN DIAGRAM — Section III" di atas.

Yang **tidak perlu diapa-apakan** (tinggal pakai file yang ada):
- Semua confusion matrix, scenario metrics, minority F1, PCA → PNG jadi di
  `reports/phase9/figures/` dan `results/figures/`.
- Semua angka tabel → CSV di `reports/phase9/tables/`, **ketik ulang** ke LaTeX
  `booktabs` (jangan screenshot tabel angka).

---

## Urutan pengerjaan yang disarankan

1. Salin `References.bib` + set style IEEE, pastikan kompilasi kosong jalan dulu.
2. Tulis Section III (Methodology) lebih dulu — paling padat karya, butuh 2 diagram.
   Selesaikan urusan TikZ/screenshot di sini.
3. Section IV (Results) — tempel tabel & gambar yang sudah jadi, tulis narasi dari
   draft `chapter4_results_and_discussion.md`.
4. Section I & II — tulis terakhir setelah tahu sisa ruang, pangkas sesuai budget.
5. Abstract + Conclusion.
6. Cek panjang. Kalau > 6 halaman (urutan pemangkasan, **Algorithm 1 + Gambar 3.1 +
   Gambar 3.2 jangan disentuh**): (a) buang gambar opsional Section IV
   (minority F1 / scenario bar), (b) buang subbab perbandingan literatur kalau
   pembimbing tidak mewajibkan, (c) lebur/buang tabel opsional Section III
   (Tabel 3.4, 3.7), (d) padatkan Section II jadi 2 paragraf, (e) padatkan narasi
   Section IV.

## Catatan penting

- **Angka hasil hanya dari full-data** (`results/metrics/`, `reports/phase9/`).
  Jangan pernah pakai angka quick-run (smoke test) — `src/reporting.py` sudah
  menolaknya, dan draft Bab IV sudah bersih dari itu.
- **Konsistensi klaim**: pertahankan nada draft Phase 9 — H2 "partially supported",
  S2 "preferred under macro objective, bukan universal winner", cacat dataset
  sebagai diskusi/limitasi (bukan kontribusi, sesuai arahan pembimbing).
- **DoS/DDoS**: selalu framing sebagai keterbatasan dataset, jangan diklaim bisa
  diperbaiki metode.
