# ===== LANGKAH 2: Persiapan Dashboard =====
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

st.set_page_config(page_title="Analisis Sentimen Karhutla", page_icon="🔥", layout="wide")
st.title("🔥 Analisis Sentimen Kebakaran Hutan dan Lahan (Karhutla)")
st.caption("Neural Network (MLP) + TF-IDF")

URUTAN = ["positif", "netral", "negatif"]
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}
WARNA_BAR = ["#2e9e5b", "#7f8c8d", "#d64541"]
STOPWORDS = {
    "yang", "di", "dan", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada", "dalam",
    "adalah", "juga", "atau", "akan", "sudah", "telah", "ada", "tidak", "tak", "nggak",
    "ga", "gak", "yg", "dg", "dgn", "sebagai", "oleh", "karena", "agar", "saat", "para",
    "mereka", "kita", "kami", "saya", "aku", "gue", "gw", "kamu", "anda", "nya", "lah",
    "kah", "pun", "si", "sih", "dong", "deh", "kok", "lagi", "masih", "sangat", "lebih",
    "bisa", "dapat", "harus", "jadi", "kalau", "kalo", "jika", "bila", "ketika", "hingga",
    "sampai", "tapi", "tetapi", "namun", "serta", "tersebut", "sebuah", "satu", "dua",
    "the", "of", "to", "and", "in", "is", "for", "on", "co", "https", "amp", "rt",
}

# ===== LANGKAH 1 (dibaca di sini): memuat tweet_data.csv =====
@st.cache_data
def muat_data():
    df = pd.read_csv("tweet_data.csv")
    df = df[["created_at", "username", "full_text"]].dropna(subset=["full_text"])
    return df.drop_duplicates(subset="full_text").reset_index(drop=True)

df = muat_data()
st.header("Langkah 1 - Data Tweet")
st.write(f"Jumlah tweet: **{len(df)}**")
st.dataframe(df.head(10))

# ===== LANGKAH 3: Pemrosesan Teks =====
def bersihkan_teks(teks):
    teks = str(teks).lower()                       # case folding
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)  # URL
    teks = re.sub(r"@\w+", " ", teks)              # mention
    teks = re.sub(r"#\w+", " ", teks)              # hashtag
    teks = re.sub(r"[^a-z\s]", " ", teks)          # simbol & angka
    return re.sub(r"\s+", " ", teks).strip()       # spasi berlebih

df["clean_text"] = df["full_text"].apply(bersihkan_teks)
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
st.header("Langkah 3 - Pemrosesan Teks")
st.dataframe(df[["full_text", "clean_text"]].head(10))

# ===== LANGKAH 4: Pelabelan Sentimen (Rule-Based) =====
KATA_POSITIF = {
    "aman", "lega", "selamat", "bantuan", "bantu", "membantu", "pulih", "berhasil",
    "sukses", "membaik", "terkendali", "dipadamkan", "padam", "apresiasi", "syukur",
    "bersyukur", "alhamdulillah", "dukung", "mendukung", "siaga", "tangguh", "relawan",
    "peduli", "solusi", "dihentikan", "menghentikan", "teratasi", "harapan", "semangat",
    "optimis", "hujan", "turun", "berkurang", "menurun", "normal", "cegah", "pencegahan",
    "sigap", "tanggap", "cepat", "gotong", "royong", "kolaborasi", "sinergi", "upaya",
    "penghargaan", "santunan", "belasungkawa", "lestari", "pelestarian", "menjaga",
    "edukasi", "sosialisasi", "penyuluhan", "patroli", "terbaik", "hebat", "salut",
    "bangga", "pahlawan", "heroik", "sehat", "damai", "terima", "kasih",
}
KATA_NEGATIF = {
    "parah", "rusak", "bahaya", "berbahaya", "kabut", "asap", "korban", "meninggal",
    "tewas", "gugur", "hangus", "musnah", "kerugian", "rugi", "dampak", "ancam",
    "mengancam", "krisis", "darurat", "sesak", "ispa", "polusi", "tercemar", "gagal",
    "lalai", "sengaja", "dibakar", "pembakaran", "marah", "kecewa", "bencana",
    "terbakar", "menyebar", "kritis", "buruk", "menderita", "gawat", "sedih", "derita",
    "kematian", "korupsi", "jahat", "salah",
}

def beri_label(teks):
    kata = teks.split()
    pos = sum(k in KATA_POSITIF for k in kata)
    neg = sum(k in KATA_NEGATIF for k in kata)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"

df["sentimen"] = df["clean_text"].apply(beri_label)
st.header("Langkah 4 - Pelabelan Sentimen")
st.dataframe(df[["clean_text", "sentimen"]].head(10))

# ===== LANGKAH 5: Distribusi Sentimen =====
st.header("Langkah 5 - Distribusi Sentimen")
hitung = df["sentimen"].value_counts().reindex(URUTAN, fill_value=0)
st.dataframe(hitung.rename("jumlah").to_frame())
col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(hitung.index, hitung.values, color=WARNA_BAR)
    ax.bar_label(bars)
    ax.set_title("Bar Chart Sentimen")
    st.pyplot(fig)
with col_b:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(hitung.values, labels=hitung.index, colors=WARNA_BAR, autopct="%1.1f%%", startangle=90)
    ax.set_title("Pie Chart Sentimen")
    st.pyplot(fig)

# ===== LANGKAH 6: WordCloud =====
st.header("Langkah 6 - WordCloud")
WARNA_WC = {"positif": "Greens", "negatif": "Reds", "netral": "Blues"}
for col, kelas in zip(st.columns(3), URUTAN):
    with col:
        st.subheader(f"{EMOJI[kelas]} {kelas.capitalize()}")
        teks_kelas = " ".join(df.loc[df["sentimen"] == kelas, "clean_text"])
        if not teks_kelas.strip():
            st.info("Tidak ada data.")
            continue
        wc = WordCloud(width=600, height=450, background_color="white",
                       colormap=WARNA_WC[kelas], stopwords=STOPWORDS,
                       max_words=80).generate(teks_kelas)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# ===== LANGKAH 7: Model Neural Network =====
st.header("Langkah 7 - Model Neural Network (MLP)")

@st.cache_resource
def latih_model(teks, label):
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2,
                          stop_words=list(STOPWORDS))
    X = vec.fit_transform(pd.Series(teks))
    y = pd.Series(label)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)       # 80:20
    model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                          solver="adam", max_iter=300, random_state=42)
    model.fit(X_tr, y_tr)
    return vec, model, y_te.to_numpy(), model.predict(X_te)

vec, model, y_uji, y_pred = latih_model(tuple(df["clean_text"]), tuple(df["sentimen"]))
st.success("Model selesai dilatih (80% data latih, 20% data uji; ReLU, Adam, 100 neuron).")

# ===== LANGKAH 8: Evaluasi Model =====
st.header("Langkah 8 - Evaluasi Model")
akurasi = accuracy_score(y_uji, y_pred)
p, r, f, s = precision_recall_fscore_support(y_uji, y_pred, labels=URUTAN, zero_division=0)
p_m, r_m, f_m, _ = precision_recall_fscore_support(y_uji, y_pred, average="macro", zero_division=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Akurasi", f"{akurasi:.2%}")
m2.metric("Presisi (macro)", f"{p_m:.2%}")
m3.metric("Recall (macro)", f"{r_m:.2%}")
m4.metric("F1-Score (macro)", f"{f_m:.2%}")

tabel = pd.DataFrame({"Presisi": p, "Recall": r, "F1-Score": f, "Support": s}, index=URUTAN)
st.dataframe(tabel.style.format({"Presisi": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}))

col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    tabel[["Presisi", "Recall", "F1-Score"]].plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Metrik per Kelas")
    st.pyplot(fig)
with col_b:
    cm = confusion_matrix(y_uji, y_pred, labels=URUTAN)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), URUTAN)
    ax.set_yticks(range(3), URUTAN)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    st.pyplot(fig)

# ===== LANGKAH 9: Prediksi Teks Baru =====
st.header("Langkah 9 - Prediksi Teks Baru")
teks_baru = st.text_area("Masukkan teks", height=120,
                         placeholder="Contoh: Kabut asap akibat karhutla makin parah, warga sesak napas")
if st.button("Prediksi", type="primary"):
    bersih = bersihkan_teks(teks_baru)
    if not bersih:
        st.warning("Teks kosong setelah dibersihkan. Coba kalimat lain.")
    else:
        proba = model.predict_proba(vec.transform([bersih]))[0]
        kelas = model.classes_[int(np.argmax(proba))]
        st.markdown(f"### {EMOJI[kelas]} Sentimen: **{kelas.upper()}**")
        st.caption(f"Teks setelah dibersihkan: {bersih}")
        hasil = pd.Series(proba, index=model.classes_).reindex(URUTAN)
        st.dataframe(hasil.rename("probabilitas").to_frame().style.format("{:.2%}"))
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(hasil.index, hasil.values, color=WARNA_BAR)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        st.pyplot(fig)
