import base64
import shutil
from pathlib import Path

import streamlit as st
from rag import create_qa_chain  # rag.py içinde senin yazdığın fonksiyon


# ============================================================
# 1) KLASÖR/YOL AYARLARI
# ============================================================
# Bu dosyanın bulunduğu klasör = proje klasörü
BASE_DIR = Path(__file__).parent

# Belgeleri okuyacağımız klasör (PDF/TXT dosyalarını buraya koyuyorsun)
DATA_DIR = BASE_DIR / "data"

# Arka plan / logo gibi görsellerin klasörü
ASSETS_DIR = BASE_DIR / "assets"

# data/ klasörü yoksa otomatik oluştur
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# 2) YARDIMCI FONKSİYON: Dosyayı base64'e çevirme
# ============================================================
# Streamlit'in arka planına resim koymak için resmi CSS'e gömüyoruz.
# Bunun için görseli base64 string'e çeviriyoruz.
def file_to_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ============================================================
# 3) STREAMLIT SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Mitoloji Ansiklopedisi (Gemini)",
    page_icon="🏛️",
    layout="centered",
)

# assets/bg.jpg varsa arka plan olarak kullanılacak
bg_b64 = file_to_b64(ASSETS_DIR / "bg.jpg")

# assets/logo.png varsa üstte logo olarak kullanılacak (opsiyonel)
logo_b64 = file_to_b64(ASSETS_DIR / "logo.png")


# ============================================================
# 4) YAN PANEL (SIDEBAR)
# ============================================================
# İSTEDİĞİN GİBİ:
# - Dosya yükleme yok
# - Yüklü dosyalar listesi yok
# Sadece: kullanıcı adı, tema, DB sıfırlama, sohbet temizleme var.
with st.sidebar:
    st.header("⚙️ Ayarlar")

    # Kullanıcı adı: yazınca session_state içine koyuyoruz ki kaybolmasın
    user_name = st.text_input("👤 İsmin:", value=st.session_state.get("user_name", ""))
    if user_name:
        st.session_state.user_name = user_name

    # Tema seçimi: sadece arka plan overlay / kart rengi / yazı rengi değişiyor
    theme = st.radio("🎨 Tema", ["🌙 Koyu", "☀️ Açık"], index=0)

    st.markdown("---")
    st.subheader("🧯 Bakım")

    # DB sıfırla:
    # Chroma bazen "hnsw index load" hatası veriyor.
    # Bu buton chroma_db_gemini* klasörlerini siler ve uygulamayı yeniden başlatır.
    if st.button("🧯 DB'yi Sıfırla (Chroma)"):
        deleted = 0
        for p in BASE_DIR.glob("chroma_db_gemini*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                deleted += 1

        # Streamlit cache'ini temizle (get_chain yeniden oluşsun)
        st.cache_resource.clear()

        st.success(f"DB sıfırlandı. Silinen klasör sayısı: {deleted}. Yenileniyor…")
        st.rerun()

    # Sohbeti temizle:
    # Sadece ekrandaki konuşma geçmişini siler. DB'ye dokunmaz.
    if st.button("🧹 Sohbeti Temizle"):
        st.session_state.messages = []


# ============================================================
# 5) CSS / GÖRÜNÜM (ARKA PLAN + KART)
# ============================================================
# Tema seçimine göre renkleri ayarlıyoruz
overlay = "rgba(0,0,0,.65)" if theme == "🌙 Koyu" else "rgba(255,255,255,.55)"
card_bg = "rgba(255,255,255,0.10)" if theme == "🌙 Koyu" else "rgba(255,255,255,0.75)"
text_color = "#f5f5f5" if theme == "🌙 Koyu" else "#111111"

# Arka plan resmi yoksa bg_b64 boş olur, yine de sorun olmaz.
st.markdown(
    f"""
    <style>
    /* Tüm sayfanın arka planı */
    .stApp {{
        background: linear-gradient({overlay}, {overlay}),
                    url("data:image/jpg;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: {text_color};
    }}

    /* İçerik kartı (cam/mermer efekti) */
    .card {{
        background: {card_bg};
        border: 1px solid rgba(212,175,55,0.40);
        box-shadow: 0 10px 35px rgba(0,0,0,0.45);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 22px;
        margin-top: 18px;
    }}

    /* Altın çizgi (başlık altında dekor) */
    .goldline {{
        height: 2px;
        background: linear-gradient(90deg, rgba(212,175,55,0),
                                    rgba(212,175,55,1),
                                    rgba(212,175,55,0));
        margin: 10px 0 18px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6) ÜST BAŞLIK (LOGO + BAŞLIK + ALT BAŞLIK)
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    # Logo yoksa emoji göster
    if logo_b64:
        st.image(f"data:image/png;base64,{logo_b64}", width=64)
    else:
        st.write("🏛️")

with col2:
    st.title("Mitoloji Ansiklopedisi Ajanı")

    # Kullanıcı adını aldıysak kişisel karşılama yaz
    if st.session_state.get("user_name"):
        st.caption(f"Hoş geldin, **{st.session_state.user_name}** ⚡")
    else:
        st.caption("Gemini API + RAG • Yunan Mitolojisi")

st.markdown('<div class="goldline"></div>', unsafe_allow_html=True)


# ============================================================
# 7) ÖRNEK SORULAR + HAZIR BUTONLAR
# ============================================================
# Bu butonlara basınca soru otomatik chat input gibi gönderilecek (pending_q)
st.subheader("💡 Örnek Sorular")
examples = [
    "Zeus kimdir?",
    "Athena neyin tanrıçasıdır?",
    "Troya Savaşı nedir?",
    "Olympos tanrıları kimlerdir?",
    "Hades yeraltı dünyasını nasıl yönetirdi?",
]
cols = st.columns(len(examples))
for c, ex in zip(cols, examples):
    if c.button(ex):
        st.session_state.pending_q = ex

st.subheader("⚡ Hazır Sorular")
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("Zeus"):
    st.session_state.pending_q = "Zeus kimdir?"
if c2.button("Hera"):
    st.session_state.pending_q = "Hera kimdir?"
if c3.button("Athena"):
    st.session_state.pending_q = "Athena kimdir?"
if c4.button("Poseidon"):
    st.session_state.pending_q = "Poseidon kimdir?"
if c5.button("Apollon"):
    st.session_state.pending_q = "Apollon kimdir?"
if c6.button("Artemis"):
    st.session_state.pending_q = "Artemis kimdir?"


# ============================================================
# 8) RAG ZİNCİRİNİ OLUŞTURMA (CACHE)
# ============================================================
# create_qa_chain() genelde şunları yapar:
# - data/ klasöründeki dosyaları okur
# - embeddings üretir
# - Chroma DB'yi yükler veya oluşturur
# - retriever ile ilgili parçaları çeker
# - Gemini'ye prompt atıp cevap döndürür
#
# @st.cache_resource ile 1 kez oluşturulur, sayfa yenilense bile tekrar tekrar kurmaz.
@st.cache_resource
def get_chain():
    return create_qa_chain()

qa = get_chain()


# ============================================================
# 9) CHAT (SOHBET ARAYÜZÜ)
# ============================================================
# Konuşma geçmişini st.session_state içinde tutuyoruz.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana bas
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Kullanıcı chat input'u
q = st.chat_input("Sorunu yaz...")

# Eğer örnek/hazır butondan soru geldiyse onu al
if (not q) and ("pending_q" in st.session_state):
    q = st.session_state.pending_q
    del st.session_state.pending_q

# Soru varsa:
# - konuşma geçmişine ekle
# - ekrana yaz
# - qa(q) ile cevabı al
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("🏛️ Olimpos'tan cevap getiriliyor..."):
            ans = qa(q)
        st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})

# Kartı kapat
st.markdown("</div>", unsafe_allow_html=True)
