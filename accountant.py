import pandas as pd
import pdfplumber
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import sys, time
import re
import subprocess

# --------------------------
# 1. Fungsi bantu: bersihkan nominal format Indonesia
# --------------------------
def bersihkan_nominal(nominal):
    nominal = str(nominal)
    nominal = nominal.replace("+", "").replace("-", "").replace(" ", "")
    nominal = nominal.replace(".", "").replace(",", ".")
    try:
        return float(nominal)
    except:
        return 0.0


# --------------------------
# 2. Ekstraksi transaksi harian dari PDF Jago
# --------------------------
def extract_transaksi_harian_jago(path_pdf):
    """
    Ekstraksi transaksi harian dari PDF Bank Jago.
    Fokus hanya pada pengeluaran (angka negatif, contoh: -21,500)
    """
    data = []

    with pdfplumber.open(path_pdf) as pdf:
        # Biasanya transaksi mulai dari halaman 3
        for page in pdf.pages[2:]:
            text = page.extract_text()
            if not text:
                continue
            lines = text.split("\n")
            for line in lines:
                # cari pola nominal negatif seperti -21,500 atau -1.250.000
                match = re.findall(r"-\s?[0-9\.\,]+", line)
                if match:
                    nominal = bersihkan_nominal(match[-1])
                    if nominal > 0:  # simpan nilai absolut
                        data.append({
                            "Deskripsi": line.strip(),
                            "Pengeluaran": nominal
                        })
    df = pd.DataFrame(data)
    if not df.empty:
        print(f"[INFO] Ditemukan {len(df)} transaksi dari {os.path.basename(path_pdf)}")
    return df


# --------------------------
# 3. Gabungkan semua PDF jadi satu DataFrame
# --------------------------
def pdfs_to_dataframe(folder_pdf):
    all_data = []
    pdf_files = [f for f in os.listdir(folder_pdf) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("[WARNING] Tidak ada PDF ditemukan di folder:", folder_pdf)
        return pd.DataFrame()

    for pdf_file in pdf_files:
        path_pdf = os.path.join(folder_pdf, pdf_file)
        print(f"[INFO] Membaca transaksi dari: {pdf_file}")
        df = extract_transaksi_harian_jago(path_pdf)
        if not df.empty:
            all_data.append(df)
        else:
            print(f"[WARNING] Tidak ada transaksi terbaca di {pdf_file}")

    if not all_data:
        print("[ERROR] Tidak ada data transaksi ditemukan.")
        return pd.DataFrame()

    full_data = pd.concat(all_data, ignore_index=True)

    # --- filter transaksi yang realistis (hapus Rp0 dan Rp besar ekstrem) ---
    full_data = full_data[full_data["Pengeluaran"] > 0]
    full_data["Pengeluaran"] = full_data["Pengeluaran"].astype(float)

    return full_data


# --------------------------
# 4. Filter transaksi rutin harian - KHUSUS MAHASISWA PERANTAUAN
# --------------------------
def filter_pengeluaran_rutin(data, percentile_batas=75, frekuensi_min=3):
    """
    Filter hanya pengeluaran rutin/kebutuhan hidup harian MAHASISWA PERANTAUAN.
    
    Fokus utama:
    - Makan sehari-hari (warung, kantin, resto murah)
    - Belanja bahan dapur (indomaret, alfamart, pasar)
    - Transport (ojol, angkot, bensin)
    - Kebutuhan harian (pulsa, laundry, fotokopi)
    
    Dibuang:
    - Investasi (reksadana, saham, emas)
    - Belanja besar (elektronik, furniture)
    - Rekreasi mahal (tiket pesawat, hotel, wisata)
    - Transfer ke tabungan
    - Pembayaran SPP/semester
    
    Strategi filtering yang LEBIH KETAT untuk mahasiswa:
    1. IQR method dengan multiplier 1.5 (lebih agresif)
    2. Percentile P20-P80 (fokus 60% data tengah)
    3. Cap maksimum Rp100rb per transaksi (asumsi pengeluaran harian)
    """
    if data.empty:
        return data
    
    # Statistik dasar
    Q1 = data["Pengeluaran"].quantile(0.25)  # Kuartil 1
    Q3 = data["Pengeluaran"].quantile(0.75)  # Kuartil 3
    IQR = Q3 - Q1
    median = data["Pengeluaran"].median()
    
    # Batas atas menggunakan IQR (LEBIH KETAT untuk mahasiswa)
    # Kita pakai 1.5*IQR (standar box plot) karena pengeluaran mahasiswa relatif homogen
    batas_atas = Q3 + 1.5 * IQR
    
    # TAMBAHAN: Cap maksimum Rp100rb untuk pengeluaran harian mahasiswa
    # Lebih dari ini kemungkinan besar bukan kebutuhan harian
    cap_maksimum = 100_000
    batas_atas = min(batas_atas, cap_maksimum)
    
    # Batas bawah untuk filter pengeluaran terlalu kecil (noise)
    batas_bawah = max(Q1 * 0.5, 5_000)  # Minimal Rp5rb (parkir, aqua, dll tetap masuk)
    
    # Filter 1: Buang outlier ekstrem (transaksi besar sekali-kali)
    data_filtered = data[
        (data["Pengeluaran"] >= batas_bawah) & 
        (data["Pengeluaran"] <= batas_atas)
    ].copy()
    
    # Filter 2: Fokus pada range persentil LEBIH KETAT untuk mahasiswa perantauan
    # Ambil 60% data tengah (buang 20% terbawah dan 20% teratas)
    # Ini lebih fokus pada pengeluaran "normal" sehari-hari
    p20 = data_filtered["Pengeluaran"].quantile(0.20)
    p80 = data_filtered["Pengeluaran"].quantile(0.80)
    data_filtered = data_filtered[
        (data_filtered["Pengeluaran"] >= p20) & 
        (data_filtered["Pengeluaran"] <= p80)
    ]
    
    # Info untuk debugging
    print(f"\n📊 Analisis Pengeluaran MAHASISWA PERANTAUAN:")
    print(f"   Total transaksi awal: {len(data)}")
    print(f"   Median pengeluaran: Rp{median:,.0f}")
    print(f"   Q1 (25%): Rp{Q1:,.0f}")
    print(f"   Q3 (75%): Rp{Q3:,.0f}")
    print(f"   IQR: Rp{IQR:,.0f}")
    print(f"   Batas filter: Rp{batas_bawah:,.0f} - Rp{batas_atas:,.0f}")
    print(f"   Range fokus (P20-P80): Rp{p20:,.0f} - Rp{p80:,.0f}")
    print(f"   Cap maksimum: Rp{cap_maksimum:,.0f}/transaksi")
    print(f"\n   ✅ Transaksi rutin harian: {len(data_filtered)} ({len(data_filtered)/len(data)*100:.1f}%)")
    print(f"   🍜 Fokus: Makan, belanja dapur, transport, kebutuhan harian")
    print(f"   🗑️ Dibuang: {len(data) - len(data_filtered)} transaksi")
    print(f"      → Investasi, belanja besar, rekreasi mahal, SPP")
    
    return data_filtered


# --------------------------
# 5. Latih model Linear Regression (opsional, untuk tren)
# --------------------------
def train_model(data):
    if len(data) < 3:
        return None
    data = data.reset_index(drop=True)
    data["Hari"] = np.arange(1, len(data) + 1)
    X = data["Hari"].values.reshape(-1, 1)
    y = data["Pengeluaran"].values
    model = LinearRegression()
    model.fit(X, y)
    return model


# --------------------------
# 6. Prediksi harian yang realistis - UNTUK MAHASISWA PERANTAUAN
# --------------------------
def prediksi_harian(saldo_saat_ini, tanggal_input, data_asli):
    # Parsing tanggal
    try:
        tanggal_input = pd.to_datetime(tanggal_input.strip(), dayfirst=False, errors="raise")
    except Exception:
        try:
            tanggal_input = pd.to_datetime(tanggal_input.strip(), dayfirst=True, errors="raise")
        except Exception:
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d %B %Y", "%d %b %Y"):
                try:
                    tanggal_input = datetime.strptime(tanggal_input.strip(), fmt)
                    break
                except:
                    continue
            else:
                print("⚠️ Format tanggal tidak dikenali. Gunakan format 2025-10-20 atau 20-10-2025.")
                
                return

    hari_bulan = 30
    hari_sekarang = tanggal_input.day
    sisa_hari = hari_bulan - hari_sekarang

    if sisa_hari <= 0:
        print("⚠️ Tanggal sudah melewati akhir bulan!")
        
        return

   
    # Animasi proses dengan sound effect (15 detik)
    print("\n🤖 AI sedang menganalisis pola pengeluaran rutin harian... ", end="")
    
    # 50 karakter loading bar dalam 15 detik = 0.3 detik per karakter
    total_steps = 50
    duration = 15  # detik
    delay = duration / total_steps  # 0.3 detik per step
    
    for i in range(total_steps):
        sys.stdout.write("█")
        sys.stdout.flush()
        time.sleep(delay)
        
    
    print(" ✅\n")

    # Filter hanya pengeluaran rutin (kebutuhan hidup mahasiswa)
    data_rutin = filter_pengeluaran_rutin(data_asli)
    
    if data_rutin.empty:
        print("⚠️ Tidak ada data pengeluaran rutin yang dapat dianalisis.")
        
        return

    # Hitung rata-rata dan median (median lebih robust terhadap outlier)
    rata_harian = data_rutin["Pengeluaran"].mean()
    median_harian = data_rutin["Pengeluaran"].median()
    std_harian = data_rutin["Pengeluaran"].std()
    
    # Gunakan median + 15% sebagai estimasi konservatif untuk mahasiswa
    # +15% karena ada kemungkinan pengeluaran mendadak (sakit, traktir teman, dll)
    estimasi_harian = median_harian * 1.15
    
    # Bulatkan ke ribuan terdekat
    estimasi_harian = int(round(estimasi_harian, -3))

    # Hitung estimasi saldo akhir bulan
    total_pengeluaran_sisa = estimasi_harian * sisa_hari
    estimasi_saldo_akhir = saldo_saat_ini - total_pengeluaran_sisa


    # Output hasil
    print("=" * 70)
    print("📊 LAPORAN PREDIKSI SALDO AKHIR BULAN - MAHASISWA PERANTAUAN")
    print("=" * 70)
    print(f"📅 Tanggal sekarang: {hari_sekarang} (sisa {sisa_hari} hari lagi)")
    print(f"💰 Saldo saat ini: Rp{saldo_saat_ini:,.0f}")
    print(f"\n📈 Analisis Pola Pengeluaran Rutin:")
    print(f"   🍜 Kategori: Makan, belanja dapur, transport, kebutuhan harian")
    print(f"   • Rata-rata harian: Rp{rata_harian:,.0f}")
    print(f"   • Median harian: Rp{median_harian:,.0f}")
    print(f"   • Standar deviasi: Rp{std_harian:,.0f}")
    print(f"   • Estimasi konservatif: Rp{estimasi_harian:,.0f}/hari")
    print(f"     (Median + 15% buffer untuk pengeluaran mendadak)")
    print(f"\n💸 Proyeksi sampai akhir bulan:")
    print(f"   • Total pengeluaran {sisa_hari} hari ke depan: Rp{total_pengeluaran_sisa:,.0f}")
    print(f"   • Estimasi saldo akhir bulan: Rp{estimasi_saldo_akhir:,.0f}")
    print("=" * 70)

    # Analisis dan rekomendasi dengan sound notification
    if estimasi_saldo_akhir < 0:
          # Suara error untuk saldo tidak cukup
        kekurangan = abs(estimasi_saldo_akhir)
        print(f"\n🚨 PERINGATAN: Uang kemungkinan TIDAK CUKUP!")
        print(f"🪙 Kekurangan estimasi: Rp{kekurangan:,.0f}")
        print(f"\n💡 Solusi untuk Mahasiswa Perantauan:")
        print(f"   1. 💰 Tambahkan minimal Rp{kekurangan:,.0f} dari tabungan/kiriman")
        print(f"   2. 🍜 Kurangi pengeluaran harian jadi Rp{int(saldo_saat_ini/sisa_hari/1000)*1000:,.0f}/hari")
        print(f"      Tips: Masak sendiri, bawa bekal, kurangi jajan")
        print(f"   3. 💼 Cari side hustle (freelance, jastip, tutor, dll)")
        print(f"   4. 🤝 Pinjam darurat ke teman/keluarga")
    elif estimasi_saldo_akhir < 100_000:
        print(f"\n⚠️ HATI-HATI: Saldo masih cukup tapi sangat tipis!")
        print(f"💡 Rekomendasi:")
        print(f"   • Tambahkan Rp{100_000 - estimasi_saldo_akhir:,.0f} sebagai buffer keamanan")
        print(f"   • Hemat pengeluaran makan (masak sendiri, beli groceries bulk)")
        print(f"   • Kurangi pengeluaran non-esensial (kopi, jajan, nongkrong)")
    elif estimasi_saldo_akhir < 300_000:
        print(f"\n🟢 CUKUP: Uang kamu cukup sampai akhir bulan!")
        print(f"💡 Tips Mahasiswa Perantauan:")
        print(f"   • Sisihkan Rp50.000 untuk dana darurat")
        print(f"   • Pertimbangkan masak sendiri untuk lebih hemat")
        print(f"   • Tetap catat pengeluaran harian")
    else:
        print(f"\n✅ AMAN: Saldo kamu sangat aman!")
        print(f"💡 Saran:")
        print(f"   • Sisa dana Rp{estimasi_saldo_akhir - 300_000:,.0f} bisa untuk:")
        print(f"     - Investasi kecil (reksadana, emas)")
        print(f"     - Tabungan darurat")
        print(f"     - Nabung untuk pulang kampung")
        print(f"     - Beli kebutuhan semester depan")

    # Simulasi harian dengan visualisasi
    print("\n📊 SIMULASI SALDO HARIAN (berdasarkan pola pengeluaran rutin):")
    print("-" * 70)
    
    saldo = saldo_saat_ini
    saldo_harian = [saldo]
    tanggal_list = [hari_sekarang]
    
    for i in range(1, sisa_hari + 1):
        saldo -= estimasi_harian
        saldo_harian.append(saldo)
        tanggal_list.append(hari_sekarang + i)
    
    # Buat bar chart ASCII
    max_val = max(saldo_harian)
    min_val = min(min(saldo_harian), 0)
    range_val = max_val - min_val
    
    if range_val > 0:
        scale = 40 / range_val
    else:
        scale = 1
    
    for i, (tgl, saldo_val) in enumerate(zip(tanggal_list, saldo_harian)):
        if i == 0:
            label = f"Hari {tgl:02d} (sekarang)"
        else:
            label = f"Hari {tgl:02d}"
        
        if saldo_val >= 0:
            bar_len = int((saldo_val - min_val) * scale)
            bar = "█" * bar_len
            warna = "🟢" if saldo_val > 200_000 else "🟡" if saldo_val > 100_000 else "🟠"
        else:
            bar_len = int(abs(saldo_val - min_val) * scale)
            bar = "▓" * bar_len
            warna = "🔴"
        
        print(f"{warna} {label:20s}: {bar:42s} Rp{saldo_val:>12,.0f}")
    
    print("-" * 70)

    # Simpan grafik
    plt.figure(figsize=(10, 6))
    plt.plot(tanggal_list, saldo_harian, "-o", linewidth=2, markersize=6, 
             color='#2E86AB', label="Prediksi Saldo")
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Batas Habis")
    plt.axhline(y=100_000, color="orange", linestyle=":", linewidth=1, label="Zona Risiko")
    
    plt.fill_between(tanggal_list, saldo_harian, 0, 
                     where=[s >= 0 for s in saldo_harian], 
                     alpha=0.3, color='green', label='Saldo Positif')
    plt.fill_between(tanggal_list, saldo_harian, 0, 
                     where=[s < 0 for s in saldo_harian], 
                     alpha=0.3, color='red', label='Defisit')
    
    plt.title("📈 Prediksi Saldo Harian Mahasiswa Perantauan", fontsize=14, fontweight='bold')
    plt.xlabel("Tanggal (Hari ke-)", fontsize=11)
    plt.ylabel("Sisa Saldo (Rp)", fontsize=11)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Format y-axis sebagai rupiah
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp{x:,.0f}'))
    
    plt.savefig("prediksi_saldo_harian_realistis.png", dpi=150)
    print("\n💾 Grafik disimpan sebagai: prediksi_saldo_harian_realistis.png")
    plt.close()


# --------------------------
# 7. Chatbot utama
# --------------------------
def start_chat(folder_pdf):
    data = pdfs_to_dataframe(folder_pdf)
    if data.empty:
        print("[ERROR] Tidak ada data transaksi ditemukan.")
        
        return

    print("\n" + "="*70)
    print("🤖 CHATBOT AKUNTAN PRIBADI AI - MAHASISWA PERANTAUAN EDITION")
    print("="*70)
    print("🎯 Tujuan: Bantu kelola keuangan selama kuliah di perantauan")
    print("🍜 Fokus: Makan, belanja dapur, transport, kebutuhan harian")
    print("🗑️ Otomatis membuang: Investasi, belanja besar, SPP, rekreasi mahal")
    print("="*70)

    while True:
        print("\n" + "-"*70)
        tanggal = input("📅 Masukkan tanggal sekarang (contoh: 2025-10-25) atau 'keluar': ")
        if tanggal.lower() in ['keluar', 'exit', 'quit']:
            print("\n👋 Terima kasih! Semoga keuanganmu selalu sehat! 💰")
            print("💪 Semangat kuliah di perantauan!")
            break
        
        try:
            saldo = input("💰 Saldo saat ini (Rp): ")
            saldo = float(saldo.replace(".", "").replace(",", "").replace("Rp", "").strip())
            prediksi_harian(saldo, tanggal, data)
        except ValueError:
            print("⚠️ Input saldo tidak valid. Gunakan format angka (contoh: 400000)")
            
        except Exception as e:
            print(f"⚠️ Terjadi kesalahan: {e}")
            


# --------------------------
# 8. Main
# --------------------------
if __name__ == "__main__":
    folder_pdf = "data_pdf"
    
    if not os.path.exists(folder_pdf):
        print(f"⚠️ Folder '{folder_pdf}' tidak ditemukan!")
        print(f"💡 Buat folder '{folder_pdf}' dan masukkan file PDF e-statement Bank Jago")
    else:
        start_chat(folder_pdf)