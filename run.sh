#!/bin/bash

# HIZLI BAŞLANGIÇ SCRIPTI
# Bu script tüm proje setup ve eğitim sürecini otomatik olarak çalıştırır

echo "======================================================================"
echo "SİNİR AĞLARI PROJESİ - OTOMATIK SETUP VE EĞİTİM"
echo "======================================================================"

# 1. Virtual Environment Kontrolü
if [ ! -d "venv" ]; then
    echo "[1/4] Virtual environment oluşturuluyor..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment mevcut, atlanıyor..."
fi

# 2. Virtual Environment Aktivasyonu
echo "[2/4] Virtual environment aktive ediliyor..."
source venv/bin/activate

# 3. Gereksinimlerin Yüklenmesi
echo "[3/4] Gereksinimler kontrol ediliyor..."
pip install -q -r requirements.txt

# 4. Model Eğitimi
echo "[4/4] Model eğitimi başlıyor..."
python src/train.py

# 5. Görselleştirmeler
echo ""
echo "Görselleştirmeler oluşturuluyor..."
python src/visualize.py

echo ""
echo "======================================================================"
echo "✅ TÜM İŞLEMLER TAMAMLANDI!"
echo "======================================================================"
echo ""
echo "Sonuçları görmek için:"
echo "  - results/ klasörüne bakın"
echo "  - README.md dosyasını okuyun"
echo ""
echo "======================================================================"
