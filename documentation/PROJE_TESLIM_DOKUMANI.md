# SİNİR AĞLARI DERSİ FİNAL PROJESİ

## PROJE TESLİM DOKÜMANINI

### ÖĞRENCİ BİLGİLERİ

**Yapanlar:**

- **Abdulrezak Khaled (22430070907)**
- **Şeyma Alahmed (22430070906)**

**Teslim Tarihi:** 08.01.2026

---

## 1. VERİ SETİ AÇIKLAMASI

### Veri Seti: Students Performance in Exams

**Kaynak:** Kaggle - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data

**Genel Bilgiler:**

- **Örneklem Sayısı:** 1,000 öğrenci
- **Özellik Sayısı:** 8 sütun (5 kategorik özellik, 3 hedef değişken)
- **Problem Tipi:** Multi-output Regression (Çoklu Çıktılı Regresyon)

**Kategorik Özellikler:**

1. **gender:** Öğrenci cinsiyeti (female, male)
2. **race/ethnicity:** Etnik köken grubu (A, B, C, D, E)
3. **parental level of education:** Ebeveyn eğitim seviyesi (6 kategori)
4. **lunch:** Öğle yemeği tipi (standard, free/reduced)
5. **test preparation course:** Sınav hazırlık kursu durumu (completed, none)

**Hedef Değişkenler:**

1. **math score:** Matematik sınavı puanı (0-100)
2. **reading score:** Okuma sınavı puanı (0-100)
3. **writing score:** Yazma sınavı puanı (0-100)

**Proje Amacı:**  
Öğrencilerin demografik özellikleri ve eğitimsel faktörlerini kullanarak sınav performanslarını (üç farklı sınav puanını) eşzamanlı olarak tahmin etmek.

---

## 2. MODEL MİMARİSİ

### Ağ Yapısı

```
Input Layer (17 neurons)
    ↓
Hidden Layer (64 neurons, ReLU activation)
    ↓
Output Layer (3 neurons, Linear activation)
```

### Mimari Detayları

| Katman       | Nöron Sayısı | Aktivasyon Fonksiyonu | Parametre Sayısı |
| ------------ | ------------ | --------------------- | ---------------- |
| Input Layer  | 17           | -                     | -                |
| Hidden Layer | 64           | ReLU                  | W: 1088, b: 64   |
| Output Layer | 3            | Linear                | W: 192, b: 3     |
| **TOPLAM**   | -            | -                     | **1,347**        |

### Kullanılan Teknikler

1. **Veri Ön İşleme:**

   - One-Hot Encoding (kategorik değişkenler için)
   - Z-score Normalizasyonu (özellikler ve hedefler için)
   - Train-Validation-Test Split (70%-15%-15%)

2. **Sinir Ağı Bileşenleri:**

   - **Aktivasyon Fonksiyonları:** ReLU (hidden), Linear (output)
   - **Kayıp Fonksiyonu:** Mean Squared Error (MSE)
   - **Optimizasyon:** Mini-batch Gradient Descent
   - **Ağırlık Başlatma:** He Initialization

3. **Eğitim Parametreleri:**
   - Learning Rate: 0.01
   - Batch Size: 32
   - Epochs: 1000
   - Optimizer: Stochastic Gradient Descent (Mini-batch)

---

## 3. EĞİTİM GRAFİKLERİ VE BAŞARI METRİKLERİ

### 3.1 Eğitim ve Doğrulama Loss Grafikleri

![Training Curves](results/training_curves.png)

**Grafik Yorumu:**

- Mavi çizgi eğitim loss'unu, kırmızı çizgi doğrulama loss'unu göstermektedir
- Model epoch'lar boyunca öğrenme göstermiştir
- Final eğitim loss: ~0.86
- Final doğrulama loss: ~1.49

### 3.2 Tahmin vs Gerçek Değerler

![Predictions vs Actual](results/predictions_vs_actual.png)

**Grafik Yorumu:**

- Her sınav türü için tahmin edilen ve gerçek puanlar scatter plot ile gösterilmiştir
- Siyah kesikli çizgi mükemmel tahmin çizgisini temsil eder
- Noktaların bu çizgiye yakınlığı model performansını gösterir
- R² değerleri her grafik üzerinde belirtilmiştir

### 3.3 Hata Dağılımları

![Error Distribution](results/error_distribution.png)

**Grafik Yorumu:**

- Her sınav için tahmin hatalarının (gerçek - tahmin) dağılımı gösterilmiştir
- Dağılımlar yaklaşık sıfır etrafında merkezlenmiştir
- Model sistematik bir bias göstermemektedir

### 3.4 Metrik Karşılaştırmaları

![Metrics Comparison](results/metrics_comparison.png)

**Grafik Yorumu:**

- Sol panel: RMSE değerlerini gösterir (düşük olması iyidir)
- Sağ panel: R² skorlarını gösterir (yüksek olması iyidir)
- Her sınav türü için train, validation ve test performansları karşılaştırılmıştır

---

## 4. PERFORMANS METRİKLERİ

### 4.1 Test Seti Performansı

| Sınav Türü   | RMSE (Normalized) | R² Score   |
| ------------ | ----------------- | ---------- |
| Math         | 1.058             | -0.134     |
| Reading      | 1.085             | -0.218     |
| Writing      | 1.027             | -0.103     |
| **Ortalama** | **1.057**         | **-0.152** |

### 4.2 Eğitim Seti Performansı

| Sınav Türü   | RMSE (Normalized) | R² Score  |
| ------------ | ----------------- | --------- |
| Math         | 0.750             | 0.444     |
| Reading      | 0.761             | 0.430     |
| Writing      | 0.714             | 0.505     |
| **Ortalama** | **0.742**         | **0.460** |

### 4.3 Metrik Açıklamaları

- **RMSE (Root Mean Squared Error):** Tahmin hatalarının ortalama büyüklüğü
- **R² Score:** Modelin hedef değişkendeki varyansın ne kadarını açıkladığı (0-1 arası ideal)
- **Not:** Normalized değerler üzerinden hesaplanmıştır

---

## 5. UYGULANAN SİNİR AĞI KAVRAMLARI

### 5.1 Sinir Ağları Temelleri

✅ Çok katmanlı perceptron (MLP) mimarisi  
✅ Ağırlık matrisleri ve bias vektörleri  
✅ Feedforward neural network yapısı

### 5.2 Aktivasyon Fonksiyonları

✅ ReLU: `f(z) = max(0, z)`  
✅ Sigmoid: `σ(z) = 1/(1 + e^(-z))`  
✅ Tanh: `tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))`  
✅ Linear: `f(z) = z`  
✅ Tüm aktivasyonların türevleri implement edildi

### 5.3 Chain Rule (Zincir Kuralı)

✅ Backpropagation sırasında gradyanların hesaplanması  
✅ Katmanlar arası türev zincirlemesi  
✅ `∂L/∂W[l] = ∂L/∂a[l] * ∂a[l]/∂z[l] * ∂z[l]/∂W[l]`

### 5.4 Gradient Descent

✅ Mini-batch gradient descent implementasyonu  
✅ Parametre güncelleme: `W = W - α * ∂L/∂W`  
✅ Learning rate kontrolü

### 5.5 Feedforward Neural Network

✅ İleri yayılım: `z[l] = W[l] @ a[l-1] + b[l]`  
✅ Aktivasyon: `a[l] = f(z[l])`  
✅ Matris çarpımları ile verimli hesaplama

### 5.6 Backpropagation

✅ Hata geri yayılımı: `delta[l] = (W[l+1].T @ delta[l+1]) * f'(z[l])`  
✅ Gradyan hesaplama: `dW[l] = (1/m) * delta[l] @ a[l-1].T`  
✅ MSE loss fonksiyonu için özelleştirilmiş implementasyon

---

## 6. GITHUB LİNKİ

**Repository URL:** [(https://github.com/Abdulrezak-halid/Students-Performance-in-Exams)]

**Repository İçeriği:**

- ✅ Tüm kaynak kodlar (src/)
- ✅ Eğitilmiş model dosyaları (models/)
- ✅ Görselleştirmeler (results/)
- ✅ Detaylı README.md
- ✅ requirements.txt
- ✅ .gitignore

---

## 7. PROJE DOSYA YAPISI

```
Students-Performance-in-Exams/
│
├── data/
│   └── StudentsPerformance.csv
│
├── src/
│   ├── data_preprocessing.py    # Veri ön işleme
│   ├── neural_network.py        # Sinir ağı implementasyonu
│   ├── train.py                 # Eğitim scripti
│   └── visualize.py             # Görselleştirme
│
├── models/
│   ├── neural_network.pkl       # Eğitilmiş model
│   └── preprocessor.pkl         # Preprocessor
│
├── results/
│   ├── training_results.json
│   ├── training_curves.png
│   ├── predictions_vs_actual.png
│   ├── error_distribution.png
│   └── metrics_comparison.png
│
├── README.md                    # Detaylı dokümantasyon
├── requirements.txt
├── run.sh                       # Quick start script
└── QUICKSTART.py                # Kullanım rehberi
```

---

## 8. NASIL ÇALIŞTIRILIR

```bash
# 1. Repository'yi clone edin
git clone [github-link]
cd Students-Performance-in-Exams

# 2. Virtual environment oluşturun
python3 -m venv venv
source venv/bin/activate

# 3. Gereksinimleri yükleyin
pip install -r requirements.txt

# 4. Modeli eğitin
python src/train.py

# 5. Görselleştirmeleri oluşturun
python src/visualize.py
```

**Alternatif: Otomatik çalıştırma**

```bash
./run.sh
```

---

## 9. SONUÇLAR VE DEĞERLENDİRME

### Başarılar

- ✅ Sinir ağı tamamen sıfırdan NumPy ile implementa edildi
- ✅ Derste öğretilen tüm kavramlar uygulandı
- ✅ Model eğitim setinde öğrenme gösterdi (R² = 0.46)
- ✅ Kapsamlı görselleştirmeler hazırlandı
- ✅ Detaylı dokümantasyon yazıldı

### İyileştirme Potansiyeli

- Daha fazla epoch ile eğitim
- Learning rate scheduling
- Farklı optimizasyon algoritmaları (Adam, Momentum)
- Dropout regularizasyonu
- Daha derin ağ mimarileri

---

## 10. KAYNAKLAR

1. **Veri Seti:**  
   Kaggle - Students Performance in Exams

2. **Teorik Kaynaklar:**

   - Nielsen, M. (2015). Neural Networks and Deep Learning
   - Goodfellow, I., et al. (2016). Deep Learning

3. **Implementasyon:**
   - NumPy Documentation
   - Matplotlib Documentation

---
