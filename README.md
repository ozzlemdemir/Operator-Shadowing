# Operator Shadowing — Operatör Gölgeleme Projesi

Fabrika / atölye ortamında kamera görüntüleri üzerinden operatörlerin tespiti, takibi ve hareket analizinin yapıldığı bilgisayarlı görü projesidir. Proje çıktısı olarak operatörlerin ısı haritaları ve bölge bazlı zaman analizleri üretilmektedir. 

**Projenin temel hedefleri:**

- Operatörlerin eğtilen görüntü işleme modelleri ile gerçek zamanlı tespiti
- ID tekrar atamasını önleyen algoritmalar ile takip edilmesi
- Mümkünse her operatörün hareket yoğunluğunun ısı haritası (heatmap) olarak görselleştirilmesi
- Belirlenen bölgelerde (yasaklı alan vb.) geçirilen sürelerin analiz edilmesi
- Tüm sonuçların JSON formatında raporlanması (zamansal analiz)

## Pipeline Genel Akışı
```
Kamera Videosu
     │
     ▼
┌─────────────────┐
│  YOLO Tespiti   │  → Her karede operatörlerin bounding box ile tespiti
│  (YOLOv8m)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BoT-SORT       │  → Her tespit edilen operatöre benzersiz ID atanması
│  + Re-ID        │    ve kareler boyunca takip edilmesi
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Koordinat      │  → Her karede her operatörün merkez koordinatları
│  Kaydetme (CSV) │    (frame, id, x, y) olarak kaydedilmesi
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│  Isı   │ │  Bölge       │
│Haritası│ │  Analizi     │
│+Overlay│ │  (JSON)      │
└────────┘ └──────────────┘
```
### Gereksinimler

```bash
pip install ultralytics #yolo modellerini kullanmak için 
pip install opencv-python #Video işleme için 
pip install pandas numpy scipy matplotlib #Isı haritasını ve görseleştirme/matemaktiksel işlemler
pip install shapely  #Polygon tanımlama ve bölge analizi için 
```

### Donanım
Model eğitimi için tercihen GPU gereklidir (Google Colab veya Kaggle ücretsiz T4 GPU kullanılabilir)(!Kaggle x2 T4 GPU sunuyor önceliklendirilebilir)

 ## 1. Veri Seti

### Kullanılan Veri Setleri

Proje sürecinde temel olarak 2 farklı veri seti değerlendirilmiştir:

| Veri Seti | Görüntü Sayısı | Sınıflar | Format | Durum |
|---|---|---|---|---|
| YOLO HighVis Dataset | 8.000 | person, high-vis | YOLOv8 (.txt) | İlk denemeler |
| Surveillance Images for Person Detection | 6.603 | person | YOLO (.txt) | **Seçilen (son model)** |
### Veri Seti Ayırma

Seçilen veri seti şu oranlarda ayrılmıştır:

- **Train:** %70 (4622 görüntü) — model eğitimi
- **Validation:** %20 (1320 görüntü) — eğitim sırasında doğrulama
- **Test:** %10 (661 görüntü) — nihai performans ölçümü

## 2. Model Eğitimi

### Eğitilen Modeller

Proje sürecinde üç farklı model eğitilmiştir:

| Model | YOLO Versiyonu | Veri Seti | Epoch | mAP50 (test) | mAP50-95 (test) |
|---|---|---|---|---|---|
| Model 1 | YOLOv8n (nano) | HighVis | 10 | %88.9 | %67.7 |
| Model 2 | YOLOv8s (small) | HighVis | 20 | %89.4 | %63.9 |
| **Model 3** | **YOLOv8m (medium)** | **Surveillance** | **20** | **%95.9** | **%70.8** |

**Not:** Model 3, hem daha güçlü bir YOLO modeli (medium) hem de daha uygun bir veri seti (gerçek CCTV görüntüleri) kullanıldığı için en iyi sonucu vermiştir.

## 3. Tespit (Detection)

Eğitilen model bir video üzerinde çalıştırılarak her karede operatörler tespit edilir.

## 4. Takip (Tracking)

Tespit edilen her operatöre benzersiz bir ID atanarak kareler boyunca takip edilir.

### Denenen Tracking Algoritmaları

| Algoritma | Hız | ID Koruma | Özellik |
|---|---|---|---|
| SORT | Çok hızlı | Zayıf | Sadece Kalman filtresi + IoU |
| DeepSORT | Orta | İyi | Görsel özellikler eklenmiş |
| ByteTrack | Hızlı | İyi | Düşük güven skorlu tespitleri de kullanır |
| **BoT-SORT + Re-ID** | **Orta** | **En iyi** | **CNN ile nesnein özelliklerini öğrenir** |

! Re-ID aynı operatöre birden fazla ID atamayı önlese de sorun hala devam etmekte 
Varsayılan `botsort.yaml` dosyasında Re-ID kapalı gelir. Re-ID'yi açmak için özel bir yaml dosyası oluşturulmalıdır (withreid: True olmalı)

## 5. Koordinat Kaydetme

Tracking sonuçlarından her karede her operatörün merkez koordinatları CSV dosyasına kaydedilir

### CSV Çıktı Formatı

```
frame, id, x, y, width, height
0, 1, 515, 572, 120, 200
1, 1, 511, 572, 118, 198
0, 2, 300, 400, 110, 190
...
```

## 6. ID Birleştirme

BoT-SORT + Re-ID kullanılmasına rağmen aynı operatöre birden fazla ID atanması sorunu devam edebilmektedir. Bu durumda video manuel olarak izlenerek hangi ID'lerin aynı kişiye ait olduğu belirlenir ve birleştirilir.

**Not:** Bu manuel birleştirme adımı, gerçek uygulamada özel Re-ID modelleri (OSNet, TransReID vb.) ile otomatize edilmesi hedeflenmektedir.

## 7. Yasaklı Bölge Analizi

Video üzerinde belirli bir alan (yasaklı bölge) tanımlanarak operatörlerin bu alanda ne kadar süre geçirdiği ölçülür. Bölge, piksel koordinatları ile polygon şeklinde tanımlanır.

## 8. Isı Haritası ve Overlay

### Isı Haritası Oluşturma Adımları

1. Video çözünürlüğünde boş bir numpy matrisi oluşturulur
2. CSV'deki her (x, y) koordinatı için ilgili pikselin değeri artırılır
3. Gaussian filtre ile yumuşatma yapılır (sigma=25)
4. Jet renk paleti uygulanır (mavi=az, kırmızı=çok)
5. Fabrika arka planı ile overlay yapılır

## 9. JSON Çıktısı

Her operatör için bölge analiz sonuçları JSON formatında kaydedilir:

```json
{
    "ID_1": {
        "toplam_sure_saniye": 98.92,
        "bolge_ici_sure_saniye": 5.68,
        "bolge_disi_sure_saniye": 93.24,
        "bolge_ici_oran": 0.057,
        "bolge_disi_oran": 0.943,
        "birlestirilen_ham_IDler": [1, 9, 22]
    },
}
```

## Kullanılan Teknolojiler

| Teknoloji | Versiyon | Kullanım Amacı |
|---|---|---|
| Python | 3.12 | Ana programlama dili |
| Ultralytics | 8.4+ | YOLO model eğitimi, tespit ve tracking |
| YOLOv8m | - | Nesne tespiti modeli |
| BoT-SORT | - | Nesne takip algoritması |
| OpenCV | 4.x | Video işleme, bölge çizimi, overlay |
| NumPy | - | Matris işlemleri, ısı haritası matrisi |
| SciPy | - | Gaussian filtre (yumuşatma) |
| Matplotlib | - | Görselleştirme, renk paleti |
| Pandas | - | Koordinat verisi yönetimi (CSV) |
| Shapely | - | Polygon içi nokta kontrolü |
| Kaggle | - | GPU ortamı (T4) |

## Bilinen Sorunlar ve Gelecek Çalışmalar

### Bilinen Sorunlar

- **ID Atlama Sorunu:** BoT-SORT + Re-ID (auto model) kullanılmasına rağmen aynı operatöre birden fazla ID atanabilmektedir. Özellikle operatör makine arkasına geçip uzun süre görünmez olduğunda yeni ID atanmaktadır.
- **Uzak Operatörler:** Kameraya uzak olan operatörlerin tespit güven skoru düşük kalmakta, bu da tracking kalitesini olumsuz etkilemektedir.

### Gelecek Çalışmalar

- **Özel Re-ID Modeli:** OSNet veya TransReID gibi özel Re-ID modellerinin entegrasyonu ile ID koruma performansının artırılması
- **İki Kamera Desteği:** Cross-camera Re-ID ile iki farklı kameradan gelen görüntülerin birleştirilmesi
- **Operatör Davranış Sınıflandırması:** Operatörün ne yaptığının sınıflandırılması (tezgah başında çalışıyor, yürüyor, bekliyor vb.)
- **Video Stabilizasyonu:** Hareketli kamera görüntülerinin yazılımsal olarak stabilize edilmesi
- **Gerçek Zamanlı Çalışma:** Pipeline'ın canlı video akışı üzerinde gerçek zamanlı çalışacak şekilde optimize edilmesi
- **Otomatik ID Birleştirme:** Manuel ID birleştirme adımının otomatize edilmesi







