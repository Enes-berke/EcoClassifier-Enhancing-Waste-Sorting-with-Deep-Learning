# Evrişimli Sinir Ağları (CNN) ile Katı Atık Tespiti

## 1) İş Problemi (Business Problem)
Katı atıkların sınıflandırılması, geri dönüşüm süreçlerini iyileştirmek ve çevresel sürdürülebilirliği artırmak için kritik öneme sahiptir. Geleneksel yöntemlerle yapılan atık sınıflandırma işlemleri zaman alıcı ve verimsiz olabilir. Bu proje, atık sınıflandırmasını otomatikleştirerek bu süreci hızlandırmayı ve doğruluğunu artırmayı amaçlar. 

## 2) Veri Anlamak (Data Understanding)
TrashNet veri seti, atık sınıflandırma problemi üzerinde çalışmak için kullanılan bir veri setidir. Bu veri seti, özellikle geri dönüşüm ve atık yönetimi alanlarında makine öğrenimi modellerinin eğitilmesi için uygundur. TrashNet veri setinin özellikleri şunlardır:

- **Veri Türü:** Görüntü verisi.
- **Sınıflar:** 6 farklı atık türünü içerir:
  - Cam (Glass)
  - Kağıt (Paper)
  - Plastik (Plastic)
  - Metal (Metal)
  - Organik atık (Organic waste)
  - E-sıkı atık (E-waste)
- **Görüntü Sayısı:** Her sınıf için belirli sayıda görüntü bulunur.
- **Görsellerin Boyutu:** Genellikle 512x384 piksel.
- **Amacı:** Atıkların doğru bir şekilde sınıflandırılması ve geri dönüşüm süreçlerinin iyileştirilmesi amacıyla derin öğrenme modelleri geliştirilmesi.

## 3) Veriyi Hazırlamak (Data Preparation)
Veri setinin işlenmesi adımlarını içerir. İşlem adımları:

1. **Veri Setini Okuma:**
   - Görüntüler OpenCV kullanılarak okunur ve yeniden boyutlandırılır.
   - Etiketler dosya yollarından çıkarılır ve sayısal değerlere dönüştürülür.
   - Veriler karıştırılır.

2. **Görüntüleri Görselleştirme:**
   - Verisetinden rastgele örnekler görselleştirilir.

3. **Veri Augmentasyonu:**
   - Eğitim veri seti için `ImageDataGenerator` kullanılarak veri artırma işlemleri yapılır. Bu, modelin genel performansını artırır.

4. **Veri Generator'ları:**
   - Eğitim ve test veri setleri için veri akışı sağlamak amacıyla `ImageDataGenerator` kullanılır.

## 4) Modelleme (Modeling)
Bir konvolüsyonel sinir ağı (CNN) modeli tanımlanır. Model yapısı:

- **Evrişimli Katmanlar (Conv2D):** Görüntüdeki özellikleri çıkarmak için kullanılır.
- **Havuzlama Katmanları (MaxPooling2D):** Özellik haritalarını küçültmek ve önemli özellikleri korumak için kullanılır.
- **Flatting Katmanı (Flatten):** Konvolüsyonel özelliklerin vektörel forma dönüştürülmesi için kullanılır.
- **Dense Katmanları (Dense):** Sonuçları sınıflandırmak için kullanılır.
- **Dropout Katmanı (Dropout):** Aşırı uyumu azaltmak için kullanılır.

Kod örneği:
```python
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(input_shape), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=6, activation='softmax'))
```

## 5) Değerlendirme (Evaluation)
Modelin performansı, doğruluk ve kayıp metrikleri kullanılarak değerlendirilir. Karışıklık matrisleri ve sınıflandırma raporları kullanılarak modelin başarımı detaylı bir şekilde analiz edilir.

Bu yapı, katı atıkların sınıflandırılmasında CNN kullanarak veri seti üzerinde bir model geliştirme sürecini kapsamaktadır.