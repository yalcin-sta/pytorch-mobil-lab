import torch
import torch.nn as nn
import torch.optim as optim

# 1. Eğitim Verisi: [Rüzgar Hızı (km/s), Batarya (%)]
X = torch.tensor([
    [10.0, 90.0],  # Düşük rüzgar, dolu batarya -> Güvenli (1)
    [15.0, 80.0],  # Normal rüzgar, iyi batarya -> Güvenli (1)
    [45.0, 20.0],  # Fırtına, düşük batarya     -> Riskli (0)
    [50.0, 15.0]   # Şiddetli rüzgar, kritik pil -> Riskli (0)
])
y = torch.tensor([[1.0], [1.0], [0.0], [0.0]])

# 2. Yapay Sinir Ağı Mimarisi (2 Girdi -> 4 Gizli Nöron -> 1 Çıktı)
model = nn.Sequential(
    nn.Linear(2, 4),   # Girişten gizli katmana (2 sensör -> 4 nöron)
    nn.ReLU(),         # Doğrusal olmayan kırılma (diyot etkisi)
    nn.Linear(4, 1),   # Gizli katmandan çıkışa (4 nöron -> 1 karar)
    nn.Sigmoid()       # Sonucu 0 ile 1 arasında olasılığa sıkıştırır
)

kayip_fonksiyonu = nn.BCELoss() # İkili sınıflandırma hata ölçeri
optimizer = optim.Adam(model.parameters(), lr=0.05)

# 3. Eğitim Döngüsü (100 Tur / Epoch)
for tur in range(1, 101):
    tahmin = model(X)
    hata = kayip_fonksiyonu(tahmin, y)

    optimizer.zero_grad() # Eski türevleri sıfırla
    hata.backward()       # Geri yayılımla tüm katmanların türevini al
    optimizer.step()      # Tüm ağırlıkları tek hamlede güncelle

    if tur % 25 == 0:
        print(f"Tur {tur:3d} | Kayıp (Loss): {hata.item():.4f}")

# 4. Yeni bir uçuş senaryosunu test edelim!
yeni_ucus = torch.tensor([[12.0, 85.0], [48.0, 18.0]])
with torch.no_grad():
    olasilik = model(yeni_ucus)
    print("\n--- YENİ UÇUŞ TEST SONUÇLARI ---")
    print(f"12 km/s rüzgar, %85 pil -> Güvenli Uçuş Olasılığı: %{olasilik[0].item()*100:.1f}")
    print(f"48 km/s rüzgar, %18 pil -> Güvenli Uçuş Olasılığı: %{olasilik[1].item()*100:.1f}")
