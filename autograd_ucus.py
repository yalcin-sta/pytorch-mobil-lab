import torch

# Girdi: Motor güç seviyesi (x = 2) ve beklenen gerçek yükselme hızı (y = 10 m/s)
x = torch.tensor(2.0)
gercek_hiz = torch.tensor(10.0)

# Modelin başlangıçta rastgele attığı ağırlık (w = 3.0). Doğrusu aslında 5.0 olmalı!
# requires_grad=True ile "kara kutuyu (türev takibini)" açıyoruz:
w = torch.tensor(3.0, requires_grad=True)

print(f"Başlangıç ağırlığı (w): {w.item():.2f}\n")

# 3 adımlık mini öğrenme (Gradient Descent) döngüsü
ogrenme_orani = 0.1

for adim in range(1, 4):
    # 1. İleri Yayılım: Tahmin yap ve hatayı (kayıp) ölç
    tahmin = w * x
    hata = (tahmin - gercek_hiz) ** 2

    # 2. Geri Yayılım: Türevi (gradyanı) otomatik hesapla!
    hata.backward()

    print(f"{adim}. Adım -> Tahmin: {tahmin.item():.2f} m/s | Hata: {hata.item():.2f} | Türev (w.grad): {w.grad.item():.2f}")

    # 3. Ağırlığı güncelle ve bir sonraki tur için eski türevi sıfırla
    with torch.no_grad():
        w -= ogrenme_orani * w.grad
        w.grad.zero_()

print(f"\n3 adım sonunda güncellenen yeni ağırlık (w): {w.item():.2f} (Hedef: 5.00)")
