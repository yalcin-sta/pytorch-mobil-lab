import torch

# Bir İHA'nın 2 farklı saniyedeki [İrtifa (m), Hız (m/s), Batarya (%)] telemetri verisi
ucus_verisi = torch.tensor([
    [120.0, 15.5, 88.0],
    [125.0, 16.0, 86.5]
])

print("Tensörümüz:\n", ucus_verisi)
print("Boyut (Shape):", ucus_verisi.shape)
