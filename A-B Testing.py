import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# Görev 1: Veriyi Hazırlama ve Analiz Etme
# Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

df = pd.read_excel("measurement_problems/datasets/ab_testing.xlsx")
df.head()

dfc = pd.read_excel("measurement_problems/datasets/ab_testing.xlsx", sheet_name="Control Group")
dfc.head()
dfc.shape

dft = pd.read_excel("measurement_problems/datasets/ab_testing.xlsx", sheet_name="Test Group")
dft.head()
dft.shape

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

def check_df(dataframe, head = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(dfc)

check_df(dft)


# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

dft['hue'] = 'T'
dfc['hue'] = 'C'
df = pd.concat([dfc, dft]).reset_index()

df.head()

#Görev 2: A/B Testinin Hipotezinin Tanımlanması

#Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2 Reklama tıklayanlar ile tıklamayanların kazançları arasında ist. ol. anlamlı bir fark yoktur.
# H1 : M1!= M2 ........... vardır.

# Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.

dfc.Purchase.mean()
dft.Purchase.mean()

# Görev 3: Hipotez Testinin Gerçekleştirilmesi
# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına
# uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.

# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ?
# Elde edilen p-value değerlerini yorumlayınız.


test_stat, pvalue = shapiro(dfc.Purchase)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Kontrol grubu için; Elde edilen p value değerine göre H0 hipotezi reddedilemez. Yani dağılım normaldir

test_stat, pvalue = shapiro(dft.Purchase)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test grubu için; Elde edilen p value değerine göre H0 hipotezi reddedilemez. Yani dağılım normaldir.
# Sonuç iki grup içinde yapılan shapiro testi dağılım normal olduğu bilgisini vermektedir. Dolayısıyla
# parametric test uygulanacaktır.


#Varyans Homojenliği :
#H0: Varyanslar homojendir.
#H1: Varyanslar homojen Değildir.
#p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
#Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
#Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = levene(dfc.Purchase, dft.Purchase)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# İki grup içinde yapılan levene test sonuçlarından elde edilen p value değerine göre H0 hipotezi reddedilemez. Yani
# varyans homojen dağılmaktadır.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

# Normallik ve varyans homojenliği varsayımları sağlandığı için parametric test olan t-test uygulanacaktır.

# Adım 3: Test sonucunda elde edilen p_value değerini gözönünde bulundurarak kontrol ve test grubu
# satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

test_stat, pvalue = ttest_ind(dfc.Purchase, dft.Purchase, equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# T testi sonucunda elde edilen p value değeri 0.3493 olduğundan dolayı, kontrol ve test grupları arasında anlamlı bir
# fark olmadığı gözlemlenmiştir.

# Görev 4: Sonuçların Analizi

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
# Normallik ve varyans homojenliği varsayımları sağlandığı için parametric test olan t test kullanıldı.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Elde edilen p value değerine göre H0 hipotezini reddemedik. Yani reklama tıklayanlar ile tıklamayanların kazançları
# arasında anlamlı bir fark olmadığı gözlemlenmiş oldu. Dolayısıyla burda elde edilen sonuçlara göre herhangi bir karar
# alınmaması gerektiği ortaya çıkmış oldu. Ortalamalar arasındaki bu farkın tesadüfi olarak ortaya çıkma ihtimali
# yüksek olduğu için şirketin burda daha farklı bir aksiyon alması gerektiği üzerine çalışması gerekmektedir.
