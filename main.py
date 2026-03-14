import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# dosya hangi ayraçla ayrılmış kont
#path = "/Users/caglaozen/PycharmProjects/PythonProject1/proje/medical_insurance.csv"
#with open(path, "r", encoding="utf-8") as f:
#    print("1. satır:\n", f.readline())
#    print("2. satır:\n", f.readline())

# Kaggle verisi standartlarına göre güncellendi (virgül ayracı ve skiprows iptali)
df = pd.read_csv(
    "medical_insurance.csv",
    sep=",",        # Kaggle verileri genelde virgülle ayrılır
    nrows= 10000 )  # ilk 10000 kullan

# person_id yoksa hata vermemesi için errors="ignore" eklendi
df = df.drop(columns=["person_id"], errors="ignore")

print(df.info())
# 0'dan 9999. kişiye kadar verimiz mevcut
# id değişkenini çıkardık, toplamda 53 değişkenimiz var
# dtypes: float64(13), int64(30), object(10)

def check_df (dataframe, head=5) :
    print("############*######## Head ##############*######")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print (dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

###################
#Kategorik değişkenler
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    print("##########################################")

    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True) # veri seti cinsiyet açısından dengeli: k 48.39, e 49.53
cat_summary(df, "alcohol_freq", plot=True) # alkol tüketim sıklığını inceleyelim
cat_summary(df, "smoker", plot=True) # alkol tüketim sıklığını inceleyelim



###################
#Sayısal değişkenler
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)
num_summary(df, "annual_medical_cost", plot=True)
num_summary(df, "income", plot=True)


########################## tüm sayısal değişkenler, yıllık tıbbi maliyetle ne kadar ilişkili? - KORELASYON #######################################

corr_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[corr_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr.loc[['annual_medical_cost']].T, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation with annual_medical_cost")
plt.show()


############# Yaş, BMI, SMOKER İLE MALİYET KARŞILAŞTIRMASI

#age

tmp = df[['age','annual_medical_cost']].dropna().sort_values('age')
tmp['smooth'] = tmp['annual_medical_cost'].rolling(window=200, min_periods=50).mean()

plt.figure(figsize=(7,4))
plt.scatter(tmp['age'], tmp['annual_medical_cost'], s=5, alpha=0.1)
plt.plot(tmp['age'], tmp['smooth'])
plt.title("Age vs Annual Medical Cost (smoothed)")
plt.show()

# bmi

tmp = df[['bmi','annual_medical_cost']].dropna().sort_values('bmi')
tmp['smooth'] = tmp['annual_medical_cost'].rolling(window=200, min_periods=50).mean()

plt.figure(figsize=(6,4))
plt.scatter(tmp['bmi'], tmp['annual_medical_cost'], s=5, alpha=0.1)
plt.plot(tmp['bmi'], tmp['smooth'])
plt.xlabel("BMI")
plt.ylabel("Annual Medical Cost")
plt.title("BMI vs Annual Medical Cost (smoothed)")
plt.show()

# smoke

plt.figure(figsize=(6,4))
sns.boxplot(x='smoker', y='annual_medical_cost', data=df)
plt.yscale('log')   # maliyet çok çarpık olduğu için
plt.xlabel("Smoker")
plt.ylabel("Annual Medical Cost")
plt.title("Smoker vs Annual Medical Cost")
plt.show()
#Sigar maliyeti ayıran GÜÇLÜÜÜ kategorik değişken



#####################################################################################################################


def grab_col_names(dataframe, cat_th=10, car_th=20): # Veri setindeki sütunları sınıflandıralımmmm

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].dtype != "O" and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].dtype == "O" and dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
#bu kod bloğuna kadar hepsini aynı anda çalıştırmayı unutma

cat_cols, num_cols, cat_but_car = grab_col_names(df)
"""
Observations: 10000
Variables: 53
cat_cols: 34
num_cols: 19
cat_but_car: 0 Kategorik görünümlü, çok fazla sınıfı olan değişken yok
num_but_cat: 24 24 değişken sayısal görünümlü ama  kategorik
"""

#########################################################################

#missingvalues

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, ratio], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n\n")

    if na_name:
        return na_columns


missing_values_table(df)

print(df.isnull( ).sum( ))


df["alcohol_freq"].isnull().sum() # değişkeninde 3077 eksik değer var
df["alcohol_freq"].value_counts(dropna=False) # yüzde kaç eksik
missing_values_table(df) # %30.77lik eksik


df["alcohol_freq"].mode() # en fazla tekrar eden kategori Occasional
df["alcohol_freq"] = df["alcohol_freq"].fillna(df["alcohol_freq"].mode()[0]) # alcohol_freq değişkenindeki eksik değerleri,
# değişkenin en sık görülen değeri ile doldur
df["alcohol_freq"].isnull().sum() #tekrar kontrol ve alcohol_freq içinde artık eksik değer yok. Occasional ile dolu

#aykırı depğer

cat_cols, num_cols, cat_but_car = grab_col_names(df) #missing sonrası kolon tipleri tekrar bak
num_cols
#10.000 gözlem
#53 değişken
#34 kategorik değişken
#19 gerçek sayısal değişken
#yüzlerce sınıf içeren string kolon yok(Kardinal kategorik değişken yok
#sayısal görünüp kategorik gibi davranan 24 değişken var

# iqr yöntemiyle alt–üst sınırlar
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

# değişkende aykırı değer var mı?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

#dağılımı koru, Aykırı değerleri sınır değere baskıla
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# HANGİ DEĞİŞKENDE AYKIRILIK VAR ?
for col in num_cols:
    if check_outlier(df, col):
        print(f"{col}: OUTLIER VAR")
    else:
        print(f"{col}: outlier YOK")

# Outlier olanları baskıla
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    if check_outlier(df, col):
        print(f"{col}:  AYKIRI")
    else:
        print(f"{col}: değil")
# TEMİZLENDİİİ


######################################################################

# ENCODING : Kategorik Değişkenlerin Sayısallaştırılması
# modelin anlayabilmesi için kategorik değişkenler sayıya çevirilir
#Binary kategorikler, 2 sınıflı: Label Encoding
#Çok sınıflı kategorikler : One-Hot Encoding

from sklearn.preprocessing import LabelEncoder

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in cat_cols if df[col].nunique() == 2] # binary kolonlar
binary_cols

for col in binary_cols:
    df = label_encoder(df, col) # kolonlar artık string deil, nümerikl

for col in binary_cols:
    df = label_encoder(df, col) # kolonlar artık string deil, nümerikl

one_cols = [col for col in cat_cols if col not in binary_cols and col not in cat_but_car]
one_cols

# one_cols listesini df'de gerçekten olanlarla sınırla
one_cols = [c for c in one_cols if c in df.columns]

df = pd.get_dummies(df, columns=one_cols, drop_first=True)
df.dtypes

df.select_dtypes(include=["object"]).shape

#########################################################
# Feature Scaling : Ölçeklendirme

df[num_cols].describe().T.head() # veri ölçeği nasıl gördük



# sağlık hizmetlerini ne kadar yoğun kullandığını gösteren ölçü
df["healthcare_utilization"] = (
    df["visits_last_year"] + df["days_hospitalized_last_3yrs"]) # ziyaret + yatış = sistem yükü


# Tek tek hastalıklar yerine TOPLAM risk yükü
chronic_cols = [
    "hypertension", "diabetes", "asthma", "copd",
    "cardiovascular_disease", "cancer_history",
    "kidney_disease", "liver_disease", "arthritis"]

df["chronic_disease_count"] = df[chronic_cols].sum(axis=1)

#AYNI  maliyet, FARKLI  gelirlerde farklı risk, +1 : sıfıra bölme hatasını önlemek için

df["cost_to_income_ratio"] = (
    df["annual_medical_cost"] / (df["income"] + 1))


# YAŞ
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 30, 45, 60, 100],
    labels=["young", "adult", "middle_age", "senior"])
df = pd.get_dummies(df, columns=["age_group"], drop_first=True)


#BMI Risk
df["bmi_risk"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["underweight", "normal", "overweight", "obese"])
df = pd.get_dummies(df, columns=["bmi_risk"], drop_first=True)

# Yüksek Sağlık Kullanımı
df["high_utilization_flag"] = (
    df["healthcare_utilization"] > 3).astype(int)

#Çoklu Kronik Hastalık
df["multi_chronic_flag"] = (
    df["chronic_disease_count"] >= 2).astype(int)




df[["healthcare_utilization",
    "chronic_disease_count",
    "cost_to_income_ratio",
    "high_utilization_flag",
    "multi_chronic_flag"]].describe().T


new_num_cols = ["healthcare_utilization",
    "chronic_disease_count",
    "cost_to_income_ratio"]

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[new_num_cols] = scaler.fit_transform(df[new_num_cols])

df[new_num_cols].describe().T



###############################################

#kontrol
target = "annual_medical_cost"

# DÜZELTME 1: cost_to_income_ratio veriyi sızdırmasın diye hedefle birlikte çıkarıldı
X = df.drop(columns=[target, "cost_to_income_ratio"])
y = df[target]

print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

bool_cols = X.select_dtypes("bool").columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)


# X - y ayrımı : 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)

print(X_train.shape, X_test.shape)


obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
print("Object kolonlar:", obj_cols)
print("Sayı:", len(obj_cols))

# say ve kat ayrımı

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numerik:", num_cols)
print("Kategorik:", cat_cols)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)])

###################

# modellerr

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# DÜZELTME 2: 3 Kere tanımlanan eval_reg fonksiyonu tek ve doğru haliyle 1 kez yazıldı
def eval_reg(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))   # sürüm bağımsız RMSE
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2


# lineer reg
from sklearn.linear_model import LinearRegression

model_lr = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())])

model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")

"""
MAE  : 184.14
RMSE : 275.63
R2   : 0.981
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# random for

# DÜZELTME 3: Modeller eval_reg() içinde çağrılmadan önce tanımlanıp eğitildi
rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))])

rf.fit(X_train, y_train)
print("RF:", eval_reg(rf, X_test, y_test))


# knn
knn = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=10))])

knn.fit(X_train, y_train)
print("KNN:", eval_reg(knn, X_test, y_test))


# model karşılaştırması
models = {
    "LinearRegression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())]),
    "RandomForest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1))]),
    "KNN": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsRegressor(
            n_neighbors=10))])
}

rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    mae, rmse, r2 = eval_reg(pipe, X_test, y_test)
    rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results = pd.DataFrame(rows).sort_values("RMSE")  # en iyi  RMSE üstte
results["MAE"] = results["MAE"].round(2)
results["RMSE"] = results["RMSE"].round(2)
results["R2"] = results["R2"].round(4)

print("\n")
print(results.to_string(index=False))

# grafikler

# mae Karşılaştırma
import matplotlib.pyplot as plt

plt.figure()
plt.barh(results["Model"], results["MAE"])
plt.xlabel("MAE")
plt.title("Model Comparison - MAE")
plt.gca().invert_yaxis()
plt.show()

# rmse karşılaştırma
plt.figure()
plt.barh(results["Model"], results["RMSE"])
plt.xlabel("RMSE")
plt.title("Model Comparison - RMSE")
plt.gca().invert_yaxis()
plt.show()

# r kare karşılaştır

plt.figure()
plt.barh(results["Model"], results["R2"])
plt.xlabel("R²")
plt.title("Model Comparison - R² Score")
plt.gca().invert_yaxis()
plt.show()

## en iyi model

best_model = results.sort_values("RMSE").iloc[0]
best_model

plt.figure()
plt.bar(results["Model"], results["RMSE"])
plt.bar(best_model["Model"], best_model["RMSE"], color='red') # En iyiyi vurguladım
plt.title("Best Model Based on RMSE")
plt.ylabel("RMSE")
plt.xticks(rotation=15) # İsimler üst üste binmesin diye yatırdım
plt.show()
# RMSE göre en düşük hata ile en başarılı model