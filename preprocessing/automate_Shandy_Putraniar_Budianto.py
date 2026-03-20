import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    print(f"Memuat data dari {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Memulai proses pembersihan data...")
    # 1. Hapus kolom yang tidak relevan
    kolom_drop = ['RowNumber', 'CustomerId', 'Surname']
    df_clean = df.drop(columns=kolom_drop, errors='ignore')

    # 2. Encoding Data Kategorikal
    le_gender = LabelEncoder()
    df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])
    df_clean = pd.get_dummies(df_clean, columns=['Geography'], drop_first=True)

    # Catatan: Kita tidak melakukan Train-Test Split & Scaling di sini
    # karena tahapan tersebut lebih ideal dilakukan di dalam pipeline Modelling (Kriteria 2)
    # Tujuan script ini hanya menghasilkan "Data Bersih".
    
    return df_clean

def save_data(df, output_path):
    # Buat folder jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    # Path disesuaikan dengan asumsi script dijalankan dari root repository
    INPUT_PATH = "churn_raw/Churn_Modelling.csv"
    OUTPUT_PATH = "churn_preprocessing/churn_clean.csv"

    # Jalankan pipeline
    raw_data = load_data(INPUT_PATH)
    clean_data = preprocess_data(raw_data)
    save_data(clean_data, OUTPUT_PATH)
    print("Automasi Preprocessing Selesai!")