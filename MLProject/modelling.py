import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
)
import mlflow
import dagshub # Tetap ada untuk opsi DagsHub
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse

DEFAULT_DAGSHUB_USERNAME = "ardenaAfif" 
DEFAULT_DAGSHUB_REPO_NAME = "SMSML_Ardena-Afif"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "Shopping_Trends_CI_Retraining"
DEFAULT_TRACKING_MODE = "dagshub" 
DEFAULT_LOCAL_MLFLOW_SERVER_URI = "http://127.0.0.1:5000"
DEFAULT_PROCESSED_DATA_PATH = "processed_shopping_trends.csv"

ARTIFACTS_TEMP_DIR = "artifacts_temp"
os.makedirs(ARTIFACTS_TEMP_DIR, exist_ok=True)


def init_mlflow(tracking_mode, dagshub_username, dagshub_repo_name, experiment_name, local_server_uri):
    """
    Inisialisasi MLflow untuk tracking berdasarkan mode yang dipilih.
    """
    mlflow_initialized_ok = False
    
    if tracking_mode == "dagshub":
        try:
            print(f"Mencoba inisialisasi MLflow dengan DagsHub (User: {dagshub_username}, Repo: {dagshub_repo_name})...")
            dagshub.init(repo_owner=dagshub_username, repo_name=dagshub_repo_name, mlflow=True)
            current_tracking_uri = mlflow.get_tracking_uri()
            if "dagshub.com" not in current_tracking_uri:
                raise Exception(f"dagshub.init berhasil, tetapi URI MLflow ({current_tracking_uri}) tidak terkonfigurasi ke DagsHub.")
            print(f"MLflow tracking URI BERHASIL diatur ke DagsHub: {mlflow.get_tracking_uri()}")
            mlflow_initialized_ok = True
        except Exception as e:
            print(f"Error saat inisialisasi DagsHub: {e}")
            print("Pastikan Anda sudah 'pip install dagshub' dan 'dagshub login' (jika menjalankan lokal) atau token DagsHub terset di environment CI.")
            print("Beralih ke opsi tracking lain jika ada.")

    if not mlflow_initialized_ok and tracking_mode == "local_server":
        try:
            print(f"Mencoba menghubungkan ke Local MLflow Server di: {local_server_uri}")
            mlflow.set_tracking_uri(local_server_uri)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if not exp:
                print(f"Eksperimen '{experiment_name}' tidak ditemukan di server lokal, membuat baru...")
                mlflow.create_experiment(experiment_name)
            print(f"MLflow tracking URI diatur ke Local Server: {mlflow.get_tracking_uri()}")
            mlflow_initialized_ok = True
        except Exception as e:
            print(f"Error saat menghubungkan ke Local MLflow Server ({local_server_uri}): {e}")
            print("Pastikan server MLflow lokal Anda berjalan (misal, dengan 'mlflow server --host ...').")
            print("Beralih ke local file-based tracking (folder mlruns).")

    if not mlflow_initialized_ok: # Fallback ke Local File-based Tracking
        try:
            print("Menggunakan local file-based tracking (folder mlruns).")
            current_uri = mlflow.get_tracking_uri()
            # Jika URI sebelumnya (DagsHub/server) gagal dan masih terset, reset ke default file-based
            if current_uri and ("dagshub.com" in current_uri or local_server_uri in current_uri):
                print(f"Resetting tracking URI dari {current_uri} ke default file-based.")
                mlflow.set_tracking_uri(None) # Reset URI untuk menggunakan ./mlruns
            
            experiment_name += "_localfiles" # Tambahkan suffix agar tidak bentrok
            print(f"MLflow tracking URI diatur ke local files: {mlflow.get_tracking_uri()} (default: ./mlruns)")
            mlflow_initialized_ok = True
        except Exception as e:
            print(f"Error fatal saat mengatur fallback local file tracking: {e}")
            # Jika ini gagal, MLflow tidak bisa digunakan sama sekali
            
    if mlflow_initialized_ok:
        mlflow.set_experiment(experiment_name)
        print(f"Eksperimen MLflow diatur ke: {mlflow.get_experiment_by_name(experiment_name).name}")
    
    return mlflow_initialized_ok

    
def load_and_split_data(data_path):
    """Memuat data yang sudah diproses dan membaginya untuk klasifikasi."""
    print(f"Memuat data dari: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: File data tidak ditemukan di {data_path}")
        raise FileNotFoundError(f"File data tidak ditemukan di {data_path}")
        
    df = pd.read_csv(data_path)
    print("Data berhasil dimuat.")
    print(f"Shape df awal: {df.shape}")
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1] 
    
    target_column_name = df.columns[-1]
    print(f"Nama kolom target yang dipilih: {target_column_name}")
    print(f"Shape X (fitur): {X.shape}, Shape y (target): {y.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"Target y harus 1D, tetapi shape-nya adalah {y.shape}. Pastikan kolom terakhir adalah target '{target_column_name}' yang sudah di-labelencode.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data dibagi menjadi: {len(X_train)} train sampel, {len(X_test)} test sampel.")
    
    class_labels_int = sorted(np.unique(y_train).astype(int).tolist())
    print(f"Label kelas unik (integer) dari y_train: {class_labels_int}")
    return X_train, X_test, y_train, y_test, X.columns.tolist(), class_labels_int


def plot_confusion_matrix(y_true, y_pred, class_names_str, run_id, artifacts_dir):
    """Membuat dan menyimpan plot confusion matrix ke direktori artefak sementara."""

    unique_labels_in_data = sorted(list(set(y_true.astype(int)) | set(y_pred.astype(int))))
    
    cm_labels_int = [int(s) for s in class_names_str] # Konversi display labels ke int untuk `labels` param
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels_int)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_str)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names_str)*1.2), max(5, len(class_names_str) * 0.9))) # Ukuran sedikit lebih besar
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - Run {run_id[:8]}")
    plot_path = os.path.join(artifacts_dir, f"confusion_matrix_run_{run_id[:8]}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Confusion matrix disimpan di: {plot_path}")
    return plot_path


def log_classification_report_as_json(y_true, y_pred, target_names_str, run_id, artifacts_dir):
    """Membuat, menyimpan, dan me-log classification report sebagai file JSON."""
    report = classification_report(y_true, y_pred, target_names=target_names_str, output_dict=True, zero_division=0)
    report_path = os.path.join(artifacts_dir, f"classification_report_run_{run_id[:8]}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Classification report disimpan di: {report_path}")
    return report_path

def train_model(X_train, y_train, X_test, y_test, feature_names, class_labels_int, params, data_file_path_info):
    """Melatih model RandomForestClassifier dengan parameter yang diberikan dan log ke MLflow."""
    
    class_labels_str = [str(label) for label in class_labels_int]

    # Memulai MLflow run dengan experiment yang sudah diatur
    with mlflow.start_run(run_name=params.get("run_name", "RF_CI_Retraining_Run")) as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        print(f"MLflow Run ID: {run_id}, Experiment ID: {experiment_id}")
        active_uri = mlflow.get_tracking_uri()
        print(f"Logging to MLflow URI: {active_uri}")
        
        # Link ke UI MLflow
        if active_uri and ("dagshub.com" in active_uri or DEFAULT_LOCAL_MLFLOW_SERVER_URI in active_uri) :
             print(f"Lihat run di UI MLflow: {active_uri}/#/experiments/{experiment_id}/runs/{run_id}")
        else: # File based
             print(f"Run disimpan di: {active_uri}. Jalankan 'mlflow ui' di direktori yang sama dengan 'mlruns' untuk melihat.")

        # Log parameter yang digunakan untuk training
        mlflow.log_params(params) # Log semua parameter yang diterima
        mlflow.log_param("data_file_used", data_file_path_info) # Info data path

        print(f"Training RandomForestClassifier with params: {params}")
        model = RandomForestClassifier(
            random_state=42, # Untuk reproduktifitas
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None), 
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight=params.get("class_weight", None) # Bisa None, 'balanced', dll.
        )
        model.fit(X_train, y_train)

        # Evaluasi model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test) # Untuk ROC AUC

        # Metrik Klasifikasi
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_macro_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        precision_macro_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        recall_macro_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("f1_macro_test", f1_macro_test)
        mlflow.log_metric("precision_macro_test", precision_macro_test)
        mlflow.log_metric("recall_macro_test", recall_macro_test)
        
        if len(class_labels_int) > 1: # ROC AUC hanya untuk >1 kelas
            roc_auc_ovr_test = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='macro', labels=class_labels_int)
            mlflow.log_metric("roc_auc_ovr_macro_test", roc_auc_ovr_test)
            print(f"ROC AUC (Test): {roc_auc_ovr_test:.4f}")
        else:
            print("ROC AUC tidak dihitung karena jumlah kelas <= 1.")

        print(f"Accuracy (Test): {accuracy_test:.4f}, F1 Macro (Test): {f1_macro_test:.4f}")
        
        # === Artefak Tambahan (Kriteria Advanced) ===
        # 1. Plot Confusion Matrix
        cm_plot_path = plot_confusion_matrix(y_test, y_pred_test, class_labels_str, run_id, ARTIFACTS_TEMP_DIR)
        mlflow.log_artifact(cm_plot_path, "plots") # Simpan di subfolder 'plots' di artefak MLflow
        
        # 2. Classification Report (sebagai JSON artifact)
        report_json_path = log_classification_report_as_json(y_test, y_pred_test, class_labels_str, run_id, ARTIFACTS_TEMP_DIR)
        mlflow.log_artifact(report_json_path, "reports") # Simpan di subfolder 'reports'

        # Log model
        # Nama artifact_path akan menjadi nama folder di dalam artefak MLflow run
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random-forest-model-ci", 
            input_example=X_train.iloc[[0]], # Contoh input untuk skema model
            # Nama model yang diregistrasi (opsional, berguna jika menggunakan MLflow Model Registry)
            registered_model_name=f"{mlflow.get_experiment_by_name(mlflow.get_experiment(run.info.experiment_id).name).name}-RF-CI"
        )
        print("Model berhasil dilog ke MLflow.")

        # Log feature importances jika model mendukung
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances_df = pd.DataFrame(
                {'feature': feature_names, 'importance': importances}
            ).sort_values(by='importance', ascending=False)
            
            # Simpan feature importances plot
            plt.figure(figsize=(10, max(6, min(20, len(feature_names)) // 2))) # Sesuaikan ukuran
            sns.barplot(x='importance', y='feature', data=feature_importances_df.head(20)) # Tampilkan top 20
            plt.title(f'Top 20 Feature Importances (CI) - Run {run_id[:8]}')
            plt.tight_layout()
            fi_plot_path = os.path.join(ARTIFACTS_TEMP_DIR, f"feature_importances_ci_run_{run_id[:8]}.png")
            plt.savefig(fi_plot_path)
            plt.close()
            mlflow.log_artifact(fi_plot_path, "plots")
            print(f"Feature importances plot (CI) dilog sebagai artefak: {fi_plot_path}")
            
            # Simpan feature importances sebagai CSV
            fi_csv_path = os.path.join(ARTIFACTS_TEMP_DIR, f"feature_importances_ci_run_{run_id[:8]}.csv")
            feature_importances_df.to_csv(fi_csv_path, index=False)
            mlflow.log_artifact(fi_csv_path, "reports")
            print(f"Feature importances (CSV CI) dilog sebagai artefak: {fi_csv_path}")

        print(f"\nEksperimen model classifier (CI) selesai. Run ID: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Project script for training a RandomForestClassifier.")
    
    # Argumen untuk konfigurasi MLflow dan DagsHub
    parser.add_argument("--data_path", type=str, default=DEFAULT_PROCESSED_DATA_PATH, help="Path to the preprocessed CSV data file.")
    parser.add_argument("--tracking_mode", type=str, default=DEFAULT_TRACKING_MODE, choices=["dagshub", "local_server", "local_files"], help="MLflow tracking mode.")
    parser.add_argument("--dagshub_user", type=str, default=DEFAULT_DAGSHUB_USERNAME, help="DagsHub username.")
    parser.add_argument("--dagshub_repo", type=str, default=DEFAULT_DAGSHUB_REPO_NAME, help="DagsHub repository name.")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_MLFLOW_EXPERIMENT_NAME, help="MLflow experiment name.")
    parser.add_argument("--local_mlflow_server_uri", type=str, default=DEFAULT_LOCAL_MLFLOW_SERVER_URI, help="URI for local MLflow server.")
    
    # Hyperparameters untuk model RandomForestClassifier
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=str, default="None", help="Maximum depth of the tree. 'None' for unlimited.") # Terima sebagai string
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum number of samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Minimum number of samples required to be at a leaf node.")
    parser.add_argument("--class_weight", type=str, default="None", help="Weights associated with classes. 'None', 'balanced', or 'balanced_subsample'.") # Terima sebagai string
    parser.add_argument("--run_name", type=str, default="RF_CI_Retraining_Run", help="Name for the MLflow run.")


    args = parser.parse_args()

    # Konversi parameter string "None" atau nilai numerik 0 untuk max_depth ke Python None
    final_max_depth = None
    if args.max_depth.lower() != 'none':
        try:
            parsed_max_depth = int(args.max_depth)
            if parsed_max_depth > 0: # Hanya jika > 0, karena 0 sering diartikan unlimited/None
                final_max_depth = parsed_max_depth
        except ValueError:
            print(f"Peringatan: Nilai max_depth '{args.max_depth}' tidak valid, menggunakan None (unlimited).")
            
    final_class_weight = None
    if args.class_weight.lower() != 'none':
        final_class_weight = args.class_weight
        if final_class_weight not in ['balanced', 'balanced_subsample']:
            print(f"Peringatan: Nilai class_weight '{args.class_weight}' mungkin tidak valid. Seharusnya 'None', 'balanced', atau 'balanced_subsample'. Menggunakan nilai yang diberikan.")


    # Kumpulkan parameter model untuk dilog dan digunakan
    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": final_max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "class_weight": final_class_weight,
        "run_name": args.run_name # Tambahkan run_name ke params agar bisa dilog
    }

    print("--- Memulai Script Pelatihan Model (MLflow Project Entry Point) ---")
    print(f"Parameter yang diterima: Data Path='{args.data_path}'")
    print(f"Parameter model: {model_params}")


    mlflow_ok = init_mlflow(
        tracking_mode=args.tracking_mode, 
        dagshub_username=args.dagshub_user, 
        dagshub_repo_name=args.dagshub_repo, 
        experiment_name=args.experiment_name,
        local_server_uri=args.local_mlflow_server_uri
    )
    
    if mlflow_ok:
        # Muat data
        load_result = load_and_split_data(args.data_path)
        # Pastikan semua elemen ada sebelum unpacking
        if load_result and len(load_result) == 6 and load_result[0] is not None:
            X_train, X_test, y_train, y_test, feature_names, class_labels_int = load_result
            # Latih model
            train_model(X_train, y_train, X_test, y_test, feature_names, class_labels_int, model_params, args.data_path)
        else:
            print(f"Gagal memuat atau membagi data dari {args.data_path}. Pelatihan dibatalkan.")
    else:
        print("Inisialisasi MLflow gagal. Script tidak dapat melanjutkan.")
    
    print("--- Script Pelatihan Model Selesai ---")