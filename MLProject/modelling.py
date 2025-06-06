import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
)
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse

DEFAULT_MLFLOW_EXPERIMENT_NAME = "Shopping_Trends_CI"
DEFAULT_PROCESSED_DATA_PATH = "processed_shopping_trends.csv"

# Folder untuk menyimpan plot/JSON sementara sebelum di-log sebagai artifact
ARTIFACTS_TEMP_DIR = "artifacts_temp"


def load_and_split_data(data_path: str):
    """
    Memuat data yang sudah diproses dan membaginya menjadi train/test.
    """
    print(f"Memuat data dari: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File data tidak ditemukan di {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data berhasil dimuat. Shape: {df.shape}. Kolom (5 terakhir): {df.columns.tolist()[-5:]}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    target_column = df.columns[-1]
    print(f"Nama kolom target yang dipilih: {target_column}")

    if y.ndim != 1:
        raise ValueError(f"Target y harus 1D, tetapi shape-nya adalah {y.shape}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    class_labels_int = sorted(np.unique(y_train).astype(int).tolist())
    print(f"Label kelas unik (integer) dari y_train: {class_labels_int}")
    return X_train, X_test, y_train, y_test, X.columns.tolist(), class_labels_int


def plot_confusion_matrix(y_true, y_pred, class_names_str, run_id, artifacts_dir):
    """
    Membuat dan menyimpan plot confusion matrix ke 'artifacts_dir'.
    """
    cm_labels_int = [int(s) for s in class_names_str]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels_int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_str)

    fig, ax = plt.subplots(
        figsize=(max(6, len(class_names_str)*1.2), max(5, len(class_names_str) * 0.9))
    )
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - Run {run_id[:8]}")
    plot_path = os.path.join(artifacts_dir, f"confusion_matrix_run_{run_id[:8]}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"✔ Confusion matrix disimpan di: {plot_path}")
    return plot_path


def log_classification_report_as_json(y_true, y_pred, target_names_str, run_id, artifacts_dir):
    """
    Membuat, menyimpan, dan mengembalikan path ke file JSON classification report.
    """
    report = classification_report(
        y_true, y_pred, target_names=target_names_str, output_dict=True, zero_division=0
    )
    report_path = os.path.join(artifacts_dir, f"classification_report_run_{run_id[:8]}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"✔ Classification report disimpan di: {report_path}")
    return report_path


def train_model(
    X_train, y_train, X_test, y_test, feature_names, class_labels_int,
    params, data_file_path_info, current_experiment_name
):
    """
    Melatih RandomForestClassifier dan mencatat ke MLflow.
    """
    print("--- [DEBUG] Memasuki train_model ---")
    class_labels_str = [str(label) for label in class_labels_int]
    
    active_run = mlflow.active_run()
    run_id_from_env = os.environ.get('MLFLOW_RUN_ID')

    if active_run:
        print(f"ℹ️ [DEBUG] Run aktif ditemukan secara otomatis di train_model: ID {active_run.info.run_id}, Experiment ID {active_run.info.experiment_id}")
        run_id = active_run.info.run_id
    elif run_id_from_env:
        print(f"⚠️ [DEBUG] TIDAK ADA run aktif otomatis, TETAPI MLFLOW_RUN_ID ({run_id_from_env}) ADA di env.")
        print(f"⚠️ [DEBUG] Mencoba memulai/bergabung dengan run ID: {run_id_from_env}")
        try:
            # Ini akan mencoba melanjutkan run yang sudah ada ATAU memulai yang baru jika run_id belum ada di backend
            # dan run tersebut tidak aktif dari perspektif proses ini.
            # Penting: Pastikan experiment sudah di-set sebelumnya di __main__
            # agar start_run ini terasosiasi dengan experiment yang benar.
            # Jika mlflow.set_experiment() dipanggil di __main__, start_run() akan menghormatinya.
            active_run = mlflow.start_run(run_id=run_id_from_env)
            print(f"✅ [DEBUG] Berhasil memulai/bergabung dengan run: ID {active_run.info.run_id}, Experiment ID {active_run.info.experiment_id}")
            run_id = active_run.info.run_id
        except Exception as e:
            print(f"🔥 [DEBUG] GAGAL memulai/bergabung dengan run ID {run_id_from_env} dari env: {e}")
            print(f"🔥 [DEBUG] train_model MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
            raise RuntimeError(f"Gagal mengaktifkan MLflow run meskipun MLFLOW_RUN_ID ada: {e}")
    else:
        print("🔥 [DEBUG] TIDAK ADA RUN AKTIF dan TIDAK ADA MLFLOW_RUN_ID di env.")
        raise RuntimeError("Script ini harus dijalankan lewat MLflow Project/`mlflow run` dan MLFLOW_RUN_ID harus ada.")

    # Pastikan run_name dari parameter MLProject digunakan sebagai tag
    # Jika run_name ada di params, gunakan itu. Jika tidak, fallback ke default.
    current_run_name = params.get("run_name", f"Run_{run_id[:8]}") # Ambil dari params jika ada
    mlflow.set_tag("mlflow.runName", current_run_name)
    print(f"[DEBUG] Tag 'mlflow.runName' diatur ke: {current_run_name}")


    print(f"Logging ke MLflow URI: {mlflow.get_tracking_uri()} (Run ID: {run_id})")

    # Log parameter yang dipakai untuk training (kecualikan run_name jika sudah jadi tag)
    params_to_log = {k: v for k, v in params.items() if k != "run_name"}
    mlflow.log_params(params_to_log)
    mlflow.log_param("data_file_used", data_file_path_info)

    # ... (sisa fungsi train_model tetap sama dari sini) ...
    # Buat dan latih model RandomForest
    print(f"▶ Training RandomForestClassifier dengan params (setelah filter run_name): {params_to_log}")
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=params_to_log.get("n_estimators", 100), # gunakan params_to_log
        max_depth=params_to_log.get("max_depth", None),
        min_samples_split=params_to_log.get("min_samples_split", 2),
        min_samples_leaf=params_to_log.get("min_samples_leaf", 1),
        class_weight=params_to_log.get("class_weight", None)
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)

    # Hitung metrik
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_macro_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    precision_macro_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    recall_macro_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)

    mlflow.log_metric("accuracy_test", accuracy_test)
    mlflow.log_metric("f1_macro_test", f1_macro_test)
    mlflow.log_metric("precision_macro_test", precision_macro_test)
    mlflow.log_metric("recall_macro_test", recall_macro_test)

    if len(class_labels_int) > 1:
        roc_auc_ovr_test = roc_auc_score(
            y_test, y_proba_test, multi_class='ovr', average='macro', labels=class_labels_int
        )
        mlflow.log_metric("roc_auc_ovr_macro_test", roc_auc_ovr_test)
        print(f"▶ ROC AUC (Test): {roc_auc_ovr_test:.4f}")
    else:
        print("ℹ ROC AUC tidak dihitung karena jumlah kelas <= 1.")

    print(f"▶ Accuracy (Test): {accuracy_test:.4f}, F1 Macro (Test): {f1_macro_test:.4f}")

    # Simpan artefak tambahan
    os.makedirs(ARTIFACTS_TEMP_DIR, exist_ok=True) 
    cm_plot_path = plot_confusion_matrix(y_test, y_pred_test, class_labels_str, run_id, ARTIFACTS_TEMP_DIR)
    mlflow.log_artifact(cm_plot_path, artifact_path="plots")

    report_json_path = log_classification_report_as_json(
        y_test, y_pred_test, class_labels_str, run_id, ARTIFACTS_TEMP_DIR
    )
    mlflow.log_artifact(report_json_path, artifact_path="reports")

    # Log model ke MLflow
    registered_model_name = f"{current_experiment_name}-RF-CI"
    print(f"Akan meregistrasikan model dengan nama: {registered_model_name}")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random-forest-model-ci",
        input_example=X_train.iloc[[0]],
        registered_model_name=registered_model_name
    )
    print("✔ Model berhasil di-log ke MLflow.")

    # Log feature importances (jika tersedia)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, max(6, min(20, len(feature_names)) // 2)))
        sns.barplot(x='importance', y='feature', data=feature_importances_df.head(20))
        plt.title(f'Top 20 Feature Importances (Run {run_id[:8]})')
        plt.tight_layout()
        fi_plot_path = os.path.join(ARTIFACTS_TEMP_DIR, f"feature_importances_ci_run_{run_id[:8]}.png")
        plt.savefig(fi_plot_path)
        plt.close()
        mlflow.log_artifact(fi_plot_path, artifact_path="plots")
        print(f"✔ Feature importances plot disimpan: {fi_plot_path}")

        fi_csv_path = os.path.join(ARTIFACTS_TEMP_DIR, f"feature_importances_ci_run_{run_id[:8]}.csv")
        feature_importances_df.to_csv(fi_csv_path, index=False)
        mlflow.log_artifact(fi_csv_path, artifact_path="reports")
        print(f"✔ Feature importances CSV disimpan: {fi_csv_path}")

    print(f"\n✔ Eksperimen selesai. Run ID: {run_id}")
    print("--- [DEBUG] Keluar dari train_model (sukses) ---")

if __name__ == "__main__":

    os.makedirs(ARTIFACTS_TEMP_DIR, exist_ok=True)
    print(f"Direktori artefak temporer '{ARTIFACTS_TEMP_DIR}' telah dipastikan ada.")

    print("--- [DEBUG] Lingkungan MLflow di awal __main__ ---")
    print(f"[DEBUG] MAIN MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    print(f"[DEBUG] MAIN MLFLOW_RUN_ID: {os.environ.get('MLFLOW_RUN_ID')}")
    print(f"[DEBUG] MAIN MLFLOW_EXPERIMENT_ID: {os.environ.get('MLFLOW_EXPERIMENT_ID')}")
    print(f"[DEBUG] MAIN MLFLOW_EXPERIMENT_NAME: {os.environ.get('MLFLOW_EXPERIMENT_NAME')}")
    print("--- [DEBUG] Akhir Lingkungan MLflow di awal __main__ ---")

    try:
        initial_active_run = mlflow.active_run()
        if initial_active_run:
            print(f"ℹ️ [DEBUG] Run aktif awal di __main__ (sebelum set_experiment): ID {initial_active_run.info.run_id}, Experiment ID {initial_active_run.info.experiment_id}")
        else:
            print("⚠️ [DEBUG] TIDAK ADA run aktif awal yang terdeteksi oleh mlflow.active_run() di __main__ (sebelum set_experiment).")
    except Exception as e:
        print(f"[WARNING] [DEBUG] Error saat memeriksa run aktif awal di __main__: {e}")


    parser = argparse.ArgumentParser(
        description="Script MLflow Project untuk RandomForestClassifier (local_files)."
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_PROCESSED_DATA_PATH)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_MLFLOW_EXPERIMENT_NAME)

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=str, default="None")
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--class_weight", type=str, default="None")
    parser.add_argument("--run_name", type=str, default="RF_CI_Default_Run")

    args = parser.parse_args()

    # Parsing max_depth
    final_max_depth = None
    if args.max_depth.lower() != 'none':
        try:
            parsed_max_depth = int(args.max_depth)
            if parsed_max_depth > 0:
                final_max_depth = parsed_max_depth
        except ValueError:
            print(f"[WARNING] Nilai max_depth '{args.max_depth}' tidak valid, menggunakan None.")

    # Parsing class_weight
    final_class_weight = None
    if args.class_weight.lower() != 'none':
        final_class_weight = args.class_weight
        if final_class_weight not in ['balanced', 'balanced_subsample']:
            print(f"[WARNING] Nilai class_weight '{args.class_weight}' tidak valid, tetap None.")

    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": final_max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "class_weight": final_class_weight,
        "run_name": args.run_name
    }

    print("--- Memulai Script Pelatihan Model (MLflow local_files) ---")
    print(f"Parameter yang diterima:\n  data_path = '{args.data_path}'\n  experiment_name = '{args.experiment_name}'")
    print(f"Parameter model: {model_params}\n")

    mlflow.set_experiment(args.experiment_name)

    actual_experiment_name = args.experiment_name
    try:
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        if experiment:
            print(f"✔ Eksperimen MLflow diatur/dikonfirmasi: {experiment.name} (ID: {experiment.experiment_id})")
            actual_experiment_name = experiment.name
        else:
            print(f"ℹ️ Eksperimen '{args.experiment_name}' telah di-set. Akan dibuat jika belum ada oleh MLflow.")
    except Exception as e:
        print(f"[WARNING] Tidak dapat mengambil detail eksperimen '{args.experiment_name}': {e}. Menggunakan nama yang diberikan.")


    try:
        X_train, X_test, y_train, y_test, feature_names, class_labels_int = load_and_split_data(args.data_path)
        
        train_model(
            X_train, y_train, X_test, y_test,
            feature_names, class_labels_int,
            model_params, args.data_path, actual_experiment_name
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Gagal memuat data: {e}. Pelatihan dibatalkan.")
        raise
    except ValueError as e:
        print(f"[ERROR] Kesalahan pada data: {e}. Pelatihan dibatalkan.")
        raise
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan tak terduga selama proses training: {e}")
        raise

    print("--- Script Pelatihan Model Selesai ---")
