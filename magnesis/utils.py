import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import inv_boxcox

from .constants import FILL_VALUES


def visualize_model(
    model,
    loader,
    y_scaler,
    lmbda,
    device="cuda",
    alpha=0.95,
    figsize=(14, 5),
):
    """Визуализация с доверительными интервалами - красивая версия для больших данных"""
    dst_labels = []
    ae_labels = []
    dst_preds = []
    ae_preds = []
    dst_errors = []
    ae_errors = []

    with torch.no_grad():
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            dst_pred, ae_pred, _ = model(x)

            preds = np.stack(
                [dst_pred.cpu().numpy().flatten(), ae_pred.cpu().numpy().flatten()],
                axis=1,
            )
            labels = np.stack(
                [
                    y[:, :, 0].cpu().numpy().flatten(),
                    y[:, :, 1].cpu().numpy().flatten(),
                ],
                axis=1,
            )

            preds = y_scaler.inverse_transform(preds)
            preds[:, 1] = inv_boxcox(preds[:, 1], lmbda)
            labels = y_scaler.inverse_transform(labels)
            labels[:, 1] = inv_boxcox(labels[:, 1], lmbda)

            dst_preds.extend(preds[:, 0])
            ae_preds.extend(preds[:, 1])
            dst_labels.extend(labels[:, 0])
            ae_labels.extend(labels[:, 1])
            dst_errors.extend(labels[:, 0] - preds[:, 0])
            ae_errors.extend(labels[:, 1] - preds[:, 1])

    dst_preds = np.array(dst_preds)
    ae_preds = np.array(ae_preds)
    dst_labels = np.array(dst_labels)
    ae_labels = np.array(ae_labels)
    dst_errors = np.array(dst_errors)
    ae_errors = np.array(ae_errors)

    dst_std = np.std(dst_errors)
    ae_std = np.std(ae_errors)
    dst_mean = np.mean(dst_errors)
    ae_mean = np.mean(ae_errors)

    import scipy.stats as stats

    z = stats.norm.ppf((1 + alpha) / 2)

    dst_ci_lower = dst_preds + (dst_mean - z * dst_std)
    dst_ci_upper = dst_preds + (dst_mean + z * dst_std)
    ae_ci_lower = ae_preds + (ae_mean - z * ae_std)
    ae_ci_upper = ae_preds + (ae_mean + z * ae_std)

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].plot(
        dst_labels, color="#1f77b4", linewidth=1.2, alpha=0.8, label="Истинные значения"
    )
    axes[0].plot(
        dst_preds, color="#ff7f0e", linewidth=1.2, alpha=0.8, label="Предсказания"
    )
    axes[0].fill_between(
        range(len(dst_preds)),
        dst_ci_lower,
        dst_ci_upper,
        color="#ff7f0e",
        alpha=0.25,
        label=f"Доверительный интервал ({int(alpha * 100)}%)",
    )
    axes[0].set_title("Индекс Dst", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Временные шаги", fontsize=12)
    axes[0].set_ylabel("Dst, nT", fontsize=12)
    axes[0].legend(
        loc="upper right", fontsize=10, frameon=True, fancybox=True, shadow=True
    )
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].set_facecolor("#f9f9f9")

    # Статистика для Dst
    dst_rmse = np.sqrt(np.mean(dst_errors**2))
    dst_mae = np.mean(np.abs(dst_errors))
    axes[0].text(
        0.02,
        0.95,
        f"RMSE = {dst_rmse:.2f} nT\nMAE = {dst_mae:.2f} nT\nStd = {dst_std:.2f} nT",
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # AE
    axes[1].plot(
        ae_labels, color="#1f77b4", linewidth=1.2, alpha=0.8, label="Истинные значения"
    )
    axes[1].plot(
        ae_preds, color="#ff7f0e", linewidth=1.2, alpha=0.8, label="Предсказания"
    )
    axes[1].fill_between(
        range(len(ae_preds)),
        ae_ci_lower,
        ae_ci_upper,
        color="#ff7f0e",
        alpha=0.25,
        label=f"Доверительный интервал ({int(alpha * 100)}%)",
    )
    axes[1].set_title("Индекс AE", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Временные шаги", fontsize=12)
    axes[1].set_ylabel("AE, nT", fontsize=12)
    axes[1].legend(
        loc="upper right", fontsize=10, frameon=True, fancybox=True, shadow=True
    )
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_facecolor("#f9f9f9")

    # Статистика для AE
    ae_rmse = np.sqrt(np.mean(ae_errors**2))
    ae_mae = np.mean(np.abs(ae_errors))
    axes[1].text(
        0.02,
        0.95,
        f"RMSE = {ae_rmse:.2f} nT\nMAE = {ae_mae:.2f} nT\nStd = {ae_std:.2f} nT",
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.suptitle(
        "Прогнозирование геомагнитных индексов с доверительными интервалами",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return {
        "dst_ci_lower": dst_ci_lower,
        "dst_ci_upper": dst_ci_upper,
        "ae_ci_lower": ae_ci_lower,
        "ae_ci_upper": ae_ci_upper,
        "dst_labels": dst_labels,
        "dst_preds": dst_preds,
        "ae_labels": ae_labels,
        "ae_preds": ae_preds,
        "dst_rmse": np.sqrt(np.mean(dst_errors**2)),
        "ae_rmse": np.sqrt(np.mean(ae_errors**2)),
        "dst_std": dst_std,
        "ae_std": ae_std,
    }


def make_omni2df(omni2_path: Path | str, one_file: bool = False):
    if one_file:
        file_list = [str(omni2_path)]
    else:
        omni2_path = Path(omni2_path)
        file_pattern = os.path.join(str(omni2_path), "omni2_*.dat*")
        file_list = glob(file_pattern)

        if not file_list:
            raise ValueError(f"Не найдено файлов по шаблону: {file_pattern}")

    file_list.sort()
    df_list = []

    for file in file_list:
        try:
            df_temp = pd.read_csv(file, header=None, sep="\\s+")
        except Exception as e:
            print(f"Ошибка при чтении файла {file}: {e}")
            continue

        if len(df_temp.columns) != 55:
            print(
                f"Предупреждение: файл {file} содержит {len(df_temp.columns)} колонок, ожидается 55"
            )
            continue

        df_temp.columns = [
            "Year",
            "Decimal Day",
            "Hour",
            "Bartels",
            "IMF_s/c_ID",
            "Plasma_s/c_ID",
            "N_IMF_points",
            "N_Plasma_points",
            "B_Magnitude_Avg",
            "B_Vector_Mag",
            "B_Lat_GSE",
            "B_Long_GSE",
            "Bx_GSE",
            "By_GSE",
            "Bz_GSE",
            "By_GSM",
            "Bz_GSM",
            "sigma_B_Mag",
            "sigma_B_Vector",
            "sigma_Bx",
            "sigma_By",
            "sigma_Bz",
            "T_proton",
            "Np_density",
            "V_plasma",
            "V_Long_GSE",
            "V_Lat_GSE",
            "Na/Np",
            "P_dyn",
            "sigma_T",
            "sigma_N",
            "sigma_V",
            "sigma_phi_V",
            "sigma_theta_V",
            "sigma_Na/Np",
            "E_field",
            "Plasma_beta",
            "Alfven_Mach",
            "Kp",
            "R_sunspot",
            "Dst",
            "AE",
            "P_flux_>1MeV",
            "P_flux_>2MeV",
            "P_flux_>4MeV",
            "P_flux_>10MeV",
            "P_flux_>30MeV",
            "P_flux_>60MeV",
            "Flag",
            "ap",
            "f10.7",
            "PC(N)",
            "AL",
            "AU",
            "Mach_num",
        ]

        df_list.append(df_temp)

    if not df_list:
        raise ValueError("Не удалось загрузить ни одного файла")

    full_dataset = pd.concat(df_list, ignore_index=True)

    columns = [
        "Year",
        "Decimal Day",
        "Hour",
        "Bz_GSM",
        "By_GSM",
        "Bx_GSE",
        "Kp",
        "f10.7",
        "AL",
        "AU",
        "T_proton",
        "Np_density",
        "V_plasma",
        "V_Long_GSE",
        "V_Lat_GSE",
        "Dst",
        "AE",
    ]
    dataset = full_dataset[columns]
    try:
        dataset["datetime"] = pd.to_datetime(
            dataset["Year"].astype(str)
            + "-"
            + dataset["Decimal Day"].astype(str)
            + " "
            + dataset["Hour"].astype(str),
            format="%Y-%j %H",
        )
    except Exception as e:
        print(f"Ошибка при создании datetime: {e}")
        # Альтернативный способ
        dataset["datetime"] = pd.to_datetime(
            dataset["Year"].astype(str) + "-01-01", format="%Y-%m-%d"
        ) + pd.to_timedelta(
            (dataset["Decimal Day"] - 1) * 24 + dataset["Hour"], unit="h"
        )

    dataset = dataset.drop(columns=["Year", "Decimal Day", "Hour"])

    cols = ["datetime"] + [col for col in dataset.columns if col != "datetime"]
    dataset = dataset[cols]
    for col in dataset.drop(columns=["datetime"]).columns:
        if col in FILL_VALUES:
            dataset[col] = dataset[col].replace(FILL_VALUES[col], np.nan)
    return dataset.copy()


def make_time_features(omni2df):
    omni2df["hour"] = omni2df["datetime"].dt.hour
    omni2df["day_of_year"] = omni2df["datetime"].dt.dayofyear
    omni2df["day_of_week"] = omni2df["datetime"].dt.dayofweek
    omni2df["month"] = omni2df["datetime"].dt.month

    omni2df["hour_sin"] = np.sin(2 * np.pi * omni2df["hour"] / 24)
    omni2df["hour_cos"] = np.cos(2 * np.pi * omni2df["hour"] / 24)
    omni2df["day_sin"] = np.sin(2 * np.pi * omni2df["day_of_year"] / 365.25)
    omni2df["day_cos"] = np.cos(2 * np.pi * omni2df["day_of_year"] / 365.25)
    omni2df["week_sin"] = np.sin(2 * np.pi * omni2df["day_of_week"] / 7)
    omni2df["week_cos"] = np.cos(2 * np.pi * omni2df["day_of_week"] / 7)

    omni2df = omni2df.drop(columns=["hour", "day_of_year", "day_of_week", "month"])
    return omni2df.copy()


def make_dataset(omni2_path: Path | str, one_file: bool = False):
    dataset = make_omni2df(omni2_path)

    clean_dataset = dataset.copy()
    features = [i for i in clean_dataset.columns if i != "datetime"]
    clean_dataset[features] = clean_dataset[features].interpolate(method="pchip")

    df = clean_dataset.copy()
    df = make_time_features(df)

    return df.copy()
