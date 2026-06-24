from pathlib import Path
from typing import Literal, TypeAlias

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from .constants import FILL_VALUES
from .dataset import GeomagneticDataset
from .model import GeomagneticModel
from .schemas import GeomagnesisResult
from .utils import make_dataset, make_omni2df, make_time_features
from .validator import GNDataValidatorImpl

deviceType: TypeAlias = Literal["cuda", "cpu"]


class GeomagneticNet:
    def __init__(self, model_dir: Path | str, device: deviceType) -> None:
        self.__data_validator = GNDataValidatorImpl()
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.__X_scaler_path = model_dir / "GN_X_scaler.joblib"
        self.__y_scaler_path = model_dir / "GN_y_scaler.joblib"
        self.__model_weights_path = model_dir / "GN_best.pt"

        assert self.__X_scaler_path.exists(), (
            f"В папке {model_dir} отсутствует файл 'GN_X_scaler.joblib'"
        )  # type: ignore
        assert self.__y_scaler_path.exists(), (
            f"В папке {model_dir} отсутствует файл 'GN_y_scaler.joblib'"
        )  # type: ignore
        assert self.__model_weights_path.exists(), (
            f"В папке {model_dir} отсутствует файл 'GN_best.pt'"
        )  # type: ignore

        self.__batch_size = 256
        self.__device = device
        self._model = GeomagneticModel(
            n_features=19,
            lstm_hidden_size=30,
            lstm_num_layers=1,
            lstm_dropout=0.0,
            dst_attention_heads=5,
            ae_attention_heads=15,
        )
        self._model.load_state_dict(
            torch.load(self.__model_weights_path, map_location=self.__device)
        )  # type: ignore
        self.__required_columns = [
            "datetime",
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
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "week_sin",
            "week_cos",
        ]
        self.__X_window_size = 24 * 2
        self.__y_window_size = 6
        self._model.to(self.__device)
        self.__X_scaler = joblib.load(self.__X_scaler_path)
        self.__y_scaler = joblib.load(self.__y_scaler_path)
        self.__omni2full_columns = [
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

    # def __preprocessing(
    #     self, geomagnetic_df: pd.DataFrame, batch_size: int, get_df_only: bool = False
    # ) -> DataLoader | pd.DataFrame:
    #     dataset = geomagnetic_df.copy()
    #     required_columns = [
    #         "Bz_GSM",
    #         "By_GSM",
    #         "Bx_GSE",
    #         "Kp",
    #         "T_proton",
    #         "Np_density",
    #         "V_plasma",
    #         "V_Long_GSE",
    #         "V_Lat_GSE",
    #         "Dst",
    #         "AE",
    #     ]

    #     missing_columns = [
    #         col for col in required_columns if col not in dataset.columns
    #     ]
    #     if missing_columns:
    #         raise ValueError(
    #             f"В датасете отсутствуют обязательные колонки: {', '.join(missing_columns)}. "
    #             f"Ожидаемые колонки: {', '.join(required_columns)}"
    #         )

    #     dataset = dataset[required_columns]

    #     for col in dataset.columns:
    #         dataset[col] = dataset[col].replace(FILL_VALIES[col], np.nan)
    #     features = [i for i in dataset.columns if i != "datetime"]
    #     # Nan interpolations
    #     if get_df_only:
    #         return dataset[features]

    #     dataset[features] = dataset[features].interpolate(method="pchip")
    #     # Sampling
    #     X, y = dataset[required_columns], dataset[["Dst", "AE"]]
    #     X_scaled = self.__X_scaler.transform(X)
    #     y_scaled = self.__y_scaler.transform(y)
    #     torch_dataset = GeomagneticDataset(
    #         X=X_scaled,
    #         y=y_scaled,
    #         X_window_size=self.__X_window_size,
    #         y_window_size=6,
    #         stride=6,
    #     )
    #     torch_dataloader = DataLoader(
    #         torch_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #     )
    #     return torch_dataloader

    # def __inference_and_postprocess(self, dataloader: DataLoader, alpha: float):
    #     dst_labels = []
    #     ae_labels = []
    #     dst_preds = []
    #     ae_preds = []
    #     dst_errors = []
    #     ae_errors = []

    #     with torch.no_grad():
    #         self._model.eval()
    #         for x, y in dataloader:
    #             x, y = x.to(self.__device), y.to(self.__device)
    #             dst_pred, ae_pred, attention_weights = self._model(x)
    #             preds = np.stack(
    #                 [dst_pred.cpu().numpy().flatten(), ae_pred.cpu().numpy().flatten()],
    #                 axis=1,
    #             )
    #             labels = np.stack(
    #                 [
    #                     y[:, :, 0].cpu().numpy().flatten(),
    #                     y[:, :, 1].cpu().numpy().flatten(),
    #                 ],
    #                 axis=1,
    #             )
    #             preds = self.__y_scaler.inverse_transform(preds)
    #             labels = self.__y_scaler.inverse_transform(labels)

    #             dst_preds.extend(preds[:, 0])
    #             ae_preds.extend(preds[:, 1])
    #             dst_labels.extend(labels[:, 0])
    #             ae_labels.extend(labels[:, 1])
    #             dst_errors.extend(labels[:, 0] - preds[:, 0])
    #             ae_errors.extend(labels[:, 1] - preds[:, 1])

    #     dst_preds = np.array(dst_preds)
    #     ae_preds = np.array(ae_preds)
    #     dst_labels = np.array(dst_labels)
    #     ae_labels = np.array(ae_labels)
    #     dst_errors = np.array(dst_errors)
    #     ae_errors = np.array(ae_errors)

    #     dst_std = np.std(dst_errors)
    #     ae_std = np.std(ae_errors)
    #     dst_mean = np.mean(dst_errors)
    #     ae_mean = np.mean(ae_errors)
    #     # данные для построения статистического доверительного интервала
    #     z = stats.norm.ppf((1 + alpha) / 2)

    #     dst_ci_lower = dst_preds + (dst_mean - z * dst_std)
    #     dst_ci_upper = dst_preds + (dst_mean + z * dst_std)
    #     ae_ci_lower = ae_preds + (ae_mean - z * ae_std)
    #     ae_ci_upper = ae_preds + (ae_mean + z * ae_std)

    #     return {
    #         "dst_ci_lower": dst_ci_lower,
    #         "dst_ci_upper": dst_ci_upper,
    #         "ae_ci_lower": ae_ci_lower,
    #         "ae_ci_upper": ae_ci_upper,
    #         "dst_labels": dst_labels,
    #         "dst_preds": dst_preds,
    #         "ae_labels": ae_labels,
    #         "ae_preds": ae_preds,
    #         "dst_rmse": np.sqrt(np.mean(dst_errors**2)),
    #         "ae_rmse": np.sqrt(np.mean(ae_errors**2)),
    #         "dst_std": dst_std,
    #         "ae_std": ae_std,
    #     }

    # def validate(
    #     self,
    #     geomagnetic_df: pd.DataFrame,
    #     batch_size: int,
    # ) -> GeomagnesisResult:
    #     self.__data_validator.validate(geomagnetic_df)
    #     dataloader = self.__preprocessing(geomagnetic_df, batch_size, get_df_only=False)
    #     result_data = self.__inference_and_postprocess(dataloader, alpha=0.95)  # type: ignore
    #     return GeomagnesisResult(**result_data)

    def predict_next(self, omni2file_path: str):
        self.__data_validator.is_omni_file(omni2file_path)

        dataset = make_omni2df(omni2file_path, one_file=True)
        dataset = make_time_features(dataset)
        dataset = dataset[self.__required_columns]

        complete_rows = dataset.notna().all(axis=1)
        last_notna_index = int(dataset[complete_rows].index[-1])
        print(
            f"Последний найденный индекс без пропусков: {last_notna_index}, "
            f"{int(last_notna_index / 24)}-й день с начала года"
        )
        geomagnetic_df = dataset.iloc[: last_notna_index + 1]

        min_required = self.__X_window_size + 5  # для t+1 нужно на 5 часов больше
        if len(geomagnetic_df) < min_required:
            raise ValueError(
                f"Недостаточно данных. Нужно минимум {min_required} строк, "
                f"получено {len(geomagnetic_df)}"
            )

        feature_columns = [
            col for col in geomagnetic_df.columns if col not in ["datetime", "Dst"]
        ]

        dst_predictions = []
        ae_predictions = []

        with torch.no_grad():
            self._model.eval()

            # 6 прогнозов: для t+1, t+2, ..., t+6
            for i in range(6):
                shift = 5 - i  # для i=0 (t+1): shift=5, для i=5 (t+6): shift=0
                end_idx = len(geomagnetic_df) - shift  # конец окна
                start_idx = end_idx - self.__X_window_size

                if start_idx < 0:
                    raise ValueError(f"Недостаточно данных для прогноза на шаг {i + 1}")

                window_data = geomagnetic_df.iloc[start_idx:end_idx]
                X_window = window_data[feature_columns].values
                X_scaled = self.__X_scaler.transform(X_window)
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.__device)

                dst_pred, ae_pred, _ = self._model(X_tensor)
                dst_pred_np = dst_pred.cpu().detach().numpy().flatten()
                ae_pred_np = ae_pred.cpu().detach().numpy().flatten()

                preds_combined = np.stack([dst_pred_np, ae_pred_np], axis=1)
                preds_original = self.__y_scaler.inverse_transform(preds_combined)
                dst_predictions.append(preds_original[0, 0])
                ae_predictions.append(preds_original[0, 1])

        print(f"ПРОГНОЗ НА СЛЕДУЮЩИЕ {self.__y_window_size} ЧАСОВ:")
        for i in range(self.__y_window_size):
            print(
                f"t+{i + 1}: Dst = {dst_predictions[i]:.2f} nT, AE = {ae_predictions[i]:.2f} nT"
            )

        return {
            "dst_predictions": np.array(dst_predictions),
            "ae_predictions": np.array(ae_predictions),
            "timestamps": [f"t+{i + 1}" for i in range(6)],
            "history_dst": geomagnetic_df["Dst"].values[-self.__X_window_size * 2 :],
            "history_ae": geomagnetic_df["AE"].values[-self.__X_window_size * 2 :],
            "history_timestamps": list(range(-self.__X_window_size * 2, 0)),
        }

    def __inference_on_loader_pipeline(
        self,
        model,
        loader,
        y_scaler,
        device="cuda",
        visualize: bool = False,
        title_prefix="",
        until=-1,
    ):
        dst_labels = []
        dst_preds = []
        ae_labels = []
        ae_preds = []

        with torch.no_grad():
            model.eval()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                dst_pred, ae_pred, _ = model(x)

                dst_pred = dst_pred.squeeze(1)
                ae_pred = ae_pred.squeeze(1)

                dst_true = y[:, 0, 0]  # [batch]
                ae_true = y[:, 0, 1]  # [batch]

                dst_preds.extend(dst_pred.cpu().numpy())
                ae_preds.extend(ae_pred.cpu().numpy())
                dst_labels.extend(dst_true.cpu().numpy())
                ae_labels.extend(ae_true.cpu().numpy())

        dst_preds = np.array(dst_preds)
        ae_preds = np.array(ae_preds)
        dst_labels = np.array(dst_labels)
        ae_labels = np.array(ae_labels)

        dst_pred_scaled = dst_preds.reshape(-1, 1)
        dst_true_scaled = dst_labels.reshape(-1, 1)
        ae_pred_scaled = ae_preds.reshape(-1, 1)
        ae_true_scaled = ae_labels.reshape(-1, 1)

        preds_combined = np.hstack([dst_pred_scaled, ae_pred_scaled])
        true_combined = np.hstack([dst_true_scaled, ae_true_scaled])

        preds_original = y_scaler.inverse_transform(preds_combined)
        true_original = y_scaler.inverse_transform(true_combined)

        dst_preds_orig = preds_original[:, 0][:until]
        ae_preds_orig = preds_original[:, 1][:until]
        dst_labels_orig = true_original[:, 0][:until]
        ae_labels_orig = true_original[:, 1][:until]

        # Метрики
        dst_rmse = np.sqrt(np.mean((dst_labels_orig - dst_preds_orig) ** 2))
        dst_mae = np.mean(np.abs(dst_labels_orig - dst_preds_orig))
        ae_rmse = np.sqrt(np.mean((ae_labels_orig - ae_preds_orig) ** 2))
        ae_mae = np.mean(np.abs(ae_labels_orig - ae_preds_orig))
        if visualize:
            _, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].plot(
                dst_labels_orig,
                label="Истинные значения",
                color="#1f77b4",
                linewidth=1.5,
            )
            axes[0].plot(
                dst_preds_orig, label="Прогноз", color="#ff7f0e", linewidth=1.5
            )
            axes[0].set_title(f"{title_prefix} Dst", fontsize=12)
            axes[0].set_xlabel("Временные шаги (часы)")
            axes[0].set_ylabel("Dst, nT")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].text(
                0.02,
                0.95,
                f"RMSE = {dst_rmse:.2f} nT\nMAE = {dst_mae:.2f} nT",
                transform=axes[0].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            axes[1].plot(
                ae_labels_orig,
                label="Истинные значения",
                color="#1f77b4",
                linewidth=1.5,
            )
            axes[1].plot(ae_preds_orig, label="Прогноз", color="#ff7f0e", linewidth=1.5)
            axes[1].set_title(f"{title_prefix} AE", fontsize=12)
            axes[1].set_xlabel("Временные шаги (часы)")
            axes[1].set_ylabel("AE, nT")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].text(
                0.02,
                0.95,
                f"RMSE = {ae_rmse:.2f} nT\nMAE = {ae_mae:.2f} nT",
                transform=axes[1].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.suptitle(f"{title_prefix} Прогноз геомагнитных индексов", fontsize=14)
            plt.tight_layout()
            plt.show()

            print(f"{title_prefix} Результаты прогноза")
            print(f"Dst: RMSE = {dst_rmse:.2f} nT, MAE = {dst_mae:.2f} nT")
            print(f"AE:  RMSE = {ae_rmse:.2f} nT, MAE = {ae_mae:.2f} nT")

        return {
            "dst_rmse": dst_rmse,
            "dst_mae": dst_mae,
            "ae_rmse": ae_rmse,
            "ae_mae": ae_mae,
            "dst_preds": dst_preds_orig,
            "dst_labels": dst_labels_orig,
            "ae_preds": ae_preds_orig,
            "ae_labels": ae_labels_orig,
        }

    def __draw_dst_storm_prediction(
        self,
        inference_on_loader_result,
        storm_threshold,
        padding,
        min_duration,
        min_depth,
        above=False,
    ):
        def find_storm_periods(
            labels, threshold, padding, min_duration, min_depth, above=False
        ):
            if above:
                mask = labels > threshold
            else:
                mask = labels < threshold

            periods = []
            i = 0
            while i < len(mask):
                if mask[i]:
                    start = i
                    while i < len(mask) and mask[i]:
                        i += 1
                    end = i - 1

                    duration = end - start + 1
                    if above:
                        max_val = np.max(labels[start : end + 1])
                        depth = max_val - threshold
                    else:
                        min_val = np.min(labels[start : end + 1])
                        depth = threshold - min_val  # Глубина для Dst

                    if duration >= min_duration and depth >= min_depth:
                        start_pad = max(0, start - padding)
                        periods.append((start_pad, end))
                else:
                    i += 1
            return periods

        labels_full = np.array(inference_on_loader_result["dst_labels"])
        dst_periods = find_storm_periods(
            labels_full,
            threshold=storm_threshold,
            padding=padding,
            min_duration=min_duration,
            min_depth=min_depth,
            above=above,  # Используем параметр функции
        )

        f1s = []
        for period in dst_periods:
            from_ = period[0]
            to = period[1]

            labels = np.array(inference_on_loader_result["dst_labels"][from_:to])
            preds = np.array(inference_on_loader_result["dst_preds"][from_:to])
            indices = np.arange(from_, to)

            # Для Dst: буря когда значения НИЖЕ порога
            storm_mask = labels < storm_threshold
            pred_storm_mask = preds < storm_threshold

            plt.figure(figsize=(18, 5))
            plt.plot(
                indices,
                labels,
                label="Истинные значения",
                color="#1f77b4",
                linewidth=1.5,
            )
            plt.plot(indices, preds, label="Прогноз", color="#ff7f0e", linewidth=1.5)

            # Фактические бури
            storm_indices = indices[storm_mask]
            storm_values = labels[storm_mask]
            plt.scatter(
                storm_indices,
                storm_values,
                color="red",
                s=50,
                zorder=5,
                label=f"Фактическая буря (Dst < {storm_threshold} nT)",
                alpha=0.7,
            )

            # Предсказанные бури
            pred_storm_indices = indices[pred_storm_mask]
            pred_storm_values = preds[pred_storm_mask]
            plt.scatter(
                pred_storm_indices,
                pred_storm_values,
                color="green",
                s=50,
                zorder=5,
                marker="s",
                label=f"Модель предсказала бурю (Dst < {storm_threshold} nT)",
                alpha=0.7,
            )

            plt.axhline(
                y=storm_threshold,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Порог {storm_threshold} nT",
            )

            # Закрашиваем области бурь
            for i in range(len(storm_mask)):
                if storm_mask[i]:
                    plt.axvspan(
                        indices[i] - 0.5,
                        indices[i] + 0.5,
                        alpha=0.15,
                        color="red",
                        zorder=0,
                    )

            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper left")
            plt.title(
                f"Dst: прогноз vs факт с выделением бурь (Dst < {storm_threshold} nT)"
            )
            plt.xlabel("Временные шаги")
            plt.ylabel("Dst, nT")
            plt.tight_layout()
            plt.show()

            # Статистика
            print(f"=== Статистика по бурям (Dst < {storm_threshold} nT) ===")
            print(f"Всего моментов с бурей по факту: {np.sum(storm_mask)}")
            print(
                f"Всего моментов, когда модель предсказала бурю: {np.sum(pred_storm_mask)}"
            )

            true_positives = np.sum(storm_mask & pred_storm_mask)
            false_positives = np.sum((~storm_mask) & pred_storm_mask)
            false_negatives = np.sum(storm_mask & (~pred_storm_mask))

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )

            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                print(f"F1: {f1:.3f}")
                f1s.append(f1)
        if f1s:
            print(f"\nСредний F1 по всем бурям: {sum(f1s) / len(f1s):.3f}")
        else:
            print("\nНе найдено ни одного периода бури!")
        return f1s

    def __call__(
        self,
        dataset: pd.DataFrame,
        visualize,
        prefix,
        until=-1,
        storm_threshold=-90,
    ):  # type: ignore
        dataset = dataset[self.__required_columns]
        X_test, y_test = (
            dataset.drop(columns=["datetime", "Dst"]),
            dataset[["Dst", "AE"]],
        )

        X_test_scaled = self.__X_scaler.transform(X_test)
        y_test_scaled = self.__y_scaler.transform(y_test)

        test_dataset = GeomagneticDataset(
            X=X_test_scaled,
            y=y_test_scaled,
            X_window_size=self.__X_window_size,
            y_window_size=self.__y_window_size,
            stride=1,
        )
        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.__batch_size, shuffle=False
        )
        result = self.__inference_on_loader_pipeline(
            model=self._model,
            loader=val_loader,
            y_scaler=self.__y_scaler,
            device=self.__device,
            visualize=visualize,
            title_prefix=prefix,
            until=until,
        )
        self.__draw_dst_storm_prediction(
            result,
            storm_threshold=storm_threshold,
            padding=5,
            min_duration=5,
            min_depth=5,
            above=False,
        )
        return result
