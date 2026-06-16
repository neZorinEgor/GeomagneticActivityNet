from pathlib import Path
from typing import Literal, TypeAlias

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from .constants import FILL_VALIES
from .dataset import GeomagneticDataset
from .model import GeomagneticModelV1
from .schemas import GeomagnesisResult
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

        self.__device = device
        self.__model = GeomagneticModelV1(
            lstm_input_size=14,
            lstm_hidden_size=64,
            lstm_num_layers=2,
            lstm_dropout=0.2,
            dst_attention_heads=2,
            ae_attention_heads=2,
            forecasts_len=6,
        )
        self.__model.load_state_dict(
            torch.load(self.__model_weights_path, map_location=self.__device)
        )  # type: ignore
        self.__X_window_size = 168
        self.__model.to(self.__device)
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

    def __preprocessing(
        self, geomagnetic_df: pd.DataFrame, batch_size: int, get_df_only: bool = False
    ) -> DataLoader | pd.DataFrame:
        dataset = geomagnetic_df.copy()
        required_columns = [
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

        missing_columns = [
            col for col in required_columns if col not in dataset.columns
        ]
        if missing_columns:
            raise ValueError(
                f"В датасете отсутствуют обязательные колонки: {', '.join(missing_columns)}. "
                f"Ожидаемые колонки: {', '.join(required_columns)}"
            )

        dataset = dataset[required_columns]

        for col in dataset.columns:
            dataset[col] = dataset[col].replace(FILL_VALIES[col], np.nan)
        features = [i for i in dataset.columns if i != "datetime"]
        # Nan interpolations
        if get_df_only:
            return dataset[features]

        dataset[features] = dataset[features].interpolate(method="pchip")
        # Sampling
        X, y = dataset[required_columns], dataset[["Dst", "AE"]]
        X_scaled = self.__X_scaler.transform(X)
        y_scaled = self.__y_scaler.transform(y)
        torch_dataset = GeomagneticDataset(
            X=X_scaled,
            y=y_scaled,
            X_window_size=self.__X_window_size,
            y_window_size=6,
            stride=6,
        )
        torch_dataloader = DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return torch_dataloader

    def __inference_and_postprocess(self, dataloader: DataLoader, alpha: float):
        dst_labels = []
        ae_labels = []
        dst_preds = []
        ae_preds = []
        dst_errors = []
        ae_errors = []

        with torch.no_grad():
            self.__model.eval()
            for x, y in dataloader:
                x, y = x.to(self.__device), y.to(self.__device)
                dst_pred, ae_pred, attention_weights = self.__model(x)
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
                preds = self.__y_scaler.inverse_transform(preds)
                labels = self.__y_scaler.inverse_transform(labels)

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
        # данные для построения статистического доверительного интервала
        z = stats.norm.ppf((1 + alpha) / 2)

        dst_ci_lower = dst_preds + (dst_mean - z * dst_std)
        dst_ci_upper = dst_preds + (dst_mean + z * dst_std)
        ae_ci_lower = ae_preds + (ae_mean - z * ae_std)
        ae_ci_upper = ae_preds + (ae_mean + z * ae_std)

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

    def validate(
        self,
        geomagnetic_df: pd.DataFrame,
        batch_size: int,
    ) -> GeomagnesisResult:
        self.__data_validator.validate(geomagnetic_df)
        dataloader = self.__preprocessing(geomagnetic_df, batch_size, get_df_only=False)
        result_data = self.__inference_and_postprocess(dataloader, alpha=0.95)  # type: ignore
        return GeomagnesisResult(**result_data)

    def predict_next(self, omni2file_path: str):
        self.__data_validator.is_omni_file(omni2file_path)
        omni2file_path = Path(omni2file_path)  # type: ignore
        omni2_df = pd.read_csv(omni2file_path, header=None, sep="\\s+")
        assert len(omni2_df.columns) == len(self.__omni2full_columns), (
            f"omni2 файл должен содержать {int(len(self.__omni2full_columns))} колонок, пример: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_2025.dat"
        )
        omni2_df.columns = self.__omni2full_columns
        omni2_df = self.__preprocessing(
            geomagnetic_df=omni2_df,
            get_df_only=True,
            batch_size=-1,
        )
        complete_rows = omni2_df.notna().all(axis=1)  # type: ignore
        last_notna_index = int(omni2_df[complete_rows].index[-1])  # type: ignore
        print(
            f"Последний найденный индекс без последующих пропусков в файле: {last_notna_index}, {int(last_notna_index / 24)}-й день с начала года"
        )
        geomagnetic_df = omni2_df.iloc[:last_notna_index]  # type: ignore
        if len(geomagnetic_df) < self.__X_window_size:
            raise Exception("TODO")
        last_batch = geomagnetic_df.tail(self.__X_window_size)
        X_scaled = self.__X_scaler.transform(last_batch)
        X = torch.FloatTensor(X_scaled).to(self.__device)
        dst_pred, ae_pred, (dst_attention, ae_attention) = self.__model(X.unsqueeze(0))
        preds = np.stack(
            [
                dst_pred.cpu().detach().numpy().flatten(),
                ae_pred.cpu().detach().numpy().flatten(),
            ],
            axis=1,
        )
        preds = self.__y_scaler.inverse_transform(preds)
        dst_preds = preds[:, 0]
        ae_preds = preds[:, 1]
        return (dst_preds, ae_preds, dst_attention, ae_attention), last_batch
