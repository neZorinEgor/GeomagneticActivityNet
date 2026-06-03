import os

import numpy as np
import torch
from scipy.special import inv_boxcox
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from tqdm import tqdm


def train_geomagnetic_model(
    model,
    loss_fn,
    optimizer,
    lr_scheduler,
    epochs,
    train_loader,
    val_loader,
    lmbda: float | None,  # for box-cox method
    y_scaler,
    patience,
    patience_delta,
    dst_loss_sigma=1.0,
    ae_loss_sigma=1.0,
    device="cuda",
    save_dir="./models",
):
    os.makedirs(save_dir, exist_ok=True)

    patience_counter = 0
    best_val_loss = float("inf")
    best_val_dst_rmse = float("inf")
    best_val_ae_rmse = float("inf")
    best_epoch = 0
    best_dst_epoch = 0
    best_ae_epoch = 0

    history_train = {
        "lr": [],
        "dst_loss": [],
        "ae_loss": [],
        "summary_loss": [],
        "ae": {"RMSE": [], "MAPE": [], "MAE": []},
        "dst": {"RMSE": [], "MAPE": [], "MAE": []},
    }
    history_val = {
        "loss": [],
        "ae": {"RMSE": [], "MAPE": [], "MAE": []},
        "dst": {"RMSE": [], "MAPE": [], "MAE": []},
    }

    model.to(device)

    for e in range(epochs):
        # ========== TRAIN ==========
        model.train()
        train_progress = tqdm(train_loader)
        running_history = {
            "dst_loss": [],
            "ae_loss": [],
            "summary_loss": [],
            "ae": {"RMSE": [], "MAPE": [], "MAE": []},
            "dst": {"RMSE": [], "MAPE": [], "MAE": []},
        }

        for x, y in train_progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            dst_y, ae_y = y[:, :, 0], y[:, :, 1]
            dst_y_pred, ae_y_pred, _ = model(x)

            dst_loss = loss_fn(dst_y_pred, dst_y)
            ae_loss = loss_fn(ae_y_pred, ae_y)
            loss = dst_loss * dst_loss_sigma + ae_loss_sigma * ae_loss

            loss.backward()
            optimizer.step()

            # Метрики для логирования
            with torch.no_grad():
                pred_data = np.stack(
                    [
                        dst_y_pred.detach().cpu().numpy().flatten(),
                        ae_y_pred.detach().cpu().numpy().flatten(),
                    ],
                    axis=1,
                )
                label_data = np.stack(
                    [
                        dst_y.detach().cpu().numpy().flatten(),
                        ae_y.detach().cpu().numpy().flatten(),
                    ],
                    axis=1,
                )
                pred_data = y_scaler.inverse_transform(pred_data)
                label_data = y_scaler.inverse_transform(label_data)
                if lmbda is not None:
                    pred_data[:, 1] = inv_boxcox(pred_data[:, 1], lmbda)
                    label_data[:, 1] = inv_boxcox(label_data[:, 1], lmbda)

                dst_rmse = root_mean_squared_error(label_data[:, 0], pred_data[:, 0])
                ae_rmse = root_mean_squared_error(label_data[:, 1], pred_data[:, 1])
                dst_mape = mean_absolute_percentage_error(
                    label_data[:, 0], pred_data[:, 0]
                )
                ae_mape = mean_absolute_percentage_error(
                    label_data[:, 1], pred_data[:, 1]
                )
                dst_mae = mean_absolute_error(label_data[:, 0], pred_data[:, 0])
                ae_mae = mean_absolute_error(label_data[:, 1], pred_data[:, 1])

                running_history["dst"]["RMSE"].append(dst_rmse)
                running_history["ae"]["RMSE"].append(ae_rmse)
                running_history["dst"]["MAPE"].append(dst_mape)
                running_history["ae"]["MAPE"].append(ae_mape)
                running_history["dst"]["MAE"].append(dst_mae)
                running_history["ae"]["MAE"].append(ae_mae)
                running_history["dst_loss"].append(dst_loss.item())
                running_history["ae_loss"].append(ae_loss.item())
                running_history["summary_loss"].append(loss.item())

            train_progress.set_description(
                f"Epoch [{e + 1:>3}/{epochs}] [TRAIN] dst-RMSE {dst_rmse:.3f} ae-RMSE {ae_rmse:.3f} Loss {loss.item():.4f} AE Loss {ae_loss.item():.4f} Dst Loss {dst_loss.item():.4f}"
            )
        history_train["lr"].append(lr_scheduler.get_last_lr())
        history_train["dst"]["RMSE"].append(np.mean(running_history["dst"]["RMSE"]))
        history_train["ae"]["RMSE"].append(np.mean(running_history["ae"]["RMSE"]))
        history_train["dst"]["MAPE"].append(np.mean(running_history["dst"]["MAPE"]))
        history_train["ae"]["MAPE"].append(np.mean(running_history["ae"]["MAPE"]))
        history_train["dst"]["MAE"].append(np.mean(running_history["dst"]["MAE"]))
        history_train["ae"]["MAE"].append(np.mean(running_history["ae"]["MAE"]))
        history_train["dst_loss"].append(np.mean(running_history["dst_loss"]))
        history_train["ae_loss"].append(np.mean(running_history["ae_loss"]))
        history_train["summary_loss"].append(np.mean(running_history["summary_loss"]))

        # ========== VALIDATION ==========
        with torch.no_grad():
            model.eval()
            val_progress = tqdm(val_loader)
            running_val = {
                "loss": [],
                "ae": {"RMSE": [], "MAPE": [], "MAE": []},
                "dst": {"RMSE": [], "MAPE": [], "MAE": []},
            }

            for x, y in val_progress:
                x, y = x.to(device), y.to(device)
                dst_y, ae_y = y[:, :, 0], y[:, :, 1]
                dst_y_pred, ae_y_pred, _ = model(x)

                # Validation loss
                dst_loss = loss_fn(dst_y_pred, dst_y)
                ae_loss = loss_fn(ae_y_pred, ae_y)
                val_loss = dst_loss + ae_loss
                running_val["loss"].append(val_loss.item())

                # Метрики для логирования
                pred_data = np.stack(
                    [
                        dst_y_pred.detach().cpu().numpy().flatten(),
                        ae_y_pred.detach().cpu().numpy().flatten(),
                    ],
                    axis=1,
                )
                label_data = np.stack(
                    [
                        dst_y.detach().cpu().numpy().flatten(),
                        ae_y.detach().cpu().numpy().flatten(),
                    ],
                    axis=1,
                )
                pred_data = y_scaler.inverse_transform(pred_data)
                label_data = y_scaler.inverse_transform(label_data)
                if lmbda is not None:
                    pred_data[:, 1] = inv_boxcox(pred_data[:, 1], lmbda)
                    label_data[:, 1] = inv_boxcox(label_data[:, 1], lmbda)

                dst_rmse = root_mean_squared_error(label_data[:, 0], pred_data[:, 0])
                ae_rmse = root_mean_squared_error(label_data[:, 1], pred_data[:, 1])
                dst_mape = mean_absolute_percentage_error(
                    label_data[:, 0], pred_data[:, 0]
                )
                ae_mape = mean_absolute_percentage_error(
                    label_data[:, 1], pred_data[:, 1]
                )
                dst_mae = mean_absolute_error(label_data[:, 0], pred_data[:, 0])
                ae_mae = mean_absolute_error(label_data[:, 1], pred_data[:, 1])

                running_val["dst"]["RMSE"].append(dst_rmse)
                running_val["ae"]["RMSE"].append(ae_rmse)
                running_val["dst"]["MAPE"].append(dst_mape)
                running_val["ae"]["MAPE"].append(ae_mape)
                running_val["dst"]["MAE"].append(dst_mae)
                running_val["ae"]["MAE"].append(ae_mae)

                val_progress.set_description(
                    f"Epoch [{e + 1:>3}/{epochs}] [VAL  ] dst-RMSE {dst_rmse:.3f} ae-RMSE {ae_rmse:.3f} Loss {val_loss.item():.4f}"
                )

            current_val_loss = np.mean(running_val["loss"])
            current_val_dst_rmse = np.mean(running_val["dst"]["RMSE"])
            current_val_ae_rmse = np.mean(running_val["ae"]["RMSE"])

            history_val["loss"].append(current_val_loss)
            history_val["dst"]["RMSE"].append(current_val_dst_rmse)
            history_val["ae"]["RMSE"].append(current_val_ae_rmse)
            history_val["dst"]["MAPE"].append(np.mean(running_val["dst"]["MAPE"]))
            history_val["ae"]["MAPE"].append(np.mean(running_val["ae"]["MAPE"]))
            history_val["dst"]["MAE"].append(np.mean(running_val["dst"]["MAE"]))
            history_val["ae"]["MAE"].append(np.mean(running_val["ae"]["MAE"]))

        lr_scheduler.step(metrics=current_val_loss)

        # Сохранение лучшей модели по общей loss
        if current_val_loss < best_val_loss - patience_delta:
            best_val_loss = current_val_loss
            best_epoch = e + 1
            best_model_path = os.path.join(
                save_dir, f"best_model_loss_{best_epoch}_loss_{best_val_loss:.6f}.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model by loss! Epoch {best_epoch}, Val Loss: {best_val_loss:.6f}"
            )
            print(f"Saved to: {best_model_path}")

        # Сохранение лучшей модели по DST RMSE
        if current_val_dst_rmse < best_val_dst_rmse - (patience_delta * 0.1):
            best_val_dst_rmse = current_val_dst_rmse
            best_dst_epoch = e + 1
            best_dst_path = os.path.join(
                save_dir,
                f"best_model_dst_epoch_{best_dst_epoch}_rmse_{best_val_dst_rmse:.3f}.pt",
            )
            torch.save(model.state_dict(), best_dst_path)
            print(
                f"New best model by DST RMSE! Epoch {best_dst_epoch}, DST RMSE: {best_val_dst_rmse:.3f}"
            )
            print(f"Saved to: {best_dst_path}")

        # Сохранение лучшей модели по AE RMSE
        if current_val_ae_rmse < best_val_ae_rmse - (patience_delta * 0.1):
            best_val_ae_rmse = current_val_ae_rmse
            best_ae_epoch = e + 1
            best_ae_path = os.path.join(
                save_dir,
                f"best_model_ae_epoch_{best_ae_epoch}_rmse_{best_val_ae_rmse:.3f}.pt",
            )
            torch.save(model.state_dict(), best_ae_path)
            print(
                f"New best model by AE RMSE! Epoch {best_ae_epoch}, AE RMSE: {best_val_ae_rmse:.3f}"
            )
            print(f"Saved to: {best_ae_path}")

        # Логика ранней остановки (на основе общей loss)
        if current_val_loss < best_val_loss - patience_delta:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {e + 1}")
            print(
                f"Best model by loss at epoch {best_epoch} with val loss: {best_val_loss:.6f}"
            )
            print(
                f"Best model by DST at epoch {best_dst_epoch} with RMSE: {best_val_dst_rmse:.3f}"
            )
            print(
                f"Best model by AE at epoch {best_ae_epoch} with RMSE: {best_val_ae_rmse:.3f}"
            )
            last_model_path = os.path.join(save_dir, "last.pt")
            torch.save(model.state_dict(), last_model_path)
            print(f"Last model saved to: {last_model_path}")
            break

    # Сохраняем финальную модель, если цикл завершился без early stopping
    else:
        final_model_path = os.path.join(save_dir, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed! Final model saved to: {final_model_path}")
        print(
            f"Best model by loss at epoch {best_epoch} with val loss: {best_val_loss:.6f}"
        )
        print(
            f"Best model by DST at epoch {best_dst_epoch} with RMSE: {best_val_dst_rmse:.3f}"
        )
        print(
            f"Best model by AE at epoch {best_ae_epoch} with RMSE: {best_val_ae_rmse:.3f}"
        )

    return model, history_train, history_val
