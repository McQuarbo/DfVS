import pandas as pd

from .pipeline import run_single_image


def evaluate_dataset(cfg):
    gt_path = cfg["paths"]["ground_truth"]
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_path}")

    df = pd.read_csv(gt_path)
    required = {"image_name", "true_width_mm", "true_height_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ground_truth.csv: {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        image_path = cfg["paths"]["images"] / row["image_name"]
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        obj_name = row.get("object", None) if "object" in df.columns else None
        print(f"Evaluating {image_path.name} (object={obj_name or 'default'})...")
        try:
            result = run_single_image(image_path, cfg, allow_manual_fallback=False, object_name=obj_name)
            pred_w = result["measurement"]["width_mm"]
            pred_h = result["measurement"]["height_mm"]
            conf = result["reliability"]["level"]
            method = result["card"]["method"]
        except Exception:
            pred_w = float("nan")
            pred_h = float("nan")
            conf = "Low"
            method = "failed_auto"

        true_w = float(row["true_width_mm"])
        true_h = float(row["true_height_mm"])
        abs_err_w = abs(pred_w - true_w) if pd.notna(pred_w) else float("nan")
        abs_err_h = abs(pred_h - true_h) if pd.notna(pred_h) else float("nan")
        pct_err_w = 100 * abs_err_w / true_w if pd.notna(abs_err_w) else float("nan")
        pct_err_h = 100 * abs_err_h / true_h if pd.notna(abs_err_h) else float("nan")

        rows.append(
            {
                "image_name": row["image_name"],
                "true_width_mm": true_w,
                "true_height_mm": true_h,
                "pred_width_mm": pred_w,
                "pred_height_mm": pred_h,
                "abs_err_width_mm": abs_err_w,
                "abs_err_height_mm": abs_err_h,
                "pct_err_width": pct_err_w,
                "pct_err_height": pct_err_h,
                "confidence": conf,
                "method": method,
            }
        )

    out = pd.DataFrame(rows)
    out_path = cfg["paths"]["tables"] / "evaluation_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved evaluation table to: {out_path}")

    valid = out[["pred_width_mm", "pred_height_mm"]].notna().all(axis=1)
    if valid.any():
        print(f"Mean absolute width error:  {out.loc[valid, 'abs_err_width_mm'].mean():.2f} mm")
        print(f"Mean absolute height error: {out.loc[valid, 'abs_err_height_mm'].mean():.2f} mm")
