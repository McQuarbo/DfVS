from pathlib import Path


def get_config():
    root = Path(__file__).resolve().parent

    cfg = {
        "paths": {
            "root": root,
            "images": root / "data" / "images",
            "results": root / "results",
            "intermediate": root / "results" / "intermediate",
            "overlays": root / "results" / "overlays",
            "tables": root / "results" / "tables",
            "ground_truth": root / "data" / "ground_truth.csv",
        },
        "card": {
            "real_width_mm": 85.60,
            "real_height_mm": 53.98,
            "true_aspect": 85.60 / 53.98,
        },
        "preprocess": {
            "gaussian_sigma": 1.0,
            "use_clahe": True,
        },
        "card_detection": {
            "use_color_first": True,
            "green_hsv_lower": [35, 40, 40],
            "green_hsv_upper": [95, 255, 255],
            "min_green_area": 1500,
            "max_green_area_frac": 0.08,
            "relaxed_max_green_area_frac": 0.14,
            "edge_min_area_frac": 0.002,
            "edge_max_area_frac": 0.30,
            "aspect_tol": 0.45,
            "relaxed_aspect_tol": 0.65,
        },
        "rectification": {
            "output_card_width_px": 500,
        },
        "segmentation": {
            "method": "adaptive",
            "adaptive_block_size": 51,
            "adaptive_C": 8,
            "min_component_area": 3000,
            "phone_target_aspect": 160.8 / 78.1,
            "target_aspects": [160.8 / 78.1, 255.0 / 75.0],
        },
        "default_object": "phone",
        "objects": {
            "phone": {
                "target_aspect": 160.8 / 78.1,
                "short_mm_range": [55.0, 95.0],
                "long_mm_range": [120.0, 220.0],
                "min_aspect": 1.5,
                "max_aspect": 3.2,
                "relax_border": False,
                "use_card_outline": False,
                "use_color_outline": False,
                "trim_percentiles": [6, 94],
            },
            "champagne": {
                "target_aspect": 255.0 / 75.0,
                "short_mm_range": [50.0, 110.0],
                "long_mm_range": [170.0, 360.0],
                "min_aspect": 2.0,
                "max_aspect": 5.2,
                "relax_border": False,
                "use_card_outline": False,
                "use_color_outline": True,
                "trim_percentiles": [1, 99],
            },
            "eevee": {
                "target_aspect": 88.0 / 63.5,
                "short_mm_range": [45.0, 90.0],
                "long_mm_range": [65.0, 125.0],
                "min_aspect": 1.2,
                "max_aspect": 1.9,
                "relax_border": True,
                "use_card_outline": True,
                "use_color_outline": False,
                "trim_percentiles": [1, 99],
            },
            "pikachu": {
                "target_aspect": 187.0 / 132.0,
                "short_mm_range": [90.0, 180.0],
                "long_mm_range": [130.0, 260.0],
                "min_aspect": 1.2,
                "max_aspect": 1.9,
                "relax_border": True,
                "use_card_outline": True,
                "use_color_outline": False,
                "trim_percentiles": [1, 99],
            },
            "generic": {
                "target_aspect": 2.0,
                "short_mm_range": [20.0, 220.0],
                "long_mm_range": [60.0, 520.0],
                "min_aspect": 1.15,
                "max_aspect": 6.0,
                "relax_border": True,
                "use_card_outline": True,
                "use_color_outline": True,
                "trim_percentiles": [1, 99],
            },
        },
        "reliability": {
            "high_threshold": 80,
            "medium_threshold": 55,
        },
        "save": {
            "intermediate": True,
            "overlay": True,
        },
    }

    cfg["rectification"]["output_card_height_px"] = int(
        cfg["rectification"]["output_card_width_px"] / cfg["card"]["true_aspect"]
    )

    return cfg
