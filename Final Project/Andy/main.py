import argparse
from pathlib import Path

from config import get_config
from src.pipeline import run_single_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Image filename inside data/images/")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI/manual fallback")
    parser.add_argument(
        "--force-manual",
        action="store_true",
        help="Always use manual card corner selection (requires GUI)",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Target object profile (e.g. phone, champagne, generic)",
    )
    args = parser.parse_args()

    cfg = get_config()

    image_path = Path(cfg["paths"]["images"]) / args.image

    result = run_single_image(
        image_path=image_path,
        cfg=cfg,
        allow_manual_fallback=not args.no_gui,
        force_manual=args.force_manual,
        object_name=args.object,
    )

    print("\n=== Measurement Result ===")
    print(f"Image: {args.image}")
    print(f"Card detection method: {result['card']['method']}")
    print(f"Width  = {result['measurement']['width_mm']:.2f} mm")
    print(f"Height = {result['measurement']['height_mm']:.2f} mm")
    print(f"Area   = {result['measurement']['area_mm2']:.2f} mm^2")
    print(f"Confidence = {result['reliability']['level']} ({result['reliability']['score']:.1f}/100)")
    print(f"Reason = {result['reliability']['reason']}")
    print(f"Overlay saved to: {result['paths']['overlay']}")


if __name__ == "__main__":
    main()
