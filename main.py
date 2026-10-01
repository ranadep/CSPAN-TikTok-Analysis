import csv
from datetime import datetime, timezone
from pathlib import Path

from feature_extractors.body_gestures import extract_body_features
from feature_extractors.facial_expressions import extract_face_features


def main():

    video_path = "videos/test.mp4"

    print("Processing:", video_path)

    print("\nExtracting body gesture features...")

    body_features = extract_body_features(video_path)

    print("\nBODY FEATURES")

    for feature, value in body_features.items():
        print(f"{feature}: {value:.4f}")

    print("\nExtracting facial expression features...")

    face_features = extract_face_features(video_path)

    print("\nFACIAL FEATURES")

    for feature, value in face_features.items():
        print(f"{feature}: {value:.4f}")

    output_path = Path("output/features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_path": video_path,
        **{f"body_{name}": value for name, value in body_features.items()},
        **{f"face_{name}": value for name, value in face_features.items()},
    }
    fieldnames = list(row)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()