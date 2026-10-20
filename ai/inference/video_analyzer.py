import os
import cv2
import json
import numpy as np
from datetime import datetime


class CropVideoAnalyzer:
    """
    Video-based crop health analyzer.

    Workflow:
        Video
          ↓
        Frame extraction
          ↓
        Blur/quality filtering
          ↓
        Disease model
          ↓
        Prediction aggregation
          ↓
        Video-level assessment
    """

    def __init__(
        self,
        predictor,
        frame_interval=10,
        max_frames=30,
        blur_threshold=80.0,
        min_confidence=0.0
    ):
        self.predictor = predictor

        # Process approximately every Nth frame.
        self.frame_interval = frame_interval

        # Prevent extremely long processing times.
        self.max_frames = max_frames

        # Lower value = more blurry.
        self.blur_threshold = blur_threshold

        self.min_confidence = min_confidence

    # =========================================================
    # IMAGE QUALITY
    # =========================================================

    def calculate_blur_score(self, frame):
        """
        Calculate image sharpness using Laplacian variance.

        Higher score generally means sharper image.
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        return float(score)

    # =========================================================
    # FRAME EXTRACTION
    # =========================================================

    def extract_frames(self, video_path):

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {video_path}"
            )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        if fps <= 0:
            fps = 30.0

        duration = (
            total_frames / fps
            if total_frames > 0
            else 0
        )

        frames = []

        frame_number = 0

        print()
        print("=" * 70)
        print("EXTRACTING VIDEO FRAMES")
        print("=" * 70)

        print("Total frames:", total_frames)
        print("FPS:", round(fps, 2))
        print("Duration:", round(duration, 2), "seconds")

        while True:

            success, frame = capture.read()

            if not success:
                break

            if frame_number % self.frame_interval == 0:

                frames.append({
                    "frame_number": frame_number,
                    "timestamp": frame_number / fps,
                    "image": frame
                })

                if len(frames) >= self.max_frames:
                    break

            frame_number += 1

        capture.release()

        print("Frames extracted:", len(frames))

        return frames

    # =========================================================
    # FRAME QUALITY FILTER
    # =========================================================

    def select_quality_frames(self, frames):

        print()
        print("=" * 70)
        print("CHECKING FRAME QUALITY")
        print("=" * 70)

        quality_frames = []

        for item in frames:

            frame = item["image"]

            blur_score = self.calculate_blur_score(
                frame
            )

            item["blur_score"] = blur_score

            if blur_score >= self.blur_threshold:

                quality_frames.append(item)

        print(
            "Usable frames:",
            len(quality_frames),
            "/",
            len(frames)
        )

        return quality_frames

    # =========================================================
    # SAVE TEMPORARY FRAME
    # =========================================================

    def save_frame(
        self,
        frame,
        output_directory,
        frame_number
    ):

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        filename = (
            f"frame_{frame_number:06d}.jpg"
        )

        filepath = os.path.join(
            output_directory,
            filename
        )

        success = cv2.imwrite(
            filepath,
            frame
        )

        if not success:
            raise RuntimeError(
                f"Could not save frame: {filepath}"
            )

        return filepath

    # =========================================================
    # EXTRACT PREDICTION INFORMATION
    # =========================================================

    def extract_prediction_info(self, prediction):

        if not isinstance(prediction, dict):
            return None

        disease = prediction.get(
            "disease"
        )

        if not isinstance(disease, dict):
            return None

        name = disease.get("name")

        # Your predictor may return "Uncertain".
        # We still keep raw_prediction if available.
        raw_name = disease.get(
            "raw_prediction",
            name
        )

        confidence = disease.get(
            "confidence"
        )

        if confidence is None:

            confidence = (
                disease.get(
                    "confidence_percent",
                    0
                ) / 100
            )

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "name": name,
            "raw_prediction": raw_name,
            "confidence": confidence,
            "confidence_percent": round(
                confidence * 100,
                2
            ),
            "confidence_level": disease.get(
                "confidence_level"
            ),
            "class_index": disease.get(
                "class_index"
            )
        }

    # =========================================================
    # RUN PREDICTION ON FRAMES
    # =========================================================

    def analyze_frames(
        self,
        quality_frames,
        crop,
        temporary_directory
    ):

        print()
        print("=" * 70)
        print("RUNNING AI PREDICTIONS ON VIDEO")
        print("=" * 70)

        results = []

        for index, item in enumerate(
            quality_frames,
            start=1
        ):

            frame = item["image"]

            frame_number = item[
                "frame_number"
            ]

            timestamp = item[
                "timestamp"
            ]

            filepath = self.save_frame(
                frame,
                temporary_directory,
                frame_number
            )

            print(
                f"[{index}/{len(quality_frames)}] "
                f"Frame {frame_number} "
                f"at {timestamp:.2f}s"
            )

            try:

                prediction = (
                    self.predictor.predict_crop_health(
                        filepath,
                        crop
                    )
                )

                prediction_info = (
                    self.extract_prediction_info(
                        prediction
                    )
                )

                if prediction_info is None:
                    print(
                        "  Prediction could not be interpreted."
                    )
                    continue

                if (
                    prediction_info["confidence"]
                    < self.min_confidence
                ):
                    continue

                results.append({
                    "frame_number": frame_number,
                    "timestamp": round(
                        timestamp,
                        3
                    ),
                    "blur_score": round(
                        item["blur_score"],
                        2
                    ),
                    "prediction": prediction_info
                })

                print(
                    f"  → "
                    f"{prediction_info['name']} "
                    f"({prediction_info['confidence_percent']}%)"
                )

            except Exception as exc:

                print(
                    "  Frame prediction error:",
                    str(exc)
                )

        return results

    # =========================================================
    # AGGREGATE VIDEO PREDICTIONS
    # =========================================================

    def aggregate_predictions(
        self,
        frame_results
    ):

        if not frame_results:

            return {
                "status": "No usable predictions",
                "disease": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "agreement": 0.0,
                "frames_analyzed": 0
            }

        groups = {}

        for result in frame_results:

            prediction = result[
                "prediction"
            ]

            name = prediction.get(
                "name"
            )

            raw_name = prediction.get(
                "raw_prediction"
            )

            # Prefer raw prediction when
            # the predictor has marked the
            # result as uncertain.
            if name == "Uncertain":
                name = raw_name

            if not name:
                continue

            if name not in groups:
                groups[name] = []

            groups[name].append(
                prediction["confidence"]
            )

        if not groups:

            return {
                "status": "Insufficient prediction data",
                "disease": None,
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "agreement": 0.0,
                "frames_analyzed": len(
                    frame_results
                )
            }

        # Select the prediction occurring
        # in the largest number of frames.
        majority_name = max(
            groups,
            key=lambda name: len(
                groups[name]
            )
        )

        confidences = groups[
            majority_name
        ]

        mean_confidence = float(
            np.mean(confidences)
        )

        frame_count = len(
            frame_results
        )

        agreement = (
            len(confidences)
            / frame_count
        )

        # The video result should consider
        # both confidence and temporal agreement.
        #
        # This is NOT a medically/agriculturally
        # validated probability.
        video_score = (
            mean_confidence *
            agreement
        )

        if video_score >= 0.75:

            status = (
                "Strong consistent visual signal"
            )

        elif video_score >= 0.50:

            status = (
                "Moderate visual signal"
            )

        else:

            status = (
                "Weak or inconsistent visual signal"
            )

        return {
            "status": status,
            "disease": majority_name,
            "mean_confidence": round(
                mean_confidence,
                4
            ),
            "confidence_percent": round(
                mean_confidence * 100,
                2
            ),
            "agreement": round(
                agreement,
                4
            ),
            "agreement_percent": round(
                agreement * 100,
                2
            ),
            "video_score": round(
                video_score,
                4
            ),
            "video_score_percent": round(
                video_score * 100,
                2
            ),
            "frames_analyzed": frame_count
        }

    # =========================================================
    # MAIN VIDEO ANALYSIS
    # =========================================================

    def analyze_video(
        self,
        video_path,
        crop,
        output_directory
    ):

        start_time = datetime.now()

        print()
        print("=" * 70)
        print("AI CROP VIDEO ANALYSIS")
        print("=" * 70)

        print("Video:", video_path)
        print("Crop:", crop)

        # -----------------------------------------------------
        # 1. Extract frames
        # -----------------------------------------------------

        frames = self.extract_frames(
            video_path
        )

        if not frames:

            raise RuntimeError(
                "No frames could be extracted from the video."
            )

        # -----------------------------------------------------
        # 2. Quality filtering
        # -----------------------------------------------------

        quality_frames = (
            self.select_quality_frames(
                frames
            )
        )

        if not quality_frames:

            return {
                "success": False,
                "error": (
                    "No sufficiently clear frames "
                    "were found in the video."
                )
            }

        # -----------------------------------------------------
        # 3. Temporary frame directory
        # -----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        frame_directory = os.path.join(
            output_directory,
            f"video_frames_{timestamp}"
        )

        os.makedirs(
            frame_directory,
            exist_ok=True
        )

        # -----------------------------------------------------
        # 4. AI prediction
        # -----------------------------------------------------

        frame_results = (
            self.analyze_frames(
                quality_frames,
                crop,
                frame_directory
            )
        )

        # -----------------------------------------------------
        # 5. Aggregate
        # -----------------------------------------------------

        aggregate = (
            self.aggregate_predictions(
                frame_results
            )
        )

        end_time = datetime.now()

        processing_seconds = (
            end_time - start_time
        ).total_seconds()

        # -----------------------------------------------------
        # 6. Final result
        # -----------------------------------------------------

        result = {
            "success": True,

            "video": {
                "path": os.path.abspath(
                    video_path
                ),
                "frames_extracted": len(
                    frames
                ),
                "usable_frames": len(
                    quality_frames
                ),
                "frames_analyzed": len(
                    frame_results
                ),
                "processing_seconds": round(
                    processing_seconds,
                    2
                )
            },

            "crop": crop,

            "visual_assessment": aggregate,

            "frame_predictions": frame_results,

            "limitations": [
                "This is a video-level aggregation of an image-only disease model.",
                "Environmental context has not yet been fused into the disease model.",
                "A high model confidence is not equivalent to a confirmed diagnosis.",
                "Independent verification is recommended before treatment decisions."
            ],

            "generated_at": (
                datetime.now().isoformat()
            )
        }

        # -----------------------------------------------------
        # 7. Save JSON
        # -----------------------------------------------------

        result_directory = os.path.join(
            output_directory,
            "video_results"
        )

        os.makedirs(
            result_directory,
            exist_ok=True
        )

        result_filename = (
            f"video_result_{timestamp}.json"
        )

        result_path = os.path.join(
            result_directory,
            result_filename
        )

        with open(
            result_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4
            )

        result["result_file"] = (
            result_path
        )

        print()
        print("=" * 70)
        print("VIDEO ANALYSIS COMPLETED")
        print("=" * 70)

        print(
            "Frames analyzed:",
            aggregate.get(
                "frames_analyzed",
                0
            )
        )

        print(
            "Visual result:",
            aggregate.get(
                "disease"
            )
        )

        print(
            "Mean confidence:",
            aggregate.get(
                "confidence_percent",
                0
            ),
            "%"
        )

        print(
            "Frame agreement:",
            aggregate.get(
                "agreement_percent",
                0
            ),
            "%"
        )

        print(
            "Video score:",
            aggregate.get(
                "video_score_percent",
                0
            ),
            "%"
        )

        print(
            "Result saved:",
            result_path
        )

        return result


# =============================================================
# STANDALONE TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CROP VIDEO ANALYZER")
    print("=" * 70)

    print()
    print(
        "This module is designed to be used by app.py"
    )

    print(
        "It requires an initialized CropHealthPredictor."
    )