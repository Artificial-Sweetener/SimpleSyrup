"""Validate decoded image effects for every P9.4 LoRA comparison."""

from __future__ import annotations

from pathlib import Path

from tools.decoded_image_comparison import compare_decoded_rgb_images

from .matrix import TextEncoderLoraCase, TextEncoderLoraSpatialMode


class TextEncoderLoraImageValidator:
    """Require every declared text-encoder LoRA to change decoded pixels."""

    def validate(
        self,
        definitions: tuple[TextEncoderLoraCase, ...],
        images: dict[str, Path],
    ) -> dict[str, dict[str, int | str]]:
        """Compare each LoRA output with its exact spatial/model reference."""

        evidence: dict[str, dict[str, int | str]] = {}
        for case in definitions:
            if case.comparison_case_id is None:
                continue
            output = _required_image(images, case.case_id)
            reference = _required_image(images, case.comparison_case_id)
            comparison = compare_decoded_rgb_images(output, reference)
            expected_size = (
                1024 if case.spatial_mode is TextEncoderLoraSpatialMode.FULL else 1536
            )
            if (comparison.width, comparison.height) != (
                expected_size,
                expected_size,
            ):
                raise ValueError(
                    f"P9.4 case {case.case_id!r} has unexpected image dimensions."
                )
            if comparison.changed_pixels == 0:
                raise ValueError(
                    f"P9.4 case {case.case_id!r} has no decoded LoRA effect "
                    f"relative to {case.comparison_case_id!r}."
                )
            evidence[case.case_id] = {
                "comparison_case_id": case.comparison_case_id,
                **comparison.as_json(),
            }
        return evidence


def _required_image(images: dict[str, Path], case_id: str) -> Path:
    """Return one existing labeled output artifact."""

    path = images.get(case_id)
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError(f"P9.4 image {case_id!r} is missing.")
    return path


TEXT_ENCODER_LORA_IMAGE_VALIDATOR = TextEncoderLoraImageValidator()
