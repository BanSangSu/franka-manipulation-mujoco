"""
Florence2Segmentor – uses Microsoft Florence-2 for open-vocabulary
segmentation of a target object (e.g. "mustard bottle") in an RGB image.

The model runs on GPU via HuggingFace transformers:
    model_id = "microsoft/Florence-2-base-ft"

Produces a binary mask (H, W) for the target object.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class Florence2Segmentor:
    """Segment a target object in an image using Florence-2.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for Florence-2.
    device : str
        Torch device string ("cuda", "cpu").
    target_object : str
        Natural-language description of the object to segment
        (e.g. "mustard bottle").
    """

    def __init__(
        self,
        target_object: str = "mustard bottle",
        model_id: str = "microsoft/Florence-2-base-ft",
        device: str | None = None,
    ):
        self.target_object = target_object
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy loading so import is fast
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading Florence-2 model: %s ...", self.model_id)
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.float32
        ).to(self.device)
        self._model.eval()
        logger.info("Florence-2 loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def segment(
        self,
        rgb: np.ndarray,
        target_object: str | None = None,
        return_bbox: bool = False,
    ) -> np.ndarray:
        """Return a binary mask (H,W bool) of the target object.

        Steps
        -----
        1. Open-vocabulary detection  → bounding box(es)
        2. Referring-expression segmentation inside the best bbox → mask
        If Florence-2 does not support pixel-level masks for the task,
        we fall back to a rectangular mask from the bbox.

        Parameters
        ----------
        rgb : ndarray (H, W, 3) uint8
            The input image.
        target_object : str, optional
            Override the default target object.
        return_bbox : bool
            If True, also return the [x1,y1,x2,y2] bounding box.

        Returns
        -------
        mask : ndarray (H, W) bool
            Binary segmentation mask of the target object.
        """
        self._load_model()
        obj = target_object or self.target_object
        H, W = rgb.shape[:2]

        # 1. Try REFERRING_EXPRESSION_SEGMENTATION
        mask = self._referring_segmentation(rgb, obj)
        bbox = None

        # 2. If it fails, fallback to bboxes
        if mask is None or not np.any(mask):
            bbox = self._detect_bbox(rgb, obj)
            mask = np.zeros((H, W), dtype=bool)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask[y1:y2, x1:x2] = True
            else:
                logger.warning("Florence-2 could not detect '%s' – returning empty mask", obj)
        elif return_bbox:
            # We have a robust mask, compute bbox directly from it
            coords = np.argwhere(mask)
            if len(coords) > 0:
                y0, y1 = coords[:, 0].min(), coords[:, 0].max()
                x0, x1 = coords[:, 1].min(), coords[:, 1].max()
                bbox = [int(x0), int(y0), int(x1), int(y1)]

        return (mask, bbox) if return_bbox else mask

    def detect(
        self,
        rgb: np.ndarray,
        target_object: str | None = None,
    ) -> Optional[list]:
        """Detect the target object and return its bounding box.

        Parameters
        ----------
        rgb : ndarray (H, W, 3) uint8
            The input image.
        target_object : str, optional
            Override the default target object.

        Returns
        -------
        bbox : list of int or None
            [x1, y1, x2, y2] bounding box if detected, else None.
        """
        self._load_model()
        obj = target_object or self.target_object
        return self._detect_bbox(rgb, obj)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_florence(self, image: Image.Image, task: str, text_input: str = ""):
        """Run a Florence-2 task and return the parsed result dict."""
        prompt = task + text_input
        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        result = self._processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        return result

    def _detect_bbox(self, rgb: np.ndarray, obj: str) -> Optional[list]:
        """Detect the target object and return [x1,y1,x2,y2] or None."""
        pil_img = Image.fromarray(rgb)
        task = "<OPEN_VOCABULARY_DETECTION>"
        # task = "<CAPTION_TO_PHRASE_GROUNDING>" # fail...
        result = self._run_florence(pil_img, task, text_input=obj)

        det = result.get(task, {})
        bboxes = det.get("bboxes", [])
        labels = det.get("bboxes_labels", det.get("labels", []))

        if not bboxes:
            return None

        # Find the best matching label
        obj_lower = obj.lower()
        best_idx = -1
        for i, lbl in enumerate(labels):
            if obj_lower in str(lbl).lower():
                best_idx = i
                break

        if best_idx == -1:
            return None

        box = bboxes[best_idx]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        return [x1, y1, x2, y2]

    def _referring_segmentation(self, rgb: np.ndarray, obj: str) -> Optional[np.ndarray]:
        """Attempt referring-expression segmentation; return (H,W) bool mask or None."""
        pil_img = Image.fromarray(rgb)
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        try:
            result = self._run_florence(pil_img, task, text_input=obj)
            seg = result.get(task, {})
            polygons = seg.get("polygons", [])
            if not polygons or len(polygons) == 0:
                return None

            H, W = rgb.shape[:2]
            mask = np.zeros((H, W), dtype=bool)

            for poly_group in polygons:
                for poly in poly_group:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    temp_img = Image.new("L", (W, H), 0)
                    draw = ImageDraw.Draw(temp_img)
                    draw.polygon(pts.flatten().tolist(), fill=1)
                    temp = np.array(temp_img, dtype=np.uint8)
                    mask = np.logical_or(mask, temp.astype(bool))
            return mask
        except Exception as e:
            logger.debug("Referring segmentation failed: %s", e)
            return None
