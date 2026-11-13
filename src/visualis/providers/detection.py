"""Object Detection Provider Architecture.

Provider interface for interchangeable detection backends:
- ONNX Runtime (YOLO models)
- External REST API providers (future)

Detection results include confidence thresholds and NMS (non-maximum suppression).
No model weights are bundled - users must supply their own ONNX models or
configure external providers via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from visualis.core.schemas import Detection, Box, ObjectCounts, OverlappingPairs


@dataclass
class DetectionConfig:
    """Configuration for object detection modules."""
    model_path: str | None = None
    conf_threshold: float = 0.35
    nms_iou_threshold: float = 0.45
    max_detections: int = 100
    input_size: tuple[int, int] = (640, 640)
    label_map: dict[str, int] | None = None  # name -> class_id
    source_name: str = "onnx"


class DetectionBackend(Protocol):
    """Protocol that all detection backends must implement."""
    
    def detect(
        self,
        image_data: Any,
        config: DetectionConfig,
    ) -> list["Detection"]:
        """Run detection on an image. Returns list of detections."""
        ...
    
    def available(self) -> bool:
        """Check if the backend can be initialized."""
        ...


class ONNXYOLODetection:
    """ONNX Runtime YOLO detector.
    
    Supports YOLOv5/v8 ONNX exports in CHW format.
    Model should be a standard YOLO export with:
    - Input: [1, 3, 640, 640] float32 normalized to [0, 1]
    - Output: [1, 84, 8400] or similar structure
    
    See https://docs.ultralytics.com/modes/export/ for export instructions.
    """
    
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._session: Any = None
        self._initialized = False
    
    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False
    
    def _load_model(self) -> None:
        """Load ONNX model into inference session."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("onnxruntime is required for ONNX detection") from e
        
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self._initialized = True
    
    def detect(
        self,
        image_data: Any,
        config: DetectionConfig,
    ) -> list["Detection"]:
        """Run YOLO detection with NMS post-processing."""
        if not self.available():
            raise ImportError("onnxruntime not installed")
        
        if not self._initialized:
            self._load_model()
        
        # TODO: Implement actual detection logic when model is provided
        # This is a skeleton - real implementation would:
        # 1. Preprocess image (resize, normalize)
        # 2. Run inference
        # 3. Post-process outputs (confidence thresholding + NMS)
        # 4. Return Detection objects
        
        return []  # Placeholder until model provided


def get_detection_backend(provider: str, config: DetectionConfig) -> DetectionBackend:
    """Factory function to get appropriate detection backend."""
    if provider == "onnx":
        if not ONNXYOLODetection().available():
            raise ImportError("onnxruntime required for ONNX detection")
        return ONNXYOLODetection(config.model_path)
    else:
        raise ValueError(f"Unknown detection provider: {provider}")


# Scene graph construction utilities

def build_scene_graph(
    detections: list["Detection"],
    ocr_results: list[Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Build a scene graph from detections and OCR results.
    
    Creates nodes for each entity and edges based on spatial relationships.
    """
    from visualis.core.schemas import SceneGraph, SceneNode, Relation, EvidenceSource, SpatialRelation
    
    nodes: list[SceneNode] = []
    relations: list[Relation] = []
    
    # Add detection nodes
    for i, det in enumerate(detections):
        nodes.append(SceneNode(
            id=i,
            kind="detection",
            label=det.label,
            box=det.box,
            source=EvidenceSource.OBJECT_DETECTION,
            confidence=det.confidence,
        ))
    
    # Add OCR text nodes
    for i, block in enumerate(ocr_results):
        nodes.append(SceneNode(
            id=len(detections) + i,
            kind="ocr_block",
            label=block.text[:50] if hasattr(block, 'text') else str(block),
            box=block.box if hasattr(block, 'box') else None,
            source=EvidenceSource.OCR_DETECTION,
            confidence=getattr(block, 'confidence', 0.5),
        ))
    
    # Compute spatial relations
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes[i+1:], i+1):
            if node_a.box is None or node_b.box is None:
                continue
            
            rel = compute_spatial_relation(node_a.box, node_b.box)
            if rel:
                relations.append(Relation(
                    kind=rel,
                    subject=node_a.label,
                    subject_id=node_a.id,
                    object=node_b.label,
                    object_id=node_b.id,
                    evidence=EvidenceSource.GEOMETRIC_ANALYSIS,
                    confidence=0.9,
                ))
    
    return SceneGraph(
        nodes=nodes,
        relations=relations,
        composition=f"{len(detections)} detected objects, {len(ocr_results)} OCR blocks",
    ).model_dump()


def compute_spatial_relation(box_a: "Box", box_b: "Box") -> str | None:
    """Determine spatial relationship between two boxes."""
    from visualis.core.schemas import SpatialRelation
    # Check containment
    if box_a.contains_box(box_b):
        return SpatialRelation.CONTAINS
    if box_b.contains_box(box_a):
        return SpatialRelation.CONTAINS
    
    # Check overlap
    if box_a.intersects(box_b):
        return SpatialRelation.OVERLAPS
    
    # Check adjacency
    center_a = box_a.center
    center_b = box_b.center
    dist = ((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)**0.5
    max_dim = max(box_a.width, box_a.height, box_b.width, box_b.height)
    
    if dist < max_dim * 1.5:
        # Closer than one object width - adjacent or near
        if abs(center_a[0] - center_b[0]) < max_dim * 0.5:
            return SpatialRelation.ADJACENT if abs(center_a[1] - center_b[1]) < max_dim * 0.5 else None
        return SpatialRelation.NEAR
    
    # Determine relative position
    if center_a[0] < center_b[0]:
        return SpatialRelation.LEFT_OF
    elif center_a[0] > center_b[0]:
        return SpatialRelation.RIGHT_OF
    elif center_a[1] < center_b[1]:
        return SpatialRelation.ABOVE
    else:
        return SpatialRelation.BELOW
