import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from src.graph_state import GraphState
from logger_config import logger

ID_TO_LABEL = {0: "Negative", 1: "Positive"}

class InferenceNode:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("models/sentiment_model")
        self.model = AutoModelForSequenceClassification.from_pretrained("models/sentiment_model")
        self.model.eval()

    def __call__(self, state: GraphState) -> GraphState:
        inputs = self.tokenizer(state["text"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs).item()
            conf = probs[0][pred_id].item() * 100
            label = ID_TO_LABEL[pred_id]

        print(f"[InferenceNode] Predicted label: {label} | Confidence: {conf:.0f}%")
        logger.info(f"Input Processed: '{state['text']}'")
        logger.info(f"Inference Node: Predicted {label} with confidence {conf:.2f}%")
        state["prediction"], state["confidence"] = label, conf
        state["log_entry"]["initial_prediction"] = label
        state["log_entry"]["initial_confidence"] = round(conf, 2)
        return state


class ConfidenceCheckNode:
    def __init__(self, threshold=80.0):
        self.threshold = threshold

    def __call__(self, state: GraphState):
        conf = state["confidence"]
        if conf < self.threshold:
            print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            logger.info(f"Confidence Check: {conf:.2f}% -> Fallback Triggered")
            state["clarification_needed"] = True
            state["log_entry"]["fallback_triggered"] = True
            return {"path": "fallback", "state": state}
        print("[ConfidenceCheckNode] Confidence sufficient. Accepting prediction.")
        logger.info(f"Confidence Check: {conf:.2f}% -> Accepted Prediction")
        state["final_label"] = state["prediction"]
        return {"path": "accept", "state": state}


class FallbackNode:
    def __init__(self, backup_threshold=70.0):
        self.backup_threshold = backup_threshold
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def __call__(self, state: GraphState) -> GraphState:
        print("[FallbackNode] Initial confidence low. Attempting with backup model...")
        logger.info("Fallback Node: Initial confidence low. Attempting with backup model...")

        try:
            result = self.zero_shot_classifier(state["text"], ["positive", "negative"])
            pred = result["labels"][0].capitalize()
            conf = result["scores"][0] * 100
            print(f"[FallbackNode] Backup model predicted: {pred} | Confidence: {conf:.0f}%")
            logger.info(f"Backup Model Prediction: {pred} (Confidence: {conf:.2f}%)")
            state["log_entry"]["backup_model_prediction"] = pred
            state["log_entry"]["backup_model_confidence"] = round(conf, 2)

            if conf >= self.backup_threshold:
                logger.info("Fallback Action: Accepted backup model prediction.")
                state["final_label"] = pred
                state["clarification_needed"] = False
            else:
                logger.info("Fallback Action: Backup model confidence too low. Requesting user clarification.")
                state["clarification_needed"] = True
        except Exception as e:
            logger.error(f"Error during backup model inference: {e}")
            state["clarification_needed"] = True

        return state
