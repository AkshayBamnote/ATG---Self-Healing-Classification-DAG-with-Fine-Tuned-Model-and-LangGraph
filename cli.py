from src.build_graph import flow
from src.graph_state import GraphState
from logger_config import logger
import numpy as np
import matplotlib.pyplot as plt

print("--- Self-Healing Text Classifier ---")

global_all_confidences = []
global_fallback_triggered_count = 0

while True:
    text = input("\nInput: ")
    if text.lower() == "exit":
        logger.info("\n--- Final Session Statistics ---")
        logger.info(f"Total Inputs Processed: {len(global_all_confidences)}")
        logger.info(f"Total Fallbacks Triggered: {global_fallback_triggered_count}")
        if global_all_confidences:
            avg = np.mean(global_all_confidences)
            logger.info(f"Average Initial Confidence: {avg:.2f}%")
        
            # BONUS 1: Confidence Curve
            plt.plot(global_all_confidences, marker='o')
            plt.title("Confidence Over Inputs")
            plt.xlabel("Input Number")
            plt.ylabel("Confidence (%)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("logs/confidence_curve.png")
            plt.close()
            print("\n Confidence plot saved to: logs/confidence_curve.png")
            logger.info("Saved confidence curve plot: logs/confidence_curve.png")
            logger.info("\n--- End of Session ---\n")

            # BONUS 2: Fallback Frequency Histogram
            total = len(global_all_confidences)
            fallbacks = global_fallback_triggered_count
            ratio = fallbacks / total
            bar_length = int(ratio * 50)

            print("\n Fallback Frequency Histogram:")
            print(f"Fallback Rate: {ratio * 100:.0f}%")
            print("[" + "#" * bar_length + "-" * (50 - bar_length) + "]")
            print(f"{fallbacks} fallback(s) out of {total} inputs \n")

        break

    state: GraphState = {
        "text": text,
        "prediction": None,
        "confidence": None,
        "clarification_needed": False,
        "user_clarification": None,
        "final_label": None,
        "log_entry": {}
    }

    current_state = flow.invoke(state)

    if current_state["confidence"]:
        global_all_confidences.append(current_state["confidence"])

    if current_state["log_entry"].get("fallback_triggered"):
        global_fallback_triggered_count += 1

    if current_state["clarification_needed"]:
        pred = current_state["prediction"]
        print(f"[FallbackNode] Could you clarify your intent? Was this a {pred.lower()} review?")
        user_input = input("User: ")
        current_state["user_clarification"] = user_input

        if "yes" in user_input.lower() or "neg" in user_input.lower():
            current_state["final_label"] = "Negative"
        elif "no" in user_input.lower() or "pos" in user_input.lower():
            current_state["final_label"] = "Positive" if pred == "Negative" else "Negative"
        else:
            current_state["final_label"] = pred

        print(f"Final Label: {current_state['final_label']} (Corrected via user clarification)")
        logger.info(f"Final Label: {current_state['final_label']} (Corrected via user clarification)")
    else:
        if not current_state["final_label"]:
            current_state["final_label"] = current_state["prediction"]
        print(f"Final Label: {current_state['final_label']}")
        logger.info(f"Final Label: {current_state['final_label']} (Accepted)")
