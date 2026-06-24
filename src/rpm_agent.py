import json
import os


class RPMAgent:

    def __init__(self):

        root = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        rpm_path = os.path.join(
            root,
            "data",
            "rpm_alerts.json"
        )

        with open(rpm_path, "r", encoding="utf-8") as f:
            self.alerts = json.load(f)

    def run(self, user_input: str):

        text = user_input.lower()

        matched = []

        keywords = {
            "chest": ["Blood Pressure", "Heart Rate"],
            "heart": ["Blood Pressure", "Heart Rate"],
            "breath": ["Oxygen"],
            "oxygen": ["Oxygen"],
            "diabetes": ["Glucose"],
            "glucose": ["Glucose"]
        }

        for alert in self.alerts:

            for word, devices in keywords.items():

                if word in text:

                    if any(
                        d.lower() in alert["device_type"].lower()
                        for d in devices
                    ):
                        matched.append(alert)

        return {
            "alerts_found": len(matched),
            "alerts": matched[:5]
        }
