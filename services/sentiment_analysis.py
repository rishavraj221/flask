import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionAnalyzer:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the EmotionAnalyzer with a pre-trained model.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name)

    def analyze_emotion(self, text):
        """
        Analyze the emotion of the input text.

        Args:
        - text (str): The input text to analyze.

        Returns:
        - dict: A dictionary containing emotions and their associated probabilities.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, padding=True, max_length=512)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the predicted emotion scores (logits)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

        # Get the emotion labels (there are 6 in this model)
        emotion_labels = [
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        ]

        # Create a dictionary with emotion labels and their corresponding probabilities
        emotion_scores = {emotion_labels[i]: probabilities[0][i].item(
        ) for i in range(len(emotion_labels))}

        # Return the emotion scores in a readable format
        return emotion_scores


# # Example usage:
# if __name__ == "__main__":
#     # Create an EmotionAnalyzer instance
#     analyzer = EmotionAnalyzer()

#     # Sample text for emotion analysis
#     text = "i will throw him out of the window, how can he say that?"

#     # Analyze the emotion
#     result = analyzer.analyze_emotion(text)

#     # Print out the emotion scores
#     print("Emotion scores:", result)
