import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def load_model_and_tokenizer(model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
    """
    Load the pretrained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def run_inference(test_data):
    """
    Perform inference on the test data.

    Args:
        test_data (pd.DataFrame): DataFrame containing 'premise' and 'hypothesis' columns.
    
    Returns:
        np.ndarray: Predicted class indices.
    """
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Tokenize input
    inputs = tokenizer(
        test_data['premise'].tolist(),
        test_data['hypothesis'].tolist(),
        truncation=True,
        padding=True,
        return_tensors="tf"
    )

    # Run inference
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)

    # Convert to class indices
    return tf.argmax(predictions, axis=1).numpy()