import os
import pandas as pd
import json
from datetime import datetime

# Constants
HISTORY_FILE = "chat_history.csv"

def save_chat_history(question, answer, timestamp=None):
    """Save a question-answer pair to chat history"""
    try:
        # Create timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Create dataframe for new entry
        new_entry = pd.DataFrame({
            "timestamp": [timestamp],
            "question": [question],
            "answer": [answer]
        })
        
        # Load existing history or create new if not exists
        if os.path.exists(HISTORY_FILE):
            history = pd.read_csv(HISTORY_FILE)
            history = pd.concat([history, new_entry], ignore_index=True)
        else:
            history = new_entry
            
        # Save to CSV
        history.to_csv(HISTORY_FILE, index=False)
        return True
        
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return False

def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists(HISTORY_FILE):
            return pd.read_csv(HISTORY_FILE)
        else:
            return pd.DataFrame(columns=["timestamp", "question", "answer"])
            
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return pd.DataFrame(columns=["timestamp", "question", "answer"])

def export_chat_history(format="csv"):
    """Export chat history in the specified format"""
    try:
        if not os.path.exists(HISTORY_FILE):
            return None
            
        history = pd.read_csv(HISTORY_FILE)
        
        if format.lower() == "csv":
            return history.to_csv(index=False)
        elif format.lower() == "json":
            return history.to_json(orient="records")
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        print(f"Error exporting chat history: {e}")
        return None

def clear_chat_history():
    """Clear all chat history"""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            return True
        return False
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return False