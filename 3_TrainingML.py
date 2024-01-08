from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Initializing embedding & recognizer
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

print("Loading face embeddings...")
try:
    with open(embeddingFile, "rb") as file:
        data = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Embedding file '{embeddingFile}' not found.")
    exit()

# Verify the structure of the loaded data
if "names" not in data or "embeddings" not in data:
    print("Error: Invalid data structure in the embedding file.")
    exit()

print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Save the trained recognizer
with open(recognizerFile, "wb") as file:
    pickle.dump(recognizer, file)

# Save the label encoder
with open(labelEncFile, "wb") as file:
    pickle.dump(labelEnc, file)

print("Training completed and models saved.")
