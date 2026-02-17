from train import Classifier

classifier = Classifier()

text = "หลักสถิติพื้นฐานและการประมวลผลข้อมูลสำหรับบุคลากรภาครัฐ"

tokens = classifier._preprocess_data(text)

print(tokens)