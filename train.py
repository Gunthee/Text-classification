import pandas as pd 

df = pd.read_csv('thaimooc_courses.csv')

category1 = len(df[df['Category'] == 'คอมพิวเตอร์และเทคโนโลยี'])
category2 = len(df[df['Category'] == 'ธุรกิจและการบริหารจัดการ'])
category3 = len(df[df['Category'] == 'สุขภาพและการแพทย์'])
category4 = len(df[df['Category'] == 'ภาษาและการสื่อสาร'])

print(f"คอมพิวเตอร์และเทคโนโลยี: {category1} courses")
print(f"ธุรกิจและการบริหารจัดการ: {category2} courses") 
print(f"สุขภาพและการแพทย์: {category3} courses")
print(f"ภาษาและการสื่อสาร: {category4} courses")