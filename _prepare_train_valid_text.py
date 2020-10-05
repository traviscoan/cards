import pandas as pd

train = pd.read_json('data/train.json')
train = train[['Paragraph_Text', 'sub_claim_combined']]
train.columns = ['text', 'label']
train.to_csv('data/train.csv', index=False)

valid = pd.read_json('data/valid.json')
valid = valid[['Paragraph_Text', 'sub_claim_combined']]
valid.columns = ['text', 'label']
valid.to_csv('data/valid.csv', index=False)

test = pd.read_json('data/test.json')
test = test[['Paragraph_Text', 'sub_claim_combined']]
test.columns = ['text', 'label']
test.to_csv('data/test.csv', index=False)

df = pd.read_csv('data/full_data/extracted_paras.csv')
df['pid'] = df.index

df = df[['pid', 'date', 'domain', 'ctt_status', 'text']]

df.to_csv('data/full_data/paragraphs_small.csv')