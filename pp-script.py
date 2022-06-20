import pandas, torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('dbmdz/bert-base-turkish-uncased')

df = pandas.read_csv('atasozleri-vk.csv', index_col=[0], na_filter=False)
list_of_tokens = df['yeni_parcalar'].apply(lambda t_list: t_list.replace(', ',' ')).astype(str).values

# https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
embeddings = model.encode(list_of_tokens,
                          show_progress_bar=True,
                          convert_to_tensor=True)

torch.save(embeddings, 'proverb_embeddings.pt')
