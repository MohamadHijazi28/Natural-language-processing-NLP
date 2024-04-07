import sys

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
model.eval()

#file_path = 'masked_sentences.txt'
file_path = sys.argv[1]
output_folder_path = sys.argv[2]
output_file_path = output_folder_path + '\\dictabert_results.txt'

with open(file_path, 'r', encoding='utf-8') as file, open(output_file_path, 'w', encoding='utf-8') as output_file:
    lines = file.readlines()

    for line in lines:
        # Replace [*] with [MASK]
        modified_line = line.strip().replace('[*]', '[MASK]')
        input_ids = tokenizer.encode(modified_line, return_tensors='pt')
        mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
        output = model(input_ids)

        predicted_tokens = []
        for mask_index in mask_token_index:
            top_1 = torch.topk(output.logits[0, mask_index, :], 1)[1]
            predicted_token = tokenizer.convert_ids_to_tokens(top_1)[0]
            predicted_tokens.append(predicted_token)

            # Replace first [MASK] with the predicted token in the modified line for display
            modified_line = modified_line.replace('[MASK]', predicted_token, 1)

        # Write to output file
        output_file.write(f'Original sentence: {line.strip()}\n')
        output_file.write(f'DictaBERT sentence: {modified_line}\n')
        output_file.write(f'DictaBERT tokens: {", ".join(predicted_tokens)}\n\n')
