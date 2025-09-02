from data_utils import CSGridMLMDataset
from GridMLM_tokenizers import CSGridMLMTokenizer

train_dir = '/media/maindisk/data/hooktheory_midi_hr/all12_train'

tokenizer = CSGridMLMTokenizer(fixed_length=80, quantization='4th', intertwine_bar_info=True, trim_start=False)
train_dataset = CSGridMLMDataset(train_dir, tokenizer, name_suffix='MLMH_bar_qt')