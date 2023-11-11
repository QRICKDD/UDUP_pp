import torch
import torch.nn as nn
ctc_loss=nn.CTCLoss()
log_probs = torch.randn(50, 1, 20).log_softmax(2).requires_grad_()#[TNC]
targets = torch.randint(1, 37, (1, 30), dtype=torch.long)

input_lengths = torch.full((16,), 50, dtype=torch.long)

target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

loss.backward()



"""



outputs=torch.log_softmax(outputs,dim=2)
>>>outputs.shape
>>>torch.Size([10, 26, 37])
bsz, seq_len = outputs.size(0), outputs.size(1) #10, 26
outputs_for_loss = outputs.permute(1, 0, 2).contiguous() 
>>>outputs_for_loss.shape
>>>torch.Size([26, 10, 37])
targets = [
            data_sample.gt_text.indexes[:seq_len]
            for data_sample in data_samples
        ]
>>>[tensor([31, 18, 27, 16, 18, 23], dtype=torch.int32), tensor([10, 23, 18, 23, 16], dtype=torch.int32),....
target_lengths = torch.IntTensor([len(t) for t in targets])
>>>tensor([6, 5, 8, 6, 7, 7, 8, 5, 5, 5])
target_lengths = torch.clamp(target_lengths, max=seq_len).long()
>>>tensor([6, 5, 8, 6, 7, 7, 8, 5, 5, 5])
input_lengths = torch.full(
            size=(bsz, ), fill_value=seq_len, dtype=torch.long)
>>>tensor([26, 26, 26, 26, 26, 26, 26, 26, 26, 26])
if self.flatten:
    targets = torch.cat(targets)
>>>targets
>>>tensor([31, 18, 27, 16, 18, 23, 10, 23, 18, 23, 16,  0,  3,  0,  9,  2,  0,  0,
         9, 10, 29, 29, 10, 12, 20, 10, 22, 14, 27, 18, 12, 10, 25, 10, 12, 18,
        15, 18, 12, 13, 10, 31, 18, 13, 28, 24, 23, 17, 24, 29, 14, 21, 17, 24,
        29, 14, 21, 16, 27, 10, 23, 13], dtype=torch.int32)

loss_ctc = self.ctc_loss(outputs_for_loss, targets, input_lengths,
                                 target_lengths)

"""


"""
关于如何得到target
        for data_sample in data_samples:
            text = data_sample.gt_text.item
            if self.letter_case in ['upper', 'lower']:
                text = getattr(text, self.letter_case)()
            indexes = self.dictionary.str2idx(text)
            indexes = torch.IntTensor(indexes)
            data_sample.gt_text.indexes = indexes
        return data_samples

"""