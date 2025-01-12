r"""目前 bert 使用的 tokenizer，功能整合 SLM 版本。"""
# %%
from transformers import AutoTokenizer

class CWSHugTokenizer:

    def __init__(
        self,
        vocab_file,
        vocab=None,
        tk_hug_name: str = 'bert-base-chinese',
        max_seq_length=32,
        segment_token='  ',
        english_token='<ENG>',
        number_token='<NUM>',
        punctuation_token='<PUNC>',
        bos_token='<BOS>',
        eos_token='</s>',
        pad_token='</s>',
        unk_token='<UNK>',
        delimiters='，。',
    ):

        self.tk_hug = None
        if tk_hug_name is not None:
            bos_token = '[CLS]'
            eos_token = '[SEP]'
            pad_token = '[PAD]'
            unk_token = '[UNK]'
            additional_tokens = ['<NUM>', '<ENG>', '<PUNC>']
            self.tk_hug = AutoTokenizer.from_pretrained(tk_hug_name, additional_special_tokens=additional_tokens)
            vocab = self.tk_hug.vocab

        self.token2id = {}
        self.id2token = {}
        if vocab:
            self.token2id = vocab
            self.id2token = {v: k for k, v in vocab.items()}
        else:
            with open(vocab_file, 'r') as fin:
                for index, word in enumerate(fin):
                    word = word.strip()
                    self.token2id[word] = index
                    self.id2token[index] = word

        self.max_seq_length = max_seq_length
        self.segment_token = segment_token
        self.english_token = english_token
        self.number_token = number_token
        self.punctuation_token = punctuation_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.delimiters = delimiters

        self.eng_id = self.token2id[english_token] # 7
        self.num_id = self.token2id[number_token] # 3
        self.punc_id = self.token2id[punctuation_token] # 2

        self.bos_id = self.token2id[bos_token]
        self.eos_id = self.token2id[eos_token] # 5
        self.pad_id = self.token2id[pad_token]
        self.unk_id = self.token2id[unk_token]

    def word2id(self, word):
        if word not in self.token2id:
            return self.token2id[self.unk_token]
        else:
            return self.token2id[word]

    def id2word(self, _id):
        return self.id2token[int(_id)]

    def convert_tokens_to_ids(self, line):
        # line = list(map(self.tokenize, line))
        if self.tk_hug:
            return self.tk_hug.convert_tokens_to_ids(line)
        else:
            return [self.word2id(tid) for tid in line]

    def convert_ids_to_tokens(self, line):
        if self.tk_hug:
            return self.tk_hug.convert_ids_to_tokens(line)
        else:
            return [self.id2word(tid) for tid in line]

    def encode(self, txt, max_len: int = 0, add_special_tokens: bool = False):
        if self.tk_hug:
            padding = True if max_len == 0 else 'max_length'
            return self.tk_hug.encode(txt, max_length=max_len, padding=padding, truncation=True, add_special_tokens=add_special_tokens)

        ids = self.convert_tokens_to_ids(txt)

        # Truncate to `max_len`.
        if max_len > 0:
            trunc_len = max_len - 2 if add_special_tokens else max_len
            ids = ids[:trunc_len]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        # Padding to `max_len`.
        pad_len = max(0, max_len - len(ids))
        ids = ids + [self.__class__.pad_id] * pad_len

        return ids

    def decode(self, ids):
        return self.de_tokenize(self.convert_ids_to_tokens(ids))

    def batch_encode(self, batch_txt, max_len, add_special_tokens=False):
        return list(map(lambda txt: self.encode(txt, max_len, add_special_tokens), batch_txt))

    def batch_decode(self, batch_ids):
        return list(map(lambda ids: self.decode(ids), batch_ids))

    def __len__(self):
        return len(self.token2id)

    def sent_tokenize(self, sent):
        sent = sent.strip()
        untokenized_segments = [len(segment) for segment in sent.split(self.segment_token)]

        uchars = list(sent.replace(self.segment_token, ''))
        uchars = [(uchar, self.tokenize(uchar)) for uchar in uchars]

        outputs = [[]]
        segments = [[0]]
        curremt_seq_length = 0
        for uchar, token in uchars:
            if len(outputs[0]) == 0:
                outputs[-1].append((uchar, token))
                segments[-1][-1] += 1
                curremt_seq_length += 1
            elif outputs[-1][-1][1] == token and token in (
                self.english_token, self.punctuation_token, self.number_token
            ):
                if token in (self.english_token, self.number_token):
                    outputs[-1][-1] = (outputs[-1][-1][0] + uchar, token)
                elif token == self.punctuation_token and outputs[-1][-1][0][-1:] == uchar:
                    outputs[-1][-1] = (outputs[-1][-1][0] + uchar, token)
                else:
                    outputs[-1][-1] = (outputs[-1][-1][0] + self.segment_token + uchar, token)
            elif curremt_seq_length == self.max_seq_length - 2:
                outputs[-1].append(('', self.eos_token))
                outputs.append([])
                segments.append([1])
                outputs[-1].append((uchar, token))
                curremt_seq_length = 1
            elif len(set(outputs[-1][-1][0]) & set(self.delimiters)) > 0:
                outputs[-1].append(('', self.eos_token))
                outputs.append([])
                segments.append([1])
                outputs[-1].append((uchar, token))
                curremt_seq_length = 1
            else:
                outputs[-1].append((uchar, token))
                segments[-1][-1] += 1
                curremt_seq_length += 1

            untokenized_segments[0] -= 1
            if untokenized_segments[0] == 0:
                del untokenized_segments[0]
                segments[-1].append(0)

        outputs[-1].append(('<\\n>', self.eos_token))

        uchars = [[self.bos_token] + [uchar for uchar, token in output] for output in outputs]
        tokens = [[self.bos_token] + [token for uchar, token in output] for output in outputs]

        for _segments in segments:
            while _segments and _segments[-1] == 0:
                del _segments[-1]

        for _uchars, _tokens, _segments in zip(uchars, tokens, segments):
            assert len(_uchars) == sum(_segments) + 2
            assert len(_uchars) == len(_tokens)

        return uchars, tokens, segments

    def restore(self, uchars, segments):
        sent = []
        start = 1
        for segment in segments:
            sent.append(''.join(uchars[start:start + segment]))
            start += segment
        sent.append(uchars[-1])
        sent = self.segment_token.join(sent)
        sent = sent.replace('<\\n>', '\n')
        return sent

    @classmethod
    def _is_chinese_char(cls, uchar):
        """Checks whether uchar is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        cp = ord(uchar)

        if (
            (cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F) or (uchar in '○')
        ):
            return True
        else:
            return False

    def de_tokenize(self, tokens):
        return ''.join(tokens)

    def tokenize(self, uchar):
        '''
        Tokenize uchar. (Check whether uchar is a Chinese character)
        '''
        cp = ord(uchar)

        # Check whether uchar is a number.
        # ０１２３４５６７８９
        # 0123456789
        # ⅢⅣⅠⅡⅤ
        # ％．＋＞∶‰+㈨℃.
        if (
            (0xff10 <= cp <= 0xff19) or (0x0030 <= cp <= 0x0039) or (0x2160 <= cp <= 0x2179) or (uchar in '％．＋＞∶‰+㈨℃.')
        ):
            return self.number_token

        # Check whether uchar is an English character.
        # ａ-ｚ
        # Ａ-Ｚ
        # A-Z
        # a-z
        # alpha, beta, gamma, ...
        # ＆
        elif (
            (0xff41 <= cp <= 0xff5a) or (0xff21 <= cp <= 0xff3a) or (0x0041 <= cp <= 0x005A) or
            (0x0061 <= cp <= 0x007A) or (0x3B1 <= cp <= 0x3D0) or (uchar == '＆')
        ):
            return self.english_token

        elif self._is_chinese_char(uchar):
            return uchar

        else:
            # if self.tk_hug is not None:
            #     return uchar
            # It is a punctuation.
            return self.punctuation_token

    def full2half(self, ustring):
        r"""把字串全形轉半形。
        角字元unicode編碼從65281~65374 （十六進位制 0xFF01 ~ 0xFF5E） 半形字元unicode編碼從33~126 （十六進位制 0x21~ 0x7E）
        空格比較特殊，全形為 12288（0x3000），半形為 32（0x20）
        (半形 + 0x7e= 全形)，對空格單獨處理。
        """
        ss = []
        for s in ustring:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全形空格直接轉換
                    inside_code = 32
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        return ''.join(ss)

    def half2full(self, ustring):
        r"""把字串半形轉全形。"""
        ss = []
        for s in ustring:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 32:  # 半形空格直接轉換
                    inside_code = 12288
                elif (inside_code >= 33 and inside_code <= 126):  # 半形字元（除空格）根據關係轉化
                    inside_code += 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        return ''.join(ss)


if __name__ == '__main__':

    vocab_file = None
    max_seq_length = 32
    segment_token = '  '
    english_token = '<ENG>'
    number_token = '<NUM>'
    punctuation_token = '<PUNC>'
    bos_token = '<BOS>'
    eos_token = '</s>'
    delimiters = '，。'

    tk = CWSHugTokenizer(
        vocab=None,
        tk_hug_name='bert-base-chinese',
        vocab_file=vocab_file,
        max_seq_length=max_seq_length,
        segment_token=segment_token,
        english_token=english_token,
        number_token=number_token,
        punctuation_token=punctuation_token,
        bos_token=bos_token,
        eos_token=eos_token
    )


    sent_uchars = []
    sent_tokens = []
    sent_segments = []

    line_count = 0

    file = 'data/as/segmented.txt'

    # with open(file, 'r') as fin:
    #     for line in fin:
    #         line_count += 1
    #         uchars, tokens, segments = tk.sent_tokenize(line)
    #         sent_uchars.extend(uchars)
    #         sent_tokens.extend(tokens)
    #         sent_segments.extend(segments)

    line = '书法、、[11插花]、土风舞班，'
    uchars, tokens, segments = tk.sent_tokenize_hug(line)

    sent = line.strip()
    # uchars = list(sent.replace(segment_token, ''))
    # uchars = [(uchar, tk.tokenize(uchar)) for uchar in uchars]

    untokenized_segments = [len(segment) for segment in sent.split(segment_token)]

    uchars = list(sent.replace(segment_token, ''))
    uchars = [(uchar, tk.tokenize(uchar)) for uchar in uchars]
    print('----')
    outputs = [[]] # [sentecne1[(uchar1, token1), (uchar2, token2)], sentence2[ (ucahr1, token1)] ]
    segments = [[0]]
    curremt_seq_length = 0
    for uchar, token in uchars:
        if len(outputs[0]) == 0: # first of all.
            outputs[-1].append((uchar, token))
            segments[-1][-1] += 1
            curremt_seq_length += 1
        elif token in (english_token, punctuation_token, number_token) and outputs[-1][-1][1] == token:

            if token in (english_token, number_token):
                outputs[-1][-1] = (outputs[-1][-1][0] + uchar, token)
            elif token == punctuation_token and outputs[-1][-1][0][-1:] == uchar:
                outputs[-1][-1] = (outputs[-1][-1][0] + uchar, token)
            else:
                outputs[-1][-1] = (outputs[-1][-1][0] + segment_token + uchar, token)
        elif curremt_seq_length == max_seq_length - 2:
            outputs[-1].append(('', eos_token))
            outputs.append([])
            segments.append([1])
            outputs[-1].append((uchar, token))
            curremt_seq_length = 1
        elif len(set(outputs[-1][-1][0]) & set(delimiters)) > 0:
            outputs[-1].append(('', eos_token))
            outputs.append([])
            segments.append([1])
            outputs[-1].append((uchar, token))
            curremt_seq_length = 1
        else:
            outputs[-1].append((uchar, token))
            segments[-1][-1] += 1
            curremt_seq_length += 1

        untokenized_segments[0] -= 1
        if untokenized_segments[0] == 0:
            del untokenized_segments[0]
            segments[-1].append(0)

    outputs[-1].append(('<\\n>', eos_token))



# %%
