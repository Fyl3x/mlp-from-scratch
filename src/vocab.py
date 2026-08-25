import pandas as pd



class Vocab:
    def __init__(self,min_count,max_len,texts):
        self.texts = texts
        self.min_count = min_count
        self.max_len = max_len
        self.count = {} # this is a counter that has {word : # of Occurences}
        for text in self.texts:
            for w in text.split():
                w = w.lower() # lower case to not mix up anything
                self.count[w] = self.count.get(w,0) + 1 # basic dictionary method to increase count
        # self.count gives us the word and how many times it occured in the narratives
        # now word2idx uses counts to see which qualifies to be a token in itself (count >= min_count)
        self.word2idx = {}
        self.word2idx["<UNK>"] = 1 # Special CHAR
        self.word2idx["<PAD>"] = 0 # Special CHAR
        i = 2 # Start count at 0 
        for key,val in self.count.items():
            if val >= self.min_count:
                self.word2idx[key] = i
                i+=1
        # Loop through {word:count} take word and test count if its eligible then give it i's component
                
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}
        
    def translate(self,input_text):
        list_text = input_text.lower().split()
        result = []
        for w in list_text:
            if self.word2idx.get(w) is None:
                result.append(self.word2idx["<UNK>"])
            else:
                result.append(self.word2idx[w])
        return result

    
    def encode(self,text):
        text_list = self.translate(text)
        result = text_list.copy()
        if len(text_list) > self.max_len:
            result = text_list[:self.max_len]
        elif len(text_list) < self.max_len:
            for w in range(self.max_len-len(text_list)):
                result.append(self.word2idx["<PAD>"])
        else:
            result = result
        return result

    
    def decode(self,sequence):
        result_text = ""
        for idx in sequence:
            word = self.idx2word.get(idx)
            if word == "<PAD>":
                continue
            else:
                result_text += word + " "
        return result_text
    
if __name__ == "__main__":

    test_data = pd.read_csv("./data/processed/train.csv")


    vocabulary = Vocab(2,400,test_data["Consumer complaint narrative"])


    narrative = test_data["Consumer complaint narrative"][1]
    encoded = vocabulary.encode(narrative)
    decoded = vocabulary.decode(encoded)
    print(len(encoded))   # should be exactly 400
    print(decoded)        # should roughly resemble the original narrative, minus punctuation/casing