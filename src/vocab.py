from collections import defaultdict
from nltk.tokenize import word_tokenize


class Vocabulary(object):

    def __init__(self):
        self._itos = []
        self._stoi = {}

    @staticmethod
    def create(sentences, vocab_size):
        self = Vocabulary()
        self._itos = ['<UNK>']
        self._stoi = {'<UNK>': 0}

        # cleaning
        sentences = sentences.encode('ascii', 'ignore').decode()

        freq = defaultdict(int)
        for w in word_tokenize(sentences):
            freq[w.lower()] += 1

        for k, v in sorted(freq.items(), key=lambda x: -x[1])[:vocab_size - 1]:
            self._stoi[k] = len(self._itos)
            self._itos.append(k)

        return self

    def __len__(self):
        return len(self._itos)

    def exists(self, word):
        return word in self._stoi

    def stoi(self, word):
        if self.exists(word):
            return self._stoi[word]
        return 0

    def itos(self, word_id):
        return self._itos[word_id]

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write('\n'.join(self._itos))

    @staticmethod
    def load(fname):
        self = Vocabulary()
        with open(fname) as f:
            for line in f:
                self._stoi[line.rstrip()] = len(self._itos)
                self._itos.append(line.rstrip())

        return self

    def sentence_to_ids(self, sentence):
        ids = []
        for word in word_tokenize(sentence):
            ids.append(self.stoi(word))

        return ids

    def ids_to_sentence(self, ids):
        sentence = []
        for word_id in ids:
            sentence.append(self.itos(word_id))

        return ' '.join(sentence)


if __name__ == '__main__':
    s = """
    THE COMPANY    Employer is a midstream service provider to the onshore Oil and Gas markets.  It is a a fast growing filtration technology company providing environmentally sound solutions to the E&P’s for water and drilling fluids management and recycling.    THE POSITION    The North Dakota Regional Technical Sales Representative reports directly to the VP of Sales and covers a territory that includes North Dakota and surrounding areas of South Dakota, Wyoming and Montana.  Specific duties for this position include but are not limited to:     Building sales volume within the established territory from existing and new accounts   Set up and maintain a strategic sales plan for the territory   Present technical presentations, product demonstrations & training   Maintain direct contact with customers, distributors and representatives   Prospect new customer contacts and referrals   Gather and record customer & competitor information   Provide accurate and updated forecasts for the territory   Identify new product opportunities   Build long-term relationships with customers, reps & distributors    CANDIDATE REQUIREMENT    The ideal candidate will possess technical degree, preferably in the oil & gas discipline and/or 5+ years of experience preferably with exploration and production companies (midstream service companies are a big plus).      Other desired requirements include but are not limited to:     Consistent record of superior sales results & experience closing sales   Proven ability to cold-call, develop relationships   Excellent written and verbal communication skills.    Strong computer skills, including Word, Excel, PowerPoint, e-mail, etc.   Strong work ethic and ability to work independently.   Must be willing to develop new business – not just maintain current accounts   Ability to travel extensively throughout assigned region    If you are a self-motivated individual with strong engineering, and leadership skills and a desire to build a stronger, more advanced organization we encourage you to apply.      Position is located in North Dakota, but sales representative could live as far away as Casper, Wyoming or Billings, Montana.     Successful candidates must pass a post offer background and drug screen.    EOE
    """
    # vocab = Vocabulary.create(s, 10)
    vocab = Vocabulary.load('vocab.txt')
    print(vocab._stoi)
    vocab.save('vocab.txt')
