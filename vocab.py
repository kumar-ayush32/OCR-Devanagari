# Paste your final cleaned vocab manually here
import numpy as np
clean_char_list = [
    'ँ','ं','ः','अ','आ','इ','ई','उ','ऊ','ऋ','ऌ','ऍ','ऎ','ए','ऐ','ऑ','ऒ','ओ','औ',
    'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
    'ऩ','प','फ','ब','भ','म','य','र','ऱ','ल','ळ','ऴ','व','श','ष','स','ह',
    '़','ऽ','ा','ि','ी','ु','ू','ृ','ॄ','ॅ','े','ै','ॉ','ॊ','ो','ौ','्',
    'क़','ख़','ग़','ज़','ड़','ढ़','फ़','य़',
    '।','॥',
    '०','१','२','३','४','५','६','७','८','९'
]

# CTC blank token
BLANK_TOKEN = "<BLANK>"

# Add blank at index 0
vocab = [BLANK_TOKEN] + clean_char_list

# Create mappings
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# Encode function
def encode(text):
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]  # ← add the if

# Decode function (basic)
def decode(indices):
    return "".join([idx_to_char[i] for i in indices if i != 0])

def beam_search_decode(log_probs, beam_width=10):
    """
    Pure Python beam search CTC decoding.
    log_probs: (T, vocab_size) — single sample, already on CPU
    """
    T, V = log_probs.shape
    probs = log_probs.exp().numpy()  # convert to regular probabilities

    # Each beam: (score, sequence)
    beams = [(0.0, [])]

    for t in range(T):
        new_beams = {}

        for score, seq in beams:
            for v in range(V):
                p = probs[t, v]
                if p < 1e-6:
                    continue

                new_seq = seq + [v]

                # CTC collapse: skip if same as last non-blank
                if v == 0:
                    # blank — don't add to sequence
                    collapsed = seq
                elif len(seq) > 0 and seq[-1] == v:
                    # repeat — collapse
                    collapsed = seq
                else:
                    collapsed = new_seq

                key = tuple(collapsed)
                new_score = score + np.log(p + 1e-10)

                if key in new_beams:
                    # log-sum-exp merge
                    existing = new_beams[key]
                    new_beams[key] = np.logaddexp(existing, new_score)
                else:
                    new_beams[key] = new_score

        # Keep top beam_width beams
        beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
        beams = [(score, list(seq)) for seq, score in beams[:beam_width]]

    # Best beam — decode
    best_seq = beams[0][1]
    return "".join([idx_to_char.get(i, "") for i in best_seq if i != 0])

def ctc_decode(indices):
    """
    Collapses repeated characters and removes blanks.
    Use this during inference/validation.
    Example: [3,3,0,3,5,5,0,5] -> 'अि'
    """
    result = []
    prev = None
    for i in indices:
        if i != prev:
            if i != 0:
                result.append(i)
        prev = i
    return "".join([idx_to_char[i] for i in result])

VOCAB_SIZE = len(vocab)   # 94