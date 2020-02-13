import sentencepiece as spm

if __name__ == "__main__":
    import sys
    sp = spm.SentencePieceProcessor()
    sp.Load("../jamodel/jawiki.model")

    for line in sys.stdin:
        print(' '.join(sp.EncodeAsPieces(line)))
