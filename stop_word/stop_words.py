def stop_word():
    sw = []
    with open('stop_words.txt', 'r') as fin:
        lines = fin.readlines()
    for line in lines:
        sw.append(line.replace('\n', ''))
    return sw
