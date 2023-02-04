if __name__ == '__main__':
    base_dir = r'D:\program\word-vector\\'
    filename = 'sgns.renmin.bigram-char'
    with open(base_dir + filename, encoding='utf-8') as in_file, open(base_dir + filename + '.correct', encoding='utf-8', mode='w') as out_file:
        for line in in_file:
            elements = line.strip().split()
            vector = elements[-300:]
            word = elements[:-300]
            new_line = ''.join(word) + ' ' + ' '.join(vector) + '\n'
            out_file.write(new_line)