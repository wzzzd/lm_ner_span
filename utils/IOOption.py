



def open_file(path, sep=' '):
    """读取文件"""
    src = []
    tgt = []
    with open(path, 'r', encoding='utf8') as f:
        content = f.readlines()#[:2000]
        tmp_src = []
        tmp_tgt = []
        for i, line in enumerate(content):
            line = line.strip().split(sep)
            # 若数据包含src和tgt
            if len(line) == 2:
                # tmp_src.append(line[0])
                tmp_src.append(line[0])
                tmp_tgt.append(line[1])
            elif i == len(content)-1:  
                # 最后一行数据     
                if tmp_src:
                    tmp_src = ' '.join(tmp_src)
                    # tmp_tgt = ' '.join(tmp_tgt)
                    src.append(tmp_src)
                    tgt.append(tmp_tgt)
            else:
                if tmp_src:
                    tmp_src = ' '.join(tmp_src)
                    # tmp_tgt = ' '.join(tmp_tgt)
                    src.append(tmp_src)
                    tgt.append(tmp_tgt)
                    tmp_src = []
                    tmp_tgt = []
    return src, tgt



def write_file(word2index, path):
    """写文件"""
    with open(path, 'w', encoding='utf8') as f:
        for k,v in word2index.items():
            string = k + ' ' + str(v) + '\n'
            f.write(string)


def write_text(text, path):
    """写文件"""
    with open(path, 'w', encoding='utf8') as f:
        for x in text:
            string = str(x) + '\n'
            f.write(string)
            
            
            