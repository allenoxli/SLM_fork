import os



def main(data, split_size = 1000):
    
    num_post = f'{split_size//1000}k' if split_size >= 1000 else split_size
    out_dir = f'data/{data}_{num_post}'
    os.makedirs(out_dir, exist_ok=True)

    print(num_post)

    def split_dset(file_path, out_path):
        fin = open(file_path, 'r').readlines()
        fout = open(out_path, 'w')
        for line in fin[:split_size]:
            fout.write(line)

        fout.close()


    file_path1 = f'data/{data}/segmented.txt'
    file_path2 = f'data/{data}/unsegmented.txt'


    out_path1 = f'{out_dir}/segmented.txt'
    out_path2 = f'{out_dir}/unsegmented.txt'
    split_dset(file_path1, out_path1)
    split_dset(file_path2, out_path2)


    os.system(f'cp data/{data}/test_gold.txt {out_dir}')
    os.system(f'cp data/{data}/test.txt {out_dir}')
    os.system(f'cp data/{data}/words.txt {out_dir}')


split_size = 3000
for data in ['as', 'cityu', 'msr', 'pku']:
    main(data, split_size)


