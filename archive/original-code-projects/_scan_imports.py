import os
root = r'D:\Projects\Communication Disorders'
files = []
for dirpath, dirnames, filenames in os.walk(root):
    if '__pycache__' in dirpath:
        continue
    for fn in filenames:
        if fn.endswith('.py'):
            files.append(os.path.join(dirpath,fn))
files = sorted(set(files))
with open(r'D:\Projects\Communication Disorders\imports_summary.txt','w',encoding='utf-8') as out:
    for f in files:
        out.write('=== ' + os.path.relpath(f,root) + ' ===\n')
        try:
            with open(f,encoding='utf-8',errors='ignore') as fh:
                for line in fh:
                    s=line.strip()
                    if s.startswith('import ') or s.startswith('from '):
                        out.write(s+'\n')
        except Exception as e:
            out.write('ERR: '+str(e)+'\n')
        out.write('\n')
print('done')
