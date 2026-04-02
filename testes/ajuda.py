import os
import pathlib

for idx, i in enumerate(list(os.listdir("./testes"))):
    if(idx %4 ==0 ):
        print(f"MODELO {i.split('.')}")

    if((".out" in i)):
        with open("testes/"+i, 'r') as f:
            for idx, line in enumerate(f):
                if("tokens gerados pelo modelo:" in line):
                    line =next(f,None).strip() 
                    while("token gerado pelo modelo" not in line):
                        print(line)
                        line =next(f,None).strip() 
