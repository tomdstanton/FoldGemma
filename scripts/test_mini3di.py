import mini3di
print(dir(mini3di))

cif_path = '/Users/tsta0015/Programming/FoldGemma/AFDB_A6TGI3/AF-A6TGI3-F1-model_v6.cif'

# Try to see if there's a main function or class
if hasattr(mini3di, 'Encoder'):
    enc = mini3di.Encoder()
    print("Encoder found")
else:
    print("No Encoder class found")
