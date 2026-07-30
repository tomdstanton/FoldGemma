import sys
from Bio.PDB import MMCIFParser
import mini3di

def main():
    cif_path = "/Users/tsta0015/Programming/FoldGemma/AFDB_A6TGI3/AF-A6TGI3-F1-model_v6.cif"
    
    # Parse CIF
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("A6TGI3", cif_path)
    chain = list(structure.get_chains())[0]
    
    # Encode with mini3di
    encoder = mini3di.Encoder()
    states = encoder.encode_chain(chain)
    true_3di = encoder.build_sequence(states)
    
    # FoldGemma output sequence
    pred_3di = "DDPPRDDDLSLLVCVSCVVVPDVPLVVSLVVCVVVSSDDVNEAYEDELVVLCVDVPDPVSVVSLVSLVDDDVAPLVSLVVDCVVPVPYAYDEDALLVVLVVLLVVSVDVPHAEEYEEDDPVSVVVVVVVVVVRHRYEYDEDDPDDDDPVSLVVVLVSCVVSVHAYYYEEEDDPPLVVSLVSCCVVPVRYYYYEYDDDACPDPVDDVPDDVSCVVVVVVVVVVVVVPDVPSVVVVVVVVVVVVVVVVVV"
    
    print("FoldGemma Output:")
    print(pred_3di)
    print("\nAlphaFold (mini3di) Output:")
    print(true_3di)
    print(f"\nLengths: FoldGemma={len(pred_3di)}, AlphaFold={len(true_3di)}")
    
    if len(pred_3di) == len(true_3di):
        matches = sum(1 for a, b in zip(pred_3di, true_3di) if a == b)
        print(f"\nExact Match Rate: {matches}/{len(pred_3di)} ({matches/len(pred_3di)*100:.2f}%)")
        
        # Print differences
        diff_str = ""
        for a, b in zip(pred_3di, true_3di):
            if a == b:
                diff_str += "|"
            else:
                diff_str += "X"
        print("\nAlignment (|=match, X=mismatch):")
        print(diff_str)
    else:
        print("Lengths don't match, can't compute exact match rate easily without alignment.")

if __name__ == "__main__":
    main()
