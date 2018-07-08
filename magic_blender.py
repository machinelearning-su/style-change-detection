import pandas as pd 

print("Reading the data...\n")
df1 = pd.read_csv('/Users/atanas/Downloads/experiment/predA.csv')
df2 = pd.read_csv('/Users/atanas/Downloads/experiment/predB.csv')

models = {
            'df1' : {'name':'lightgbm_freak.csv',
                    'score':87.10,
                    'df':df1 },
            'df2' : {'name':'stacker.csv',
                    'score':72.00,
                    'df':df2 },

         }

isa_hm = 0  # harmonic
isa_am = 0  # arithmetic
isa_gm = 1  # geometric

print("Blending...\n")
for df in models.keys() :
    isa_hm += 1/(models[df]['df'].is_multi_author)
    isa_am += models[df]['df'].is_multi_author
    isa_gm *= models[df]['df'].is_multi_author

num_models = len(models)

isa_hm = num_models/isa_hm
isa_am = isa_am/num_models
isa_gm = (isa_gm)**(1/num_models)

print("Isa harmo\n")
print(isa_hm[:2])
print()
print("Isa arith\n")
print(isa_am[:2])
print()
print("Isa geom\n")
print(isa_gm[:2])

isa_fin = (isa_hm + isa_am + isa_gm)/3

sub_fin = pd.DataFrame()
sub_fin['is_multi_author'] = isa_fin
print("Writing...")
sub_fin.to_csv('/Users/atanas/Downloads/experiment/predC.csv', index=False, float_format='%.9f')
